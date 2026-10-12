"""Backend selection must override HF's import-time optional-package dispatch."""

import functools
import inspect
import os
import sys
import types

import pytest
import torch
from test_hf_continuation import tiny_engine
from transformers.models.qwen3_5 import modeling_qwen3_5 as module

import rcuf.engine as implementation
from rcuf.engine import check_cache_paths, configure_delta_backend
from rcuf.state import recurrent_states


def test_torch_overrides_import_time_external_dispatch(monkeypatch):
    names = (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    monkeypatch.setattr(implementation, "_TORCH_FUNCTIONS", None)
    for name in names:
        original = inspect.unwrap(getattr(module, name))

        @functools.wraps(original)
        def automatic_gpu_dispatch(*args, **kwargs):
            pytest.fail("A CPU test reached an optional GPU implementation")

        monkeypatch.setattr(module, name, automatic_gpu_dispatch)
    # Executes actual HF CPU convolutions, chunk prefill and single-token recurrence.
    engine = tiny_engine()
    cache, _ = engine.forward([3, 4, 5])
    engine.forward([6], cache)


def test_fla_to_torch_round_trip_restores_reference_functions(monkeypatch):
    configure_delta_backend("torch")
    originals = {
        name: getattr(module, name)
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule")
    }
    calls = []

    def external(q, k, v, **kwargs):
        calls.append((q.is_contiguous(), kwargs))
        return q, None

    fake = types.ModuleType("fla.ops.gated_delta_rule")
    fake.chunk_gated_delta_rule = fake.fused_recurrent_gated_delta_rule = external
    monkeypatch.setitem(sys.modules, "fla.ops.gated_delta_rule", fake)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name, function in originals.items():
        monkeypatch.setattr(module, name, function)
    configure_delta_backend("fla")
    tensor = torch.zeros(1, 2, 1, 8)
    module.torch_chunk_gated_delta_rule(tensor, tensor, tensor, chunk_size=64, cu_seqlens=None)
    assert calls == [(True, {})]
    configure_delta_backend("torch")
    assert all(getattr(module, name) is function for name, function in originals.items())


@pytest.mark.skipif(os.environ.get("RCUF_GPU_TESTS") != "1", reason="Set RCUF_GPU_TESTS=1 on the A100 server")
def test_actual_cuda_fla_cache_continuation_and_edit():
    assert torch.cuda.is_available(), "RCUF_GPU_TESTS requires a CUDA GPU"
    # Use real-model key/value head dimensions for FLA, with tiny random model weights.
    engine = tiny_engine(chunk_size=11, head_k=128, head_v=128)
    engine.device = torch.device("cuda:0")
    engine.model = engine.model.to(device=engine.device, dtype=torch.bfloat16)
    try:
        configure_delta_backend("fla")
        report = check_cache_paths(engine, list(range(3, 61)), split=37)
        assert report["perturbation_max_logits_effect"] > 1e-6
        cache, _ = engine.forward([3, 4, 5])
        assert all(state.dtype == torch.float32 for state in recurrent_states(cache).values())
    finally:
        configure_delta_backend("torch")
