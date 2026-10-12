"""Explicit cached prefill and greedy decoding; no stale first-token logits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import platform
import time
from pathlib import Path

import torch

from .io import digest
from .state import capture_states, make_anchor, recurrent_states

_TORCH_FUNCTIONS: dict | None = None


def configure_delta_backend(backend: str) -> None:
    global _TORCH_FUNCTIONS
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    if backend not in {"torch", "fla"}:
        raise ValueError("delta_backend must be 'fla' or 'torch'.")
    if _TORCH_FUNCTIONS is None:
        # HF's decorated functions may automatically dispatch to installed GPU packages.
        # Retain the undecorated reference implementations before installing our wrappers.
        _TORCH_FUNCTIONS = {
            name: inspect.unwrap(getattr(module, name))
            for name in (
                "torch_chunk_gated_delta_rule",
                "torch_recurrent_gated_delta_rule",
                "causal_conv1d_fn",
                "causal_conv1d_update",
            )
        }
    # Both modes intentionally use the HF convolution reference path.
    for name in ("causal_conv1d_fn", "causal_conv1d_update"):
        setattr(module, name, _TORCH_FUNCTIONS[name])
    if backend == "torch":
        for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
            setattr(module, name, _TORCH_FUNCTIONS[name])
        return
    if not torch.cuda.is_available():
        raise RuntimeError("FLA requires CUDA. For CPU checks use delta_backend=torch.")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    # Use the official FLA operator functions, preserving HF's surrounding projections/conv/cache code.
    # HF calls these globals with query/key/value names; FLA calls the tensors q/k/v.
    def chunk(query, key, value, **kwargs):
        kwargs.pop("chunk_size", None)
        kwargs.pop("cu_seqlens", None)  # this runner has batch=1, no packing or padding
        return chunk_gated_delta_rule(query.contiguous(), key.contiguous(), value.contiguous(), **kwargs)

    def recurrent(query, key, value, **kwargs):
        kwargs.pop("cu_seqlens", None)
        return fused_recurrent_gated_delta_rule(
            query.contiguous(), key.contiguous(), value.contiguous(), **kwargs
        )

    module.torch_chunk_gated_delta_rule = chunk
    module.torch_recurrent_gated_delta_rule = recurrent


def environment() -> dict:
    versions = {}
    for package in (
        "torch",
        "transformers",
        "flash-linear-attention",
        "triton",
        "numpy",
        "langid",
        "datasets",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "modeling_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
    }


class Engine:
    def __init__(self, config: dict, tokenizer_sha256: str | None = None):
        import transformers
        from transformers import AutoConfig, AutoTokenizer, Qwen3_5ForCausalLM

        if transformers.__version__ != "5.19.0":
            raise RuntimeError("This implementation is pinned to transformers==5.19.0.")
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu" and config["delta_backend"] != "torch":
            raise RuntimeError("GPU experiments require CUDA. Use the dedicated CPU tests on this computer.")
        configure_delta_backend(config["delta_backend"])
        full_config = AutoConfig.from_pretrained(config["model"], revision=config["model_revision"])
        text_config = full_config.get_text_config()
        if text_config.model_type != "qwen3_5_text" or "linear_attention" not in text_config.layer_types:
            raise ValueError("This runner requires the dense Qwen3.5 GDN hybrid, not Qwen3-4B.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model"], revision=config["model_revision"], use_fast=True
        )
        if tokenizer_sha256 and digest(self.tokenizer.backend_tokenizer.to_str()) != tokenizer_sha256:
            raise ValueError("Tokenizer differs from the prepared data.")
        torch.manual_seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        self.model, loading_info = Qwen3_5ForCausalLM.from_pretrained(
            config["model"],
            revision=config["model_revision"],
            config=text_config,
            dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
            attn_implementation=config["attention_backend"],
            output_loading_info=True,
        )
        if (
            loading_info.get("missing_keys")
            or loading_info.get("mismatched_keys")
            or loading_info.get("error_msgs")
        ):
            raise RuntimeError(f"Text checkpoint did not load completely: {loading_info}")
        self.model = self.model.to(self.device).eval()
        self.loading_audit = {
            key: sorted(str(item) for item in value)
            for key, value in loading_info.items()
            if isinstance(value, (list, set, tuple))
        }
        self.model.requires_grad_(False)
        self.model_config_sha256 = digest(self.model.config.to_dict())
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        if self.tokenizer.eos_token_id is not None:
            self.eos.add(self.tokenizer.eos_token_id)
        end_of_message = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if end_of_message is not None and end_of_message != self.tokenizer.unk_token_id:
            self.eos.add(end_of_message)

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def forward(self, token_ids: list[int], cache=None):
        if not token_ids:
            raise ValueError("Cannot forward an empty sequence.")
        inputs = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        start = cache.get_seq_length() if cache is not None else 0
        positions = torch.arange(start, start + len(token_ids), device=self.device).unsqueeze(0)
        output = self.model(
            input_ids=inputs, position_ids=positions, past_key_values=cache, use_cache=True, logits_to_keep=1
        )
        return output.past_key_values, output.logits[0, -1].float()

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], cache=None):
        if not token_ids:
            raise ValueError("Cannot prefill an empty sequence.")
        self.synchronize()
        started = time.perf_counter()
        chunk_size = self.config["prefill_chunk_size"]
        logits = None
        for start in range(0, len(token_ids), chunk_size):
            cache, logits = self.forward(token_ids[start : start + chunk_size], cache)
        self.synchronize()
        return cache, logits, time.perf_counter() - started

    @torch.inference_mode()
    def generate(self, cache, suffix: list[int]) -> dict:
        if not suffix:
            raise ValueError("Generation requires an unprocessed suffix after the intervention.")
        self.synchronize()
        started = time.perf_counter()
        cache, logits, suffix_seconds = self.prefill(suffix, cache)
        generated = []
        stopped = False
        for _ in range(self.config["max_new_tokens"]):
            token = int(logits.argmax())
            if token in self.eos:
                stopped = True
                break
            generated.append(token)
            cache, logits = self.forward([token], cache)
        self.synchronize()
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True),
            "generated_ids": generated,
            "generated_tokens": len(generated),
            "truncated": not stopped,
            "suffix_tokens": len(suffix),
            "suffix_seconds": suffix_seconds,
            "continuation_seconds": time.perf_counter() - started,
        }

    @torch.inference_mode()
    def anchors(self, reference_tokens: dict[str, list[int]], rank: int):
        captures = {}
        for language in ("French", "English"):
            cache, _, _ = self.prefill(reference_tokens[language])
            captures[language] = capture_states(cache)
            del cache
        result = {}
        for layer, french in captures["French"].items():
            result[layer] = make_anchor(
                french.to(self.device), captures["English"][layer].to(self.device), rank
            )
        return result


@torch.inference_mode()
def check_cache_paths(engine, prompt_ids: list[int], split: int) -> dict:
    """Runtime gate: independent branches, continuation vs full prefill, edit effect on fresh logits."""
    from .state import clone_cache

    prefix, suffix = prompt_ids[:split], prompt_ids[split:]
    base, _, _ = engine.prefill(prefix)
    snapshot = clone_cache(base, "cpu")
    del base
    first = clone_cache(snapshot, engine.device)
    second = clone_cache(snapshot, engine.device)
    first, first_logits, _ = engine.prefill(suffix, first)
    second, second_logits, _ = engine.prefill(suffix, second)
    repeat_error = float((first_logits - second_logits).abs().max())
    if not torch.equal(first_logits, second_logits):
        raise RuntimeError(f"Identical cache forks do not reproduce logits: max error {repeat_error}")
    del first, second
    full, full_logits = engine.forward(prompt_ids)
    full_error = float((full_logits - first_logits).abs().max())
    first_token_matches = int(full_logits.argmax()) == int(first_logits.argmax())
    if not torch.allclose(full_logits, first_logits, atol=0.15, rtol=0.02) or not first_token_matches:
        raise RuntimeError(
            f"Split vs full prefill mismatch: max logits error={full_error}, argmax match={first_token_matches}"
        )
    del full
    edited = clone_cache(snapshot, engine.device)
    control = clone_cache(snapshot, engine.device)
    control, control_logits = engine.forward(suffix[:1], control)
    del control
    states = recurrent_states(edited)
    # Deterministic small perturbation on every recurrent layer; only a wiring check, not a method result.
    for state in states.values():
        state.add_(0.05)
    edited, edited_logits = engine.forward(suffix[:1], edited)
    edit_effect = float((edited_logits - control_logits).abs().max())
    if not torch.isfinite(edited_logits).all() or edit_effect <= 1e-6:
        raise RuntimeError("Edited recurrent states did not affect fresh suffix logits.")
    return {
        "repeat_max_logits_error": repeat_error,
        "split_full_max_logits_error": full_error,
        "split_full_first_token_matches": first_token_matches,
        "perturbation_max_logits_effect": edit_effect,
        "state_shapes": {str(i): list(s.shape) for i, s in states.items()},
        "state_layout": "batch,value_head,key_dim,value_dim",
        "model_forward_signature": str(inspect.signature(engine.model.forward)),
    }
