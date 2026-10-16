"""Two CUDA workers, independent full prefills and registered staged control tests."""

from __future__ import annotations

import os
import time

import torch

from .confirm import bind_runtime, runtime_signature
from .controls_data import POLICIES, load_controls
from .controls_scoring import score
from .engine import Engine, check_cache_paths, environment
from .io import (
    append_jsonl,
    bind_metadata,
    digest,
    provenance,
    read_json,
    read_jsonl,
    recover_jsonl,
    write_json,
)
from .processes import exclusive_lock


def context(manifest):
    return {"manifest_sha256": manifest["manifest_sha256"], "source_sha256": provenance()["source_sha256"]}


def cells(examples, phase, shard=None):
    return {
        e["cell_id"]: e
        for e in examples
        if e["family"] == phase and (shard is None or e["doc_index"] % 2 == shard)
    }


def validate_row(row, example, manifest, phase, shard, runtime, device):
    fields = {k: v for k, v in example.items() if k != "input_ids"}
    if (
        any(row.get(k) != v for k, v in fields.items())
        or row["phase"] != phase
        or row["shard"] != shard
        or row["manifest_sha256"] != manifest["manifest_sha256"]
        or row["runtime_sha256"] != digest(runtime)
        or row["gpu_uuid"] != device["gpu_uuid"]
        or row["output_sha256"] != digest(row["text"])
        or row["row_sha256"] != digest({k: v for k, v in row.items() if k != "row_sha256"})
        or row["suffix_tokens"] != len(example["input_ids"]) - example["question_cut"]
        or row["prefix_schedule"] != {"start": 0, "end": example["question_cut"], "chunk_size": 2048}
        or row["generated_tokens"] != len(row["generated_ids"])
    ):
        raise ValueError("Controls result input/output/runtime fingerprint mismatch.")


def validated_results(data_dir, run_dir, phases=("language", "format")):
    manifest, examples = load_controls(data_dir)
    if read_json(run_dir / "experiment.json") != context(manifest):
        raise ValueError("Controls run source or input differs; use its original checkout and data.")
    rows, missing, slots = [], {}, {}
    for phase in phases:
        expected, seen = cells(examples, phase), set()
        for shard in (0, 1):
            directory = run_dir / phase
            previous = read_jsonl(directory / f"results-{shard}.jsonl")
            if not previous:
                continue
            runtime = read_json(directory / f"runtime-{shard}.json")
            device = read_json(directory / f"device-{shard}.json")
            metadata = read_json(directory / f"metadata-{shard}.json")
            if (
                runtime != read_json(run_dir / "runtime.json")
                or device["gpu_uuid"] != read_json(run_dir / f"device-slot-{shard}.json")["gpu_uuid"]
                or metadata["expected_cells"] != sorted(cells(examples, phase, shard))
                or any(metadata[k] != v for k, v in context(manifest).items())
            ):
                raise ValueError("Controls worker runtime/device/registry mismatch.")
            if shard in slots and slots[shard] != device["gpu_uuid"]:
                raise ValueError("Controls physical GPU slot changed between phases.")
            slots[shard] = device["gpu_uuid"]
            for row in previous:
                cell = row["cell_id"]
                if cell in seen or cell not in expected or expected[cell]["doc_index"] % 2 != shard:
                    raise ValueError("Duplicate or unexpected controls result.")
                validate_row(row, expected[cell], manifest, phase, shard, runtime, device)
                seen.add(cell)
                rows.append(row)
        missing[phase] = sorted(expected.keys() - seen)
    if len(slots) == 2 and len(set(slots.values())) != 2:
        raise ValueError("Controls workers used the same physical GPU.")
    return manifest, rows, missing


def require_language_gate(data_dir, run_dir, reviews=None):
    from .controls_analysis import apply_reviews, evaluate_family

    manifest, rows, missing = validated_results(data_dir, run_dir, phases=("language",))
    if any(missing.values()):
        raise ValueError("Complete the 288 language answers before launching format conditions.")
    _, examples = load_controls(data_dir)
    scored, provenance_review = apply_reviews(rows, reviews, {e["cell_id"] for e in examples})
    if reviews is not None and not provenance_review["review_complete"]:
        raise ValueError(
            "Complete all 288 language reviews before using --reviews to launch format conditions."
        )
    gate = evaluate_family(scored, manifest["config"], "language")
    if not gate["eligible"]:
        raise ValueError(
            f"Language gate did not pass: {gate['conditions']}. Inspect controls-summary.json and blind "
            "reviews; preserve the data and protocol. Format inference was not started."
        )
    return {
        **context(manifest),
        "language_results_sha256": digest(rows),
        "gate_basis": "human"
        if provenance_review["human_review_complete"]
        else "reviewed_provisional"
        if provenance_review["review_complete"]
        else "automatic_provisional",
        "reviews_sha256": provenance_review["reviews_sha256"],
        "conditions": gate["conditions"],
    }


def launch_controls(data_dir, run_dir, phase, devices, reviews=None):
    from .controls_analysis import read_reviews, summarize
    from .launch import launch

    if phase not in (*POLICIES, "all"):
        raise ValueError("Unknown controls phase.")
    read_reviews(reviews)  # Fail on a typo before loading either GPU.
    with exclusive_lock(run_dir / "controls-launcher.lock"):
        # A rerun of --phase all summarizes A before B's worker starts. Recover
        # B's interrupted tail first, or that intermediate summary cannot read it.
        manifest, _ = load_controls(data_dir)
        bind_metadata(run_dir / "experiment.json", context(manifest))
        for existing_phase in POLICIES:
            for shard in (0, 1):
                path = run_dir / existing_phase / f"results-{shard}.jsonl"
                if path.exists():
                    with (
                        exclusive_lock(run_dir / f"gpu-slot-{shard}.lock"),
                        exclusive_lock(run_dir / existing_phase / f"worker-{shard}.lock"),
                    ):
                        recover_jsonl(path)
        for step in ("language", "format") if phase == "all" else (phase,):
            if step == "format":
                bind_metadata(run_dir / "format-gate.json", require_language_gate(data_dir, run_dir, reviews))
            launch(data_dir, run_dir, step, devices, controls=True, reviews=reviews)
            summarize(data_dir, run_dir, reviews)


def generate_answer(engine, example):
    """A clean token-zero prefill for each policy; no state or token intervention."""
    cut = example["question_cut"]
    cache, _, seconds = engine.prefill(example["input_ids"][:cut])
    output = engine.generate(cache, example["input_ids"][cut:])
    return {**output, "prefix_seconds": seconds}


def worker(data_dir, run_dir, phase, shard, reviews=None):
    if phase not in POLICIES or shard not in (0, 1):
        raise ValueError("Controls requires two workers and a registered phase.")
    with (
        exclusive_lock(run_dir / f"gpu-slot-{shard}.lock"),
        exclusive_lock(run_dir / phase / f"worker-{shard}.lock"),
    ):
        _worker(data_dir, run_dir, phase, shard, reviews)


def _worker(data_dir, run_dir, phase, shard, reviews):
    manifest, examples = load_controls(data_dir)
    bind_metadata(run_dir / "experiment.json", context(manifest))
    if phase == "format":
        bind_metadata(run_dir / "format-gate.json", require_language_gate(data_dir, run_dir, reviews))
    selected = cells(examples, phase, shard)
    directory = run_dir / phase
    directory.mkdir(parents=True, exist_ok=True)
    bind_metadata(
        directory / f"metadata-{shard}.json",
        {
            **context(manifest),
            "phase": phase,
            "shard": shard,
            "expected_cells": sorted(selected),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    previous = recover_jsonl(directory / f"results-{shard}.jsonl")
    seen = {r["cell_id"] for r in previous}
    if len(seen) != len(previous) or not seen.issubset(selected):
        raise ValueError("Duplicate or unexpected resumed controls cells.")
    if previous:
        runtime = read_json(directory / f"runtime-{shard}.json")
        device = read_json(directory / f"device-{shard}.json")
        if (
            runtime != read_json(run_dir / "runtime.json")
            or device["gpu_uuid"] != read_json(run_dir / f"device-slot-{shard}.json")["gpu_uuid"]
        ):
            raise ValueError("Resumed controls runtime/device differs.")
        for row in previous:
            validate_row(row, selected[row["cell_id"]], manifest, phase, shard, runtime, device)
    if seen == selected.keys():
        print(f"{phase} shard {shard}: already complete", flush=True)
        return
    engine = Engine(manifest["engine_config"], manifest["tokenizer_sha256"])
    if engine.device.type != "cuda":
        raise RuntimeError("Formal controls inference requires CUDA; use CPU unit tests locally.")
    env = environment()
    runtime = runtime_signature(engine, env)
    bind_runtime(run_dir, runtime)
    bind_metadata(directory / f"runtime-{shard}.json", runtime)
    properties = torch.cuda.get_device_properties(0)
    device = {
        "gpu_uuid": str(properties.uuid),
        "name": properties.name,
        "total_memory": properties.total_memory,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    bind_metadata(run_dir / f"device-slot-{shard}.json", {"gpu_uuid": device["gpu_uuid"]})
    other = run_dir / f"device-slot-{1 - shard}.json"
    if other.exists() and read_json(other)["gpu_uuid"] == device["gpu_uuid"]:
        raise ValueError("Both controls workers refer to one physical GPU.")
    bind_metadata(directory / f"device-{shard}.json", device)
    write_json(
        directory / f"environment-{shard}.json",
        {
            **env,
            "weight_loading_audit": engine.loading_audit,
            "provenance": provenance(),
        },
    )
    short = next(e for e in selected.values() if e["background"] == "none")
    write_json(
        directory / f"cache-checks-{shard}.json",
        check_cache_paths(engine, short["input_ids"], short["question_cut"]),
    )
    started = time.perf_counter()
    for cell, example in selected.items():
        if cell in seen:
            continue
        torch.cuda.reset_peak_memory_stats()
        output = generate_answer(engine, example)
        row = {
            **{k: v for k, v in example.items() if k != "input_ids"},
            **output,
            "phase": phase,
            "shard": shard,
            "manifest_sha256": manifest["manifest_sha256"],
            "runtime_sha256": digest(runtime),
            "gpu_uuid": device["gpu_uuid"],
            "prefix_schedule": {"start": 0, "end": example["question_cut"], "chunk_size": 2048},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "output_sha256": digest(output["text"]),
            "scores": score(output["text"], example, output["truncated"]),
        }
        row["row_sha256"] = digest(row)
        append_jsonl(directory / f"results-{shard}.jsonl", row)
        seen.add(cell)
        print(
            f"{phase} shard={shard} {len(seen)}/{len(selected)} {example['doc_id']} "
            f"{example['policy']} {example['background']} query={example['question_language']}",
            flush=True,
        )
    append_jsonl(
        directory / f"costs-{shard}.jsonl",
        {
            "invocation_seconds": time.perf_counter() - started,
            "resumed_existing_cells": len(previous),
            "complete": seen == selected.keys(),
        },
    )
