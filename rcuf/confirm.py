"""A registered, tuning-free confirmation with crossed two-GPU replay."""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

import torch

from .confirm_data import load_confirmation
from .engine import Engine, check_cache_paths, environment
from .io import append_jsonl, bind_metadata, digest, provenance, read_json, recover_jsonl, write_json
from .processes import exclusive_lock
from .scoring import score
from .state import clone_cache, edit_cache

PHASES = ("smoke", "crossover-a", "crossover-b", "confirm")
STATE_METHODS = ("add_lowrank", "replace", "replace_norm_matched")


def specifications(example, phase):
    if phase not in PHASES:
        raise ValueError("Unknown confirmation phase.")
    if phase.startswith("crossover-"):
        if example["cohort"] != "replay":
            return []
        return [
            {"method": m, "strength": 0.0 if m == "baseline" else 1.0, "delay": 4096}
            for m in ("baseline", "add_lowrank", "replace")
        ]
    if example["cohort"] != "fresh":
        return []
    if phase == "smoke":
        if example["doc_index"] >= 2 or example["length"] != 0:
            return []
        return [
            {"method": m, "strength": 0.0 if m == "baseline" else 1.0, "delay": 0}
            for m in ("baseline", "add_lowrank", "replace_norm_matched")
        ]
    baseline = {"method": "baseline", "strength": 0.0, "delay": 0}
    if example["background"] != "en" or example["length"] != 32768:
        return [baseline]
    return [
        {"method": m, "strength": 1.0 if m in STATE_METHODS else 0.0, "delay": d}
        for d in (0, 4096)
        for m in ("baseline", *STATE_METHODS, "reminder")
    ]


def confirmation_cells(examples, phase):
    cells = {}
    for example in examples:
        for spec in specifications(example, phase):
            identity = {
                "phase": phase,
                "cohort": example["cohort"],
                "doc_id": example["doc_id"],
                "background": example["background"],
                "length": example["length"],
                **spec,
            }
            key = digest(identity)[:24]
            if key in cells:
                raise ValueError("Duplicate confirmation cell.")
            cells[key] = example, spec
    return cells


def assigned_shard(example, phase):
    # A: original even/odd assignment. B: each exact document runs on the other GPU.
    return (example["doc_index"] + (phase == "crossover-b")) % 2


def context(manifest):
    return {"manifest_sha256": manifest["manifest_sha256"], "source_sha256": provenance()["source_sha256"]}


def runtime_signature(engine, env):
    # Keep this identical to the original runner for comparison with the pilot.
    return {
        "versions": env["versions"],
        "modeling_source_sha256": env["modeling_source_sha256"],
        "model_config_sha256": engine.model_config_sha256,
        "delta_backend": engine.config["delta_backend"],
        "attention_backend": engine.config["attention_backend"],
    }


def bind_runtime(run_dir, runtime):
    # Compare across workers and phases before generating, not only at final analysis.
    # The short lease makes the first-writer check atomic even for concurrent workers.
    with (run_dir / "runtime.lock").open("a") as lock:
        # This brief metadata critical section waits, unlike worker ownership leases.
        fcntl.flock(lock, fcntl.LOCK_EX)
        bind_metadata(run_dir / "runtime.json", runtime)


def validate_result(row, example, spec, manifest, phase, shard, runtime, device):
    """The same checks apply to resumed rows and to the final report."""
    if (
        row["phase"] != phase
        or row["shard"] != shard
        or any(row[k] != v for k, v in spec.items())
        or any(
            row[k] != example[k]
            for k in ("doc_id", "doc_index", "cohort", "background", "length", "target", "input_sha256")
        )
        or row["manifest_sha256"] != manifest["manifest_sha256"]
        or row["output_sha256"] != digest(row["text"])
    ):
        raise ValueError("Result differs from its registered input/spec/output.")
    cut = example["question_cut"] - spec["delay"]
    extra = example["reminder_ids"] if spec["method"] == "reminder" else []
    if (
        cut <= 0
        or row["intervention_cut"] != cut
        or row["inserted_ids"] != extra
        or row["prompt_tokens_actual"] != len(example["input_ids"]) + len(extra)
        or row["actual_input_sha256"]
        != digest(example["input_ids"][:cut] + extra + example["input_ids"][cut:])
        or row["prefix_schedule"] != {"start": 0, "end": cut, "chunk_size": 2048}
    ):
        raise ValueError("Actual token stream or prefill schedule differs from the plan.")
    if row["runtime_sha256"] != digest(runtime) or row["gpu_uuid"] != device["gpu_uuid"]:
        raise ValueError("Result/runtime/device fingerprint mismatch.")


def validate_resumed(previous, selected, manifest, directory, phase, shard):
    seen = {r["cell_id"] for r in previous}
    if len(seen) != len(previous) or not seen.issubset(selected):
        raise ValueError("Unexpected or duplicate resumed confirmation output.")
    if previous:
        runtime = read_json(directory / f"runtime-{shard}.json")
        if read_json(directory.parent / "runtime.json") != runtime:
            raise ValueError("Resumed worker differs from the experiment runtime.")
        device = read_json(directory / f"device-{shard}.json")
        if read_json(directory.parent / f"device-slot-{shard}.json")["gpu_uuid"] != device["gpu_uuid"]:
            raise ValueError("Resumed worker differs from its physical GPU slot.")
        for row in previous:
            example, spec = selected[row["cell_id"]]
            validate_result(row, example, spec, manifest, phase, shard, runtime, device)
    return seen


def launch_confirmation(data_dir, run_dir, phase, devices):
    from .launch import launch

    if phase not in (*PHASES, "all"):
        raise ValueError("Unknown confirmation phase.")
    with exclusive_lock(run_dir / "confirmation-launcher.lock"):
        for step in ("smoke", "crossover-a", "crossover-b", "confirm") if phase == "all" else (phase,):
            launch(data_dir, run_dir, step, devices, confirmation=True)


def worker(data_dir: Path, run_dir: Path, phase: str, shard: int):
    if phase not in PHASES or shard not in (0, 1):
        raise ValueError("Confirmation requires exactly two workers.")
    with (
        exclusive_lock(run_dir / f"gpu-slot-{shard}.lock"),
        exclusive_lock(run_dir / phase / f"worker-{shard}.lock"),
    ):
        _worker(data_dir, run_dir, phase, shard)


def generate_branch(engine, snapshot, example, spec, cut, anchors):
    """Every method starts from a new copy; edits never contaminate the base path."""
    branch = clone_cache(snapshot, engine.device)
    engine.synchronize()
    started = time.perf_counter()
    audit = {}
    if spec["method"] in STATE_METHODS:
        audit = edit_cache(branch, anchors, spec["method"], spec["strength"])
    engine.synchronize()
    edit_seconds = time.perf_counter() - started
    suffix = example["input_ids"][cut:]
    extra = example["reminder_ids"] if spec["method"] == "reminder" else []
    # Reminders deliberately add tokens. Log their exact token stream and length;
    # all state comparisons use the unchanged original input stream.
    output = engine.generate(branch, extra + suffix)
    return {
        **output,
        **audit,
        "edit_seconds": edit_seconds,
        "inserted_ids": extra,
        "prompt_tokens_actual": len(example["input_ids"]) + len(extra),
        "actual_input_sha256": digest(example["input_ids"][:cut] + extra + suffix),
    }


def _worker(data_dir, run_dir, phase, shard):
    started = time.perf_counter()
    manifest, examples = load_confirmation(data_dir)
    bind_metadata(run_dir / "experiment.json", context(manifest))
    all_cells = confirmation_cells(examples, phase)
    selected = {k: v for k, v in all_cells.items() if assigned_shard(v[0], phase) == shard}
    directory = run_dir / phase
    directory.mkdir(parents=True, exist_ok=True)
    bind_metadata(
        directory / f"metadata-{shard}.json",
        {
            **context(manifest),
            "phase": phase,
            "shard": shard,
            "shards": 2,
            "expected_cells": sorted(selected),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    previous = recover_jsonl(directory / f"results-{shard}.jsonl")
    seen = validate_resumed(previous, selected, manifest, directory, phase, shard)
    if seen == selected.keys():
        print(f"{phase} shard={shard}: already complete", flush=True)
        return
    engine = Engine(manifest["engine_config"], manifest["tokenizer_sha256"])
    if engine.device.type != "cuda":
        raise RuntimeError("Server confirmation requires CUDA; run CPU unit tests locally.")
    env = environment()
    runtime = runtime_signature(engine, env)
    bind_runtime(run_dir, runtime)
    bind_metadata(directory / f"runtime-{shard}.json", runtime)
    # UUID detects accidentally launching both device aliases on one physical GPU.
    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = str(properties.uuid)
    bind_metadata(run_dir / f"device-slot-{shard}.json", {"gpu_uuid": gpu_uuid})
    other = run_dir / f"device-slot-{1 - shard}.json"
    if other.exists() and read_json(other)["gpu_uuid"] == gpu_uuid:
        raise ValueError("Both worker slots refer to the same physical GPU.")
    bind_metadata(
        directory / f"device-{shard}.json",
        {
            "gpu_uuid": gpu_uuid,
            "name": properties.name,
            "total_memory": properties.total_memory,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    )
    write_json(
        directory / f"environment-{shard}.json",
        {
            **env,
            "gpu_uuid": gpu_uuid,
            "provenance": provenance(),
            "weight_loading_audit": engine.loading_audit,
        },
    )
    short = next(x for x in examples if x["cohort"] == "fresh" and x["length"] == 0)
    write_json(
        directory / f"cache-checks-{shard}.json",
        check_cache_paths(engine, short["input_ids"], short["question_cut"]),
    )
    anchors = engine.anchors(manifest["reference_tokens"], manifest["config"]["rank"])
    write_json(
        directory / f"anchor-audit-{shard}.json",
        {
            "rank": manifest["config"]["rank"],
            "difference_norm_per_layer": {str(i): a.difference.norm().item() for i, a in anchors.items()},
            "energy_per_head": {str(i): a.retained_energy.cpu().tolist() for i, a in anchors.items()},
        },
    )
    groups = {}
    for key, (example, spec) in selected.items():
        group_key = example["cohort"], example["doc_id"], example["background"], example["length"]
        groups.setdefault(group_key, (example, []))[1].append((key, spec))
    prefix_seconds_total = 0.0
    for example, entries in groups.values():
        cuts = sorted({example["question_cut"] - spec["delay"] for _, spec in entries})
        for cut in cuts:
            pending = [
                (k, s) for k, s in entries if k not in seen and example["question_cut"] - s["delay"] == cut
            ]
            if not pending:
                continue
            if cut <= 0:
                raise ValueError("Intervention is before the start of the input.")
            # Each cut is prefetched independently from token zero. Baseline N=32K
            # and resume behavior therefore cannot inherit another delay's chunk boundaries.
            base, _, prefix_seconds = engine.prefill(example["input_ids"][:cut])
            prefix_seconds_total += prefix_seconds
            snapshot = clone_cache(base, engine.config["snapshot_device"])
            del base
            for key, spec in pending:
                torch.cuda.reset_peak_memory_stats()
                branch_started = time.perf_counter()
                output = generate_branch(engine, snapshot, example, spec, cut, anchors)
                row = {
                    "schema_version": 1,
                    "cell_id": key,
                    "phase": phase,
                    "shard": shard,
                    **{
                        k: example[k]
                        for k in (
                            "cohort",
                            "background",
                            "doc_id",
                            "doc_index",
                            "length",
                            "reference_tokens",
                            "question_cut",
                            "target",
                            "input_sha256",
                        )
                    },
                    **spec,
                    **output,
                    "intervention_cut": cut,
                    "gpu_uuid": gpu_uuid,
                    "prefix_schedule": {"start": 0, "end": cut, "chunk_size": 2048},
                    "branch_seconds": time.perf_counter() - branch_started,
                    "prefix_seconds_shared": prefix_seconds,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "manifest_sha256": manifest["manifest_sha256"],
                    "runtime_sha256": digest(runtime),
                    "output_sha256": digest(output["text"]),
                }
                row["scores"] = score(row["text"], row["target"], row["truncated"])
                append_jsonl(directory / f"results-{shard}.jsonl", row)
                seen.add(key)
                print(
                    f"{phase} shard={shard} {len(seen)}/{len(selected)} {example['doc_id']} "
                    f"{example['background']} N={example['length']} {spec}",
                    flush=True,
                )
            del snapshot
    if seen != selected.keys():
        raise RuntimeError("Worker finished with missing registered cells.")
    append_jsonl(
        directory / f"costs-{shard}.jsonl",
        {
            "invocation_seconds": time.perf_counter() - started,
            "prefix_seconds_actual": prefix_seconds_total,
            "resumed_existing_cells": len(previous),
            "complete": True,
        },
    )
