"""Document-sharded workers; methods never get assigned to separate GPUs."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from .data import load_data
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
from .scoring import score
from .state import clone_cache, edit_cache

STAGES = ("smoke", "baseline", "tune", "compare", "delay")


def method_specs(stage: str, config: dict, length: int, selection: dict | None = None) -> list[dict]:
    baseline = {"method": "baseline", "strength": 0.0, "delay": 0}
    if stage == "smoke":
        return [
            baseline,
            {"method": "add_lowrank", "strength": 1.0, "delay": 0},
            {"method": "replace", "strength": 1.0, "delay": 0},
        ]
    if stage == "baseline":
        return [baseline]
    if stage == "tune":
        if length not in (config["lengths"][0], config["lengths"][-1]):
            return []
        specs = [baseline]
        for method, grid in (
            ("add_full", config["gamma_grid"]),
            ("add_lowrank", config["gamma_grid"]),
            ("replace", config["lambda_grid"]),
        ):
            specs.extend({"method": method, "strength": float(value), "delay": 0} for value in grid)
        return specs
    if selection is None:
        raise ValueError("Compare/delay requires frozen dev-only selection.json; run tune and select first.")
    if stage == "compare":
        if length not in (config["lengths"][0], config["lengths"][-1]):
            return []
        specs = [baseline, {"method": "reminder", "strength": 0.0, "delay": 0}]
        specs.extend(
            {"method": method, "strength": selection["strengths"][method], "delay": 0}
            for method in ("add_full", "add_lowrank", "replace")
        )
        specs.append({"method": "wrong_anchor", "strength": selection["strengths"]["replace"], "delay": 0})
        return specs
    if stage == "delay":
        if length != config["lengths"][-1]:
            return []
        return [
            {
                "method": method,
                "strength": 0.0 if method == "baseline" else selection["strengths"][method],
                "delay": delay,
            }
            for delay in config["delays"]
            for method in ("baseline", "add_lowrank", "replace")
        ]
    raise ValueError(f"Unknown stage: {stage}")


def cell_id(example: dict, spec: dict) -> str:
    return digest({"doc_id": example["doc_id"], "length": example["length"], **spec})[:24]


def expected_cells(stage: str, manifest: dict, examples: list[dict], selection: dict | None):
    config = manifest["config"]
    split = "dev" if stage == "tune" else "test"
    result = {}
    for example in examples:
        if example["split"] != split:
            continue
        if stage == "smoke" and (example["doc_index"] >= 2 or example["length"] != 0):
            continue
        for spec in method_specs(stage, config, example["length"], selection):
            result[cell_id(example, spec)] = (example, spec)
    return result


def load_selection(run_dir: Path, manifest: dict) -> dict:
    selection = read_json(run_dir / "selection.json")
    if selection["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Frozen selection belongs to different data/model/config.")
    if selection["source_sha256"] != provenance()["source_sha256"]:
        raise ValueError("Source changed since tuning. Use a new experiment directory.")
    if selection["split"] != "dev":
        raise ValueError("Only development selection is allowed.")
    return selection


def stage_rows(directory: Path) -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(directory.glob("results-*.jsonl")):
        for row in read_jsonl(path):
            if row["cell_id"] in seen:
                raise ValueError(f"Duplicate result cell in {directory}: {row['cell_id']}")
            seen.add(row["cell_id"])
            rows.append(row)
    return rows


def require_previous_gate(run_dir: Path, stage: str, manifest: dict) -> None:
    previous = {"tune": "baseline", "compare": "baseline", "delay": "compare"}.get(stage)
    if previous is None:
        return
    report = read_json(run_dir / previous / "summary.json")
    scored = read_jsonl(run_dir / previous / "scored-results.jsonl")
    if (
        report["manifest_sha256"] != manifest["manifest_sha256"]
        or report["source_sha256"] != provenance()["source_sha256"]
        or report["results_sha256"] != digest(scored)
    ):
        raise ValueError("Prior gate report/results differ from this experiment.")
    if report["decision"] != "continue_pilot":
        raise ValueError(
            f"{stage} requires a reviewed positive {previous} gate; current decision={report['decision']}."
        )


def run_worker(data_dir: Path, run_dir: Path, stage: str, shard: int, shards: int) -> None:
    if stage not in STAGES or not 0 <= shard < shards:
        raise ValueError("Invalid stage/shard.")
    # This lease belongs to the worker, so even a SIGKILLed launcher cannot allow
    # a replacement worker to overwrite an orphan that is still writing results.
    with exclusive_lock(run_dir / stage / f"worker-{shard}.lock"):
        _run_worker(data_dir, run_dir, stage, shard, shards)


def intervention_cuts(stage: str, config: dict, example: dict, selection: dict | None) -> list[int]:
    # Freeze the prefill schedule independently of which cells remain after resuming.
    return sorted(
        {
            example["question_cut"] - spec["delay"]
            for spec in method_specs(stage, config, example["length"], selection)
        }
    )


def _run_worker(data_dir: Path, run_dir: Path, stage: str, shard: int, shards: int) -> None:
    started = time.perf_counter()
    manifest, examples = load_data(data_dir)
    config = manifest["config"]
    require_previous_gate(run_dir, stage, manifest)
    selection = load_selection(run_dir, manifest) if stage in {"compare", "delay"} else None
    expected = expected_cells(stage, manifest, examples, selection)
    selected = {
        key: (example, spec)
        for key, (example, spec) in expected.items()
        if example["doc_index"] % shards == shard
    }
    stage_dir = run_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    run_context = {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": provenance()["source_sha256"],
    }
    # Coordinator owns experiment.json; direct single-worker calls safely bind it too.
    bind_metadata(run_dir / "experiment.json", run_context)
    metadata = {
        **run_context,
        "stage": stage,
        "shard": shard,
        "shards": shards,
        "selection_sha256": digest(selection) if selection else None,
        "expected_cells": sorted(selected),
    }
    bind_metadata(stage_dir / f"metadata-{shard}.json", metadata)
    result_path = stage_dir / f"results-{shard}.jsonl"
    previous = recover_jsonl(result_path)
    seen = {row["cell_id"] for row in previous}
    if len(seen) != len(previous) or not seen.issubset(selected):
        raise ValueError("Duplicate/unexpected result records; refusing to resume.")
    if seen == set(selected):
        print(f"{stage} shard {shard}: already complete", flush=True)
        return
    engine = Engine(config, manifest["tokenizer_sha256"])
    env = environment()
    stable_runtime = {
        "versions": env["versions"],
        "modeling_source_sha256": env["modeling_source_sha256"],
        "model_config_sha256": engine.model_config_sha256,
        "delta_backend": config["delta_backend"],
        "attention_backend": config["attention_backend"],
    }
    # Bind each shard separately to avoid simultaneous file creation; reports compare all runtime fingerprints.
    bind_metadata(stage_dir / f"runtime-{shard}.json", stable_runtime)
    write_json(
        stage_dir / f"environment-{shard}.json",
        {
            **env,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "provenance": provenance(),
            "weight_loading_audit": engine.loading_audit,
        },
    )
    runtime_sha = digest(stable_runtime)
    if selection and selection["runtime_sha256"] != runtime_sha:
        raise ValueError("Runtime/model differs from the frozen development selection.")
    first_example = min((example for example, _ in selected.values()), key=lambda row: row["length"])
    # A short actual prompt check on every worker before spending on long contexts.
    short = next(row for row in examples if row["split"] == first_example["split"] and row["length"] == 0)
    checks = check_cache_paths(engine, short["input_ids"], short["question_cut"])
    write_json(stage_dir / f"cache-checks-{shard}.json", checks)
    anchors = None
    if stage in {"tune", "compare", "delay", "smoke"}:
        anchors = engine.anchors(manifest["reference_tokens"], config["rank"])
        write_json(
            stage_dir / f"anchor-audit-{shard}.json",
            {
                "rank": config["rank"],
                "state_layout": "batch,value_head,key_dim,value_dim",
                "energy_per_head": {str(i): a.retained_energy.cpu().tolist() for i, a in anchors.items()},
                "effective_rank_per_head": {
                    str(i): a.effective_rank.cpu().tolist() for i, a in anchors.items()
                },
                "difference_norm_per_layer": {str(i): a.difference.norm().item() for i, a in anchors.items()},
            },
        )
    reused_baselines = {}
    if stage == "compare":
        for row in read_jsonl(run_dir / "baseline" / "scored-results.jsonl"):
            if row["runtime_sha256"] != runtime_sha:
                raise ValueError("Baseline runtime differs from comparison runtime.")
            reused_baselines[row["cell_id"]] = row
    groups = {}
    for key, (example, spec) in selected.items():
        if key in seen:
            continue
        if spec["method"] == "baseline" and key in reused_baselines:
            reused = dict(reused_baselines[key], stage=stage, reused_from="baseline", shard=shard)
            append_jsonl(result_path, reused)
            seen.add(key)
            continue
        group_key = (example["doc_id"], example["length"])
        groups.setdefault(group_key, (example, []))[1].append((key, spec))
    cumulative_prefix_seconds = 0.0
    cumulative_prefix_tokens = 0
    for example, entries in groups.values():
        ids = example["input_ids"]
        cut_positions = intervention_cuts(stage, config, example, selection)
        last_cut = 0
        original_snapshot = None
        for cut in cut_positions:
            if cut <= 0:
                raise ValueError("Delay puts intervention before the prompt starts.")
            base = clone_cache(original_snapshot, engine.device) if original_snapshot is not None else None
            base, _, prefix_seconds = engine.prefill(ids[last_cut:cut], base)
            cumulative_prefix_seconds += prefix_seconds
            cumulative_prefix_tokens += cut - last_cut
            copy_started = time.perf_counter()
            original_snapshot = clone_cache(base, config["snapshot_device"])
            engine.synchronize()
            snapshot_seconds = time.perf_counter() - copy_started
            del base
            for key, spec in entries:
                if example["question_cut"] - spec["delay"] != cut:
                    continue
                if engine.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                engine.synchronize()
                branch_started = time.perf_counter()
                branch = clone_cache(original_snapshot, engine.device)
                engine.synchronize()
                restore_seconds = time.perf_counter() - branch_started
                edit_started = time.perf_counter()
                edit = {}
                if spec["method"] not in {"baseline", "reminder"}:
                    edit = edit_cache(branch, anchors, spec["method"], spec["strength"])
                engine.synchronize()
                edit_seconds = time.perf_counter() - edit_started
                suffix = ids[cut:]
                if spec["method"] == "reminder":
                    suffix = example["reminder_ids"] + suffix
                output = engine.generate(branch, suffix)
                engine.synchronize()
                row = {
                    "schema_version": 1,
                    "cell_id": key,
                    "stage": stage,
                    "shard": shard,
                    "doc_id": example["doc_id"],
                    "split": example["split"],
                    "length": example["length"],
                    "reference_tokens": example["reference_tokens"],
                    "prompt_tokens": len(ids),
                    "question_cut": example["question_cut"],
                    "intervention_cut": cut,
                    "target": example["target"],
                    **spec,
                    **output,
                    **edit,
                    "snapshot_seconds": snapshot_seconds,
                    "restore_seconds": restore_seconds,
                    "edit_seconds": edit_seconds,
                    "branch_seconds": time.perf_counter() - branch_started,
                    "prefix_seconds_shared": prefix_seconds,
                    "prefix_tokens_shared": cut - last_cut,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                    if engine.device.type == "cuda"
                    else None,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "runtime_sha256": runtime_sha,
                    "output_sha256": digest(output["text"]),
                }
                row["scores"] = score(row["text"], row["target"], row["truncated"])
                append_jsonl(result_path, row)
                seen.add(key)
                print(
                    f"{stage} shard={shard} {len(seen)}/{len(selected)} {example['doc_id']} N={example['length']} "
                    f"{spec} J={int(row['scores']['joint_ok'])} tokens={row['generated_tokens']}",
                    flush=True,
                )
                del branch
            last_cut = cut
        del original_snapshot
    if seen != set(selected):
        raise RuntimeError("Worker ended with missing result cells.")
    # This invocation's measured costs include startup/anchors/checks; resumed invocations get separate cost records.
    costs = {
        "stage": stage,
        "shard": shard,
        "runtime_sha256": runtime_sha,
        "invocation_seconds": time.perf_counter() - started,
        "prefix_tokens_actual": cumulative_prefix_tokens,
        "prefix_seconds_actual": cumulative_prefix_seconds,
        "resumed_existing_cells": len(previous),
        "complete": True,
    }
    append_jsonl(stage_dir / f"costs-{shard}.jsonl", costs)
