"""Complete-cell validation, blind review, dev selection, and paired pilot decisions."""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_data
from .io import append_jsonl, digest, provenance, read_json, read_jsonl, write_json, write_jsonl
from .runner import expected_cells, load_selection, stage_rows


def required_reviews(rows: list[dict]) -> set[str]:
    required = {row["cell_id"] for row in rows if row["scores"]["needs_review"]}
    # Review all samples in documents whose automatic language labels change across length/method/delay.
    documents = defaultdict(list)
    for row in rows:
        documents[row["doc_id"]].append(row)
    for group in documents.values():
        if len({r["scores"]["language_ok"] for r in group}) > 1:
            required.update(r["cell_id"] for r in group)
    return required - {row["cell_id"] for row in rows if row["scores"].get("reviewed")}


def apply_reviews(rows: list[dict], path: Path | None) -> tuple[list[dict], list[str]]:
    reviews = read_jsonl(path) if path is not None else []
    by_cell = {row["cell_id"]: row for row in rows}
    complete_reviews = {}
    seen = set()
    for review in reviews:
        cell = review["cell_id"]
        if cell in seen:
            raise ValueError(f"Duplicate review for {cell}")
        seen.add(cell)
        if cell not in by_cell or review["output_sha256"] != by_cell[cell]["output_sha256"]:
            raise ValueError(f"Review belongs to a different output: {cell}")
        if type(review.get("language_ok")) is bool and type(review.get("content_ok")) is bool:
            complete_reviews[cell] = review
    required = required_reviews(rows)
    result = []
    for original in rows:
        row = dict(original, scores=dict(original["scores"]))
        review = complete_reviews.get(row["cell_id"])
        if review:
            row["scores"].update(
                language_ok=review["language_ok"],
                content_ok=review["content_ok"],
                joint_ok=review["language_ok"] and review["content_ok"],
                reviewed=True,
            )
        result.append(row)
    return result, sorted(required - complete_reviews.keys())


def validated_rows(data_dir: Path, run_dir: Path, stage: str):
    manifest, examples = load_data(data_dir)
    context = read_json(run_dir / "experiment.json")
    if context != {
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": provenance()["source_sha256"],
    }:
        raise ValueError("The experiment was generated with different data or source code.")
    selection = load_selection(run_dir, manifest) if stage in {"compare", "delay"} else None
    expected = expected_cells(stage, manifest, examples, selection)
    rows = stage_rows(run_dir / stage)
    unexpected = {row["cell_id"] for row in rows} - expected.keys()
    if unexpected:
        raise ValueError(f"Unexpected result cells in {stage}: {sorted(unexpected)}")
    runtime_hashes = {row["runtime_sha256"] for row in rows}
    if len(runtime_hashes) > 1:
        raise ValueError("Workers/stages have different runtime/model fingerprints; comparisons are invalid.")
    for row in rows:
        if row["manifest_sha256"] != manifest["manifest_sha256"]:
            raise ValueError("Result data fingerprint mismatch.")
        example, spec = expected[row["cell_id"]]
        if any(row[key] != value for key, value in spec.items()) or row["target"] != example["target"]:
            raise ValueError("Result spec/target differs from the registered cell.")
        if row["output_sha256"] != digest(row["text"]):
            raise ValueError("Result output was changed after generation.")
    missing = sorted(expected.keys() - {row["cell_id"] for row in rows})
    return manifest, rows, missing


def export_reviews(data_dir: Path, run_dir: Path, stage: str, output: Path, review_all=False) -> int:
    _, rows, _ = validated_rows(data_dir, run_dir, stage)
    if output.exists():
        raise ValueError(f"Review file already exists: {output}; keep existing labels or choose a new path.")
    needed = required_reviews(rows)
    chosen = rows if review_all else [row for row in rows if row["cell_id"] in needed]
    random.Random(913).shuffle(chosen)
    for index, row in enumerate(chosen):
        # Method, parameter, length and automatic label are intentionally hidden.
        append_jsonl(
            output,
            {
                "review_id": f"R{index:04d}",
                "cell_id": row["cell_id"],
                "output_sha256": row["output_sha256"],
                "answer": row["text"],
                "expected_code": row["target"]["code"],
                "expected_budget": row["target"]["budget"],
                "language_ok": None,
                "content_ok": None,
                "notes": "",
            },
        )
    if not chosen:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
    return len(chosen)


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        **{
            key: {
                "successes": sum(int(row["scores"][key]) for row in rows),
                "rate": sum(int(row["scores"][key]) for row in rows) / n if n else None,
            }
            for key in ("language_ok", "content_ok", "joint_ok")
        },
    }


def paired(left: list[dict], right: list[dict], metric: str, config: dict) -> dict:
    """left - right, matched by document, with document-level bootstrap."""
    a = {row["doc_id"]: int(row["scores"][metric]) for row in left}
    b = {row["doc_id"]: int(row["scores"][metric]) for row in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys() or not a:
        raise ValueError("Paired comparison requires one result per matching document.")
    docs = sorted(a)
    difference = np.array([a[doc] - b[doc] for doc in docs])
    rng = np.random.default_rng(config["seed"])
    draws = rng.choice(difference, size=(config["bootstrap_samples"], len(docs)), replace=True).mean(1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "n": len(docs),
        "failure_to_success": int((difference == 1).sum()),
        "success_to_failure": int((difference == -1).sum()),
        "net_gains": int(difference.sum()),
        "difference_rate": float(difference.mean()),
        "paired_bootstrap_95ci": [float(lo), float(hi)],
    }


def group(rows, method, length, delay=0, strength=None):
    return [
        row
        for row in rows
        if row["method"] == method
        and row["length"] == length
        and row["delay"] == delay
        and (strength is None or row["strength"] == strength)
    ]


def select(data_dir: Path, run_dir: Path, reviews: Path | None = None) -> dict:
    manifest, raw, missing = validated_rows(data_dir, run_dir, "tune")
    rows, unreviewed = apply_reviews(raw, reviews)
    if missing or unreviewed:
        raise ValueError(
            f"Cannot freeze selection: {len(missing)} missing cells, {len(unreviewed)} required reviews unfinished."
        )
    config = manifest["config"]
    short_length, long_length = config["lengths"][0], config["lengths"][-1]
    base_short = metrics(group(rows, "baseline", short_length))["joint_ok"]["successes"]
    choices, audit = {}, {}
    for method, grid in (
        ("add_full", config["gamma_grid"]),
        ("add_lowrank", config["gamma_grid"]),
        ("replace", config["lambda_grid"]),
    ):
        candidates = []
        for strength in grid:
            short = metrics(group(rows, method, short_length, strength=strength))
            long = metrics(group(rows, method, long_length, strength=strength))
            admissible = short["joint_ok"]["successes"] >= base_short
            candidates.append({"strength": strength, "short": short, "long": long, "admissible": admissible})
        # Prioritize zero observed short-task harm, then long joint/content success, then smaller edit strength.
        winner = max(
            candidates,
            key=lambda c: (
                c["admissible"],
                c["long"]["joint_ok"]["successes"],
                c["long"]["content_ok"]["successes"],
                -c["strength"],
            ),
        )
        choices[method] = winner["strength"]
        audit[method] = {"candidates": candidates, "selected_admissible": winner["admissible"]}
    result = {
        "schema_version": 1,
        "split": "dev",
        "strengths": choices,
        "rank": config["rank"],
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": provenance()["source_sha256"],
        "reviewed_results_sha256": digest(rows),
        "runtime_sha256": rows[0]["runtime_sha256"],
        "audit": audit,
    }
    selection_path = run_dir / "selection.json"
    if selection_path.exists():
        if read_json(selection_path) != result:
            raise ValueError("A different frozen selection already exists; use a new experiment directory.")
    else:
        write_json(selection_path, result)
    return result


def summarize(
    data_dir: Path,
    run_dir: Path,
    stage: str,
    reviews: Path | None = None,
    compare_reviews: Path | None = None,
) -> dict:
    manifest, raw, missing = validated_rows(data_dir, run_dir, stage)
    rows, unreviewed = apply_reviews(raw, reviews)
    config = manifest["config"]
    short, long = config["lengths"][0], config["lengths"][-1]
    table_groups = defaultdict(list)
    for row in rows:
        table_groups[(row["method"], row["strength"], row["length"], row["delay"])].append(row)
    table = [
        {"method": key[0], "strength": key[1], "length": key[2], "delay": key[3], **metrics(values)}
        for key, values in sorted(table_groups.items())
    ]
    report = {
        "stage": stage,
        "rows": len(rows),
        "missing_cells": missing,
        "required_unfinished_reviews": unreviewed,
        "complete": not missing,
        "review_complete": not unreviewed,
        "metrics": table,
        "decision": "incomplete" if missing else "pending_review" if unreviewed else "ready",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_sha256": provenance()["source_sha256"],
        "results_sha256": digest(rows),
        "note": "Pilot thresholds guide investment; they are not proof of mechanisms or publication quality.",
    }
    write_jsonl(run_dir / stage / "scored-results.jsonl", rows)
    if missing:
        write_json(run_dir / stage / "summary.json", report)
        return report
    gates = config["gates"]
    if stage == "baseline":
        short_rows, long_rows = group(rows, "baseline", short), group(rows, "baseline", long)
        drift = paired(long_rows, short_rows, "language_ok", config)
        short_scores, long_scores = metrics(short_rows), metrics(long_rows)
        conditions = {
            "short_language_capable": short_scores["language_ok"]["successes"]
            >= gates["short_min_successes"],
            "short_content_capable": short_scores["content_ok"]["successes"] >= gates["short_min_successes"],
            "large_natural_drift": -drift["net_gains"] >= gates["drift_min_net_losses"],
            "long_content_preserved": long_scores["content_ok"]["successes"]
            >= gates["long_min_content_successes"],
        }
        report.update(conditions=conditions, drift=drift, provisional_gate_pass=all(conditions.values()))
    elif stage == "compare":
        base_short, base_long = group(rows, "baseline", short), group(rows, "baseline", long)
        replacement = group(rows, "replace", long)
        gain = paired(replacement, base_long, "joint_ok", config)
        content = paired(replacement, base_long, "content_ok", config)
        short_harm = paired(group(rows, "replace", short), base_short, "joint_ok", config)
        low_rank = paired(replacement, group(rows, "add_lowrank", long), "joint_ok", config)
        reminder = paired(replacement, group(rows, "reminder", long), "joint_ok", config)
        anchor_direction = paired(replacement, group(rows, "wrong_anchor", long), "language_ok", config)
        conditions = {
            "large_joint_recovery": gain["net_gains"] >= gates["repair_min_net_gains"],
            "long_content_preserved": content["net_gains"] >= -gates["max_content_net_losses"],
            "short_joint_preserved": short_harm["net_gains"] >= -gates["max_short_joint_net_losses"],
        }
        report.update(
            conditions=conditions,
            provisional_gate_pass=all(conditions.values()),
            recovery=gain,
            content_change=content,
            short_change=short_harm,
            replacement_vs_lowrank=low_rank,
            replacement_vs_reminder=reminder,
            french_vs_english_anchor=anchor_direction,
        )
    elif stage == "delay":
        _, compare_raw, compare_missing = validated_rows(data_dir, run_dir, "compare")
        compare, compare_unreviewed = apply_reviews(compare_raw, compare_reviews)
        if compare_missing:
            raise ValueError("Delay interpretation requires a complete immediate comparison.")
        if {row["runtime_sha256"] for row in rows + compare}.__len__() != 1:
            raise ValueError("Delay and comparison runtime fingerprints differ.")
        report["required_unfinished_compare_reviews"] = compare_unreviewed
        base = group(compare, "baseline", long)
        immediate = paired(group(compare, "replace", long), base, "joint_ok", config)
        delayed = []
        for delay in config["delays"]:
            repair = group(rows, "replace", long, delay)
            gain = paired(repair, base, "joint_ok", config)
            content = paired(repair, base, "content_ok", config)
            vs_add = paired(repair, group(rows, "add_lowrank", long, delay), "joint_ok", config)
            delayed.append(
                {
                    "delay": delay,
                    "recovery": gain,
                    "content_change": content,
                    "replacement_vs_lowrank": vs_add,
                }
            )
        final = delayed[-1]
        conditions = {
            "immediate_gain_positive": immediate["net_gains"] > 0,
            "retains_half_gain": final["recovery"]["net_gains"]
            >= gates["delay_min_retained_gain_fraction"] * immediate["net_gains"],
            "long_content_preserved": final["content_change"]["net_gains"]
            >= -gates["max_content_net_losses"],
        }
        report.update(
            conditions=conditions,
            immediate_recovery=immediate,
            delayed=delayed,
            provisional_gate_pass=all(conditions.values()),
        )
        unreviewed += compare_unreviewed
    if stage in {"baseline", "compare", "delay"}:
        report["decision"] = (
            "pending_review"
            if unreviewed
            else "continue_pilot"
            if report["provisional_gate_pass"]
            else "stop_current_configuration"
        )
    write_json(run_dir / stage / "summary.json", report)
    return report
