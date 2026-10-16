"""Paired confirmation reports with explicit human/model review provenance."""

from __future__ import annotations

import random
from collections import defaultdict

from .analysis import metrics, paired
from .confirm import assigned_shard, confirmation_cells, context, validate_result
from .confirm_data import load_confirmation
from .io import digest, read_json, read_jsonl, write_json, write_jsonl

FORMAL_PHASES = ("crossover-a", "crossover-b", "confirm")


def validated_results(data_dir, run_dir):
    manifest, examples = load_confirmation(data_dir)
    if read_json(run_dir / "experiment.json") != context(manifest):
        raise ValueError("Confirmation was run with different input or source code.")
    rows, missing = [], {}
    for phase in FORMAL_PHASES:
        expected = confirmation_cells(examples, phase)
        seen = set()
        for path in sorted((run_dir / phase).glob("results-*.jsonl")):
            for row in read_jsonl(path):
                cell = row["cell_id"]
                if cell in seen or cell not in expected:
                    raise ValueError("Duplicate or unregistered confirmation cell.")
                seen.add(cell)
                example, spec = expected[cell]
                shard = assigned_shard(example, phase)
                if path.name != f"results-{shard}.jsonl":
                    raise ValueError("Result is in the wrong shard file.")
                directory = run_dir / phase
                runtime = read_json(directory / f"runtime-{shard}.json")
                device = read_json(directory / f"device-{shard}.json")
                if read_json(run_dir / "runtime.json") != runtime:
                    raise ValueError("Worker differs from the experiment runtime.")
                validate_result(row, example, spec, manifest, phase, shard, runtime, device)
                rows.append(row)
        missing[phase] = sorted(expected.keys() - seen)
    if len({r["runtime_sha256"] for r in rows}) > 1:
        raise ValueError("Confirmation workers have different runtime/model fingerprints.")
    # The same physical device must occupy each slot across BOTH crossover rounds.
    slots = {}
    for phase in FORMAL_PHASES:
        for shard in (0, 1):
            path = run_dir / phase / f"device-{shard}.json"
            if path.exists():
                uuid = read_json(path)["gpu_uuid"]
                if shard in slots and slots[shard] != uuid:
                    raise ValueError("Physical GPU slot changed between phases; crossover is invalid.")
                slots[shard] = uuid
    if len(slots) == 2 and len(set(slots.values())) != 2:
        raise ValueError("Both workers used the same physical GPU.")
    return manifest, rows, missing


def export_confirmation_reviews(data_dir, run_dir, output):
    _, rows, _ = validated_results(data_dir, run_dir)
    if output.exists():
        raise ValueError("Review output exists; preserve it or choose another path.")
    random.Random(913).shuffle(rows)
    write_jsonl(
        output,
        [
            {
                "review_id": f"C{i:04d}",
                "cell_id": r["cell_id"],
                "output_sha256": r["output_sha256"],
                "answer": r["text"],
                "expected_project": r["target"]["project"],
                "expected_code": r["target"]["code"],
                "expected_budget": r["target"]["budget"],
                "language_ok": None,
                "content_ok": None,
                "reviewer_type": None,
                "reviewer_id": None,
                "notes": "",
            }
            for i, r in enumerate(rows)
        ],
    )
    return {
        "rows": len(rows),
        "output": str(output),
        "instruction": "Fill both booleans, reviewer_type ('human'/'model') and reviewer_id. "
        "Keep answer, expected fields, cell and output hash unchanged.",
    }


def reviewed_results(rows, path):
    by_cell = {r["cell_id"]: r for r in rows}
    reviews = read_jsonl(path) if path else []
    labels, seen = {}, set()
    for review in reviews:
        cell = review["cell_id"]
        row = by_cell.get(cell)
        if cell in seen or row is None or review["output_sha256"] != row["output_sha256"]:
            raise ValueError("Duplicate review or review/output binding mismatch.")
        seen.add(cell)
        if review["answer"] != row["text"] or any(
            review[f"expected_{k}"] != row["target"][k] for k in ("project", "code", "budget")
        ):
            raise ValueError("Review answer or expected target was changed.")
        if not all(type(review.get(k)) is bool for k in ("language_ok", "content_ok")):
            continue
        if (
            review.get("reviewer_type") not in {"human", "model"}
            or not str(review.get("reviewer_id") or "").strip()
        ):
            raise ValueError("Completed reviews need an explicit human/model type and reviewer identity.")
        if review.get("human_reviewed", review["reviewer_type"] == "human") != (
            review["reviewer_type"] == "human"
        ):
            raise ValueError("Review provenance fields contradict each other.")
        labels[cell] = review
    scored = []
    for original in rows:
        row = dict(original, scores=dict(original["scores"]))
        if row["cell_id"] in labels:
            label = labels[row["cell_id"]]
            row["scores"].update(
                language_ok=label["language_ok"],
                content_ok=label["content_ok"],
                joint_ok=label["language_ok"] and label["content_ok"],
            )
        scored.append(row)
    return scored, {
        "completed_reviews": len(labels),
        "required_reviews": len(rows),
        "review_complete": len(labels) == len(rows),
        "human_review_complete": bool(rows)
        and len(labels) == len(rows)
        and all(r["reviewer_type"] == "human" for r in labels.values()),
        "reviewer_types": sorted({r["reviewer_type"] for r in labels.values()}),
        "unfinished_cells": sorted(by_cell.keys() - labels.keys()),
    }


def select_rows(rows, phase="confirm", background="en", length=32768, method="baseline", delay=0):
    return [
        r
        for r in rows
        if (r["phase"], r["background"], r["length"], r["method"], r["delay"])
        == (phase, background, length, method, delay)
    ]


def evaluate(rows, config):
    gates = config["gates"]
    bootstrap = {"seed": 1729, "bootstrap_samples": config["bootstrap_samples"]}

    def comparison(left, right, metric="joint_ok"):
        return paired(left, right, metric, bootstrap)

    def disagreements(left, right, metric):
        effect = comparison(left, right, metric)
        return effect["failure_to_success"] + effect["success_to_failure"]

    def content_preserved(method, delay, phase="confirm"):
        values = fresh(method=method, delay=delay, phase=phase)
        return (
            metrics(values)["content_ok"]["successes"] >= gates["long_min_content_successes"]
            and comparison(values, fresh(delay=delay, phase=phase), "content_ok")["net_gains"]
            >= -gates["max_content_net_losses"]
        )

    fresh = lambda **kw: select_rows(rows, **kw)
    short = fresh(background="common", length=0)
    long = fresh()
    crossover = []
    for method in ("baseline", "add_lowrank", "replace"):
        a = select_rows(rows, phase="crossover-a", method=method, delay=4096)
        b = select_rows(rows, phase="crossover-b", method=method, delay=4096)
        a_by = {r["doc_id"]: r for r in a}
        b_by = {r["doc_id"]: r for r in b}
        joint = comparison(b, a)
        crossover.append(
            {
                "method": method,
                "a": metrics(a),
                "b": metrics(b),
                "a_by_shard": {str(s): metrics([r for r in a if r["shard"] == s]) for s in (0, 1)},
                "b_by_shard": {str(s): metrics([r for r in b if r["shard"] == s]) for s in (0, 1)},
                "b_minus_a": joint,
                "joint_disagreements": joint["failure_to_success"] + joint["success_to_failure"],
                "language_disagreements": sum(
                    a_by[d]["scores"]["language_ok"] != b_by[d]["scores"]["language_ok"] for d in a_by
                ),
                "content_disagreements": sum(
                    a_by[d]["scores"]["content_ok"] != b_by[d]["scores"]["content_ok"] for d in a_by
                ),
                "token_output_disagreements": sum(
                    a_by[d]["generated_ids"] != b_by[d]["generated_ids"] for d in a_by
                ),
            }
        )
    native_gap = comparison(fresh(method="add_lowrank", delay=4096), fresh(method="replace", delay=4096))
    matched_gap = comparison(
        fresh(method="add_lowrank", delay=4096), fresh(method="replace_norm_matched", delay=4096)
    )
    delayed_recovery = comparison(fresh(method="add_lowrank", delay=4096), fresh(delay=4096))
    replacement_delay_effect = comparison(fresh(method="replace", delay=4096), fresh(method="replace"))
    baseline_disagreements = {
        metric: disagreements(fresh(delay=4096), long, metric)
        for metric in ("language_ok", "content_ok", "joint_ok")
    }
    replay_gaps = [
        {
            "phase": phase,
            "additive_minus_replace": comparison(
                select_rows(rows, phase=phase, method="add_lowrank", delay=4096),
                select_rows(rows, phase=phase, method="replace", delay=4096),
            ),
            "language_gap": comparison(
                fresh(phase=phase, method="add_lowrank", delay=4096),
                fresh(phase=phase, method="replace", delay=4096),
                "language_ok",
            ),
            "additive_minus_no_op": comparison(
                fresh(phase=phase, method="add_lowrank", delay=4096), fresh(phase=phase, delay=4096)
            ),
        }
        for phase in ("crossover-a", "crossover-b")
    ]
    conditions = {
        "cross_gpu_joint_stable": all(
            x["joint_disagreements"] <= gates["crossover_max_disagreements_per_metric"] for x in crossover
        ),
        "cross_gpu_language_content_stable": all(
            x[f"{metric}_disagreements"] <= gates["crossover_max_disagreements_per_metric"]
            for x in crossover
            for metric in ("language", "content")
        ),
        "no_op_cut_stable": all(
            n <= gates["same_cut_baseline_max_disagreements_per_metric"]
            for n in baseline_disagreements.values()
        ),
        "replay_method_gap_replicated": all(
            x["additive_minus_replace"]["net_gains"] >= gates["method_gap_min_net_gains"] for x in replay_gaps
        ),
        "replay_recovery_and_language_gap": all(
            x["language_gap"]["net_gains"] >= gates["method_gap_min_net_gains"]
            and x["additive_minus_no_op"]["net_gains"] >= gates["recovery_min_net_gains"]
            for x in replay_gaps
        ),
        "replay_content_preserved": all(
            content_preserved(method, 4096, phase)
            for phase in ("crossover-a", "crossover-b")
            for method in ("add_lowrank", "replace")
        ),
        "short_language_capable": metrics(short)["language_ok"]["successes"] >= gates["short_min_successes"],
        "short_content_capable": metrics(short)["content_ok"]["successes"] >= gates["short_min_successes"],
        "natural_drift_replicated": -comparison(long, short, "language_ok")["net_gains"]
        >= gates["drift_min_net_losses"],
        "long_content_preserved": metrics(long)["content_ok"]["successes"]
        >= gates["long_min_content_successes"],
        "delayed_additive_effective": metrics(fresh(method="add_lowrank", delay=4096))["joint_ok"][
            "successes"
        ]
        >= gates["additive_min_successes"],
        "delayed_additive_beats_no_op": delayed_recovery["net_gains"] >= gates["recovery_min_net_gains"],
        "both_native_methods_restore_immediately": all(
            metrics(fresh(method=method))["joint_ok"]["successes"] >= gates["immediate_min_successes"]
            and comparison(fresh(method=method), long)["net_gains"] >= gates["recovery_min_net_gains"]
            for method in ("add_lowrank", "replace")
        ),
        "replacement_degrades_with_delay": -replacement_delay_effect["net_gains"]
        >= gates["replacement_delay_min_net_losses"],
        "native_method_gap_replicated": native_gap["net_gains"] >= gates["method_gap_min_net_gains"],
        "native_language_gap_replicated": comparison(
            fresh(method="add_lowrank", delay=4096), fresh(method="replace", delay=4096), "language_ok"
        )["net_gains"]
        >= gates["method_gap_min_net_gains"],
        "native_methods_content_preserved": all(
            content_preserved(method, delay) for method in ("add_lowrank", "replace") for delay in (0, 4096)
        ),
    }
    controls = []
    for delay in (0, 4096):
        for method in ("add_lowrank", "replace", "replace_norm_matched", "reminder"):
            controls.append(
                {
                    "delay": delay,
                    "method": method,
                    "vs_same_cut_baseline": comparison(fresh(method=method, delay=delay), fresh(delay=delay)),
                    "content_change": comparison(
                        fresh(method=method, delay=delay), fresh(delay=delay), "content_ok"
                    ),
                }
            )
    onset = {}
    for language in ("en", "fr"):
        points = []
        for length in config["lengths"][1:]:
            values = fresh(background=language, length=length)
            effect = comparison(values, short, "language_ok")
            points.append(
                {
                    "length": length,
                    "metrics": metrics(values),
                    "vs_short": effect,
                    "substantial_drop": -effect["net_gains"] >= gates["drift_min_net_losses"],
                }
            )
        onset[language] = {
            "points": points,
            "first_sampled_drop": next((p["length"] for p in points if p["substantial_drop"]), None),
        }
    return {
        "conditions": conditions,
        "provisional_decision": "confirmed" if all(conditions.values()) else "not_confirmed",
        "crossover": crossover,
        "replay_method_gaps": replay_gaps,
        "native_additive_minus_replace_4096": native_gap,
        "equal_norm_additive_minus_replace_4096": matched_gap,
        "equal_norm_gap_supported": (
            all(conditions.values())
            and matched_gap["net_gains"] >= gates["method_gap_min_net_gains"]
            and comparison(
                fresh(method="add_lowrank", delay=4096),
                fresh(method="replace_norm_matched", delay=4096),
                "language_ok",
            )["net_gains"]
            >= gates["method_gap_min_net_gains"]
            and content_preserved("replace_norm_matched", 4096)
        ),
        "delayed_additive_minus_no_op": delayed_recovery,
        "replacement_4096_minus_0": replacement_delay_effect,
        "controls": controls,
        "onset": onset,
        "french_minus_english_32k_language": comparison(fresh(background="fr"), long, "language_ok"),
        "delayed_reminder_minus_additive": comparison(
            fresh(method="reminder", delay=4096), fresh(method="add_lowrank", delay=4096)
        ),
        "same_cut_baseline_4096_minus_0": comparison(fresh(delay=4096), long),
        "same_cut_baseline_disagreements": baseline_disagreements,
    }


def summarize_confirmation(data_dir, run_dir, reviews=None):
    manifest, raw, missing = validated_results(data_dir, run_dir)
    rows, review = reviewed_results(raw, reviews)
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("phase", "background", "length", "method", "delay"))].append(row)
    table = [
        dict(zip(("phase", "background", "length", "method", "delay"), key), **metrics(values))
        for key, values in sorted(groups.items())
    ]
    complete = not any(missing.values())
    report = {
        **context(manifest),
        "protocol": manifest["protocol"],
        "rows": len(rows),
        "expected_rows": 768,
        "complete": complete,
        "missing_cells": missing,
        **review,
        "metrics": table,
        "metrics_status": "reviewed" if complete and review["review_complete"] else "preliminary",
        "results_sha256": digest(raw),
        "reviews_sha256": digest(read_jsonl(reviews)) if reviews else None,
        "decision": "incomplete" if not complete else "pending_review",
        "limitations": [
            (
                "English/French Wikipedia corpora are not sentence-aligned translations; their comparison "
                "does not isolate language from topic/style."
            ),
            "Norm matching is per layer, uses each unedited state, and is not an idempotent replacement.",
            "Every delay has its own independent prefill from token zero and its own no-op baseline.",
            (
                "Delay comparisons edit different input positions and states, not the same restored state at "
                "two later readouts; they establish position-conditioned robustness, not a forgetting rate."
            ),
            "Bootstrap intervals condition on 32 documents; 32/32 is not a population guarantee.",
            "State editing demonstrates behavioral control, not exclusive natural storage in recurrence.",
        ],
    }
    report["required_reviews"] = 768
    report["review_complete"] = complete and review["review_complete"]
    report["human_review_complete"] = complete and review["human_review_complete"]
    if complete and review["review_complete"]:
        report.update(evaluate(rows, manifest["config"]))
        report["decision"] = (
            report["provisional_decision"] if review["human_review_complete"] else "pending_human_review"
        )
    write_json(run_dir / "confirmation-summary.json", report)
    return report
