"""Document-paired control loss, baseline language preferences and explicit review provenance."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np

from .controls_data import BACKGROUND, POLICIES, QUESTIONS
from .io import digest, read_jsonl, write_json, write_jsonl

LANGUAGES = {"en", "fr", "mixed", "other", "unknown"}


def apply_reviews(rows, path, registered_cells=None):
    by_cell = {r["cell_id"]: r for r in rows}
    known = set(by_cell) if registered_cells is None else set(registered_cells)
    labels, seen = {}, set()
    for review in read_jsonl(path) if path else []:
        cell = review["cell_id"]
        if cell in seen or cell not in known:
            raise ValueError("Duplicate or unregistered controls review.")
        seen.add(cell)
        if cell not in by_cell:
            continue  # a language gate may read a review file containing format cells too
        row = by_cell[cell]
        if (
            review["output_sha256"] != row["output_sha256"]
            or review["answer"] != row["text"]
            or review["expected_target"] != row["target"]
        ):
            raise ValueError("Controls review answer/target/output binding mismatch.")
        complete = type(review.get("content_ok")) is bool
        if row["family"] == "language":
            complete &= review.get("response_language") in LANGUAGES
        if not complete:
            continue
        if (
            review.get("reviewer_type") not in {"human", "model"}
            or not str(review.get("reviewer_id") or "").strip()
        ):
            raise ValueError("Completed controls reviews require reviewer_type and reviewer_id.")
        if review.get("human_reviewed", review["reviewer_type"] == "human") != (
            review["reviewer_type"] == "human"
        ):
            raise ValueError("Contradictory controls review provenance.")
        labels[cell] = review
    scored = []
    for original in rows:
        row = dict(original, scores=dict(original["scores"]))
        if row["cell_id"] in labels:
            label = labels[row["cell_id"]]
            row["scores"].update(content_ok=label["content_ok"], reviewed=True)
            if row["family"] == "language":
                observed = label["response_language"]
                rule = None if row["policy"] == "none" else observed == row["policy"] and not row["truncated"]
                row["scores"].update(response_language=observed, rule_ok=rule)
            row["scores"]["joint_ok"] = row["scores"]["rule_ok"] is True and label["content_ok"]
        scored.append(row)
    return scored, {
        "completed_reviews": len(labels),
        "required_reviews": len(rows),
        "review_complete": bool(rows) and len(labels) == len(rows),
        "human_review_complete": bool(rows)
        and len(labels) == len(rows)
        and all(r["reviewer_type"] == "human" for r in labels.values()),
        "reviewer_types": sorted({r["reviewer_type"] for r in labels.values()}),
    }


def evaluate_family(rows, config, family):
    values = [r for r in rows if r["family"] == family]
    expected_n = 288 if family == "language" else 192
    if len(values) != expected_n:
        return {"complete": False, "rows": len(values), "expected_rows": expected_n}
    groups = defaultdict(list)
    for row in values:
        groups[row["policy"], row["background"], row["question_language"]].append(row)
    expected = {(p, b, q) for p in POLICIES[family] for b in BACKGROUND for q in QUESTIONS}
    if set(groups) != expected or any(
        len(g) != 16 or len({r["doc_id"] for r in g}) != 16 for g in groups.values()
    ):
        raise ValueError("Incomplete or duplicated document pairing in controls analysis.")
    lookup = {(r["doc_id"], r["policy"], r["background"], r["question_language"]): r for r in values}
    policies = ("en", "fr") if family == "language" else ("json", "xml")
    docs = sorted({r["doc_id"] for r in values})
    switches, joint_switches = {}, {}
    for doc in docs:
        for background in BACKGROUND:
            for question in QUESTIONS:
                pair = [lookup[doc, policy, background, question]["scores"] for policy in policies]
                switches[doc, background, question] = int(all(s["rule_ok"] is True for s in pair))
                joint_switches[doc, background, question] = int(all(s["joint_ok"] for s in pair))
    # Each document contributes ONE mean loss across both backgrounds and both queries.
    # Positive values mean reduced system control after adding background.
    loss = np.array(
        [
            np.mean([switches[d, "none", q] - switches[d, b, q] for b in ("en", "fr") for q in QUESTIONS])
            for d in docs
        ]
    )
    question_loss = {
        q: float(np.mean([switches[d, "none", q] - switches[d, b, q] for d in docs for b in ("en", "fr")]))
        for q in QUESTIONS
    }
    rng = np.random.default_rng(1729)
    draws = rng.choice(loss, size=(config["bootstrap_samples"], len(loss)), replace=True).mean(1)
    gates = config["gates"]
    short_capable = all(
        sum(r["scores"]["rule_ok"] is True and r["scores"]["content_ok"] for r in group)
        >= gates["short_min_successes"]
        for (policy, background, _), group in groups.items()
        if background == "none" and policy != "none"
    )
    content_preserved = all(
        sum(r["scores"]["content_ok"] for r in group) >= gates["content_min_successes"]
        for group in groups.values()
    )
    conditions = {
        "short_capable": short_capable,
        "content_preserved": content_preserved,
        "control_loss_at_least_25_points": float(loss.mean()) >= gates["min_control_loss"],
        "at_least_four_affected_documents": int((loss > 0).sum()) >= gates["min_affected_documents"],
        "both_query_languages_decline": all(x > 0 for x in question_loss.values()),
    }
    table = [
        {
            "policy": p,
            "background": b,
            "question_language": q,
            "n": len(group),
            "rule_successes": None if p == "none" else sum(r["scores"]["rule_ok"] is True for r in group),
            "content_successes": sum(r["scores"]["content_ok"] for r in group),
            "joint_successes": None if p == "none" else sum(r["scores"]["joint_ok"] for r in group),
            **(
                {
                    "response_languages": {
                        lang: sum(r["scores"]["response_language"] == lang for r in group)
                        for lang in sorted(LANGUAGES)
                    }
                }
                if family == "language"
                else {}
            ),
        }
        for (p, b, q), group in sorted(groups.items())
    ]
    result = {
        "complete": True,
        "rows": len(values),
        "conditions": conditions,
        "eligible": all(conditions.values()),
        "metrics": table,
        "mean_control_loss": float(loss.mean()),
        "affected_documents": int((loss > 0).sum()),
        "loss_by_query_language": question_loss,
        "document_losses": dict(zip(docs, loss.tolist())),
        "paired_document_bootstrap_95ci": np.quantile(draws, [0.025, 0.975]).tolist(),
        "switch_successes": [
            {
                "background": b,
                "question_language": q,
                "rule": sum(switches[d, b, q] for d in docs),
                "joint": sum(joint_switches[d, b, q] for d in docs),
                "n": len(docs),
            }
            for b in BACKGROUND
            for q in QUESTIONS
        ],
    }
    if family == "language":
        # Instruction benefit over the no-language-rule condition at the SAME input.
        result["instruction_benefit"] = [
            {
                "target": p,
                "background": b,
                "question_language": q,
                "net_language_gains": sum(
                    int(lookup[d, p, b, q]["scores"]["response_language"] == p)
                    - int(lookup[d, "none", b, q]["scores"]["response_language"] == p)
                    for d in docs
                ),
                "n": len(docs),
            }
            for p in policies
            for b in BACKGROUND
            for q in QUESTIONS
        ]
    return result


def review_export(data_dir, run_dir, output):
    from .controls import validated_results

    _, rows, _ = validated_results(data_dir, run_dir)
    if output.exists():
        raise ValueError("Review file already exists; preserve existing labels.")
    random.Random(26101719).shuffle(rows)
    write_jsonl(
        output,
        [
            {
                "review_id": f"L{i:04d}",
                "cell_id": r["cell_id"],
                "output_sha256": r["output_sha256"],
                "answer": r["text"],
                "expected_target": r["target"],
                "response_language": None,
                "content_ok": None,
                "reviewer_type": None,
                "reviewer_id": None,
                "notes": "",
            }
            for i, r in enumerate(rows)
        ],
    )
    return {"rows": len(rows), "output": str(output)}


def summarize(data_dir, run_dir, reviews=None):
    from .controls import context, validated_results

    manifest, raw, missing = validated_results(data_dir, run_dir)
    rows, review = apply_reviews(raw, reviews)
    a = evaluate_family(rows, manifest["config"], "language")
    b = evaluate_family(rows, manifest["config"], "format")
    if not a["complete"]:
        candidate = "incomplete_language"
    elif not a["conditions"]["short_capable"] or not a["conditions"]["content_preserved"]:
        candidate = "language_task_invalid_for_attribution"
    elif not a["eligible"]:
        candidate = "language_effect_not_replicated_under_controls"
    elif not b["complete"]:
        candidate = "ready_for_format_gate"
    elif not b["conditions"]["short_capable"] or not b["conditions"]["content_preserved"]:
        candidate = "format_task_invalid_for_attribution"
    elif b["eligible"]:
        candidate = "beyond_language_supported_at_8k"
    else:
        candidate = "language_only_evidence_at_8k"
    report = {
        **context(manifest),
        "protocol": manifest["protocol"],
        "rows": len(rows),
        "expected_rows": 480,
        "missing_cells": missing,
        "complete": not any(missing.values()),
        "language": a,
        "format": b,
        **review,
        "results_sha256": digest(raw),
        "reviews_sha256": digest(read_jsonl(reviews)) if reviews else None,
        "provisional_interpretation": candidate,
        "decision": candidate if review["human_review_complete"] else "pending_human_review",
        "metrics_status": "reviewed" if review["review_complete"] else "preliminary",
        "limitations": [
            "Sixteen passages are the statistical units; repeated questions and policies are paired conditions.",
            "OPUS Books contains translated literary text and imperfect sentence alignments, not Wikipedia.",
            "Passages can share a source book; intervals do not establish corpus-wide or model-wide generality.",
            "Length matching does not remove all translation or style differences.",
            "An 8K format null result does not establish stability at longer lengths or for other rules.",
            "These behavioral tests do not identify a recurrent or attention mechanism.",
        ],
    }
    write_json(run_dir / "controls-summary.json", report)
    return report
