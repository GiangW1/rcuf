#!/usr/bin/env python3
"""Verify the published pilot evidence using only Python's standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def counts(rows):
    n = len(rows)
    return {
        "n": n,
        **{
            key: {"successes": sum(row["scores"][key] for row in rows),
                  "rate": sum(row["scores"][key] for row in rows) / n}
            for key in ("language_ok", "content_ok", "joint_ok")
        },
    }


def group(rows, method, length, delay=0):
    return [r for r in rows if (r["method"], r["length"], r["delay"]) == (method, length, delay)]


def net(left, right, metric="joint_ok"):
    a = {r["doc_id"]: int(r["scores"][metric]) for r in left}
    b = {r["doc_id"]: int(r["scores"][metric]) for r in right}
    require(len(a) == len(left) and len(b) == len(right) and a.keys() == b.keys(), "Pairing mismatch")
    return sum(a[k] - b[k] for k in a)


def verify(root, source_root):
    listed = set()
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, relative = line.split("  ", 1)
        path = root / relative
        require(path.resolve().is_relative_to(root.resolve()), "Checksum path outside bundle")
        require(relative not in listed, f"Duplicate checksum entry: {relative}")
        listed.add(relative)
        require(hashlib.sha256(path.read_bytes()).hexdigest() == checksum, f"File hash mismatch: {relative}")
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
    require(listed == actual, "Checksum manifest does not cover exactly the bundle files")
    artifacts = root / "artifacts"
    experiment = read(artifacts / "experiment.json")
    manifest = read(root / "data-manifest.json")
    require(manifest["manifest_sha256"] == digest({k: v for k, v in manifest.items()
                                                 if k != "manifest_sha256"}), "Manifest hash mismatch")
    require(digest(manifest["config"]) == manifest["config_sha256"], "Config hash mismatch")
    require(read(root / "config-frozen.json") == manifest["config"], "Frozen config differs")
    require(experiment["manifest_sha256"] == manifest["manifest_sha256"], "Experiment/data mismatch")
    source = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((source_root / "rcuf").glob("*.py"))}
    require(source and digest(source) == experiment["source_sha256"], "Source fingerprint mismatch")
    selection = read(artifacts / "selection.json")
    result = read(artifacts / "pilot-result-model-reviewed.json")
    require(selection["split"] == "dev", "Parameters were not selected on development split")
    require(result["selection"] == selection, "Final/frozen selection mismatch")
    runtime_hashes = set()
    scored_stages = {}
    new_count, reused_count = 0, 0
    for stage, expected in (("baseline", 96), ("tune", 160), ("compare", 384), ("delay", 192)):
        directory = artifacts / stage
        rows = [r for path in sorted(directory.glob("results-*.jsonl")) for r in read_lines(path)]
        registry = [cell for path in sorted(directory.glob("metadata-*.json"))
                    for cell in read(path)["expected_cells"]]
        require(len(rows) == expected and len({r["cell_id"] for r in rows}) == expected,
                f"{stage}: incomplete/duplicate cells")
        require(len(registry) == expected and set(registry) == {r["cell_id"] for r in rows},
                f"{stage}: registered cells differ")
        reviews = read_lines(artifacts / f"{stage}-model-review.jsonl")
        by_cell = {r["cell_id"]: r for r in reviews}
        require(len(reviews) == len(by_cell), f"{stage}: duplicate reviews")
        original = read_lines(artifacts / f"{stage}-review.jsonl")
        require(all(r["language_ok"] is None and r["content_ok"] is None for r in original),
                f"{stage}: original review export changed")
        require({r["cell_id"] for r in original} == by_cell.keys(), f"{stage}: review export mismatch")
        scored = []
        for original_row in rows:
            row = dict(original_row, scores=dict(original_row["scores"]))
            cell = row["cell_id"]
            spec = {key: row[key] for key in ("doc_id", "length", "method", "strength", "delay")}
            require(digest(spec)[:24] == cell, f"{stage}: wrong cell ID")
            require(row["output_sha256"] == digest(row["text"]), f"{stage}: altered output")
            require(row["manifest_sha256"] == manifest["manifest_sha256"], f"{stage}: data fingerprint")
            runtime_hashes.add(row["runtime_sha256"])
            review = by_cell.get(cell)
            if review:
                require(review["output_sha256"] == row["output_sha256"] and review["answer"] == row["text"],
                        f"{stage}: review/output binding mismatch")
                require((review["expected_project"], review["expected_code"], review["expected_budget"]) ==
                        (row["target"]["project"], row["target"]["code"], row["target"]["budget"]),
                        f"{stage}: review target mismatch")
                require(review["human_reviewed"] is False and review["reviewer_type"] == "AI model (Codex)",
                        f"{stage}: review provenance mismatch")
                require(all(type(review[k]) is bool for k in ("language_ok", "content_ok")),
                        f"{stage}: unfinished labels")
                row["scores"].update(language_ok=review["language_ok"], content_ok=review["content_ok"],
                                     joint_ok=review["language_ok"] and review["content_ok"], reviewed=True)
            require(row["scores"].get("reviewed"), f"{stage}: unreviewed cell")
            if row.get("reused_from"):
                reused_count += 1
                require(stage == "compare" and row["reused_from"] == "baseline", "Unexpected reuse")
                baseline = next(r for r in scored_stages["baseline"] if r["cell_id"] == cell)
                require(row["scores"] == baseline["scores"] and row["text"] == baseline["text"], "Reuse differs")
            else:
                new_count += 1
                require(cell in by_cell, f"{stage}: missing new-output audit")
            require(row["scores"]["joint_ok"] == (row["scores"]["language_ok"] and row["scores"]["content_ok"]),
                    f"{stage}: joint score mismatch")
            scored.append(row)
        require(scored == read_lines(directory / "scored-results.jsonl"), f"{stage}: scored rows mismatch")
        summary = read(directory / "summary.json")
        require(digest(scored) == summary["results_sha256"], f"{stage}: scored hash mismatch")
        require(summary == result["stages"][stage], f"{stage}: final/individual summary differs")
        require(summary["human_review_complete"] is False and summary["complete"] and summary["review_complete"],
                f"{stage}: misleading completeness fields")
        groups = defaultdict(list)
        for row in scored:
            groups[tuple(row[k] for k in ("method", "strength", "length", "delay"))].append(row)
        table = [dict(zip(("method", "strength", "length", "delay"), key), **counts(values))
                 for key, values in sorted(groups.items())]
        require(table == summary["metrics"], f"{stage}: metric table mismatch")
        scored_stages[stage] = scored
    require(len(runtime_hashes) == 1 and selection["runtime_sha256"] in runtime_hashes, "Runtime mismatch")
    require(digest(scored_stages["tune"]) == selection["reviewed_results_sha256"], "Selection binding mismatch")
    require(new_count == 768 and reused_count == 64, "Generation/reuse count mismatch")
    gates = manifest["config"]["gates"]
    baseline = scored_stages["baseline"]
    short, long = group(baseline, "baseline", 0), group(baseline, "baseline", 32768)
    conditions = {
        "short_language_capable": counts(short)["language_ok"]["successes"] >= gates["short_min_successes"],
        "short_content_capable": counts(short)["content_ok"]["successes"] >= gates["short_min_successes"],
        "large_natural_drift": -net(long, short, "language_ok") >= gates["drift_min_net_losses"],
        "long_content_preserved": counts(long)["content_ok"]["successes"] >= gates["long_min_content_successes"],
    }
    compare = scored_stages["compare"]
    base, repair = group(compare, "baseline", 32768), group(compare, "replace", 32768)
    immediate = net(repair, base)
    compare_conditions = {
        "large_joint_recovery": immediate >= gates["repair_min_net_gains"],
        "long_content_preserved": net(repair, base, "content_ok") >= -gates["max_content_net_losses"],
        "short_joint_preserved": net(group(compare, "replace", 0), group(compare, "baseline", 0))
        >= -gates["max_short_joint_net_losses"],
    }
    delay = scored_stages["delay"]
    delayed_base, delayed_repair = group(delay, "baseline", 32768, 4096), group(delay, "replace", 32768, 4096)
    delay_conditions = {
        "immediate_gain_positive": immediate > 0,
        "retains_half_gain": net(delayed_repair, delayed_base)
        >= gates["delay_min_retained_gain_fraction"] * immediate,
        "long_content_preserved": net(delayed_repair, delayed_base, "content_ok") >= -gates["max_content_net_losses"],
    }
    for stage, evaluated in (("baseline", conditions), ("compare", compare_conditions), ("delay", delay_conditions)):
        summary = result["stages"][stage]
        expected_decision = "continue_pilot" if all(evaluated.values()) else "stop_current_configuration"
        require(summary["conditions"] == evaluated and summary["decision"] == expected_decision,
                f"{stage}: gate decision mismatch")
    print(f"Verified {len(listed)} file hashes, frozen source/config, 768 new outputs + 64 reused rows,")
    print("model-review bindings, scored rows, metric tables, selection binding and stage gates.")
    print("Human blind review remains incomplete. Delay gate: stop_current_configuration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    verify(Path(__file__).resolve().parent, args.source_root)
