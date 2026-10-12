"""Synthetic result fixtures exercise reporting/gates only; these are not model experiment results."""

from copy import deepcopy
from pathlib import Path

import pytest

from rcuf.analysis import select, summarize
from rcuf.io import append_jsonl, digest, file_digest, load_config, provenance, read_jsonl, write_json
from rcuf.runner import expected_cells, require_previous_gate


def make_data(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.json")
    data, run = tmp_path / "data", tmp_path / "runs"
    examples = []
    for split in ("dev", "test"):
        for i in range(2):
            for length in config["lengths"]:
                row = {
                    "doc_id": f"{split}-{i:03d}",
                    "doc_index": i,
                    "split": split,
                    "length": length,
                    "target": {"code": "ZUM-482", "budget": 42700},
                }
                examples.append(row)
                append_jsonl(data / "examples.jsonl", row)
    manifest = {
        "config": config,
        "counts": {"dev": 2, "test": 2},
        "examples_sha256": file_digest(data / "examples.jsonl"),
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(data / "manifest.json", manifest)
    write_json(
        run / "experiment.json",
        {"manifest_sha256": manifest["manifest_sha256"], "source_sha256": provenance()["source_sha256"]},
    )
    return data, run, manifest, examples


def add_results(run, manifest, examples, stage, selection=None, missing_last=False):
    expected = expected_cells(stage, manifest, examples, selection)
    items = list(expected.items())
    if missing_last:
        items = items[:-1]
    for cell, (example, spec) in items:
        language = example["length"] == 0 or (
            spec["method"] not in {"baseline", "wrong_anchor"} and spec["strength"] >= 1
        )
        text = f"synthetic fixture {cell}"
        append_jsonl(
            run / stage / "results-0.jsonl",
            {
                **example,
                **spec,
                "cell_id": cell,
                "text": text,
                "output_sha256": digest(text),
                "manifest_sha256": manifest["manifest_sha256"],
                "runtime_sha256": "fixture-runtime",
                "scores": {
                    "language_ok": language,
                    "content_ok": True,
                    "joint_ok": language,
                    "needs_review": False,
                },
            },
        )


def reviews_for(run, stage):
    path = run / f"{stage}-reviews.jsonl"
    for row in read_jsonl(run / stage / "results-0.jsonl"):
        append_jsonl(
            path,
            {
                "cell_id": row["cell_id"],
                "output_sha256": row["output_sha256"],
                "language_ok": row["scores"]["language_ok"],
                "content_ok": True,
            },
        )
    return path


def test_incomplete_and_unreviewed_results_cannot_unlock_next_stage(tmp_path):
    data, run, manifest, examples = make_data(tmp_path)
    add_results(run, manifest, examples, "baseline", missing_last=True)
    report = summarize(data, run, "baseline")
    assert report["decision"] == "incomplete"
    assert len(report["missing_cells"]) == 1
    with pytest.raises(ValueError):
        require_previous_gate(run, "tune", manifest)


def test_complete_reviewed_pipeline_freezes_dev_selection_and_compares_delays(tmp_path):
    data, run, manifest, examples = make_data(tmp_path)
    add_results(run, manifest, examples, "baseline")
    assert summarize(data, run, "baseline")["decision"] == "pending_review"
    base_reviews = reviews_for(run, "baseline")
    assert summarize(data, run, "baseline", base_reviews)["decision"] == "continue_pilot"
    require_previous_gate(run, "tune", manifest)
    add_results(run, manifest, examples, "tune")
    with pytest.raises(ValueError):
        select(data, run)
    tuned = select(data, run, reviews_for(run, "tune"))
    assert tuned["split"] == "dev"
    assert tuned["strengths"] == {"add_full": 1.0, "add_lowrank": 1.0, "replace": 1.0}
    add_results(run, manifest, examples, "compare", tuned)
    compare_reviews = reviews_for(run, "compare")
    assert summarize(data, run, "compare", compare_reviews)["decision"] == "continue_pilot"
    require_previous_gate(run, "delay", manifest)
    add_results(run, manifest, examples, "delay", tuned)
    delayed = summarize(data, run, "delay", reviews_for(run, "delay"), compare_reviews)
    assert delayed["decision"] == "continue_pilot"
    assert delayed["delayed"][-1]["recovery"]["net_gains"] == 2
    modified = deepcopy(tuned)
    modified["split"] = "test"
    write_json(run / "selection.json", modified)
    with pytest.raises(ValueError):
        summarize(data, run, "delay")
