"""Crossed-cell fixtures validate inference and decisions, not experimental outcomes."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_hf_continuation import tiny_engine

from rcuf.controls import cells, context, generate_answer, require_language_gate, validated_results
from rcuf.controls_analysis import apply_reviews, evaluate_family, review_export, summarize
from rcuf.controls_data import BACKGROUND, POLICIES, QUESTIONS, collect_pairs, load_controls, validate_config
from rcuf.controls_scoring import parse_format, score
from rcuf.io import digest, file_digest, read_json, read_jsonl, write_json, write_jsonl


def config():
    return read_json(Path(__file__).resolve().parents[1] / "configs/controls.json")


def examples():
    rows = []
    for i in range(16):
        record = {"project": f"TESTX-{100 + i}", "code": "ZUM-482", "budget": 42700}
        for family, policies in POLICIES.items():
            for background in BACKGROUND:
                for q in QUESTIONS:
                    for p in policies:
                        ids = [3, 4, 5, 6, 7 + len(rows) % 100]
                        row = {
                            "doc_id": f"controls-{i:03d}",
                            "doc_index": i,
                            "family": family,
                            "policy": p,
                            "background": background,
                            "question_language": q,
                            "target": record,
                            "records": [record],
                            "target_position": 0,
                            "input_ids": ids,
                            "input_sha256": digest(ids),
                            "question_cut": 3,
                            "reference_tokens": 0 if background == "none" else 8192,
                            "reference_sha256": digest(
                                "" if background == "none" else f"fixture {i} {background}"
                            ),
                        }
                        row["cell_id"] = digest(row)[:24]
                        rows.append(row)
    return rows


def sealed_data(directory):
    rows = examples()
    write_jsonl(directory / "examples.jsonl", rows)
    write_jsonl(
        directory / "backgrounds.jsonl",
        [
            {
                "doc_id": f"controls-{i:03d}",
                "doc_index": i,
                "texts": {lang: f"fixture {i} {lang}" for lang in QUESTIONS},
                "tokens": {lang: 8192 for lang in QUESTIONS},
                "units": [{"id": str(i), "sha256": digest(i)}],
            }
            for i in range(16)
        ],
    )
    manifest = {
        "protocol": config()["protocol"],
        "config": config(),
        "examples_sha256": file_digest(directory / "examples.jsonl"),
        "backgrounds_sha256": file_digest(directory / "backgrounds.jsonl"),
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(directory / "manifest.json", manifest)
    return manifest, rows


def test_registered_design_and_frozen_crossing(tmp_path):
    manifest, rows = sealed_data(tmp_path)
    assert len(load_controls(tmp_path)[1]) == 480
    assert {p: len(cells(rows, p)) for p in POLICIES} == {"language": 288, "format": 192}
    assert len(cells(rows, "language", 0)) == len(cells(rows, "language", 1)) == 144
    assert len(cells(rows, "format", 0)) == len(cells(rows, "format", 1)) == 96
    changed = config()
    changed["gates"]["min_control_loss"] = 0.01
    with pytest.raises(ValueError, match="registered"):
        validate_config(changed)
    rows[-1]["input_ids"][0] = 100
    write_jsonl(tmp_path / "examples.jsonl", rows)
    with pytest.raises(ValueError, match="fingerprint"):
        load_controls(tmp_path)
    assert manifest["protocol"] == "rcuf-language-controls-v1"


@pytest.mark.parametrize(
    "policy,text",
    [
        ("json", '{"code":"ZUM-482","budget_eur":42700}'),
        ("xml", "<answer><code>ZUM-482</code><budget_eur>42700</budget_eur></answer>"),
    ],
)
def test_format_parser_and_truth_are_distinct(policy, text):
    e = dict(examples()[0], family="format", policy=policy)
    assert parse_format(text, policy) == {"code": "ZUM-482", "budget_eur": 42700}
    assert score(text, e)["joint_ok"]
    wrong = score(text.replace("42700", "42800"), e)
    assert wrong["rule_ok"] and not wrong["content_ok"]
    wrong = score(text.replace("ZUM-482", "wrong").replace("42700", "-7"), e)
    assert wrong["rule_ok"] and not wrong["content_ok"]
    assert not score(text, e, truncated=True)["rule_ok"]


@pytest.mark.parametrize(
    "policy,text",
    [
        ("json", '{"code":"ZUM-482","code":"XXX-111","budget_eur":42700}'),
        ("json", '{"code":"ZUM-482","budget_eur":true}'),
        ("json", '{"code":"ZUM-482","budget_eur":"42700"}'),
        ("json", '```json\n{"code":"ZUM-482","budget_eur":42700}\n```'),
        ("json", '{"code":"ZUM-482","budget_eur":42700,"note":"extra"}'),
        ("xml", "<answer><code>ZUM-482</code><budget_eur>42700</budget_eur><code>ZUM-482</code></answer>"),
        ("xml", "<!DOCTYPE answer><answer><code>ZUM-482</code><budget_eur>42700</budget_eur></answer>"),
        ("xml", '<answer x="1"><code>ZUM-482</code><budget_eur>42700</budget_eur></answer>'),
        ("xml", "<answer><code>ZUM-482</code>explanation<budget_eur>42700</budget_eur></answer>"),
    ],
)
def test_format_parser_rejects_ambiguous_or_extra_output(policy, text):
    assert parse_format(text, policy) is None


def test_both_output_languages_and_no_rule_are_scored():
    e = examples()[0]
    english = "The authorization code is ZUM-482. The approved budget is 42,700 EUR."
    french = "Le code d'autorisation est ZUM-482. Le budget approuvé est de 42 700 euros."
    assert score(english, dict(e, policy="en"))["joint_ok"]
    assert score(french, dict(e, policy="fr"))["joint_ok"]
    assert not score(english, dict(e, policy="fr"))["rule_ok"]
    neutral = score(english, dict(e, policy="none"))
    assert neutral["rule_ok"] is None and neutral["response_language"] == "en"
    mixed = "Le code d'autorisation est ZUM-482. The approved budget is 42,700 EUR."
    assert score(mixed, dict(e, policy="fr"))["response_language"] == "mixed"
    assert not score(mixed, dict(e, policy="en"))["rule_ok"]


def labeled_rows(format_decline=True):
    rows = []
    for e in examples():
        long = e["background"] != "none"
        observed = e["background"] if long else e["policy"]
        if e["policy"] == "none":
            observed = e["question_language"] if not long else e["background"]
        rule = (
            (None if e["policy"] == "none" else observed == e["policy"])
            if e["family"] == "language"
            else not (long and format_decline and e["policy"] == "json")
        )
        row = {
            **e,
            "phase": e["family"],
            "shard": e["doc_index"] % 2,
            "text": e["cell_id"],
            "output_sha256": digest(e["cell_id"]),
            "generated_ids": [5],
            "generated_tokens": 1,
            "truncated": False,
            "suffix_tokens": 2,
            "scores": {
                "rule_ok": rule,
                "content_ok": True,
                "joint_ok": rule is True,
                **({"response_language": observed} if e["family"] == "language" else {}),
            },
        }
        rows.append(row)
    return rows


def test_document_paired_gate_and_language_only_result():
    rows = labeled_rows()
    a = evaluate_family(rows, config(), "language")
    assert a["eligible"] and a["mean_control_loss"] == 1.0
    assert len(a["document_losses"]) == 16 and a["affected_documents"] == 16
    assert all(x == 1 for x in a["loss_by_query_language"].values())
    assert evaluate_family(rows, config(), "format")["eligible"]
    assert not evaluate_family(labeled_rows(False), config(), "format")["eligible"]
    # Missing capability/content must not be called instruction maintenance failure.
    for r in rows:
        if r["background"] == "none" and r["policy"] == "fr":
            r["scores"].update(rule_ok=False, joint_ok=False)
    assert not evaluate_family(rows, config(), "language")["conditions"]["short_capable"]


def test_effect_from_only_one_query_language_does_not_pass():
    rows = labeled_rows()
    for r in rows:
        if r["family"] == "language" and r["question_language"] == "fr" and r["policy"] != "none":
            r["scores"].update(rule_ok=True, joint_ok=True, response_language=r["policy"])
    report = evaluate_family(rows, config(), "language")
    assert report["mean_control_loss"] == 0.5
    assert not report["eligible"] and not report["conditions"]["both_query_languages_decline"]


def reviews(rows, reviewer_type="model"):
    return [
        {
            "cell_id": r["cell_id"],
            "output_sha256": r["output_sha256"],
            "answer": r["text"],
            "expected_target": r["target"],
            "content_ok": r["scores"]["content_ok"],
            "response_language": r["scores"].get("response_language"),
            "reviewer_type": reviewer_type,
            "reviewer_id": "fixture-reviewer",
        }
        for r in rows
    ]


def stored_run(data_dir, run_dir):
    manifest, examples_ = sealed_data(data_dir)
    write_json(run_dir / "experiment.json", context(manifest))
    write_json(run_dir / "runtime.json", {"fixture": "runtime"})
    by_cell = {r["cell_id"]: r for r in labeled_rows()}
    for phase in POLICIES:
        for s in (0, 1):
            directory = run_dir / phase
            write_json(directory / f"runtime-{s}.json", {"fixture": "runtime"})
            write_json(directory / f"device-{s}.json", {"gpu_uuid": f"GPU-{s}"})
            write_json(run_dir / f"device-slot-{s}.json", {"gpu_uuid": f"GPU-{s}"})
            write_json(
                directory / f"metadata-{s}.json",
                {**context(manifest), "expected_cells": sorted(cells(examples_, phase, s))},
            )
            rows = []
            for key in cells(examples_, phase, s):
                row = {k: v for k, v in by_cell[key].items() if k != "input_ids"}
                row.update(
                    manifest_sha256=manifest["manifest_sha256"],
                    runtime_sha256=digest({"fixture": "runtime"}),
                    gpu_uuid=f"GPU-{s}",
                    prefix_schedule={"start": 0, "end": 3, "chunk_size": 2048},
                )
                row["row_sha256"] = digest(row)
                rows.append(row)
            write_jsonl(directory / f"results-{s}.jsonl", rows)
    return manifest


def test_full_result_integrity_reviews_and_stage_gate(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    stored_run(data, run)
    assert len(validated_results(data, run)[1]) == 480
    gate = require_language_gate(data, run)
    assert gate["gate_basis"] == "automatic_provisional"
    review_path = tmp_path / "reviews.jsonl"
    review_export(data, run, review_path)
    exported = read_jsonl(review_path)
    assert len(exported) == 480 and not any("policy" in r or "background" in r for r in exported)
    report = summarize(data, run)
    assert report["provisional_interpretation"] == "beyond_language_supported_at_8k"
    assert report["decision"] == "pending_human_review"
    raw = validated_results(data, run)[1]
    write_jsonl(review_path, reviews(raw))
    assert summarize(data, run, review_path)["decision"] == "pending_human_review"
    write_jsonl(review_path, reviews(raw, "human"))
    assert summarize(data, run, review_path)["decision"] == "beyond_language_supported_at_8k"
    broken = reviews(raw)
    broken[0]["answer"] = "different"
    write_jsonl(review_path, broken)
    with pytest.raises(ValueError, match="binding"):
        apply_reviews(raw, review_path)
    path = run / "language/results-0.jsonl"
    rows = read_jsonl(path)
    rows[0]["text"] = "corrupted"
    write_jsonl(path, rows)
    with pytest.raises(ValueError, match="fingerprint"):
        validated_results(data, run)


def test_language_gate_cannot_be_bypassed_by_missing_or_reviewed_failed_rows(tmp_path):
    data, run = tmp_path / "data", tmp_path / "run"
    stored_run(data, run)
    path = run / "language/results-0.jsonl"
    original = read_jsonl(path)
    write_jsonl(path, original[:-1])
    with pytest.raises(ValueError, match="Complete the 288"):
        require_language_gate(data, run)
    write_jsonl(path, original)
    raw = validated_results(data, run)[1]
    labels = reviews(raw, "human")
    by_cell = {r["cell_id"]: r for r in raw}
    for label in labels:
        row = by_cell[label["cell_id"]]
        if row["family"] == "language" and row["background"] == "none":
            label["content_ok"] = False
    review_path = tmp_path / "failed-reviews.jsonl"
    write_jsonl(review_path, labels)
    with pytest.raises(ValueError, match="Language gate did not pass"):
        require_language_gate(data, run, review_path)


def test_background_pair_reuse_rejected_even_with_resealed_files(tmp_path):
    manifest, _ = sealed_data(tmp_path)
    pairs = read_jsonl(tmp_path / "backgrounds.jsonl")
    pairs[1]["units"] = pairs[0]["units"]
    write_jsonl(tmp_path / "backgrounds.jsonl", pairs)
    manifest["backgrounds_sha256"] = file_digest(tmp_path / "backgrounds.jsonl")
    manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="reused source units"):
        load_controls(tmp_path)


def test_controls_launcher_dispatches_independent_workers_and_reviews(tmp_path, monkeypatch):
    import rcuf.launch as module

    data, run = tmp_path / "data", tmp_path / "run"
    sealed_data(data)
    calls = []

    class Finished:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    def spawn(command, **kwargs):
        calls.append((command, kwargs["env"]))
        return Finished()

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module, "provenance", lambda: {"source_sha256": "fixture"})
    labels = tmp_path / "labels.jsonl"
    module.launch(data, run, "format", "3,5", controls=True, reviews=labels)
    assert len(calls) == 2
    for shard, (command, env) in enumerate(calls):
        assert command[3] == "controls-worker"
        assert command[command.index("--shard") + 1] == str(shard)
        assert command[command.index("--phase") + 1] == "format"
        assert command[command.index("--reviews") + 1] == str(labels)
        assert "--shards" not in command
        assert env["CUDA_VISIBLE_DEVICES"] == ("3", "5")[shard]
    with pytest.raises(ValueError, match="exactly two"):
        module.launch(data, run, "language", "3", controls=True)


def test_actual_hf_independent_full_prefills():
    engine = tiny_engine()
    engine.eos = {2}
    engine.tokenizer = type("Decoder", (), {"decode": lambda self, ids, **kwargs: str(ids)})()
    e = {"input_ids": list(range(3, 35)), "question_cut": 19}
    first = generate_answer(engine, e)
    second = generate_answer(engine, e)
    assert first["generated_ids"] == second["generated_ids"]
    assert first["suffix_tokens"] == len(e["input_ids"]) - e["question_cut"]


def test_collect_pairs_preserves_whole_units_and_does_not_reuse_them(monkeypatch):
    import rcuf.scoring as module

    class Tokenizer:
        all_special_ids = (999,)

        def encode(self, text, **kwargs):
            return list(text)

    class Identifier:
        def classify(self, text):
            return ("en" if text.startswith("English") else "fr", 1.0)

    monkeypatch.setattr(module, "language_identifier", Identifier)
    cfg = deepcopy(config())
    cfg["target_tokens"] = 80
    source = [
        {
            "id": str(i),
            "translation": {
                "en": (f"English passage {i:02d} ").ljust(80, "x"),
                "fr": (f"French passage {i:02d} ").ljust(80, "x"),
            },
        }
        for i in range(16)
    ]
    selected, count = collect_pairs(iter(source), Tokenizer(), cfg)
    assert count == len(selected) == 16
    assert {p["units"][0]["id"] for p in selected} == {str(i) for i in range(16)}
    assert all(p["tokens"] == {"en": 80, "fr": 80} for p in selected)
    assert all(p["texts"]["en"] in {r["translation"]["en"] for r in source} for p in selected)
