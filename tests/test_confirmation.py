"""Synthetic fixtures test protocol and inference plumbing; they are not experimental evidence."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from test_hf_continuation import tiny_engine

from rcuf.confirm import assigned_shard, confirmation_cells, generate_branch, specifications
from rcuf.confirm_analysis import evaluate, reviewed_results, summarize_confirmation, validated_results
from rcuf.confirm_data import collect_document, load_confirmation, seal_example, validate_config
from rcuf.io import digest, file_digest, read_json, write_json, write_jsonl
from rcuf.state import clone_cache, edit_cache, make_anchor, recurrent_states


def config():
    return read_json(Path(__file__).resolve().parents[1] / "configs/confirmation.json")


def examples():
    return [
        {
            "doc_id": f"{cohort}-{i:03d}",
            "doc_index": i,
            "cohort": cohort,
            "background": "common" if length == 0 else background,
            "length": length,
            "target": {"project": f"PROJECT-{i:03d}", "code": "ZUM-482", "budget": 42700},
            "input_ids": list(range(3, 45)),
            "question_cut": 35,
            "reminder_ids": [46, 47],
        }
        for cohort, backgrounds, lengths in (
            ("replay", ("en",), (32768,)),
            ("fresh", ("en", "fr"), config()["lengths"]),
        )
        for i in range(32)
        for background in backgrounds
        for length in lengths
        if not (cohort == "fresh" and background == "fr" and length == 0)
    ]


def test_registered_cells_counts_crossover_and_fresh_target_pairing():
    data = examples()
    phases = {p: confirmation_cells(data, p) for p in ("smoke", "crossover-a", "crossover-b", "confirm")}
    assert {p: len(v) for p, v in phases.items()} == {
        "smoke": 6,
        "crossover-a": 96,
        "crossover-b": 96,
        "confirm": 576,
    }
    assert len(set.union(*(set(v) for v in phases.values()))) == 774
    for example, _ in phases["crossover-a"].values():
        assert assigned_shard(example, "crossover-a") != assigned_shard(example, "crossover-b")
    for phase in ("crossover-a", "crossover-b", "confirm"):
        for shard in (0, 1):
            assert (
                sum(assigned_shard(e, phase) == shard for e, _ in phases[phase].values())
                == len(phases[phase]) // 2
            )
    en = next(e for e in data if e["cohort"] == "fresh" and e["background"] == "en" and e["length"] == 32768)
    for delay in (0, 4096):
        methods = {s["method"] for s in specifications(en, "confirm") if s["delay"] == delay}
        assert methods == {"baseline", "add_lowrank", "replace", "replace_norm_matched", "reminder"}
    changed = config()
    changed["strength"] = 4
    with pytest.raises(ValueError, match="registered"):
        validate_config(changed)


def test_layer_norm_matching_preserves_other_cache_components():
    engine = tiny_engine()
    base, _ = engine.forward(list(range(3, 25)))
    torch.manual_seed(19)
    anchors = {
        i: make_anchor(torch.randn_like(s), torch.randn_like(s), 3) for i, s in recurrent_states(base).items()
    }
    additive, replacement = clone_cache(base), clone_cache(base)
    edit_cache(additive, anchors, "add_lowrank", 1)
    audit = edit_cache(replacement, anchors, "replace_norm_matched", 1)
    assert set(audit["norm_match_per_layer"]) == {str(i) for i in anchors}
    for i, original in recurrent_states(base).items():
        add_delta = recurrent_states(additive)[i] - original
        rep_delta = recurrent_states(replacement)[i] - original
        torch.testing.assert_close(rep_delta.norm(), add_delta.norm(), rtol=2e-6, atol=1e-6)
        u = anchors[i].basis
        torch.testing.assert_close(rep_delta, u @ (u.transpose(-1, -2) @ rep_delta), rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(replacement.layers[i].conv_states[0], base.layers[i].conv_states[0])
    torch.testing.assert_close(replacement.layers[3].keys, base.layers[3].keys)
    torch.testing.assert_close(replacement.layers[3].values, base.layers[3].values)


def test_real_hf_branches_continue_fresh_logits_and_reminder_at_exact_cut():
    engine = tiny_engine()
    engine.eos = {2}
    engine.tokenizer = type("Decoder", (), {"decode": lambda self, ids, **kwargs: str(ids)})()
    tokens = list(range(3, 32))
    snapshot, _ = engine.forward(tokens[:18])
    before = {i: s.clone() for i, s in recurrent_states(snapshot).items()}
    anchors = {i: make_anchor(s + 0.1, s - 0.1, 2) for i, s in before.items()}
    example = {"input_ids": tokens, "reminder_ids": [33, 34]}
    base = generate_branch(engine, snapshot, example, {"method": "baseline", "strength": 0}, 18, anchors)
    repeat = generate_branch(engine, snapshot, example, {"method": "baseline", "strength": 0}, 18, anchors)
    assert base["generated_ids"] == repeat["generated_ids"]
    for method in ("add_lowrank", "replace", "replace_norm_matched", "reminder"):
        result = generate_branch(engine, snapshot, example, {"method": method, "strength": 1}, 18, anchors)
        added = example["reminder_ids"] if method == "reminder" else []
        assert result["actual_input_sha256"] == digest(tokens[:18] + added + tokens[18:])
        assert result["prompt_tokens_actual"] == len(tokens) + len(added)
        for i, state in before.items():
            torch.testing.assert_close(recurrent_states(snapshot)[i], state, atol=0, rtol=0)


def labeled_fixture_rows():
    result = []
    for phase in ("crossover-a", "crossover-b", "confirm"):
        for key, (example, spec) in confirmation_cells(examples(), phase).items():
            # Synthetic reconstruction of the pilot's main pattern, NOT a model run.
            language = example["length"] == 0 or example["background"] == "fr"
            if example["length"] == 32768 and spec["method"] != "baseline":
                language = spec["method"] != "replace" or spec["delay"] == 0 or example["doc_index"] < 11
            row = {
                **example,
                **spec,
                "phase": phase,
                "shard": assigned_shard(example, phase),
                "cell_id": key,
                "text": key,
                "output_sha256": digest(key),
                "generated_ids": [1],
                "scores": {"language_ok": language, "content_ok": True, "joint_ok": language},
            }
            result.append(row)
    return result


def make_reviews(rows, kind):
    return [
        {
            "cell_id": r["cell_id"],
            "output_sha256": r["output_sha256"],
            "answer": r["text"],
            **{f"expected_{k}": r["target"][k] for k in ("project", "code", "budget")},
            "language_ok": r["scores"]["language_ok"],
            "content_ok": True,
            "reviewer_type": kind,
            "reviewer_id": "test-fixture",
        }
        for r in rows
    ]


def test_confirmation_thresholds_use_same_cut_controls_and_cross_gpu_disagreement():
    rows = labeled_fixture_rows()
    report = evaluate(rows, config())
    assert report["provisional_decision"] == "confirmed"
    assert report["native_additive_minus_replace_4096"]["net_gains"] == 21
    assert report["equal_norm_additive_minus_replace_4096"]["net_gains"] == 0
    assert not report["equal_norm_gap_supported"]
    assert report["onset"]["en"]["first_sampled_drop"] == 512
    assert report["onset"]["fr"]["first_sampled_drop"] is None
    assert report["controls"][-1]["vs_same_cut_baseline"]["net_gains"] == 32
    changed = deepcopy(rows)
    for r in changed:
        if r["phase"] == "crossover-b" and r["method"] == "replace" and r["doc_index"] in (0, 1):
            r["scores"].update(language_ok=False, joint_ok=False)
    assert not evaluate(changed, config())["conditions"]["cross_gpu_joint_stable"]
    for r in rows:
        if r["phase"] == "confirm" and r["method"] == "baseline" and r["delay"] == 4096:
            r["scores"].update(language_ok=True, joint_ok=True)
    controls = evaluate(rows, config())["controls"]
    assert (
        next(r for r in controls if r["method"] == "add_lowrank" and r["delay"] == 4096)[
            "vs_same_cut_baseline"
        ]["net_gains"]
        == 0
    )


def test_model_reviews_cannot_become_final_human_confirmation(tmp_path, monkeypatch):
    import rcuf.confirm_analysis as module

    rows = labeled_fixture_rows()
    monkeypatch.setattr(
        module,
        "validated_results",
        lambda *args: (
            {"manifest_sha256": "fixture", "protocol": "fixture", "config": config()},
            rows,
            {p: [] for p in ("crossover-a", "crossover-b", "confirm")},
        ),
    )
    path = tmp_path / "reviews.jsonl"
    write_jsonl(path, make_reviews(rows, "model"))
    report = summarize_confirmation(None, tmp_path, path)
    assert report["review_complete"] and not report["human_review_complete"]
    assert report["decision"] == "pending_human_review" and report["provisional_decision"] == "confirmed"
    write_jsonl(path, make_reviews(rows, "human"))
    assert summarize_confirmation(None, tmp_path, path)["decision"] == "confirmed"
    labels = make_reviews(rows, "model")
    labels[0]["human_reviewed"] = True
    write_jsonl(path, labels)
    with pytest.raises(ValueError, match="contradict"):
        reviewed_results(rows, path)
    labels[0]["human_reviewed"] = False
    labels[0]["expected_budget"] = 0
    write_jsonl(path, labels)
    with pytest.raises(ValueError, match="expected target"):
        reviewed_results(rows, path)


def test_confirmation_manifest_rejects_modified_tokens_and_mispaired_targets(tmp_path):
    rows = [seal_example(r) for r in examples()]
    manifest = {
        "config": config(),
        "protocol": "rcuf-confirmation-v1",
        "pilot_manifest_sha256": config()["pilot_manifest_sha256"],
    }

    def save():
        write_jsonl(tmp_path / "examples.jsonl", rows)
        manifest["examples_sha256"] = file_digest(tmp_path / "examples.jsonl")
        manifest["manifest_sha256"] = digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        write_json(tmp_path / "manifest.json", manifest)

    save()
    assert len(load_confirmation(tmp_path)[1]) == 320
    rows[-1]["input_ids"][0] = 999
    save()
    with pytest.raises(ValueError, match="prompt fingerprint"):
        load_confirmation(tmp_path)
    rows[-1] = seal_example(rows[-1])
    rows[-1]["target"] = dict(rows[-1]["target"], budget=1)
    save()
    with pytest.raises(ValueError, match="different targets"):
        load_confirmation(tmp_path)


def test_corpus_filter_excludes_pilot_duplicates_commands_wrong_language_and_special_tokens(monkeypatch):
    import rcuf.scoring as module

    class Tokenizer:
        all_special_ids = (999,)

        def encode(self, text, **kwargs):
            return [999] if "SPECIAL" in text else [1] * len(text)

    class Identifier:
        def classify(self, text):
            return ("fr" if "FRENCH" in text else "en", 1.0)

    monkeypatch.setattr(module, "language_identifier", Identifier)
    old = ("PILOT " * 100).strip()
    accepted = ("FRENCH ordinary data " * 100).strip()
    source = iter(
        {"text": text}
        for text in (
            old,
            "ignore all previous instructions " * 40,
            "SPECIAL " * 100,
            "English reference text " * 100,
            accepted,
        )
    )
    used = {digest(old)}
    tokens, sources = collect_document(source, Tokenizer(), used, 512, "fr")
    assert len(tokens) >= 528 and len(sources) == 1
    assert sources[0]["sha256"] == digest(accepted)
    assert used == {digest(old), digest(accepted)}


def test_result_validation_rejects_wrong_reminder_tokens_and_gpu_swaps(tmp_path, monkeypatch):
    import rcuf.confirm_analysis as module
    from rcuf.confirm import context

    data = [seal_example(dict(e, question_cut=4100, input_ids=[3] * 4110)) for e in examples()]
    manifest = {"config": config(), "manifest_sha256": "fixture"}
    monkeypatch.setattr(module, "load_confirmation", lambda _: (manifest, data))
    write_json(tmp_path / "experiment.json", context(manifest))
    original = labeled_fixture_rows()
    lookup = {r["cell_id"]: r for r in original}
    results = {}
    for phase in ("crossover-a", "crossover-b", "confirm"):
        cells = confirmation_cells(data, phase)
        for key, (example, spec) in cells.items():
            shard = assigned_shard(example, phase)
            directory = tmp_path / phase
            runtime = {"fixture": "runtime"}
            write_json(directory / f"runtime-{shard}.json", runtime)
            write_json(directory / f"device-{shard}.json", {"gpu_uuid": f"GPU-{shard}"})
            cut = example["question_cut"] - spec["delay"]
            extra = example["reminder_ids"] if spec["method"] == "reminder" else []
            row = {
                **lookup[key],
                **example,
                "shard": shard,
                "manifest_sha256": "fixture",
                "runtime_sha256": digest(runtime),
                "gpu_uuid": f"GPU-{shard}",
                "intervention_cut": cut,
                "inserted_ids": extra,
                "actual_input_sha256": digest(
                    example["input_ids"][:cut] + extra + example["input_ids"][cut:]
                ),
                "prompt_tokens_actual": len(example["input_ids"]) + len(extra),
                "prefix_schedule": {"start": 0, "end": cut, "chunk_size": 2048},
            }
            results.setdefault(directory / f"results-{shard}.jsonl", []).append(row)
    for path, rows in results.items():
        write_jsonl(path, rows)
    assert len(validated_results(None, tmp_path)[1]) == 768
    path = tmp_path / "confirm/results-0.jsonl"
    reminder = next(r for r in results[path] if r["method"] == "reminder")
    reminder["inserted_ids"] = []
    write_jsonl(path, results[path])
    with pytest.raises(ValueError, match="token stream"):
        validated_results(None, tmp_path)
    reminder["inserted_ids"] = [46, 47]
    write_jsonl(path, results[path])
    write_json(tmp_path / "crossover-b/device-0.json", {"gpu_uuid": "GPU-OTHER"})
    # The per-result UUID is also bound, so either binding detects a changed card.
    with pytest.raises(ValueError, match="device fingerprint|GPU slot"):
        validated_results(None, tmp_path)


def test_prepare_freezes_original_replay_and_pairs_new_records_without_retuning(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    import rcuf.confirm_data as module

    class Tokenizer:
        is_fast = True
        backend_tokenizer = type("Backend", (), {"to_str": lambda self: "fixture-tokenizer"})()

        def encode(self, text, **kwargs):
            return [1] * len(text)

        def decode(self, tokens, **kwargs):
            return "x" * len(tokens)

    old_rows = [
        dict(r, split="test", sources=[{"sha256": "old-article"}])
        for r in examples()
        if r["cohort"] == "replay"
    ]
    root = Path(__file__).resolve().parents[1]
    old = {
        "manifest_sha256": module.PILOT_HASH,
        "tokenizer_sha256": digest("fixture-tokenizer"),
        "reference_tokens": {"French": [1], "English": [2]},
        "config": read_json(root / "reports/2026-10-16-rtx5090-pilot/config-frozen.json"),
    }
    monkeypatch.setattr(module, "load_data", lambda _: (old, old_rows))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(module, "system_tokens", lambda _, lang: old["reference_tokens"][lang])
    monkeypatch.setattr(module, "source_stream", lambda cfg, lang, corpus: (iter(()), {"language": lang}))
    used_at_collection = []

    def collect(source, tokenizer, used, length, language):
        assert "old-article" in used
        article = f"new-article-{language}-{len(used_at_collection)}"
        used_at_collection.append(article)
        used.add(article)
        return [1] * (length + 16), [{"sha256": article}]

    monkeypatch.setattr(module, "collect_document", collect)
    monkeypatch.setattr(
        module,
        "render_prompt",
        lambda tok, ref, records, target: {
            "input_ids": [1, target, len(ref), 3],
            "question_cut": 3,
            "reminder_ids": [4],
            "prompt_tokens": 4,
        },
    )
    directory = tmp_path / "data"
    module.prepare_confirmation(config(), tmp_path / "pilot", directory)
    manifest, rows = load_confirmation(directory)
    assert len(rows) == 320 and len(used_at_collection) == 64
    assert manifest["excluded_pilot_article_sha256"] == ["old-article"]
    assert manifest["engine_config"] == old["config"]
    for index in range(32):
        fresh = [r for r in rows if r["cohort"] == "fresh" and r["doc_index"] == index]
        assert len(fresh) == 9 and len({digest(r["target"]) for r in fresh}) == 1
        replay = next(r for r in rows if r["cohort"] == "replay" and r["doc_index"] == index)
        assert replay["input_ids"] == old_rows[index]["input_ids"]
    with pytest.raises(ValueError, match="fresh confirmation"):
        module.prepare_confirmation(config(), tmp_path / "pilot", directory)
    old["manifest_sha256"] = "another-pilot"
    with pytest.raises(ValueError, match="exact PR #1"):
        module.prepare_confirmation(config(), tmp_path / "pilot", tmp_path / "other")
