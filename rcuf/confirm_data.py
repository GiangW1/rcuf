"""Frozen confirmation inputs: exact pilot replay plus independent bilingual backgrounds."""

from __future__ import annotations

import re

from .data import SUSPICIOUS, load_data, records_for, render_prompt, system_tokens
from .io import append_jsonl, digest, file_digest, read_json, write_json

PROTOCOL = "rcuf-confirmation-v2"
PILOT_HASH = "9d608544c7a85426f7d69023b998f9bb320f3f05e6b5b8167f83a381d03cc919"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FRENCH_INSTRUCTION = re.compile(
    r"ignorez? (?:les |toutes les )?instructions|r[ée]pondez? (?:en |uniquement en )(?:fran[çc]ais|anglais)",
    re.IGNORECASE,
)


def validate_config(config):
    # This is a fixed confirmation, without a search grid or test-dependent selection.
    fixed = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "pilot_manifest_sha256": PILOT_HASH,
        "model_revision": MODEL_REVISION,
        "documents": 32,
        "record_seed": 26101617,
        "lengths": [0, 512, 2048, 8192, 32768],
        "delays": [0, 4096],
        "rank": 4,
        "strength": 1.0,
        "dataset": "wikimedia/wikipedia",
        "dataset_revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "backgrounds": {"en": "20231101.en", "fr": "20231101.fr"},
        "bootstrap_samples": 10000,
        "gates": {
            "crossover_max_disagreements_per_metric": 1,
            "same_cut_baseline_max_disagreements_per_metric": 1,
            "short_min_successes": 31,
            "drift_min_net_losses": 7,
            "long_min_content_successes": 29,
            "additive_min_successes": 29,
            "immediate_min_successes": 29,
            "recovery_min_net_gains": 7,
            "replacement_delay_min_net_losses": 7,
            "method_gap_min_net_gains": 7,
            "max_content_net_losses": 1,
        },
    }
    if any(config.get(k) != v for k, v in fixed.items()):
        raise ValueError(
            "Confirmation protocol differs from the registered v2 configuration; prepare fresh data."
        )
    if set(config["backgrounds"]) != {"en", "fr"}:
        raise ValueError("Both English and French background sources are required.")
    if config["bootstrap_samples"] < 100:
        raise ValueError("Too few bootstrap draws.")


def stream_jsonl(path):
    import json

    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_stream(config, language, corpus):
    if corpus is not None:
        return iter(stream_jsonl(corpus)), {"kind": "local_jsonl", "sha256": file_digest(corpus)}
    from datasets import load_dataset

    return iter(
        load_dataset(
            config["dataset"],
            config["backgrounds"][language],
            revision=config["dataset_revision"],
            split="train",
            streaming=True,
        )
    ), {
        "kind": "huggingface",
        "dataset": config["dataset"],
        "config": config["backgrounds"][language],
        "revision": config["dataset_revision"],
    }


def collect_document(source, tokenizer, used, length, language):
    from .scoring import language_identifier

    pieces, sources = [], []
    tokens = []
    while len(tokens) < length + 16:
        try:
            article = next(source)
        except StopIteration as exc:
            raise ValueError("Not enough independent corpus text; supply a larger corpus JSONL.") from exc
        text = str(article["text"]).strip()
        sha = digest(text)
        if sha in used or len(text) < 500 or SUSPICIOUS.search(text) or FRENCH_INSTRUCTION.search(text):
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if set(encoded).intersection(tokenizer.all_special_ids):
            continue
        detected, probability = language_identifier().classify(text[:4000])
        if detected != language or probability < 0.9:
            continue
        used.add(sha)
        pieces.append(text)
        sources.append(
            {
                "title": article.get("title"),
                "url": article.get("url"),
                "sha256": sha,
                "detected_language": detected,
                "language_confidence": float(probability),
            }
        )
        tokens = tokenizer.encode("\n\n".join(pieces), add_special_tokens=False)
    return tokens, sources


def seal_example(row):
    return dict(row, input_sha256=digest(row["input_ids"]))


def prepare_confirmation(config, pilot_data, output, english_corpus=None, french_corpus=None, model=None):
    from transformers import AutoTokenizer

    validate_config(config)
    old, old_examples = load_data(pilot_data)
    if old["manifest_sha256"] != PILOT_HASH:
        raise ValueError("Replay needs the exact PR #1 prepared data; its manifest fingerprint differs.")
    engine_config = dict(old["config"])
    if (
        engine_config["model_revision"] != MODEL_REVISION
        or engine_config["rank"] != config["rank"]
        or engine_config["prefill_chunk_size"] != 2048
    ):
        raise ValueError("Pilot model, rank or chunk schedule differs from PR #1.")
    if model:
        engine_config["model"] = model
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh confirmation data directory.")
    tokenizer = AutoTokenizer.from_pretrained(
        engine_config["model"],
        revision=MODEL_REVISION,
        use_fast=True,
    )
    if not tokenizer.is_fast or digest(tokenizer.backend_tokenizer.to_str()) != old["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from the pilot.")
    anchors = {lang: system_tokens(tokenizer, lang) for lang in ("French", "English")}
    if anchors != old["reference_tokens"]:
        raise ValueError("System anchor tokens differ from the pilot.")
    replay = [x for x in old_examples if x["split"] == "test" and x["length"] == 32768]
    if len(replay) != config["documents"]:
        raise ValueError("Expected all 32 original long-context test prompts.")
    output.mkdir(parents=True, exist_ok=True)
    for row in replay:
        append_jsonl(output / "examples.jsonl", seal_example(dict(row, cohort="replay", background="en")))
    used = {s["sha256"] for row in old_examples for s in row["sources"]}
    excluded = sorted(used)
    previous_targets = {digest(x["target"]) for x in old_examples}
    corpus_metadata = {}
    for language, corpus in (("en", english_corpus), ("fr", french_corpus)):
        source, metadata = source_stream(config, language, corpus)
        corpus_metadata[language] = metadata
        for index in range(config["documents"]):
            tokens, sources = collect_document(source, tokenizer, used, config["lengths"][-1], language)
            records, target = records_for(index, config["record_seed"])
            if digest(records[target]) in previous_targets:
                raise ValueError("A confirmation target overlaps the pilot.")
            for length in config["lengths"]:
                if length == 0 and language == "fr":
                    continue  # identical empty-background condition is generated only once
                reference = tokenizer.decode(tokens[:length], skip_special_tokens=False) if length else ""
                actual = len(tokenizer.encode(reference, add_special_tokens=False))
                if abs(actual - length) > 8:
                    raise ValueError("Reference token round-trip changed length unexpectedly.")
                row = {
                    "cohort": "fresh",
                    "background": language if length else "common",
                    "doc_id": f"confirmation-{index:03d}",
                    "doc_index": index,
                    "split": "confirmation",
                    "length": length,
                    "reference_tokens": actual,
                    "reference_sha256": digest(reference),
                    "records": records,
                    "target": records[target],
                    "target_position": target,
                    "sources": sources,
                    **render_prompt(tokenizer, reference, records, target),
                }
                append_jsonl(output / "examples.jsonl", seal_example(row))
            print(f"prepared confirmation-{index:03d} background={language}", flush=True)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "config": config,
        "engine_config": engine_config,
        "pilot_manifest_sha256": PILOT_HASH,
        "tokenizer_sha256": old["tokenizer_sha256"],
        "reference_tokens": anchors,
        "corpus_sources": corpus_metadata,
        "excluded_pilot_article_sha256": excluded,
        "examples_sha256": file_digest(output / "examples.jsonl"),
    }
    manifest["manifest_sha256"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def load_confirmation(directory):
    from .io import read_jsonl

    manifest = read_json(directory / "manifest.json")
    validate_config(manifest["config"])
    if (
        manifest["protocol"] != PROTOCOL
        or manifest["pilot_manifest_sha256"] != PILOT_HASH
        or digest({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        != manifest["manifest_sha256"]
        or file_digest(directory / "examples.jsonl") != manifest["examples_sha256"]
    ):
        raise ValueError("Confirmation input fingerprint mismatch.")
    rows = read_jsonl(directory / "examples.jsonl")
    expected = {
        (cohort, i, background, length)
        for cohort, backgrounds, lengths in (
            ("replay", ("en",), (32768,)),
            ("fresh", ("en", "fr"), manifest["config"]["lengths"]),
        )
        for i in range(32)
        for background in backgrounds
        for length in lengths
        if not (cohort == "fresh" and background == "fr" and length == 0)
    }
    keys = {
        (r["cohort"], r["doc_index"], "en" if r["background"] == "common" else r["background"], r["length"])
        for r in rows
    }
    if len(rows) != len(expected) or keys != expected:
        raise ValueError("Missing or duplicate confirmation prompts.")
    targets = {}
    for row in rows:
        if row["input_sha256"] != digest(row["input_ids"]) or not 0 < row["question_cut"] < len(
            row["input_ids"]
        ):
            raise ValueError("Invalid prompt fingerprint or question cut.")
        key = row["cohort"], row["doc_index"]
        if key in targets and targets[key] != row["target"]:
            raise ValueError("Paired prompts have different targets.")
        targets[key] = row["target"]
    return manifest, rows
