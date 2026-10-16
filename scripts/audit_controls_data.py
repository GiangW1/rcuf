"""CPU-only source/tokenizer replay; exports metadata, never source text or tokens."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from rcuf.controls_data import QUESTIONS, candidate_pair, load_controls, parallel_source, render
from rcuf.io import digest, read_jsonl, write_json


def audit(data_dir, corpus=None):
    manifest, examples = load_controls(data_dir)
    config = manifest["config"]
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["engine_config"]["model"], revision=config["model_revision"], use_fast=True
    )
    if digest(tokenizer.backend_tokenizer.to_str()) != manifest["tokenizer_sha256"]:
        raise ValueError("Audit tokenizer differs from the prepared inputs.")
    pairs = read_jsonl(data_dir / "backgrounds.jsonl")
    needed = {unit["id"] for pair in pairs for unit in pair["units"]}
    source, metadata = parallel_source(config, corpus)
    if metadata != manifest["source"]:
        raise ValueError("Audit source differs from the prepared inputs.")
    recovered, scanned = {}, 0
    for index, row in enumerate(source):
        scanned += 1
        source_id = str(row.get("id", index))
        if source_id not in needed:
            continue
        if source_id in recovered:
            raise ValueError("Duplicate selected source ID in the raw corpus.")
        texts = {lang: str(row.get("translation", row)[lang]).strip() for lang in QUESTIONS}
        recovered[source_id] = {"id": source_id, "sha256": digest(texts), "texts": texts}
    if recovered.keys() != needed:
        raise ValueError("Selected source units are missing from the raw corpus.")
    lower = math.ceil(config["target_tokens"] * (1 - config["length_tolerance"]))
    upper = math.floor(config["target_tokens"] * (1 + config["length_tolerance"]))
    documents, passage_hashes = [], set()
    for pair in pairs:
        units = [recovered[unit["id"]] for unit in pair["units"]]
        replay = candidate_pair(units, tokenizer, config, lower, upper)
        if replay is None or any(replay[key] != pair[key] for key in replay):
            raise ValueError(f"Source/tokenizer/language replay failed for {pair['doc_id']}.")
        for text in replay["texts"].values():
            text_hash = digest(text)
            if text_hash in passage_hashes:
                raise ValueError("Repeated whole background text, despite distinct source IDs.")
            passage_hashes.add(text_hash)
        sizes = pair["tokens"]
        documents.append(
            {
                "doc_id": pair["doc_id"],
                "tokens": sizes,
                "relative_length_gap": abs(sizes["en"] - sizes["fr"]) / max(sizes.values()),
                "source_unit_count": len(units),
                "first_source_id": units[0]["id"],
                "last_source_id": units[-1]["id"],
                "source_units_sha256": digest(pair["units"]),
                "source_columns": pair["source_columns"],
                "language_blocks": {lang: len(pair["language_blocks"][lang]) for lang in QUESTIONS},
                "min_block_confidence": min(
                    block["confidence"] for blocks in pair["language_blocks"].values() for block in blocks
                ),
            }
        )
    by_document = {pair["doc_id"]: pair for pair in pairs}
    for row in examples:
        reference = (
            "" if row["background"] == "none" else by_document[row["doc_id"]]["texts"][row["background"]]
        )
        replay = render(
            tokenizer,
            reference,
            row["records"],
            row["target_position"],
            row["family"],
            row["policy"],
            row["question_language"],
        )
        if any(replay[key] != row[key] for key in replay):
            raise ValueError(f"Full prompt template replay failed for {row['cell_id']}.")
    report = {
        "audit": "rcuf-controls-source-tokenizer-replay-v1",
        "model_generation_performed": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "backgrounds_sha256": manifest["backgrounds_sha256"],
        "examples_sha256": manifest["examples_sha256"],
        "config": config,
        "source": metadata,
        "scanned_aligned_units": scanned,
        "eligible_passages": manifest["eligible_passages"],
        "selected_documents": len(documents),
        "selected_distinct_source_units": len(needed),
        "replayed_examples": len(examples),
        "families": dict(Counter(row["family"] for row in examples)),
        "input_tokens": {
            "min": min(len(r["input_ids"]) for r in examples),
            "max": max(len(r["input_ids"]) for r in examples),
        },
        "documents": documents,
        "limitations": [
            "Source-unit and template replay checks integrity, not semantic translation equivalence.",
            "Language classification is an automated screen; inspect the paired passages before inference.",
            (
                "Eligible passage count comes from preparation's full registered-file scan; this audit "
                "rechecks selected passages, not the unselected candidate pool."
            ),
        ],
    }
    report["audit_sha256"] = digest(report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--parallel-corpus-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite an existing audit report.")
    result = audit(args.data_dir, args.parallel_corpus_jsonl)
    write_json(args.output, result)
    print(
        f"Verified {result['selected_documents']} pairs and {result['replayed_examples']} prompts; "
        f"audit {result['audit_sha256']}"
    )


if __name__ == "__main__":
    main()
