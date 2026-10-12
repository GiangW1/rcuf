import json
from pathlib import Path

import pytest

from rcuf.analysis import apply_reviews, paired, required_reviews
from rcuf.io import append_jsonl, digest, load_config, recover_jsonl
from rcuf.runner import expected_cells, intervention_cuts, method_specs


def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/default.json")


def test_protocol_counts_and_document_sharding():
    cfg = config()
    manifest = {"config": cfg}
    examples = [
        {"doc_id": f"{split}-{i:03d}", "doc_index": i, "split": split, "length": length}
        for split, n in (("dev", 8), ("test", 32))
        for i in range(n)
        for length in cfg["lengths"]
    ]
    selection = {"strengths": {"add_full": 1.0, "add_lowrank": 1.0, "replace": 1.0}}
    assert len(expected_cells("baseline", manifest, examples, None)) == 96
    assert len(expected_cells("tune", manifest, examples, None)) == 160
    assert len(expected_cells("compare", manifest, examples, selection)) == 384
    assert len(expected_cells("delay", manifest, examples, selection)) == 192
    # Baselines at endpoints are reused. Unique formal responses total 96+384-64+192=608.
    sets = [
        set(expected_cells(stage, manifest, examples, selection))
        for stage in ("baseline", "compare", "delay")
    ]
    assert len(set.union(*sets)) == 608
    cells = expected_cells("compare", manifest, examples, selection)
    shard_docs = [
        {example["doc_id"] for example, _ in cells.values() if example["doc_index"] % 2 == i} for i in (0, 1)
    ]
    assert not shard_docs[0] & shard_docs[1]


def test_selection_is_required_and_no_middle_length_methods():
    cfg = config()
    assert method_specs("tune", cfg, cfg["lengths"][1]) == []
    with pytest.raises(ValueError):
        method_specs("compare", cfg, 0)


def test_delay_controls_share_cuts_and_resumed_prefill_schedule():
    cfg = config()
    selection = {"strengths": {"add_lowrank": 1.0, "replace": 0.5}}
    example = {"question_cut": 33000, "length": cfg["lengths"][-1]}
    specs = method_specs("delay", cfg, example["length"], selection)
    for delay in cfg["delays"]:
        methods = {s["method"] for s in specs if s["delay"] == delay}
        assert methods == {"baseline", "add_lowrank", "replace"}
    # The function does not accept the remaining cell list; completed cells cannot move chunk boundaries.
    assert intervention_cuts("delay", cfg, example, selection) == [33000 - 4096, 33000 - 512]


def row(doc, language, content=True):
    return {
        "doc_id": doc,
        "cell_id": digest([doc, language])[:24],
        "text": str(language),
        "output_sha256": digest(str(language)),
        "scores": {
            "language_ok": language,
            "content_ok": content,
            "joint_ok": language and content,
            "needs_review": False,
        },
    }


def test_paired_net_gains_include_new_failures():
    cfg = config()
    right = [row("a", False), row("b", True), row("c", False), row("d", True)]
    left = [row("a", True), row("b", False), row("c", True), row("d", True)]
    result = paired(left, right, "joint_ok", cfg)
    assert result["failure_to_success"] == 2
    assert result["success_to_failure"] == 1
    assert result["net_gains"] == 1
    with pytest.raises(ValueError):
        paired(left[:-1], right, "joint_ok", cfg)


def test_reviews_cannot_silently_relabel_different_outputs(tmp_path):
    rows = [row("a", True), row("a", False)]
    assert required_reviews(rows) == {r["cell_id"] for r in rows}
    path = tmp_path / "review.jsonl"
    append_jsonl(
        path,
        {"cell_id": rows[0]["cell_id"], "output_sha256": "wrong", "language_ok": True, "content_ok": True},
    )
    with pytest.raises(ValueError):
        apply_reviews(rows, path)


def test_recover_only_incomplete_final_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"cell_id": "a"})
    with path.open("ab") as stream:
        stream.write(b'{"cell_id": "b')
    assert recover_jsonl(path) == [{"cell_id": "a"}]
    assert json.loads(path.read_text()) == {"cell_id": "a"}
    with path.open("ab") as stream:
        stream.write(b"broken\n")
    with pytest.raises(ValueError):
        recover_jsonl(path)
