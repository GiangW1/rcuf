from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="RCUF staged instruction-lifetime pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="Freeze model/data revisions and paired token streams")
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/default.json"))
    prepare_parser.add_argument("--data-dir", type=Path, default=Path("data/pilot"))
    prepare_parser.add_argument("--corpus-jsonl", type=Path)
    controls_prepare = sub.add_parser("controls-prepare", help="Freeze paired bilingual instruction controls")
    controls_prepare.add_argument("--config", type=Path, default=Path("configs/controls.json"))
    controls_prepare.add_argument("--data-dir", type=Path, default=Path("data/language-controls"))
    controls_prepare.add_argument("--parallel-corpus-jsonl", type=Path)
    controls_prepare.add_argument("--model")
    for command in ("controls-launch", "controls-worker", "controls-review", "controls-summarize"):
        cp = sub.add_parser(command)
        cp.add_argument("--data-dir", type=Path, default=Path("data/language-controls"))
        cp.add_argument("--run-dir", type=Path, default=Path("runs/language-controls"))
        if command in {"controls-launch", "controls-worker"}:
            cp.add_argument(
                "--phase",
                choices=["language", "format"] + (["all"] if command == "controls-launch" else []),
                default="language" if command == "controls-launch" else None,
                required=command == "controls-worker",
            )
            cp.add_argument("--reviews", type=Path)
        if command == "controls-launch":
            cp.add_argument("--devices", default="0,1")
        if command == "controls-worker":
            cp.add_argument("--shard", type=int, choices=(0, 1), required=True)
        if command == "controls-review":
            cp.add_argument("--output", type=Path, required=True)
        if command == "controls-summarize":
            cp.add_argument("--reviews", type=Path)
    confirm_prepare = sub.add_parser(
        "confirm-prepare", help="Freeze exact pilot replay and fresh confirmation"
    )
    confirm_prepare.add_argument("--config", type=Path, default=Path("configs/confirmation.json"))
    confirm_prepare.add_argument("--pilot-data-dir", type=Path, required=True)
    confirm_prepare.add_argument("--data-dir", type=Path, default=Path("data/confirmation"))
    confirm_prepare.add_argument("--english-corpus-jsonl", type=Path)
    confirm_prepare.add_argument("--french-corpus-jsonl", type=Path)
    confirm_prepare.add_argument("--model", help="Optional relocated local model directory or fixed model ID")
    for command in ("confirm-launch", "confirm-worker", "confirm-review", "confirm-summarize"):
        cp = sub.add_parser(command)
        cp.add_argument("--data-dir", type=Path, default=Path("data/confirmation"))
        cp.add_argument("--run-dir", type=Path, default=Path("runs/confirmation"))
        if command in {"confirm-launch", "confirm-worker"}:
            choices = ["smoke", "crossover-a", "crossover-b", "confirm"]
            cp.add_argument(
                "--phase",
                choices=choices + (["all"] if command == "confirm-launch" else []),
                default="all" if command == "confirm-launch" else None,
                required=command == "confirm-worker",
            )
        if command == "confirm-launch":
            cp.add_argument("--devices", default="0,1")
        if command == "confirm-worker":
            cp.add_argument("--shard", type=int, choices=(0, 1), required=True)
        if command == "confirm-review":
            cp.add_argument("--output", type=Path, required=True)
        if command == "confirm-summarize":
            cp.add_argument("--reviews", type=Path)
    for command in ("launch", "worker", "summarize", "review", "select"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--data-dir", type=Path, default=Path("data/pilot"))
        command_parser.add_argument("--run-dir", type=Path, default=Path("runs/pilot"))
        if command != "select":
            command_parser.add_argument(
                "--stage", choices=("smoke", "baseline", "tune", "compare", "delay"), required=True
            )
        if command == "launch":
            command_parser.add_argument("--devices", default="0,1")
        if command == "worker":
            command_parser.add_argument("--shard", type=int, default=0)
            command_parser.add_argument("--shards", type=int, default=1)
        if command in {"summarize", "select"}:
            command_parser.add_argument("--reviews", type=Path)
        if command == "summarize":
            command_parser.add_argument("--compare-reviews", type=Path)
        if command == "review":
            command_parser.add_argument("--output", type=Path, required=True)
            command_parser.add_argument("--all", action="store_true", dest="review_all")
    args = parser.parse_args()
    if args.command == "controls-prepare":
        from .controls_data import prepare
        from .io import read_json

        result = prepare(read_json(args.config), args.data_dir, args.parallel_corpus_jsonl, args.model)
    elif args.command == "controls-launch":
        from .controls import launch_controls

        launch_controls(args.data_dir, args.run_dir, args.phase, args.devices, args.reviews)
        return
    elif args.command == "controls-worker":
        from .controls import worker

        worker(args.data_dir, args.run_dir, args.phase, args.shard, args.reviews)
        return
    elif args.command == "controls-review":
        from .controls_analysis import review_export

        result = review_export(args.data_dir, args.run_dir, args.output)
    elif args.command == "controls-summarize":
        from .controls_analysis import summarize

        result = summarize(args.data_dir, args.run_dir, args.reviews)
    elif args.command == "confirm-prepare":
        from .confirm_data import prepare_confirmation
        from .io import read_json

        result = prepare_confirmation(
            read_json(args.config),
            args.pilot_data_dir,
            args.data_dir,
            args.english_corpus_jsonl,
            args.french_corpus_jsonl,
            args.model,
        )
    elif args.command == "confirm-launch":
        from .confirm import launch_confirmation

        launch_confirmation(args.data_dir, args.run_dir, args.phase, args.devices)
        return
    elif args.command == "confirm-worker":
        from .confirm import worker

        worker(args.data_dir, args.run_dir, args.phase, args.shard)
        return
    elif args.command == "confirm-review":
        from .confirm_analysis import export_confirmation_reviews

        result = export_confirmation_reviews(args.data_dir, args.run_dir, args.output)
    elif args.command == "confirm-summarize":
        from .confirm_analysis import summarize_confirmation

        result = summarize_confirmation(args.data_dir, args.run_dir, args.reviews)
    elif args.command == "prepare":
        from .data import prepare
        from .io import load_config

        result = prepare(load_config(args.config), args.data_dir, args.corpus_jsonl)
    elif args.command == "launch":
        from .launch import launch

        launch(args.data_dir, args.run_dir, args.stage, args.devices)
        return
    elif args.command == "worker":
        from .runner import run_worker

        run_worker(args.data_dir, args.run_dir, args.stage, args.shard, args.shards)
        return
    elif args.command == "select":
        from .analysis import select

        result = select(args.data_dir, args.run_dir, args.reviews)
    elif args.command == "review":
        from .analysis import export_reviews

        result = {
            "review_count": export_reviews(
                args.data_dir, args.run_dir, args.stage, args.output, args.review_all
            ),
            "output": str(args.output),
            "instruction": "Fill language_ok and content_ok with JSON true/false; preserve cell/output IDs.",
        }
    else:
        from .analysis import summarize

        result = summarize(args.data_dir, args.run_dir, args.stage, args.reviews, args.compare_reviews)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
