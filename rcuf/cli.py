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
    if args.command == "prepare":
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
