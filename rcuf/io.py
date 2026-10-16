"""Small, auditable JSON storage and run fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: str | Path) -> str:
    # Local Wikipedia exports can be many GB: do not load the entire file into RAM.
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def recover_jsonl(path: str | Path) -> list[dict]:
    """Recover only an unfinished final line after a killed worker, never an earlier error."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows = []
    valid_bytes = 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt completed JSONL record in {path}, line {i + 1}")
            with path.open("r+b") as stream:
                stream.truncate(valid_bytes)
            break
        rows.append(row)
        valid_bytes += len(line)
    if rows and raw and not raw.endswith(b"\n") and valid_bytes == len(raw):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def bind_metadata(path: Path, expected: dict) -> None:
    if path.exists():
        if read_json(path) != expected:
            raise ValueError(f"Run metadata differs at {path}; use a new run directory.")
    else:
        write_json(path, expected)


def provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    source = {str(p.relative_to(root)): file_digest(p) for p in sorted((root / "rcuf").glob("*.py"))}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
        commit = commit.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    return {"source_sha256": digest(source), "git_commit": commit}


def load_config(path: str | Path) -> dict:
    config = read_json(path)
    lengths = config["lengths"]
    if lengths != sorted(set(lengths)) or len(lengths) != 3 or lengths[0] != 0:
        raise ValueError("This pilot requires three distinct sorted lengths, starting with zero.")
    delays = config["delays"]
    if (
        config["rank"] <= 0
        or not delays
        or delays != sorted(set(delays))
        or any(d <= 0 or d >= lengths[-1] for d in delays)
    ):
        raise ValueError("Invalid rank or delays.")
    if config["dev_documents"] < 1 or config["test_documents"] < 1:
        raise ValueError("Both splits need documents.")
    if config["max_new_tokens"] < 1 or config["prefill_chunk_size"] < 1:
        raise ValueError("Token limits must be positive.")
    return config
