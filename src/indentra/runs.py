"""Shared I/O for evaluation runs: load a split, pick few-shot examples, write and read predictions."""

from __future__ import annotations

import hashlib
import json
import random
from decimal import Decimal
from pathlib import Path

from indentra.task import Fact, answer_json

SPLITS = {
    "train": Path("data/processed/train.jsonl"),
    "val": Path("data/processed/val.jsonl"),
    "test": Path("data/processed/test.jsonl"),
    "eightk": Path("data/eightk/test.jsonl"),
    "eightk_fresh": Path("data/eightk_fresh/test.jsonl"),  # frozen 2026-10-14, sha256 in docs/RESULTS.md run 030
    # Label version 2 (exploratory, RESULTS run 035): decided after version 1 results were seen; never a replacement test.
    "eightk_fresh_v2": Path("data/eightk_fresh/v2/test.jsonl"),
    "eightk_v2": Path("data/eightk/v2/test.jsonl"),
    "audit_v1": Path("data/audit/v1/audit.jsonl"),  # the 100 audited 10-K paragraphs as a split, version 1 gold
    "audit_v2": Path("data/audit/v2/audit.jsonl"),
}
RUNS = Path("runs")
AUDIT = Path("data/audit/test_audit.jsonl")
SHOT_MAX_CHARS = 450


def load_split(split: str, limit: int | None = None) -> list[dict]:
    records = [json.loads(line) for line in SPLITS[split].open()]
    return records[:limit] if limit else records


def load_audit() -> list[dict]:
    return [json.loads(line) for line in AUDIT.open()]


def gold_set(record: dict) -> set[Fact]:
    return {(field, Decimal(value)) for field, value in record["gold"]}


def select_shots(train: list[dict], seed: int = 0) -> list[tuple[str, str]]:
    """Three short training examples: a notes issuance, a credit facility, and a hard negative."""
    rng = random.Random(seed)
    short = sorted((r for r in train if len(r["text"]) <= SHOT_MAX_CHARS), key=lambda r: r["id"])

    def fields(r):
        return {field for field, _ in r["gold"]}

    picks = [
        rng.choice([r for r in short if {"face_amount", "stated_rate"} <= fields(r)]),
        rng.choice([r for r in short if {"facility_size", "spread"} <= fields(r)]),
        rng.choice([r for r in short if not r["gold"] and r["other_facts"]]),
    ]
    return [(r["text"], answer_json(gold_set(r))) for r in picks]


def write_predictions(name: str, config: dict, rows: list[dict]) -> Path:
    out = RUNS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    with open(out / "predictions.jsonl", "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)
    return out


def read_predictions(name: str) -> list[dict]:
    return [json.loads(line) for line in (RUNS / name / "predictions.jsonl").open()]


def file_digest(path: Path, chunk: int = 1 << 24) -> str:
    """Short sha256 of a model file, or of every file in a model directory."""
    digest = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        with open(file, "rb") as f:
            while block := f.read(chunk):
                digest.update(block)
    return digest.hexdigest()[:16]
