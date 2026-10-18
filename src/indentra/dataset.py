"""Split rules: companies never cross splits, and test filings are strictly later than training filings.

Llama 3.2's pretraining data runs through December 2023, so val (filed 2025) and test (filed 2026)
text was published after the base model's cutoff.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict

SHINGLE_SIZE = 3
MAX_SHINGLE_DOC_FREQUENCY = 200  # boilerplate shingles shared by many paragraphs are too common to index
SPLIT_BUCKETS = (("train", 65), ("val", 75), ("test", 100))
WINDOWS = {
    "train": ("2022-01-01", "2025-12-31"),
    "val": ("2025-01-01", "2025-12-31"),
    "test": ("2026-01-01", "2026-12-31"),
}


def assign_split(cik: int) -> str:
    bucket = int(hashlib.sha256(str(cik).encode()).hexdigest(), 16) % 100
    return next(name for name, upper in SPLIT_BUCKETS if bucket < upper)


def in_window(split: str, filing_date: str) -> bool:
    start, end = WINDOWS[split]
    return start <= filing_date <= end


def subsample_negatives(records: list[dict], fraction: float, seed: int) -> list[dict]:
    """Every record with gold facts plus a seeded `fraction` of those without, in the original order."""
    if not 0 <= fraction <= 1:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    negatives = [i for i, r in enumerate(records) if not r["gold"]]
    kept = set(random.Random(seed).sample(negatives, round(len(negatives) * fraction)))
    return [r for i, r in enumerate(records) if r["gold"] or i in kept]


NUMERIC_TOKEN = re.compile(r"^[\$\(\)\d,.%—–\-]+$")
SENTENCE = re.compile(r"[a-z]{2,}[^.]*\.(?:\s|$)")
TABULAR_RATIO, TABULAR_RATIO_WITHOUT_SENTENCE = 0.5, 0.15


def looks_tabular(text: str) -> bool:
    """Tables laid out with <div>s flatten into number runs with no sentence; narrative has sentences."""
    tokens = text.split()
    ratio = sum(bool(NUMERIC_TOKEN.match(t)) for t in tokens) / len(tokens)
    return ratio >= TABULAR_RATIO or (SENTENCE.search(text) is None and ratio >= TABULAR_RATIO_WITHOUT_SENTENCE)


def normalize_text(text: str) -> str:
    """Lowercase, numbers collapsed to '#': boilerplate with different figures counts as the same text."""
    return " ".join(re.sub(r"\d[\d,.]*", "#", text.lower()).split())


def shingles(text: str) -> set[tuple[str, ...]]:
    words = normalize_text(text).split()
    return {tuple(words[i : i + SHINGLE_SIZE]) for i in range(max(len(words) - SHINGLE_SIZE + 1, 1))}


class ShingleIndex:
    """Word-shingle Jaccard lookup against a reference set (the training split)."""

    def __init__(self, texts: list[str]) -> None:
        self._sets = [shingles(t) for t in texts]
        postings = defaultdict(list)
        for doc, items in enumerate(self._sets):
            for item in items:
                postings[item].append(doc)
        self._postings = {k: v for k, v in postings.items() if len(v) <= MAX_SHINGLE_DOC_FREQUENCY}

    def max_jaccard(self, text: str) -> float:
        query = shingles(text)
        candidates = {doc for item in query for doc in self._postings.get(item, ())}
        return max((len(query & self._sets[d]) / len(query | self._sets[d]) for d in candidates), default=0.0)
