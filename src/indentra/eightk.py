"""8-K paragraph selection: the narrative blocks of a filing's HTML, filtered to those that state debt terms.

The same filter built the 8-K test set (scripts/build_eightk.py) and selects paragraphs to extract from (scripts/extract.py).
The fresh 8-K test set (scripts/build_eightk_fresh.py) adds filing selection, deduplication and a capped sample.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable

import lxml.html

from indentra import dataset, edgar

BLOCKS = ("p", "div", "li")
DEBT_TERMS = re.compile(r"notes?\b|credit agreement|facility|term loan|principal|interest|sofr|borrow|indenture|loan", re.I)
MONEY_OR_RATE = re.compile(r"\$\s?\d|\d\s?%|\d\s?percent|basis points", re.I)
MIN_CHARS, MAX_CHARS = 80, 3000


def narrative_paragraphs(html: bytes) -> list[str]:
    root = lxml.html.fromstring(html)
    paragraphs = []
    for el in root.iter(*BLOCKS):
        if any(d.tag in (*BLOCKS, "table") for d in el.iterdescendants()) or any(a.tag == "table" for a in el.iterancestors()):
            continue
        text = " ".join(el.text_content().split())
        if text:
            paragraphs.append(text)
    return paragraphs


def is_debt_term_paragraph(text: str) -> bool:
    return (
        MIN_CHARS <= len(text) <= MAX_CHARS
        and DEBT_TERMS.search(text) is not None
        and MONEY_OR_RATE.search(text) is not None
        and not dataset.looks_tabular(text)
    )


def item_203_filings(filings: Iterable[edgar.Filing], start: str, end: str) -> list[edgar.Filing]:
    """Original 8-Ks (not 8-K/A) reporting Item 2.03, filed from `start` to `end` inclusive, oldest first."""
    selected = [f for f in filings if f.form == "8-K" and "2.03" in f.items and start <= f.filing_date <= end]
    return sorted(selected, key=lambda f: (f.filing_date, f.accession))


class NoveltyFilter:
    """Admits each normalized text once, and never one that appears in the reference texts (earlier sets)."""

    def __init__(self, reference_texts: Iterable[str]) -> None:
        self._reference = {dataset.normalize_text(t) for t in reference_texts}
        self._seen: set[str] = set()

    def check(self, text: str) -> str | None:
        """None if the text is admitted, else why not: "duplicate" (within this set) or "leakage" (in the reference)."""
        key = dataset.normalize_text(text)
        if key in self._seen:
            return "duplicate"
        self._seen.add(key)
        return "leakage" if key in self._reference else None


def capped_sample(candidates: list[dict], size: int, per_filing: int, per_company: int, seed: int) -> list[dict]:
    """Up to `size` candidates in seeded shuffle order, skipping any whose filing or company has reached its cap."""
    order = list(candidates)
    random.Random(seed).shuffle(order)
    per_accession, per_cik = defaultdict(int), defaultdict(int)
    sample = []
    for c in order:
        if len(sample) == size:
            break
        if per_accession[c["accession"]] < per_filing and per_cik[c["cik"]] < per_company:
            per_accession[c["accession"]] += 1
            per_cik[c["cik"]] += 1
            sample.append(c)
    return sample
