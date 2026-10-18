"""Non-LLM baselines: regex candidates, then keyword rules or TF-IDF + logistic regression assign a field."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal

from indentra.task import FIELDS, Fact, canonical

MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)(?:\s*(thousand|million|billion))?", re.IGNORECASE)
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?(%|percent\b)", re.IGNORECASE)
BASIS_POINTS = re.compile(r"(\d+(?:\.\d+)?)\s?(?:basis points?|bps)\b", re.IGNORECASE)
SCALE = {None: 1, "thousand": 10**3, "million": 10**6, "billion": 10**9}
LEFT_WORDS, RIGHT_WORDS = 12, 6
NONE = "none"

# Keyword rules: the field whose keyword sits closest to the number (in words) wins; ties go to the earlier rule.
USD_RULES = (
    ("facility_size", re.compile(r"facilit|revolv|line of credit|borrowing capacity|borrowings? of up to|up to", re.I)),
    ("face_amount", re.compile(r"principal|aggregate|notes?\b|debentures|issued|face|term loan", re.I)),
)
PERCENT_RULES = (
    ("commitment_fee", re.compile(r"commitment fee|unused", re.I)),
    ("spread", re.compile(r"sofr|libor|plus|margin|spread|above|base rate|prime|euribor|sonia", re.I)),
    ("effective_rate", re.compile(r"effective", re.I)),
    ("stated_rate", re.compile(r"interest|coupon|bear|notes?\b|rate of", re.I)),
)


@dataclass(frozen=True)
class Candidate:
    kind: str
    value: Decimal
    start: int
    left: str
    right: str


def candidates(text: str) -> list[Candidate]:
    found = []
    for m in MONEY.finditer(text):
        value = Decimal(m.group(1).replace(",", "")) * SCALE[(m.group(2) or "").lower() or None]
        found.append(_candidate(text, "usd", value, m))
    for m in PERCENT.finditer(text):
        found.append(_candidate(text, "percent", Decimal(m.group(1)), m))
    for m in BASIS_POINTS.finditer(text):
        found.append(_candidate(text, "percent", Decimal(m.group(1)) / 100, m))
    return sorted(found, key=lambda c: c.start)


def rules_predict(text: str) -> set[Fact]:
    predicted = set()
    for c in candidates(text):
        rules = USD_RULES if c.kind == "usd" else PERCENT_RULES
        scored = [
            (distance, order, field)
            for order, (field, pattern) in enumerate(rules)
            if (distance := _keyword_distance(pattern, c.left, c.right)) is not None
        ]
        if scored:
            predicted.add((min(scored)[2], c.value))
    return predicted


class TfidfBaseline:
    """Classifies each regex candidate into a field (or none) from the words around it."""

    def __init__(self, seed: int = 0) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self._classifier = LogisticRegression(max_iter=2000, C=4.0, random_state=seed)

    def fit(self, records: list[dict]) -> TfidfBaseline:
        targets = training_targets(records)
        matrix = self._vectorizer.fit_transform([features for features, _ in targets])  # one document per candidate
        rows = [i for i, (_, labels) in enumerate(targets) for _ in labels]
        self._classifier.fit(matrix[rows], [label for _, labels in targets for label, _ in labels],
                             sample_weight=[weight for _, labels in targets for _, weight in labels])
        return self

    def label_candidates(self, text: str) -> list[str]:
        """One label per regex candidate, in candidate order (NONE where predict would drop it): predict's labels, kept per position."""
        found = candidates(text)
        if not found:
            return []
        labels = self._classifier.predict(self._vectorizer.transform([_features(c) for c in found]))
        return [str(label) if label != NONE and _fits(label, c) else NONE for label, c in zip(labels, found, strict=True)]

    def digest(self) -> str:
        """sha256 over the fitted vocabulary, idf, classes and coefficients: equal digests mean the same model."""
        digest = hashlib.sha256()
        for term, index in sorted(self._vectorizer.vocabulary_.items()):
            digest.update(f"{term}\t{index}\n".encode())
        for array in (self._vectorizer.idf_, self._classifier.classes_.astype(str), self._classifier.coef_, self._classifier.intercept_):
            digest.update(array.tobytes())
        return digest.hexdigest()

    def predict(self, text: str) -> set[Fact]:
        found = candidates(text)
        if not found:
            return set()
        labels = self._classifier.predict(self._vectorizer.transform([_features(c) for c in found]))
        return {(label, c.value) for label, c in zip(labels, found, strict=True) if label != NONE and _fits(label, c)}


def training_targets(records: list[dict]) -> list[tuple[str, list[tuple[str, float]]]]:
    """Each regex candidate's features with its training labels: every gold field of the candidate's own unit whose
    value it equals, sorted, each weighted 1/k so a candidate counts once; NONE when no such field matches."""
    targets = []
    for r in records:
        gold = {(field, canonical(Decimal(value))) for field, value in r["gold"]}
        for c in candidates(r["text"]):
            fields = sorted(field for field, value in gold if value == c.value and _fits(field, c))
            targets.append((_features(c), [(field, 1 / len(fields)) for field in fields] or [(NONE, 1.0)]))
    return targets


def _candidate(text: str, kind: str, value: Decimal, m: re.Match) -> Candidate:
    left = " ".join(text[: m.start()].split()[-LEFT_WORDS:])
    right = " ".join(text[m.end() :].split()[:RIGHT_WORDS])
    return Candidate(kind, canonical(value), m.start(), left, right)


def _features(c: Candidate) -> str:
    left = " ".join(f"L_{w}" for w in c.left.lower().split())
    right = " ".join(f"R_{w}" for w in c.right.lower().split())
    return f"KIND_{c.kind} {left} {right}"


def _keyword_distance(pattern: re.Pattern, left: str, right: str) -> int | None:
    """Words between the number and the nearest keyword match on either side, or None if none."""
    distances = [len(left[m.end() :].split()) for m in pattern.finditer(left)]
    distances += [len(right[: m.start()].split()) for m in pattern.finditer(right)]
    return min(distances, default=None)


def _fits(field: str, c: Candidate) -> bool:
    return FIELDS[field][1] == c.kind
