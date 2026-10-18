"""Candidates format: regex finds the numbers (as in the baselines), the LLM only says which are facts and of which field.

The prompt is the minimal prompt plus a numbered list of the paragraph's candidates; the answer names candidates by
number, so every predicted value is grounded in the text and normalized by code. A paragraph without candidates has
the fixed empty answer and needs no model call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

from indentra import baselines, task
from indentra.task import FIELDS, Fact

PROMPT = task.MINIMAL_PROMPT + "\n\nCandidates:\n{candidates}"
EMPTY_ANSWER = '{"facts": []}'
PATTERNS = {"usd": (baselines.MONEY,), "percent": (baselines.PERCENT, baselines.BASIS_POINTS)}


@dataclass(frozen=True)
class Candidate:
    kind: str
    value: Decimal
    surface: str


def candidates(text: str) -> list[Candidate]:
    """The baselines' dollar, percent and basis-point candidates in text order, with the text each was read from."""
    return [Candidate(c.kind, c.value, _surface(text, c)) for c in baselines.candidates(text)]


def messages(text: str, found: list[Candidate]) -> list[dict[str, str]]:
    lines = "\n".join(f"{number}. {c.surface} -> {format(c.value, 'f')}" for number, c in enumerate(found, 1))
    return [{"role": "system", "content": task.SYSTEM}, {"role": "user", "content": PROMPT.format(text=text, candidates=lines)}]


def answer_json(found: list[Candidate], gold: set[Fact]) -> str:
    """Each candidate is labeled with every gold field of its value and unit kind; unlabeled candidates are left out."""
    return json.dumps({"facts": [[number, field] for number, c in enumerate(found, 1) for field in _gold_fields(c, gold)]})


def covered(found: list[Candidate], gold: set[Fact]) -> set[Fact]:
    """The gold facts this format can express: some candidate has the value and the field's unit kind."""
    return {(field, c.value) for c in found for field in _gold_fields(c, gold)}


def parse_output(raw: str, found: list[Candidate]) -> task.Parsed:
    """Strict: any unknown field, unknown candidate number or schema violation makes the whole answer invalid and empty."""
    pairs = _decoded_pairs(raw)
    if pairs is None or not all(_well_formed(pair, len(found)) for pair in pairs):
        return task.Parsed(False, frozenset())
    return task.Parsed(True, frozenset((field, found[number - 1].value) for number, field in pairs))


def labeled_pairs(raw: str, found: list[Candidate]) -> list[tuple[int, str]] | None:
    """The (candidate number, field) pairs of a valid answer under the same strict rules as parse_output, or None."""
    pairs = _decoded_pairs(raw)
    if pairs is None or not all(_well_formed(pair, len(found)) for pair in pairs):
        return None
    return [(number, field) for number, field in pairs]


def parse_output_lenient(raw: str, found: list[Candidate]) -> set[Fact]:
    """Secondary view: keep every well-formed pair and drop the rest."""
    return {(pair[1], found[pair[0] - 1].value) for pair in _decoded_pairs(raw) or [] if _well_formed(pair, len(found))}


def _gold_fields(c: Candidate, gold: set[Fact]) -> list[str]:
    return sorted({field for field, value in gold if value == c.value and FIELDS[field][1] == c.kind})


def _surface(text: str, c: baselines.Candidate) -> str:
    match = next(m for pattern in PATTERNS[c.kind] if (m := pattern.match(text, c.start)))
    return " ".join(match.group(0).split())


def _decoded_pairs(raw: str) -> list | None:
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except ValueError:  # JSONDecodeError, or an integer too long for int()
        return None
    pairs = obj.get("facts") if isinstance(obj, dict) else None
    return pairs if isinstance(pairs, list) else None


def _well_formed(pair, count: int) -> bool:
    return (
        isinstance(pair, list)
        and len(pair) == 2
        and type(pair[0]) is int  # not bool, not 1.0
        and 1 <= pair[0] <= count
        and isinstance(pair[1], str)
        and pair[1] in FIELDS
    )
