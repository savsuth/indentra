"""The extraction task: target fields, gold answers from XBRL facts, the prompt, and strict output parsing.

One prompt format is used for few-shot baselines, fine-tuning data and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal

# field -> (us-gaap concepts, unit kind, description shown to the model). A field lists more than one
# concept only when the standard taxonomy has two elements for the same idea.
FIELDS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "face_amount": (("us-gaap:DebtInstrumentFaceAmount",), "usd", "principal (face or par) amount of a debt instrument"),
    "stated_rate": (("us-gaap:DebtInstrumentInterestRateStatedPercentage",), "percent", "stated (coupon) interest rate of a debt instrument"),
    "effective_rate": (("us-gaap:DebtInstrumentInterestRateEffectivePercentage",), "percent", "effective interest rate of a debt instrument"),
    "spread": (("us-gaap:DebtInstrumentBasisSpreadOnVariableRate1",), "percent", "margin added to a variable reference rate such as SOFR"),
    "facility_size": (("us-gaap:LineOfCreditFacilityMaximumBorrowingCapacity",), "usd", "maximum borrowing capacity of a credit facility"),
    "commitment_fee": (
        ("us-gaap:LineOfCreditFacilityCommitmentFeePercentage", "us-gaap:LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage"),
        "percent",
        "fee rate charged on the unused part of a credit facility",
    ),
}
CONCEPT_TO_FIELD = {concept: field for field, (concepts, _, _) in FIELDS.items() for concept in concepts}

Fact = tuple[str, Decimal]
# No debt term reaches a quintillion dollars; larger numbers are degenerate generations, and beyond 28 digits
# Decimal's default context cannot canonicalize them at all.
MAX_VALUE = Decimal(10) ** 18

SYSTEM = "You extract debt terms from SEC filings. Reply with JSON only."
PROMPT = """Extract every value stated in the paragraph for these fields:
{fields}

Write dollar amounts in full ($1.5 billion -> 1500000000) and rates as percentages (5.25% -> 5.25; 175 basis points -> 1.75).
Reply as {{"facts": [{{"field": "<field>", "value": <number>}}]}}, or {{"facts": []}} if the paragraph states none.

Paragraph:
{text}"""
# The fine-tuned model learns the fields from data, so it is trained and evaluated with the paragraph alone;
# this cuts each example from about 350 to about 150 tokens (training on the M4 is compute-bound).
MINIMAL_PROMPT = "Paragraph:\n{text}"
FIELD_LINES = "\n".join(f"- {field}: {description}" for field, (_, _, description) in FIELDS.items())


@dataclass(frozen=True)
class Parsed:
    valid: bool
    facts: frozenset[Fact]


def gold_facts(facts: list[dict]) -> set[Fact]:
    """Target facts of a paragraph in answer units; tagged facts outside FIELDS are ignored."""
    gold = set()
    for f in facts:
        field = CONCEPT_TO_FIELD.get(f["name"])
        if field is None or f["value"] is None:
            continue
        value = Decimal(f["value"])
        if FIELDS[field][1] == "percent":
            value *= 100
        gold.add((field, canonical(value)))
    return gold


def answer_json(facts: set[Fact]) -> str:
    items = ", ".join(
        f'{{"field": "{field}", "value": {format(value, "f")}}}' for field, value in sorted(facts)
    )
    return f'{{"facts": [{items}]}}'


def messages(text: str, shots: list[tuple[str, str]] = (), instructions: bool = True) -> list[dict[str, str]]:
    def user(paragraph: str) -> str:
        return PROMPT.format(fields=FIELD_LINES, text=paragraph) if instructions else MINIMAL_PROMPT.format(text=paragraph)

    turns = [{"role": "system", "content": SYSTEM}]
    for shot_text, shot_answer in shots:
        turns.append({"role": "user", "content": user(shot_text)})
        turns.append({"role": "assistant", "content": shot_answer})
    turns.append({"role": "user", "content": user(text)})
    return turns


def parse_output(raw: str) -> Parsed:
    """Strict: any schema violation makes the whole answer invalid and empty."""
    items = _decoded_items(raw)
    if items is None or not all(_well_formed(item) for item in items):
        return Parsed(False, frozenset())
    return Parsed(True, frozenset((item["field"], canonical(item["value"])) for item in items))


def parse_output_lenient(raw: str) -> set[Fact]:
    """Secondary view: keep every well-formed item and drop the rest, to separate content errors from format errors."""
    items = _decoded_items(raw) or []
    return {(item["field"], canonical(item["value"])) for item in items if _well_formed(item)}


def _decoded_items(raw: str) -> list | None:
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder(parse_float=Decimal, parse_int=Decimal).raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    items = obj.get("facts") if isinstance(obj, dict) else None
    return items if isinstance(items, list) else None


def _well_formed(item) -> bool:
    return (
        isinstance(item, dict)
        and item.get("field") in FIELDS
        and isinstance(item.get("value"), Decimal)
        and item["value"].is_finite()
        and item["value"].copy_abs() < MAX_VALUE  # copy_abs is exact; abs() rounds and can overflow
    )


def canonical(value: Decimal) -> Decimal:
    normalized = value.normalize()
    return normalized.quantize(Decimal(1)) if normalized == normalized.to_integral_value() else normalized
