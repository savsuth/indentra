from decimal import Decimal

import pytest

from indentra import task


def fact(name, value, unit="iso4217:USD"):
    return {"name": name, "value": value, "unit": unit}


def test_gold_maps_concepts_to_fields_and_normalizes_units():
    facts = [
        fact("us-gaap:DebtInstrumentFaceAmount", "500000000"),
        fact("us-gaap:DebtInstrumentInterestRateStatedPercentage", "0.05250", "xbrli:pure"),
        fact("us-gaap:DebtInstrumentBasisSpreadOnVariableRate1", "0.0175", "xbrli:pure"),
        fact("us-gaap:InterestExpenseDebt", "12000000"),
    ]
    assert task.gold_facts(facts) == {
        ("face_amount", Decimal("500000000")),
        ("stated_rate", Decimal("5.25")),
        ("spread", Decimal("1.75")),
    }


def test_gold_treats_both_standard_commitment_fee_concepts_as_one_field():
    facts = [
        fact("us-gaap:LineOfCreditFacilityCommitmentFeePercentage", "0.0020", "xbrli:pure"),
        fact("us-gaap:LineOfCreditFacilityUnusedCapacityCommitmentFeePercentage", "0.0035", "xbrli:pure"),
    ]
    assert task.gold_facts(facts) == {("commitment_fee", Decimal("0.2")), ("commitment_fee", Decimal("0.35"))}


def test_gold_deduplicates_repeated_tags_of_the_same_number():
    facts = [fact("us-gaap:DebtInstrumentFaceAmount", "500000000")] * 2
    assert task.gold_facts(facts) == {("face_amount", Decimal("500000000"))}


def test_answer_json_is_canonical_and_sorted():
    gold = {("stated_rate", Decimal("5.25")), ("face_amount", Decimal("500000000"))}
    assert task.answer_json(gold) == '{"facts": [{"field": "face_amount", "value": 500000000}, {"field": "stated_rate", "value": 5.25}]}'
    assert task.answer_json(set()) == '{"facts": []}'


@pytest.mark.parametrize(
    "raw, valid, expected",
    [
        ('{"facts": [{"field": "face_amount", "value": 500000000}]}', True, {("face_amount", Decimal("500000000"))}),
        ('{"facts": [{"field": "stated_rate", "value": 5.250}]}', True, {("stated_rate", Decimal("5.25"))}),
        ('Sure! {"facts": []} Hope that helps.', True, set()),
        ('{"facts": [{"field": "face_amount", "value": "500000000"}]}', False, set()),
        ('{"facts": [{"field": "maturity", "value": 2031}]}', False, set()),
        ('{"facts": [{"field": "face_amount", "value": 5e8}]}', True, {("face_amount", Decimal("500000000"))}),
        ('{"facts": [{"field": "face_amount"', False, set()),
        ("no json here", False, set()),
    ],
)
def test_parse_output_is_strict_about_schema(raw, valid, expected):
    parsed = task.parse_output(raw)
    assert parsed.valid is valid
    assert parsed.facts == expected


@pytest.mark.parametrize(
    "raw",
    [
        '{"facts": [{"field": "face_amount", "value": 1e30}]}',
        '{"facts": [{"field": "face_amount", "value": 1e999999999}]}',
        '{"facts": [{"field": "face_amount", "value": ' + "9" * 40 + "}]}",
        '{"facts": [{"field": "face_amount", "value": NaN}]}',
        '{"facts": [{"field": "face_amount", "value": Infinity}]}',
    ],
)
def test_parse_output_marks_degenerate_numbers_invalid_instead_of_raising(raw):
    assert task.parse_output(raw) == task.Parsed(False, frozenset())


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"facts": [{"field": "facility_size", "value": 2000000000}, {"field": "stated_rate", "value": null}]}',
         {("facility_size", Decimal("2000000000"))}),
        ('{"facts": [{"field": "face_amount", "value": 1}, {"field": "maturity_date", "value": 2026}]}',
         {("face_amount", Decimal("1"))}),
        ('{"facts": [{"field": "spread", "value": "1.75"}, {"field": "spread", "value": [1, 2]}]}', set()),
        ('{"facts": [{"field": "face_amount", "value": 5', set()),
        ("no json", set()),
    ],
)
def test_lenient_parse_keeps_well_formed_items_and_drops_the_rest(raw, expected):
    assert task.parse_output_lenient(raw) == expected


def test_parse_output_rejects_whole_answer_when_any_field_is_unknown():
    raw = '{"facts": [{"field": "face_amount", "value": 1}, {"field": "bogus", "value": 2}]}'
    assert task.parse_output(raw).valid is False


def test_minimal_prompt_drops_instructions_but_keeps_system_and_paragraph():
    full = task.messages("Target text.")
    minimal = task.messages("Target text.", instructions=False)
    assert minimal[0] == full[0]
    assert minimal[1]["content"] == "Paragraph:\nTarget text."
    assert len(minimal[1]["content"]) < len(full[1]["content"]) // 4


def test_messages_put_paragraph_last_and_few_shots_as_prior_turns():
    shots = [("Example text.", '{"facts": []}')]
    messages = task.messages("Target text.", shots)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[-1]["content"].endswith("Target text.")
    assert messages[2]["content"] == '{"facts": []}'
