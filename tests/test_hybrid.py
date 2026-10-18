from decimal import Decimal

import pytest

from indentra import hybrid, task

TEXT = (
    "In March 2024 we issued $500 million aggregate principal amount of 5.25% senior notes due 2031. "
    "Our $1.25 billion revolving credit facility bears interest at SOFR plus 175 basis points, "
    "with a commitment fee of 0.20% on unused amounts."
)
FOUND = hybrid.candidates(TEXT)


def test_candidates_keep_the_surface_text_and_canonical_value_in_text_order():
    assert [(c.kind, c.surface, c.value) for c in FOUND] == [
        ("usd", "$500 million", Decimal("500000000")),
        ("percent", "5.25%", Decimal("5.25")),
        ("usd", "$1.25 billion", Decimal("1250000000")),
        ("percent", "175 basis points", Decimal("1.75")),
        ("percent", "0.20%", Decimal("0.2")),
    ]


@pytest.mark.parametrize(
    "text, surface, value",
    [
        ("SOFR plus 175 basis points", "175 basis points", Decimal("1.75")),
        ("an applicable margin of 1 basis point", "1 basis point", Decimal("0.01")),
        ("a fee of 12.5 bps on unused amounts", "12.5 bps", Decimal("0.125")),
        ("a $1.5\n billion term loan", "$1.5 billion", Decimal("1500000000")),
    ],
)
def test_candidates_normalize_basis_points_and_whitespace(text, surface, value):
    assert [(c.surface, c.value) for c in hybrid.candidates(text)] == [(surface, value)]


def test_messages_extend_the_minimal_prompt_with_a_numbered_candidate_list():
    messages = hybrid.messages("We issued $500 million of 5.25% notes.", hybrid.candidates("We issued $500 million of 5.25% notes."))
    minimal = task.messages("We issued $500 million of 5.25% notes.", instructions=False)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0] == minimal[0]
    assert messages[1]["content"] == (
        "Paragraph:\nWe issued $500 million of 5.25% notes.\n\nCandidates:\n1. $500 million -> 500000000\n2. 5.25% -> 5.25"
    )
    assert messages[1]["content"].startswith(minimal[1]["content"])


def test_answer_names_matching_candidates_and_omits_the_rest():
    gold = {("face_amount", Decimal("500000000")), ("stated_rate", Decimal("5.25")), ("spread", Decimal("1.75")),
            ("effective_rate", Decimal("5.6"))}
    assert hybrid.answer_json(FOUND, gold) == '{"facts": [[1, "face_amount"], [2, "stated_rate"], [4, "spread"]]}'


def test_answer_labels_a_candidate_with_every_matching_field_and_every_repeat_of_the_value():
    text = "the $500 million notes and the $500 million revolving facility"
    gold = {("face_amount", Decimal("500000000")), ("facility_size", Decimal("500000000"))}
    assert hybrid.answer_json(hybrid.candidates(text), gold) == (
        '{"facts": [[1, "face_amount"], [1, "facility_size"], [2, "face_amount"], [2, "facility_size"]]}'
    )


def test_answer_skips_a_field_whose_unit_does_not_fit_the_candidate():
    found = hybrid.candidates("a $5 fee on notes that bear interest at 5%")
    assert hybrid.answer_json(found, {("stated_rate", Decimal("5"))}) == '{"facts": [[2, "stated_rate"]]}'


def test_covered_gold_facts_need_a_candidate_with_the_value_and_unit_kind():
    found = hybrid.candidates("a $5 fee on $500 million of notes")
    gold = {("face_amount", Decimal("500000000")), ("stated_rate", Decimal("5")), ("spread", Decimal("1.75"))}
    assert hybrid.covered(found, gold) == {("face_amount", Decimal("500000000"))}


def test_gold_answer_parses_back_to_the_covered_gold_facts():
    gold = {("face_amount", Decimal("500000000")), ("commitment_fee", Decimal("0.2")), ("stated_rate", Decimal("6"))}
    parsed = hybrid.parse_output(hybrid.answer_json(FOUND, gold), FOUND)
    assert parsed == task.Parsed(True, frozenset(gold - {("stated_rate", Decimal("6"))}))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"facts": [[1, "face_amount"], [4, "spread"]]}', {("face_amount", Decimal("500000000")), ("spread", Decimal("1.75"))}),
        ('Sure! {"facts": [[2, "stated_rate"]]} Hope that helps.', {("stated_rate", Decimal("5.25"))}),
        ('{"facts": [[5, "commitment_fee"], [5, "commitment_fee"]]}', {("commitment_fee", Decimal("0.2"))}),
        ('{"facts": [[1, "stated_rate"]]}', {("stated_rate", Decimal("500000000"))}),
        ('{"facts": []}', set()),
    ],
)
def test_parse_output_accepts_well_formed_answers(raw, expected):
    assert hybrid.parse_output(raw, FOUND) == task.Parsed(True, frozenset(expected))


@pytest.mark.parametrize(
    "raw",
    [
        '{"facts": [[1, "face_amount"], [2, "maturity"]]}',
        '{"facts": [[0, "face_amount"]]}',
        '{"facts": [[6, "face_amount"]]}',
        '{"facts": [[-1, "face_amount"]]}',
        '{"facts": [["1", "face_amount"]]}',
        '{"facts": [[1.0, "face_amount"]]}',
        '{"facts": [[true, "face_amount"]]}',
        '{"facts": [[1, "face_amount", "extra"]]}',
        '{"facts": [[1, ["face_amount"]]]}',
        '{"facts": [{"field": "face_amount", "value": 500000000}]}',
        '{"facts": [[1, "face_amount"]',
        '{"facts": {"1": "face_amount"}}',
        '{"answer": []}',
        '[[1, "face_amount"]]',
        "no json here",
    ],
)
def test_parse_output_rejects_the_whole_answer_on_any_violation(raw):
    assert hybrid.parse_output(raw, FOUND) == task.Parsed(False, frozenset())


def test_parse_output_marks_numbers_too_long_to_convert_invalid_instead_of_raising():
    raw = '{"facts": [[' + "9" * 5000 + ', "face_amount"]]}'
    assert hybrid.parse_output(raw, FOUND) == task.Parsed(False, frozenset())
    assert hybrid.parse_output_lenient(raw, FOUND) == set()


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('{"facts": [[1, "face_amount"], [2, "maturity"], [9, "spread"], ["4", "spread"]]}', {("face_amount", Decimal("500000000"))}),
        ('{"facts": [[1, "face_amount"]', set()),
        ("no json", set()),
    ],
)
def test_lenient_parse_keeps_well_formed_pairs_and_drops_the_rest(raw, expected):
    assert hybrid.parse_output_lenient(raw, FOUND) == expected


def test_zero_candidates_get_the_fixed_empty_answer():
    found = hybrid.candidates("The notes are senior unsecured obligations of the Company.")
    assert found == []
    assert hybrid.answer_json(found, {("face_amount", Decimal("500000000"))}) == hybrid.EMPTY_ANSWER == '{"facts": []}'
    assert hybrid.parse_output(hybrid.EMPTY_ANSWER, found) == task.Parsed(True, frozenset())
    assert hybrid.parse_output('{"facts": [[1, "face_amount"]]}', found) == task.Parsed(False, frozenset())
