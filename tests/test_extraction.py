import json
import threading
from decimal import Decimal

import pytest

from indentra import baselines, extraction, hybrid

TEXT = ("In March 2024 we issued $500 million aggregate principal amount of 5.25% senior notes due 2031. "
        "Our $1.25 billion revolving credit facility bears interest at SOFR plus 175 basis points, "
        "with a commitment fee of 0.20% on unused amounts; the $500 million notes are unsecured.")


def test_paragraph_check_rejects_empty_and_too_long_input_without_truncating():
    with pytest.raises(extraction.InputError, match="Paste a paragraph"):
        extraction.check_paragraph("   \n ")
    with pytest.raises(extraction.InputError, match="3,001 characters"):
        extraction.check_paragraph("x" * 3001)
    assert extraction.check_paragraph("x" * 3000) == "x" * 3000
    assert extraction.check_paragraph("  The notes\n bear  5% interest. ") == "The notes bear 5% interest."


def test_candidate_spans_point_at_the_text_each_candidate_was_read_from():
    spans = extraction.candidate_spans(TEXT)
    assert [s.text for s in spans] == ["$500 million", "5.25%", "$1.25 billion", "175 basis points", "0.20%", "$500 million"]
    assert all(TEXT[s.start:s.end] == s.text for s in spans)


def test_grouped_facts_carry_every_labeled_span_and_count_unlabeled_repeats():
    facts = extraction.group_facts(TEXT, [(0, "face_amount"), (1, "stated_rate"), (2, "facility_size"), (3, "spread")])
    by_field = {f.field: f for f in facts}
    principal = by_field["face_amount"]
    assert principal.value == "500000000" and principal.display == "$500,000,000" and principal.label == "Principal amount"
    assert [s.text for s in principal.spans] == ["$500 million"] and principal.other_occurrences == 1
    assert by_field["spread"].display == "1.75%" and by_field["spread"].spans[0].text == "175 basis points"
    both = extraction.group_facts(TEXT, [(0, "face_amount"), (5, "face_amount")])
    assert len(both) == 1 and len(both[0].spans) == 2 and both[0].other_occurrences == 0


def test_values_are_formatted_for_reading_and_kept_exact():
    assert extraction.format_value("facility_size", Decimal("1250000000")) == "$1,250,000,000"
    assert extraction.format_value("face_amount", Decimal("2850")) == "$2,850"
    assert extraction.format_value("stated_rate", Decimal("7.125")) == "7.125%"
    assert extraction.format_value("spread", Decimal("-0.5")) == "-0.5%"


@pytest.mark.parametrize("raw", ['{"facts": [[1, "face_amount"], [2, "stated_rate"]]}', '{"facts": []}', '{"facts": [[9, "face_amount"]]}',
                                 '{"facts": [[1, "coupon"]]}', "not json", '{"facts": [[true, "spread"]]}'])
def test_labeled_pairs_agree_with_the_strict_parser(raw):
    found = hybrid.candidates(TEXT)
    pairs, parsed = hybrid.labeled_pairs(raw, found), hybrid.parse_output(raw, found)
    assert (pairs is not None) == parsed.valid
    if pairs is not None:
        assert {(field, found[n - 1].value) for n, field in pairs} == set(parsed.facts)


def test_tfidf_candidate_labels_reproduce_predict():
    train = [{"text": "we issued $300 million of notes bearing 4% interest", "gold": [["face_amount", "300000000"], ["stated_rate", "4"]]},
             {"text": "the facility provides up to $900 million at SOFR plus 1.5%", "gold": [["facility_size", "900000000"], ["spread", "1.5"]]}] * 20
    model = baselines.TfidfBaseline(seed=0).fit(train)
    labels = model.label_candidates(TEXT)
    found = baselines.candidates(TEXT)
    assert len(labels) == len(found)
    assert {(label, c.value) for c, label in zip(found, labels, strict=True) if label != baselines.NONE} == model.predict(TEXT)


class FakeGenerate:
    def __init__(self, answer: str, truncated: bool = False):
        self.answer, self.truncated, self.calls = answer, truncated, 0

    def __call__(self, ids: list[int]) -> tuple[str, bool]:
        self.calls += 1
        return self.answer, self.truncated


def make_b3(generate) -> extraction.B3Extractor:
    b3 = extraction.B3Extractor()
    b3.use_generator(generate, render=lambda messages: [1, 2, 3], identity={"adapter": "test"})
    return b3


def test_b3_without_candidates_answers_empty_without_calling_the_model():
    generate = FakeGenerate('{"facts": [[1, "face_amount"]]}')
    result = make_b3(generate).extract("The notes are senior unsecured obligations of the Company.")
    assert result.facts == () and result.model_called is False and generate.calls == 0 and result.valid


def test_b3_maps_the_models_candidate_numbers_to_facts_with_spans():
    result = make_b3(FakeGenerate('{"facts": [[1, "face_amount"], [2, "stated_rate"], [6, "face_amount"]]}')).extract(TEXT)
    assert result.valid and result.model_called and result.engine == "b3"
    assert {(f.field, f.value) for f in result.facts} == {("face_amount", "500000000"), ("stated_rate", "5.25")}
    assert len(next(f for f in result.facts if f.field == "face_amount").spans) == 2


def test_b3_malformed_answer_gives_no_facts_and_says_so():
    result = make_b3(FakeGenerate('{"facts": [[1, "coupon"]]}')).extract(TEXT)
    assert result.valid is False and result.facts == () and "format" in result.note


def test_b3_refuses_to_load_when_files_are_missing(tmp_path):
    b3 = extraction.B3Extractor(model_dir=tmp_path / "model", adapter_dir=tmp_path / "adapter", eval_config=tmp_path / "config.json")
    with pytest.raises(extraction.LoadError, match="missing"):
        b3.load()


def test_b3_refuses_to_load_a_model_other_than_the_one_evaluated(tmp_path):
    model, adapter = tmp_path / "model", tmp_path / "adapter"
    for d, names in ((model, extraction.MODEL_FILES), (adapter, extraction.ADAPTER_FILES)):
        d.mkdir()
        for name in names:
            (d / name).write_text("x")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"model_digest": "0" * 16, "adapter_digest": "0" * 16}))
    with pytest.raises(extraction.LoadError, match="does not match"):
        extraction.B3Extractor(model_dir=model, adapter_dir=adapter, eval_config=config).load()


def test_b3_generates_on_one_dedicated_thread_whatever_thread_calls_it():
    # MLX binds its GPU stream to the thread that created it; a web server calls extract from many threads.
    seen = []

    def generate(ids):
        seen.append(threading.get_ident())
        return '{"facts": []}', False

    b3 = make_b3(generate)
    callers = [threading.Thread(target=b3.extract, args=(TEXT,)) for _ in range(3)]
    for t in callers:
        t.start()
    for t in callers:
        t.join()
    assert len(seen) == 3 and len(set(seen)) == 1 and seen[0] not in {t.ident for t in callers}
