import os
import subprocess
import sys
from decimal import Decimal

from indentra import baselines

TEXT = (
    "In March 2024 we issued $500 million aggregate principal amount of 5.25% senior notes due 2031. "
    "Our $1.25 billion revolving credit facility bears interest at SOFR plus 175 basis points, "
    "with a commitment fee of 0.20% on unused amounts."
)


def test_candidates_normalize_money_percent_and_basis_points():
    found = [(c.kind, c.value) for c in baselines.candidates(TEXT)]
    assert found == [
        ("usd", Decimal("500000000")),
        ("percent", Decimal("5.25")),
        ("usd", Decimal("1250000000")),
        ("percent", Decimal("1.75")),
        ("percent", Decimal("0.2")),
    ]


def test_candidates_handle_full_dollar_figures_and_thousands():
    found = [c.value for c in baselines.candidates("a $1,500,000 term loan and a $250 thousand note")]
    assert found == [Decimal("1500000"), Decimal("250000")]


def test_rules_assign_fields_from_nearby_words():
    assert baselines.rules_predict(TEXT) == {
        ("face_amount", Decimal("500000000")),
        ("stated_rate", Decimal("5.25")),
        ("facility_size", Decimal("1250000000")),
        ("spread", Decimal("1.75")),
        ("commitment_fee", Decimal("0.2")),
    }


def test_tfidf_baseline_learns_field_from_context():
    train = [
        {"text": "we issued $300 million of notes bearing 4% interest", "gold": [["face_amount", "300000000"], ["stated_rate", "4"]]},
        {"text": "the facility provides up to $900 million of borrowings at SOFR plus 1.5%", "gold": [["facility_size", "900000000"], ["spread", "1.5"]]},
    ] * 20
    model = baselines.TfidfBaseline(seed=0).fit(train)
    assert ("face_amount", Decimal("700000000")) in model.predict("we issued $700 million of notes bearing 6% interest")


def test_training_targets_label_every_matching_field_of_the_candidates_unit_with_split_weight():
    record = {"text": "a $500 million term loan and a $500 million revolving facility, with notes at 5% interest",
              "gold": [["facility_size", "500000000"], ["face_amount", "500000000"], ["stated_rate", "5"]]}
    targets = baselines.training_targets([record])
    assert [labels for _, labels in targets] == [
        [("face_amount", 0.5), ("facility_size", 0.5)],
        [("face_amount", 0.5), ("facility_size", 0.5)],
        [("stated_rate", 1.0)],
    ]


def test_training_targets_ignore_gold_fields_of_the_other_unit():
    record = {"text": "an arrangement fee of $5 on the 5% notes", "gold": [["stated_rate", "5"]]}
    assert [labels for _, labels in baselines.training_targets([record])] == [[(baselines.NONE, 1.0)], [("stated_rate", 1.0)]]


def test_tfidf_fit_is_identical_under_any_hash_seed():
    # Values shared by two fields made the old first-match-from-a-set label depend on the per-process string hash seed.
    script = (
        "from indentra import baselines\n"
        "train = [{'text': 'a $500 million term loan and a $500 million revolving facility at 5% interest',"
        " 'gold': [['facility_size', '500000000'], ['face_amount', '500000000'], ['stated_rate', '5']]},"
        " {'text': 'we issued $300 million of notes bearing 4% interest', 'gold': [['face_amount', '300000000'], ['stated_rate', '4']]},"
        " {'text': 'the facility provides up to $900 million at SOFR plus 1.5%', 'gold': [['facility_size', '900000000'], ['spread', '1.5']]}] * 10\n"
        "print(baselines.TfidfBaseline(seed=0).fit(train).digest())\n"
    )
    digests = {
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
                       env={**os.environ, "PYTHONHASHSEED": seed}).stdout
        for seed in ("0", "1", "2", "3")
    }
    assert len(digests) == 1
