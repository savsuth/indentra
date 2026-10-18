from collections import Counter

import pytest

from indentra import dataset


def test_split_is_deterministic_per_company():
    assert dataset.assign_split(320193) == dataset.assign_split(320193)


def test_split_proportions_are_close_to_targets():
    counts = Counter(dataset.assign_split(cik) for cik in range(1, 20001))
    assert abs(counts["train"] / 20000 - 0.65) < 0.02
    assert abs(counts["val"] / 20000 - 0.10) < 0.02
    assert abs(counts["test"] / 20000 - 0.25) < 0.02


def test_normalize_text_ignores_case_numbers_and_spacing():
    assert dataset.normalize_text("We issued $500 million  of Notes.") == dataset.normalize_text("we issued $750 million of notes.")


def test_near_duplicates_are_found_against_reference_set():
    reference = [
        "the revolving credit facility bears interest at sofr plus a margin and matures in 2029 under the agreement",
        "completely unrelated text about leases and office space in several cities across the country",
    ]
    index = dataset.ShingleIndex(reference)
    near = "the revolving credit facility bears interest at sofr plus a margin and matures in 2030 under the agreement"
    assert index.max_jaccard(near) > 0.7
    assert index.max_jaccard("a new paragraph describing convertible notes issued to investors in a private placement") < 0.3


@pytest.mark.parametrize(
    "text, tabular",
    [
        ("Available Capacity $10 million revolving credit arrangement (due December 2026) $3,172 $3,172", True),
        ("Amount Outstanding as of December 31, 2023 Interest Rate (1) Date (Dollars in millions) Loan $ 75.0 $ 60.9 4.10 % June", True),
        ("In millions20212020Current maturities of long-term debt$568 $350 Commercial paper210 — Total short-term debt$778 $350", True),
        ("As of January 31, 2024 and January 31, 2023, the convertible note was $48,861 and $0, respectively.", False),
        ("In March 2024 we issued $500 million of 5.25% senior notes due 2031.", False),
        ("The facility bears interest at SOFR plus a margin of 1.25% to 1.75%, depending on our leverage ratio.", False),
    ],
)
def test_looks_tabular_separates_flattened_tables_from_narrative(text, tabular):
    assert dataset.looks_tabular(text) is tabular


def test_filing_windows_keep_test_strictly_after_training_time():
    assert dataset.in_window("train", "2022-01-03")
    assert dataset.in_window("train", "2025-12-30")
    assert not dataset.in_window("train", "2026-01-02")
    assert dataset.in_window("val", "2025-06-01")
    assert not dataset.in_window("val", "2024-06-01")
    assert dataset.in_window("test", "2026-03-01")
    assert not dataset.in_window("test", "2025-12-31")


def test_negative_subsampling_keeps_every_positive_and_a_seeded_share_of_negatives():
    records = [{"id": str(i), "gold": [["face_amount", "1"]] if i % 2 else []} for i in range(600)]
    kept = dataset.subsample_negatives(records, fraction=1 / 3, seed=0)
    assert [r["id"] for r in kept if r["gold"]] == [r["id"] for r in records if r["gold"]]
    assert sum(not r["gold"] for r in kept) == 100
    assert kept == dataset.subsample_negatives(records, fraction=1 / 3, seed=0)
    assert kept != dataset.subsample_negatives(records, fraction=1 / 3, seed=1)
    assert [r["id"] for r in kept] == sorted((r["id"] for r in kept), key=int)  # original order preserved
    with pytest.raises(ValueError):
        dataset.subsample_negatives(records, fraction=1.5, seed=0)
