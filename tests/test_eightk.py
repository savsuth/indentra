from collections import Counter

from indentra.edgar import Filing
from indentra.eightk import NoveltyFilter, capped_sample, is_debt_term_paragraph, item_203_filings, narrative_paragraphs

HTML = b"""<html><body>
<div><p>Item 2.03 Creation of a Direct Financial Obligation.</p></div>
<div><div>On March 3, 2026, the Company issued $500,000,000 aggregate principal amount of 5.25% Senior Notes due 2031 under an indenture with the trustee.</div></div>
<table><tr><td>Notes issued in a table should be skipped: $100,000,000 at 4.00% interest</td></tr></table>
<p>The Company held its annual meeting of stockholders, where 51% of shares voted in favor of the proposal to elect directors.</p>
<p>Forward-looking statements are subject to risks.</p>
</body></html>"""


def test_narrative_paragraphs_take_leaf_blocks_and_skip_tables():
    paragraphs = narrative_paragraphs(HTML)
    assert any(p.startswith("On March 3, 2026") for p in paragraphs)
    assert not any("should be skipped" in p for p in paragraphs)
    assert "Item 2.03 Creation of a Direct Financial Obligation." in paragraphs


def test_debt_term_filter_needs_debt_words_and_a_money_or_rate_mention():
    paragraphs = narrative_paragraphs(HTML)
    selected = [p for p in paragraphs if is_debt_term_paragraph(p)]
    assert len(selected) == 1 and selected[0].startswith("On March 3, 2026")
    assert not is_debt_term_paragraph("The notes are senior unsecured obligations of the Company and rank equally.")  # no $ or %
    assert not is_debt_term_paragraph("$5")  # too short


def _filing(accession, form="8-K", filing_date="2025-06-02", items=("2.03", "9.01")):
    return Filing(form, accession, filing_date, "", "doc.htm", False, items)


def test_item_203_filings_keep_original_8ks_reporting_item_203_in_window_oldest_first():
    filings = [
        _filing("late", filing_date="2026-10-14"),
        _filing("early", filing_date="2025-01-01"),
        _filing("amendment", form="8-K/A"),
        _filing("other item", items=("1.01", "9.01")),
        _filing("before window", filing_date="2024-12-31"),
        _filing("after window", filing_date="2026-10-15"),
        _filing("ten-k", form="10-K"),
    ]
    selected = item_203_filings(filings, "2025-01-01", "2026-10-14")
    assert [f.accession for f in selected] == ["early", "late"]


def test_novelty_filter_drops_repeats_within_the_set_and_reference_texts_after_normalizing():
    novelty = NoveltyFilter(["The Notes bear interest at 5.25% per annum."])
    assert novelty.check("The notes bear interest at 6.10% per annum.") == "leakage"  # same text once figures are collapsed
    assert novelty.check("The Term Loan matures in 2031 and bears interest at SOFR plus 2.00%.") is None
    assert novelty.check("The term loan matures in 2030 and bears interest at SOFR plus 1.75%.") == "duplicate"
    assert novelty.check("The notes bear interest at 6.10% per annum.") == "duplicate"


def _candidate(n, accession, cik):
    return {"id": f"{accession}-{n}", "accession": accession, "cik": cik}


def test_capped_sample_respects_filing_and_company_caps_and_is_seeded():
    candidates = (
        [_candidate(n, "a1", cik=1) for n in range(6)]  # one filing with many candidates
        + [_candidate(n, f"b{n}", cik=2) for n in range(6)]  # one company with many filings
        + [_candidate(n, f"c{n}", cik=10 + n) for n in range(6)]  # one candidate per company
    )
    sample = capped_sample(candidates, size=10, per_filing=2, per_company=3, seed=0)
    assert len(sample) == 10
    assert max(Counter(c["accession"] for c in sample).values()) <= 2
    assert max(Counter(c["cik"] for c in sample).values()) <= 3
    assert capped_sample(candidates, size=10, per_filing=2, per_company=3, seed=0) == sample
    assert capped_sample(candidates, size=10, per_filing=2, per_company=3, seed=1) != sample


def test_capped_sample_returns_fewer_than_size_when_the_caps_bind():
    candidates = [_candidate(n, "a1", cik=1) for n in range(6)]
    assert len(capped_sample(candidates, size=5, per_filing=3, per_company=3, seed=0)) == 3
