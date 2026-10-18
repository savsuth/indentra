import pytest

from indentra import edgar

FORM_INDEX = """Description:           Daily Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    March 31, 2025

Form Type   Company Name                                                  CIK         Date Filed  File Name
---------------------------------------------------------------------------------------------------------------------------------------------
10-K        ACME HOLDINGS INC                                             1234567     2025-02-14  edgar/data/1234567/0001234567-25-000010.txt
10-K/A      ACME HOLDINGS INC                                             1234567     2025-03-01  edgar/data/1234567/0001234567-25-000020.txt
10-K        BETA & SONS, L.P.                                             89          2025-03-28  edgar/data/89/0000950170-25-012345.txt
SC 13G      SOMEONE                                                       555         2025-01-02  edgar/data/555/0000555-25-000001.txt
"""


def test_parse_form_index_keeps_exact_form_only():
    entries = edgar.parse_form_index(FORM_INDEX, "10-K")
    assert [(e.cik, e.company, e.date_filed, e.accession) for e in entries] == [
        (1234567, "ACME HOLDINGS INC", "2025-02-14", "0001234567-25-000010"),
        (89, "BETA & SONS, L.P.", "2025-03-28", "0000950170-25-012345"),
    ]


def test_urls_use_sec_layout():
    assert edgar.submissions_url(89) == "https://data.sec.gov/submissions/CIK0000000089.json"
    assert (
        edgar.document_url(89, "0000950170-25-012345", "beta-20241231.htm")
        == "https://www.sec.gov/Archives/edgar/data/89/000095017025012345/beta-20241231.htm"
    )


def test_parse_filings_reads_column_arrays():
    columns = {
        "form": ["8-K", "10-K", "4"],
        "accessionNumber": ["a-1", "a-2", "a-3"],
        "filingDate": ["2025-03-01", "2025-02-14", "2025-01-05"],
        "reportDate": ["2025-02-27", "2024-12-31", ""],
        "primaryDocument": ["ev.htm", "k.htm", "x.xml"],
        "isInlineXBRL": [1, 1, 0],
        "items": ["2.03,9.01", "", ""],
    }
    filings = edgar.parse_filings(columns)
    assert filings[0] == edgar.Filing("8-K", "a-1", "2025-03-01", "2025-02-27", "ev.htm", True, ("2.03", "9.01"))
    assert filings[1].form == "10-K" and filings[1].items == ()
    assert filings[2].inline_xbrl is False


def test_parse_filings_tolerates_missing_optional_columns():
    columns = {"form": ["10-K"], "accessionNumber": ["a"], "filingDate": ["2022-02-01"], "primaryDocument": ["k.htm"]}
    assert edgar.parse_filings(columns) == [edgar.Filing("10-K", "a", "2022-02-01", "", "k.htm", False, ())]


@pytest.mark.parametrize("user_agent", ["", "indentra", "   "])
def test_client_requires_contact_email(user_agent):
    with pytest.raises(ValueError, match="contact email"):
        edgar.EdgarClient(user_agent)
