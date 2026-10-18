"""Polite SEC EDGAR client and helpers for EDGAR's index and URL layout.

SEC fair-access rules: declare a User-Agent with a contact email, stay under 10 requests/second.
https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

MAX_REQUESTS_PER_SECOND = 5
MAX_ATTEMPTS = 4
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class IndexEntry:
    form: str
    company: str
    cik: int
    date_filed: str
    accession: str


@dataclass(frozen=True)
class Filing:
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str
    inline_xbrl: bool
    items: tuple[str, ...]


class EdgarClient:
    def __init__(self, user_agent: str, cache_dir: Path | None = None) -> None:
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with a contact email, e.g. 'Jane Doe jane@example.com'")
        self._user_agent = user_agent.strip()
        self._cache_dir = cache_dir
        self._last_request = 0.0

    @classmethod
    def from_env(cls, cache_dir: Path | None = None) -> EdgarClient:
        return cls(os.environ.get("SEC_USER_AGENT", ""), cache_dir)

    def get(self, url: str, cache: bool = True) -> bytes:
        cached = self._cache_path(url) if cache else None
        if cached is not None and cached.exists():
            return cached.read_bytes()
        data = self._fetch(url)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(data)
        return data

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url))

    def filings(self, cik: int, since: str) -> tuple[dict, list[Filing]]:
        """Company metadata and its filings dated on or after `since`, following EDGAR's pagination."""
        submissions = self.get_json(submissions_url(cik))
        filings = parse_filings(submissions["filings"]["recent"])
        for page in submissions["filings"].get("files", []):
            if page.get("filingTo", "") >= since:
                filings += parse_filings(self.get_json(f"https://data.sec.gov/submissions/{page['name']}"))
        meta = {k: submissions.get(k) for k in ("name", "sic", "sicDescription", "entityType", "fiscalYearEnd")}
        return meta, [f for f in filings if f.filing_date >= since]

    def _fetch(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": self._user_agent})
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self._throttle()
            try:
                with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                if error.code not in RETRY_STATUSES or attempt == MAX_ATTEMPTS:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == MAX_ATTEMPTS:
                    raise
            time.sleep(2**attempt)
        raise AssertionError("unreachable")

    def _throttle(self) -> None:
        wait = self._last_request + 1 / MAX_REQUESTS_PER_SECOND - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(url.encode()).hexdigest()[:24]
        return self._cache_dir / f"{digest}{Path(urlparse(url).path).suffix}"


def parse_form_index(text: str, form: str) -> list[IndexEntry]:
    """Entries of exactly `form` from an EDGAR full-index form.idx file."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("---")) + 1
    entries = []
    for line in lines[start:]:
        fields = line.rsplit(None, 3)
        if len(fields) != 4:
            continue
        left, cik, date_filed, file_name = fields
        head = left.split(None, 1)
        if len(head) != 2 or head[0] != form:
            continue
        accession = Path(file_name).stem
        entries.append(IndexEntry(form, head[1].strip(), int(cik), date_filed, accession))
    return entries


def parse_filings(columns: dict[str, list]) -> list[Filing]:
    """Rows of a submissions JSON block, which EDGAR stores as parallel column arrays."""
    n = len(columns["form"])
    report_dates = columns.get("reportDate", [""] * n)
    inline = columns.get("isInlineXBRL", [0] * n)
    items = columns.get("items", [""] * n)
    return [
        Filing(
            form=columns["form"][i],
            accession=columns["accessionNumber"][i],
            filing_date=columns["filingDate"][i],
            report_date=report_dates[i] or "",
            primary_document=columns["primaryDocument"][i],
            inline_xbrl=bool(inline[i]),
            items=tuple(item for item in (items[i] or "").split(",") if item),
        )
        for i in range(n)
    ]


def form_index_url(year: int, quarter: int) -> str:
    return f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/form.idx"


def submissions_url(cik: int) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik:010d}.json"


def document_url(cik: int, accession: str, primary_document: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{primary_document}"
