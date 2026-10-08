"""What followed an alert, quoted from the filing (src/live/excerpts.py).

Display only, extractive by design: the dashboard must never put words in a
company's mouth, so every quotation is a substring of the filing's own text.
"""
from __future__ import annotations

import pytest

from src import db
from src.live import excerpts
from src.utils.config import load_config

FILING = b"""<html><body>
<p>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</p>
<p><b>Item 5.02 Departure of Directors or Certain Officers; Election of
Directors; Appointment of Certain Officers; Compensatory Arrangements of
Certain Officers.</b></p>
<p>On</p><p>September 29, 2026, Jane Doe resigned as Chief Executive Officer of
Example Inc. and as a member of its Board. The Board appointed John Roe, Mr.
Roe being a current director, as interim Chief Executive Officer.</p>
<p>The information set forth in Item 7.01 is incorporated herein by
reference.</p>
<p><b>Item 7.01 Regulation FD Disclosure.</b></p>
<p>The information in this Item 7.01 is being furnished and shall not be
deemed filed.</p>
<p><b>Item 9.01 Financial Statements and Exhibits.</b></p>
<p>Exhibit 99.1 Press release.</p>
<p>SIGNATURES</p></body></html>"""


@pytest.fixture
def cfg():
    return load_config()


def test_quotes_the_announcement_not_the_title_or_boilerplate(cfg):
    out = excerpts.excerpt_items(FILING, ["5.02", "7.01", "9.01"], cfg)
    assert set(out) == {"5.02"}                       # 7.01 is all boilerplate
    quote = out["5.02"]
    assert quote.startswith("On September 29, 2026, Jane Doe resigned")
    assert "Mr. Roe being a current director" in quote   # no split at "Mr."
    assert "Departure of Directors" not in quote
    assert "incorporated" not in quote


def test_every_quote_is_the_filing_own_words(cfg):
    text = " ".join(excerpts.to_text(FILING).split())
    for quote in excerpts.excerpt_items(FILING, ["5.02"], cfg).values():
        assert quote.rstrip("…") in text


def test_long_items_are_cut_at_the_limit(cfg):
    long = (b"<p>Item 8.01 Other Events.</p><p>"
            + b"On October 1, 2026, the Company announced a plan. " * 40 + b"</p>")
    small = {**cfg, "live": {**cfg["live"], "excerpts": {
        **cfg["live"]["excerpts"], "max_chars": 120}}}
    quote = excerpts.excerpt_items(long, ["8.01"], small)["8.01"]
    assert len(quote) <= 121


def test_backfill_records_unfetchable_filings_and_never_refetches(cfg, tmp_path):
    conn = db.get_conn(tmp_path / "x.db")
    db.upsert_filings(conn, [{"accession_no": "0000000001-26-000001",
                              "cik": "1", "ticker": "EX", "form": "8-K",
                              "items": "5.02,9.01", "acceptance_utc": 1,
                              "primary_doc": "ex.htm"}])

    class Client:
        calls = 0

        def get_bytes(self, url):
            Client.calls += 1
            return FILING

    out = excerpts.backfill(cfg, conn, client=Client(),
                            extra=["0000000001-26-000001"])
    assert out == {"fetched": 1, "quoted": 1, "empty": 0, "failed": 0}
    again = excerpts.backfill(cfg, conn, client=Client(),
                              extra=["0000000001-26-000001"])
    assert again["fetched"] == 0 and Client.calls == 1

    path = tmp_path / "x.csv"
    assert excerpts.export_csv(conn, path) == 1
    fresh = db.get_conn(tmp_path / "y.db")
    assert excerpts.export_csv(fresh, path) == 1        # never loses a row
