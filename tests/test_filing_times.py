"""Filing times from headers, and one grader for every live hit rate.

Found 2026-10-07: the submissions JSON's `acceptanceDateTime` drifted by the
Eastern offset, so the live run stored filing times up to five hours wrong,
and the dashboard and `live_vs_chance` — each grading against the filings it
happened to hold — disagreed on 412 of 14,576 alerts.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import db
from src.collectors.edgar import EdgarRequestError, parse_header_acceptance
from src.eval.live_vs_chance import KINDS, use_committed_hits
from src.live.alertlog import append
from src.live.monitor import Alert
from src.live.outcomes import (backfill, filing_kind, kinds_followed, regrade)
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts

HOUR = 3600
ET = "America/New_York"


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------
# the header
# --------------------------------------------------------------------------
def test_header_time_is_eastern_in_summer_and_in_winter():
    """Checked against SEC's index, which carries an explicit offset."""
    summer = parse_header_acceptance(b"<ACCEPTANCE-DATETIME>20261001161515", ET)
    winter = parse_header_acceptance(b"<ACCEPTANCE-DATETIME>20260102163052", ET)
    assert summer == iso_utc_to_ts("2026-10-01T20:15:15Z")      # NKE, EDT
    assert winter == iso_utc_to_ts("2026-01-02T21:30:52Z")      # AAPL, EST


def test_the_escaped_form_in_the_html_page_also_parses():
    body = b"&lt;ACCEPTANCE-DATETIME&gt;20261001073111"
    assert parse_header_acceptance(body, ET) == iso_utc_to_ts("2026-10-01T11:31:11Z")


def test_a_page_without_the_field_is_an_error_not_a_time():
    with pytest.raises(EdgarRequestError):
        parse_header_acceptance(b"<html>Request Rate Threshold Exceeded</html>", ET)


# --------------------------------------------------------------------------
# one grader
# --------------------------------------------------------------------------
def _setup(tmp_path, filings):
    conn = db.get_conn(tmp_path / "g.db")
    db.upsert_filings(conn, [
        {"accession_no": a, "cik": "1", "ticker": "AAA", "form": "8-K",
         "items": items, "acceptance_utc": ts} for a, items, ts in filings])
    return conn


def test_kind_is_any_filing_of_that_kind_in_the_window_not_the_first(cfg, tmp_path):
    t = date_str_to_ts("2026-09-21") + 14 * HOUR
    conn = _setup(tmp_path, [("R", "5.07", t + 1 * HOUR),          # routine first
                             ("U", "5.02,9.01", t + 20 * HOUR)])   # then a departure
    assert kinds_followed(cfg, conn, "AAA", t, t + 48 * HOUR) == \
        {"filed_scheduled": 0, "filed_unscheduled": 1}
    assert filing_kind("5.07", set(cfg["items"]["exclude"]),
                       set(cfg["items"]["scheduled"])) == "routine"


def test_regrade_follows_a_filing_whose_time_was_corrected(cfg, tmp_path):
    """Stored 4 hours early, the filing sat BEFORE the alert: a miss. At its
    true time it follows the alert, and the regrade says so."""
    alert_ts = date_str_to_ts("2026-10-01") + 18 * HOUR + 1800      # 18:30Z
    true_ts = date_str_to_ts("2026-10-01") + 20 * HOUR + 900        # 20:15Z
    conn = _setup(tmp_path, [("N", "2.02,9.01", true_ts - 4 * HOUR)])
    append(conn, [Alert(alert_ts, "AAA", "cusum", 3.0, 1.5)], raised_utc=alert_ts)
    backfill(cfg, conn, horizon=true_ts + 100 * HOUR)
    assert conn.execute("SELECT filed FROM alert_outcomes").fetchone()[0] == 0

    conn.execute("UPDATE filings SET acceptance_utc = ? WHERE accession_no='N'",
                 (true_ts,))
    out = regrade(cfg, conn)
    row = conn.execute("SELECT filed, filed_scheduled, filed_unscheduled "
                       "FROM alert_outcomes").fetchone()
    assert tuple(row) == (1, 1, 0) and out == {"regraded": 1, "filed_changed": 1}


def test_the_chance_table_reads_the_committed_grades():
    graded = pd.DataFrame({"alert_id": ["a", "b", "c"],
                           **{f"hit_{k}": [False] * 3 for k in KINDS}})
    committed = pd.DataFrame({"alert_id": ["a", "b", "c"], "filed": [1, 0, 1],
                              "filed_scheduled": [1, 0, None],
                              "filed_unscheduled": [0, 0, None]})
    out = use_committed_hits(graded, committed)
    assert list(out["alert_id"]) == ["a", "b"]          # c has no kind grade
    assert out["hit_any"].tolist() == [True, False]
    assert out["hit_scheduled"].tolist() == [True, False]
