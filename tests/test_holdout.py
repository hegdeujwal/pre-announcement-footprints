"""The post-study holdout: the study's rules, applied without touching `events`."""

import copy

import pandas as pd
import pytest

from src import db
from src.pipeline import holdout
from src.pipeline.evalset import _positive_windows
from src.utils.config import load_config
from src.utils.timeutils import get_market_calendar, iso_utc_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


@pytest.fixture
def conn(tmp_path):
    return db.get_conn(tmp_path / "holdout.db")


def hourly_bars(cfg, conn, ticker, start, end, jump_at=None, jump=0.0):
    """Hourly bars on every real session in [start, end]; price 100, then
    100*(1+jump) from `jump_at` on."""
    cal = get_market_calendar(cfg["market"]["calendar"])
    rows = []
    for session in cal.sessions_in_range(start, end):
        o = int(cal.session_open(session).timestamp())
        c = int(cal.session_close(session).timestamp())
        ts = o
        while ts < c:
            p = 100.0 * (1 + jump) if jump_at and ts >= jump_at else 100.0
            rows.append((ticker, ts, p, p, p, p, 1e6, cfg["market"]["interval"]))
            ts += HOUR
    db.upsert_bars(conn, rows)


def seed(cfg, conn):
    bench = cfg["market"]["benchmark"]
    db.upsert_companies(conn, [
        {"cik": f"C{t}", "ticker": t, "in_universe": int(t != "OUT")}
        for t in ("EARN", "EXH", "FLAT", "NEWS", "OUT")])
    acc = {"EARN": "2026-08-12T20:30:00Z", "EXH": "2026-08-13T20:30:00Z",
           "FLAT": "2026-08-14T20:30:00Z", "NEWS": "2026-08-18T20:30:00Z",
           "OUT": "2026-08-19T20:30:00Z"}
    items = {"EARN": "2.02,9.01", "EXH": "9.01", "FLAT": "5.02",
             "NEWS": "8.01", "OUT": "8.01"}
    db.upsert_filings(conn, [{
        "accession_no": f"0000000000-26-{i:06d}", "cik": f"C{t}", "ticker": t,
        "form": "8-K", "items": items[t], "acceptance_utc": iso_utc_to_ts(a),
        "filing_date_utc": iso_utc_to_ts(a)} for i, (t, a) in enumerate(acc.items())])
    # An article an hour before NEWS's filing: its t0 must come from the news.
    db.upsert_news(conn, [{
        "url": "http://x/news", "ticker": "NEWS", "title": "t",
        "source_name": "Benzinga", "source_domain": "x.com", "source_tier": 2,
        "published_utc": iso_utc_to_ts(acc["NEWS"]) - HOUR, "api": "finnhub"}])
    hourly_bars(cfg, conn, bench, "2026-06-01", "2026-09-30")
    # The reaction comes AFTER the filing (the next session), so materiality
    # measures a 20% move from the last bar at or before t0.
    for t in ("EARN", "EXH", "NEWS", "OUT"):
        hourly_bars(cfg, conn, t, "2026-06-01", "2026-09-30",
                    jump_at=iso_utc_to_ts(acc[t]) + HOUR, jump=0.2)
    hourly_bars(cfg, conn, "FLAT", "2026-06-01", "2026-09-30")
    return acc


def test_holdout_window_must_not_overlap_the_study(cfg):
    cfg["holdout"]["start"] = "2026-07-15"
    with pytest.raises(ValueError, match="inside the study window"):
        holdout.holdout_window(cfg)


def test_build_applies_the_study_rules(cfg, conn):
    seed(cfg, conn)
    f = holdout.build(cfg, conn).set_index("ticker")
    assert "OUT" not in f.index                                # not in universe
    assert f.loc["EARN", "usable"] == 1 and f.loc["EARN", "is_scheduled"] == 1
    assert f.loc["EXH", "exclude_reason"] == "only_excluded_items"
    assert f.loc["FLAT", "exclude_reason"] == "immaterial"
    assert f.loc["NEWS", "t0_source"] == "news"
    assert f.loc["NEWS", "t0_utc"] < f.loc["NEWS", "t0_filing_utc"]
    assert f.loc["NEWS", "is_scheduled"] == 0


def test_build_never_writes_the_study_events_table(cfg, conn):
    seed(cfg, conn)
    holdout.build(cfg, conn)
    assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_build_refuses_an_empty_window(cfg, conn):
    seed(cfg, conn)
    cfg["holdout"].update(start="2026-09-20", end="2026-09-25")
    with pytest.raises(SystemExit, match="no in-universe filings"):
        holdout.build(cfg, conn)


def test_eval_positives_can_come_from_a_frame(cfg):
    events = pd.DataFrame({
        "event_id": ["a", "b", "c"], "ticker": ["X", "X", "Y"],
        "items": ["8.01", "5.02", "8.01"], "usable": [1, 0, 1],
        "t0_utc": [100, 200, 300], "t0_filing_utc": [110, 210, 310],
        "is_scheduled": [0, 0, 0]})
    got = _positive_windows(None, cfg, "X", 0, 1000, events=events)
    assert [g["event_id"] for g in got] == ["a"]
    got = _positive_windows(None, cfg, "X", 0, 1000, "t0_filing_utc", events=events)
    assert got[0]["t0_utc"] == 110
