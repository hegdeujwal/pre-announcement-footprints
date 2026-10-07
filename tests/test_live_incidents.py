"""Two ways the live log was over-counted, found 2026-10-07.

Repeats: the monitor logs every bar a detector fires on, so one elevated stock
writes an alert per hour, while the alert budget and the offline evaluation
both allow one flag per `decision.horizon_hours`-bar episode. 71% of the live
log was repeats.

Incidents: 2026-10-06 was scored on hourly bars Yahoo later corrected. Those
alerts stay in the append-only log, and every rate leaves them out through one
helper, so the exclusion cannot differ between the hit rate, the chance
comparison and the dashboard.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("streamlit", reason="dashboard extras not installed")

from app import data  # noqa: E402
from src import db  # noqa: E402
from src.live.alertlog import append  # noqa: E402
from src.live.incidents import in_incident, sql_exclusion  # noqa: E402
from src.live.monitor import Alert  # noqa: E402
from src.live.outcomes import hit_rates  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.timeutils import date_str_to_ts  # noqa: E402

HOUR = 3600


def _bar(day: str, hour_utc: float) -> int:
    return date_str_to_ts(day) + int(hour_utc * HOUR)


@pytest.fixture
def cfg():
    return load_config()


# --------------------------------------------------------------------------
# episodes
# --------------------------------------------------------------------------
def test_a_stock_elevated_all_session_is_one_episode_not_seven():
    stamps = [_bar("2026-10-05", 13.5 + i) for i in range(7)]
    df = pd.DataFrame({"detector": "cusum", "ticker": "AAA", "ts_utc": stamps})
    assert data.episode_starts(df).tolist() == [True] + [False] * 6


def test_the_episode_re_arms_after_the_horizon_counted_in_bars(cfg):
    """48 bars is ~7 sessions, not 48 hours and not two calendar days."""
    horizon = cfg["decision"]["horizon_hours"]
    first = _bar("2026-09-21", 13.5)                       # Monday open
    # Bar 47 after it is still inside the episode; bar 48 opens the next one.
    # 7 bars per session: 47 = 6 sessions + 5 bars, 48 = 6 sessions + 6 bars.
    inside = _bar("2026-09-29", 13.5 + 5)
    after = _bar("2026-09-29", 13.5 + 6)
    df = pd.DataFrame({"detector": "cusum", "ticker": "AAA",
                       "ts_utc": [first, inside, after]})
    assert horizon == 48
    assert data.episode_starts(df).tolist() == [True, False, True]


def test_episodes_are_per_detector_and_per_stock():
    t = _bar("2026-10-05", 14.5)
    df = pd.DataFrame({"detector": ["cusum", "volume_zscore", "cusum"],
                       "ticker": ["AAA", "AAA", "BBB"], "ts_utc": [t, t, t]})
    assert data.episode_starts(df).all()


# --------------------------------------------------------------------------
# incidents
# --------------------------------------------------------------------------
def test_the_configured_incident_covers_its_whole_day_and_nothing_else(cfg):
    stamps = [_bar("2026-10-05", 19.5), _bar("2026-10-06", 13.5),
              _bar("2026-10-06", 19.5), _bar("2026-10-07", 13.5)]
    early = [s - 1 for s in stamps]                  # raised by no listed run
    assert in_incident(cfg, stamps, early).tolist() == [False, True, True, False]


def test_no_incidents_means_no_exclusion(cfg):
    clean = {**cfg, "live": {**cfg["live"], "data_incidents": []}}
    assert sql_exclusion(clean) == ("", [])
    assert not in_incident(clean, [_bar("2026-10-06", 13.5)]).any()


def test_a_run_can_be_excluded_by_its_raise_time_whatever_its_bars(cfg):
    """The 2026-10-07 repair run raised alerts on bars weeks old."""
    from src.utils.timeutils import iso_utc_to_ts
    run = iso_utc_to_ts("2026-10-07T19:36:04Z")
    bars = [_bar("2026-08-24", 14.5), _bar("2026-10-07", 14.5)]
    assert in_incident(cfg, bars, [run, run]).all()
    assert not in_incident(cfg, bars, [run + 86400, run - 86400]).any()
    with pytest.raises(ValueError, match="raise time"):
        in_incident(cfg, bars)


def test_hit_rates_leave_incident_alerts_out(cfg, tmp_path):
    conn = db.get_conn(tmp_path / "h.db")
    good, bad = _bar("2026-10-05", 14.5), _bar("2026-10-06", 14.5)
    append(conn, [Alert(good, "AAA", "cusum", 3.0, 1.5),
                  Alert(bad, "BBB", "cusum", 30.0, 1.5)], raised_utc=bad + 12 * HOUR)
    rows = conn.execute("SELECT alert_id FROM alerts").fetchall()
    conn.executemany("INSERT INTO alert_outcomes (alert_id, checked_utc, filed) "
                     "VALUES (?, ?, 0)", [(r[0], bad + 99 * HOUR) for r in rows])
    conn.commit()

    assert hit_rates(conn, cfg)["cusum"]["scored"] == 1
    clean = {**cfg, "live": {**cfg["live"], "data_incidents": []}}
    assert hit_rates(conn, clean)["cusum"]["scored"] == 2


def test_dashboard_rates_and_budget_leave_incidents_and_repeats_out():
    day = [_bar("2026-10-06", 13.5 + i) for i in range(3)]
    ok = [_bar("2026-10-05", 13.5 + i) for i in range(3)]
    df = pd.DataFrame({"detector": "cusum", "ticker": "AAA",
                       "ts_utc": ok + day, "filed": [1, 0, 0, 1, 1, 1],
                       "item_code": "8.01"})
    df["episode_start"] = data.episode_starts(df)
    df["raised_utc"] = df["ts_utc"] + 12 * HOUR
    df["incident"] = in_incident(load_config(), df["ts_utc"], df["raised_utc"])

    resolved, filed, _ = data.hit_rate(df)
    assert (resolved, filed) == (3, 1)
    b = data.budget_line(df)
    assert b["used"] == 1 and b["repeats"] == 2
