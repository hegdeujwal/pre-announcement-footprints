"""The clock-only null, the same-hour view, and the session-close helper."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.baselines.clock_only import ClockOnly
from src.baselines.compare import comparison_table
from src.eval import same_hour as SH
from src.utils.config import load_config
from src.utils.timeutils import iso_utc_to_ts, seconds_to_session_close

HOUR = 3600


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


def session_bars(day: str, open_utc: str = "13:30", n: int = 7) -> list[int]:
    """Bar starts of one session: hourly from the open (EDT open = 13:30 UTC)."""
    start = iso_utc_to_ts(f"{day}T{open_utc}:00Z")
    return [start + i * HOUR for i in range(n)]


# -- seconds_to_session_close --------------------------------------------

def test_full_day_closing_bar_is_a_half_bar_from_the_close():
    bars = session_bars("2025-10-01")
    out = seconds_to_session_close(bars)
    assert out[-1] == 1800 and out[-2] == 5400 and out[0] == 6.5 * HOUR


def test_half_day_close_comes_from_the_calendar():
    # 2025-11-28, the day after Thanksgiving: 09:30-13:00 ET = 14:30-18:00 UTC.
    bars = session_bars("2025-11-28", open_utc="14:30", n=4)
    assert seconds_to_session_close(bars)[-1] == 1800


def test_a_timestamp_outside_every_session_is_refused():
    with pytest.raises(ValueError, match="outside every"):
        seconds_to_session_close([iso_utc_to_ts("2025-10-04T15:00:00Z")])  # Saturday


# -- frames ----------------------------------------------------------------

def frame(days=("2025-10-01", "2025-10-02", "2025-10-03"), positives_end_at_close=True):
    """Quiet single-bar windows on every bar, plus one 7-bar positive per day
    whose final bar is the closing bar (or the 14:30 bar)."""
    rows = []
    for d, day in enumerate(days):
        bars = session_bars(day)
        for ts in bars:
            rows.append({"window_id": f"bar:A:{ts}", "ticker": "A", "ts_utc": ts,
                         "t0_utc": None, "is_scheduled": None, "item_code": None})
        end = len(bars) if positives_end_at_close else len(bars) - 1
        t0 = bars[end - 1] + HOUR
        for ts in bars[:end]:
            rows.append({"window_id": f"E{d}", "ticker": "B", "ts_utc": ts,
                         "t0_utc": t0, "is_scheduled": False, "item_code": "8.01"})
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


# -- clock_only ------------------------------------------------------------

def test_clock_only_ranks_the_closing_bar_highest(cfg):
    f = frame()
    s = ClockOnly(cfg).score(f)
    to_close = seconds_to_session_close(f["ts_utc"].astype("int64"))
    order = np.argsort(-s.to_numpy(), kind="stable")
    assert (np.diff(to_close[order]) >= 0).all()


def test_clock_only_never_reads_the_label_columns(cfg):
    f = frame()
    scrubbed = f.assign(t0_utc=pd.NA, is_scheduled=pd.NA, item_code=pd.NA)
    pd.testing.assert_series_equal(ClockOnly(cfg).score(f), ClockOnly(cfg).score(scrubbed))


def test_clock_only_jitter_is_seeded_and_cannot_reorder_bars(cfg):
    f = frame()
    a, b = ClockOnly(cfg).score(f), ClockOnly(cfg).score(f)
    pd.testing.assert_series_equal(a, b)
    assert (a - a.round()).abs().max() < 1e-2


# -- same-hour view ----------------------------------------------------------

def test_view_keeps_exactly_the_windows_scored_on_the_closing_bar(cfg):
    f = frame()
    kept = set(SH.closing_bar_windows(f, cfg))
    closing = {f"bar:A:{session_bars(d)[-1]}" for d in ("2025-10-01", "2025-10-02", "2025-10-03")}
    assert kept == closing | {"E0", "E1", "E2"}


def test_a_positive_is_judged_by_its_final_bar_not_any_bar(cfg):
    # Each positive here ends on the 14:30 bar; its earlier bars never decide.
    f = frame(positives_end_at_close=False)
    kept = set(SH.closing_bar_windows(f, cfg))
    assert not ({"E0", "E1", "E2"} & kept)


def test_view_refuses_a_population_with_one_class(cfg):
    with pytest.raises(ValueError, match="needs both"):
        SH.same_hour_view(frame(positives_end_at_close=False), cfg)


def test_inside_the_view_the_clock_carries_no_information(cfg):
    view = SH.scored_bars(SH.same_hour_view(frame(), cfg))
    s = ClockOnly(cfg).score(view)
    assert s.round().nunique() == 1       # only jitter separates the windows


def test_an_off_session_bar_is_unreadable_not_an_error(cfg):
    f = frame().iloc[:1].copy()
    f["ts_utc"] = pd.array([iso_utc_to_ts("2025-10-04T15:00:00Z")], dtype="Int64")
    assert ClockOnly(cfg).score(f).isna().all()
    assert len(SH.closing_bar_windows(f, cfg)) == 0


def test_comparison_table_labels_its_view(cfg):
    f = frame()
    model = ClockOnly(cfg)
    pred = model.predict(f, threshold=float("inf"))
    t = comparison_table(cfg, {model.name: SH.same_hour_view(pred, cfg)},
                         "news_adjusted", view="closing_bar")
    assert (t["view"] == "closing_bar").all()
    t2 = comparison_table(cfg, {model.name: pred}, "news_adjusted")
    assert (t2["view"] == "all_hours").all()
