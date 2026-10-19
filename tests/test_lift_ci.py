"""Paired lift intervals: stocks resampled, both detectors on the same draw."""

import numpy as np
import pandas as pd
import pytest

from src.eval import lift_ci as L


def windows(detector, hits_by_ticker, quiet_alerts=2, sched=True):
    """Each ticker: one alerted positive per hit, `quiet_alerts` alerted quiet
    windows, and one un-alerted positive (a miss)."""
    rows = []
    for t, hits in hits_by_ticker.items():
        for i in range(hits):
            rows.append((detector, f"P{t}{i}", t, True, sched, 1.0, True))
        rows.append((detector, f"M{t}", t, True, sched, 0.0, False))
        for i in range(quiet_alerts):
            rows.append((detector, f"Q{t}{i}", t, False, None, 1.0, True))
    return pd.DataFrame(rows, columns=L.WINDOW_COLUMNS).astype(
        {"is_scheduled": "boolean"})


TICKERS = [f"T{i}" for i in range(30)]


def test_identical_detectors_have_a_ratio_of_exactly_one():
    w = pd.concat([windows("a", {t: 1 for t in TICKERS}),
                   windows("b", {t: 1 for t in TICKERS})])
    r = L.paired_intervals(w, ["b"], n_boot=200, seed=0)
    row = r[(r.slice == "scheduled") & (r.detector == "a")].iloc[0]
    assert row.lift_ratio == pytest.approx(1.0)
    assert row.ci_lo == pytest.approx(1.0) and row.ci_hi == pytest.approx(1.0)
    assert row.p_better == 0.0


def test_a_detector_better_on_every_stock_is_better_in_every_draw():
    w = pd.concat([windows("a", {t: 2 for t in TICKERS}),
                   windows("b", {t: 1 for t in TICKERS})])
    row = L.paired_intervals(w, ["b"], n_boot=200, seed=0).query(
        "slice == 'scheduled'").iloc[0]
    # a: 2 hits of 4 alerts per stock, b: 1 of 3.
    assert row.lift_ratio == pytest.approx((2 / 4) / (1 / 3))
    assert row.p_better == 1.0 and row.ci_lo > 1


def test_a_gain_carried_by_one_stock_gets_a_wide_interval():
    lucky = {t: 0 for t in TICKERS} | {"T0": 20}
    w = pd.concat([windows("a", lucky), windows("b", {t: 1 for t in TICKERS})])
    row = L.paired_intervals(w, ["b"], n_boot=500, seed=0).query(
        "slice == 'scheduled'").iloc[0]
    assert row.ci_lo < 1 < row.ci_hi


def test_seeds_pool_into_one_group():
    w = pd.concat([windows("m[s1]", {t: 2 for t in TICKERS}),
                   windows("m[s2]", {t: 0 for t in TICKERS}),
                   windows("b", {t: 1 for t in TICKERS})])
    r = L.paired_intervals(w, ["b"], n_boot=100, seed=0)
    row = r[r.slice == "scheduled"].iloc[0]
    assert row.detector == "m" and row.n_seeds == 2
    assert row.precision == pytest.approx(np.mean([2 / 4, 0 / 2]))


def test_a_slice_counts_quiet_alerts_but_not_the_other_slices_hits():
    sched = windows("a", {t: 1 for t in TICKERS}, sched=True)
    unsched = windows("a", {t: 1 for t in TICKERS}, quiet_alerts=0, sched=False)
    unsched["window_id"] = "U" + unsched["window_id"]
    w = pd.concat([sched, unsched, windows("b", {t: 1 for t in TICKERS})])
    c = L.stock_counts(w, "scheduled", pd.Index(TICKERS))
    # 1 scheduled hit + 2 quiet alerts; the unscheduled hit is not spent here.
    assert c.loc[("a", "T0")].tolist() == [3, 1]


def test_alerted_windows_uses_the_budget_threshold():
    # 1 ticker-month -> a budget of 2 alerts: the two highest windows.
    ts = 1_760_000_000
    pred = pd.DataFrame({
        "window_id": ["P", "Q1", "Q2", "Q3"], "ticker": "A",
        "ts_utc": [ts, ts + 3600, ts + 7200, ts + 10800],
        "score": [3.0, 2.0, 1.0, 0.0], "t0_utc": [ts + 3600, None, None, None],
        "is_scheduled": [True, None, None, None], "item_code": ["2.02", None, None, None],
        "action": "WAIT"})
    w = L.alerted_windows("a", pred)
    assert w.set_index("window_id")["alerted"].to_dict() == {
        "P": True, "Q1": True, "Q2": False, "Q3": False}


def test_an_unknown_reference_is_refused():
    with pytest.raises(ValueError, match="reference"):
        L.paired_intervals(windows("a", {"T0": 1}), ["nope"], n_boot=10, seed=0)
