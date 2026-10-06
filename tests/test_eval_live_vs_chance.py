"""Live alerts against chance — the controls must measure what they claim.

The test that matters most: an "alert" that only picks filing-prone stocks, at
random hours, must score ~1x against the same-stock-random-hour control. That
control is the one the timing claim rests on.
"""

import numpy as np
import pandas as pd
import pytest

from src.eval.live_vs_chance import (FilingIndex, filing_kind, grade,
                                     stock_only_order, summarise)
from src.utils.config import load_config

HOUR = 3600
SPAN = 48 * HOUR
EXCL, SCHED = {"9.01", "5.07"}, {"2.02"}


def test_filing_kind_drops_excluded_items_first():
    assert filing_kind("2.02,9.01", EXCL, SCHED) == "scheduled"
    assert filing_kind("1.01, 9.01", EXCL, SCHED) == "unscheduled"
    assert filing_kind("9.01,5.07", EXCL, SCHED) == "routine"
    assert filing_kind(None, EXCL, SCHED) == "routine"
    # Earnings plus a contract is still a filing whose date was known.
    assert filing_kind("2.02,1.01", EXCL, SCHED) == "scheduled"


def index_of(rows) -> FilingIndex:
    return FilingIndex(pd.DataFrame(rows, columns=["ticker", "t0", "kind"]), SPAN)


def test_followed_is_strictly_after_and_inside_the_window():
    idx = index_of([("A", 1000, "unscheduled")])
    assert not idx.followed("any", "A", 1000)          # same instant: not predicted
    assert idx.followed("any", "A", 999)
    assert idx.followed("any", "A", 1000 - SPAN)        # window end is inclusive
    assert not idx.followed("any", "A", 999 - SPAN)
    assert not idx.followed("any", "B", 999)


def test_a_kind_slice_asks_whether_that_kind_followed():
    """A routine filing first must not hide the earnings filing after it."""
    idx = index_of([("A", 1000, "routine"), ("A", 2000, "scheduled")])
    assert idx.followed("scheduled", "A", 500)
    assert not idx.followed("unscheduled", "A", 500)
    assert idx.followed("any", "A", 500)


def make_world(rng, n_stocks=40, n_hours=200, timing=False):
    """Stocks S0.. file at rates that fall with their index. Alerts pick the
    frequent filers; with `timing`, they also pick the hour just before a filing."""
    hours = np.arange(n_hours) * 24 * HOUR
    filings, stocks = [], [f"S{i}" for i in range(n_stocks)]
    for i, s in enumerate(stocks):
        rate = 0.3 / (1 + i)
        for h in hours[rng.random(n_hours) < rate]:
            filings.append((s, int(h + 10 * HOUR), "unscheduled"))
    fil = pd.DataFrame(filings, columns=["ticker", "t0", "kind"])
    alerts = []
    for d, h in enumerate(hours):
        for s in stocks[:5]:
            if timing:
                due = ((fil.ticker == s) & (fil.t0 == h + 10 * HOUR)).any()
                if not due and rng.random() > 0.2:
                    continue
            elif rng.random() > 0.5:
                continue
            alerts.append(("det", s, int(h)))
    return (fil, pd.DataFrame(alerts, columns=["detector", "ticker", "ts_utc"]),
            stocks)


def run(fil, alerts, stocks, rng):
    idx = FilingIndex(fil, SPAN)
    graded = grade(alerts, idx, stocks, stocks, random_hours=30, rng=rng)
    return summarise(graded, n_boot=300, seed=0).set_index(["detector", "followed_by"])


def test_stock_selection_alone_scores_about_one_against_the_timing_control():
    rng = np.random.default_rng(1)
    t = run(*make_world(rng), rng).loc[("det", "any")]
    assert t["A_ratio"] > 1.5                     # it does beat a random stock...
    assert 0.8 < t["B_ratio"] < 1.25              # ...but not the same stock later
    assert t["B_ratio_lo"] < 1 < t["B_ratio_hi"]


def test_real_timing_beats_the_timing_control():
    rng = np.random.default_rng(2)
    t = run(*make_world(rng, timing=True), rng).loc[("det", "any")]
    assert t["B_ratio_lo"] > 1.5


def test_summary_carries_every_kind_and_a_pooled_row():
    rng = np.random.default_rng(3)
    table = run(*make_world(rng), rng)
    assert set(table.index.get_level_values(1)) == {"any", "scheduled", "unscheduled"}
    assert "ALL" in table.index.get_level_values(0)


def test_a_zero_control_gives_nan_not_infinity():
    rng = np.random.default_rng(4)
    table = run(*make_world(rng), rng)
    row = table.loc[("det", "scheduled")]          # no scheduled filings at all
    assert np.isnan(row["B_ratio"]) and np.isnan(row["B_ratio_lo"])


def test_stock_only_order_ranks_by_train_count_with_seeded_ties():
    pos = pd.DataFrame({"ticker": ["B", "B", "B", "C"], "window_id": ["1", "2", "3", "4"]})
    order = stock_only_order(pos, ["A", "B", "C", "D"], np.random.default_rng(0))
    assert order[:2] == ["B", "C"] and set(order[2:]) == {"A", "D"}


def test_config_carries_the_knobs():
    knobs = load_config()["eval"]["live_vs_chance"]
    assert {"random_hours", "bootstrap", "seed", "ci"} <= set(knobs)
