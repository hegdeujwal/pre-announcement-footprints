"""Within-stock AUC — timing credit only, never stock-selection credit."""

import numpy as np
import pandas as pd
import pytest

from src.eval.within_stock import stock_pairs, within_stock_auc
from src.utils.config import load_config


def windows(rows) -> pd.DataFrame:
    """rows: (ticker, peak_score, is_positive, is_scheduled)."""
    df = pd.DataFrame(rows, columns=["ticker", "peak_score", "is_positive",
                                     "is_scheduled"])
    df.index = [f"W{i}" for i in range(len(df))]
    df["is_scheduled"] = df["is_scheduled"].astype("boolean")
    return df


def test_perfect_timing_scores_one():
    w = windows([("A", 9, True, False), ("A", 1, False, None), ("A", 2, False, None)])
    assert within_stock_auc(stock_pairs(w), 50, 0)["auc"] == 1.0


def test_stock_selection_alone_earns_nothing():
    """HOT's every bar outscores COLD's, positive or not. Ranked across stocks
    this looks like a detector; ranked within each stock it is a coin flip."""
    w = windows([("HOT", 10, True, False), ("HOT", 10, False, None),
                 ("COLD", 1, True, False), ("COLD", 1, False, None)])
    assert within_stock_auc(stock_pairs(w), 50, 0)["auc"] == 0.5


def test_ties_count_half_and_inverted_timing_scores_zero():
    tie = windows([("A", 5, True, False), ("A", 5, False, None)])
    inv = windows([("A", 1, True, False), ("A", 5, False, None)])
    assert within_stock_auc(stock_pairs(tie), 50, 0)["auc"] == 0.5
    assert within_stock_auc(stock_pairs(inv), 50, 0)["auc"] == 0.0


def test_slice_filters_positives_but_keeps_every_quiet_window():
    w = windows([("A", 9, True, True), ("A", 0, True, False),
                 ("A", 5, False, None), ("A", 4, False, None)])
    sched = stock_pairs(w, "scheduled")
    unsched = stock_pairs(w, "unscheduled")
    assert sched.pairs.sum() == 2 and unsched.pairs.sum() == 2
    assert within_stock_auc(sched, 50, 0)["auc"] == 1.0
    assert within_stock_auc(unsched, 50, 0)["auc"] == 0.0


def test_a_stock_without_both_classes_contributes_no_pair():
    w = windows([("A", 9, True, False), ("B", 1, False, None),
                 ("C", 9, True, False), ("C", 1, False, None)])
    pairs = stock_pairs(w)
    assert list(pairs.ticker) == ["C"]


def test_random_scores_sit_near_half_and_the_interval_covers_it():
    rng = np.random.default_rng(3)
    rows = [(f"T{t}", rng.random(), i == 0, False if i == 0 else None)
            for t in range(200) for i in range(30)]
    res = within_stock_auc(stock_pairs(windows(rows)), 500, 0)
    assert abs(res["auc"] - 0.5) < 0.06
    assert res["ci_lo"] < 0.5 < res["ci_hi"]


def test_empty_input_is_nan_not_an_error():
    res = within_stock_auc(stock_pairs(windows([("A", 1, False, None)])), 50, 0)
    assert np.isnan(res["auc"]) and res["n_stocks"] == 0


def test_unknown_slice_is_refused():
    with pytest.raises(ValueError, match="unknown slice"):
        stock_pairs(windows([("A", 1, True, False)]), "earnings")


def test_config_carries_the_knobs():
    knobs = load_config()["eval"]["within_stock"]
    assert {"bootstrap", "seed", "ci"} <= set(knobs)
