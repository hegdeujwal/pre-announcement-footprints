"""The discovery analyses: earnings leak, item types, learning curve."""

import numpy as np
import pandas as pd
import pytest

from src.baselines import discovery_review as R
from src.eval import by_item as B
from src.eval import earnings_leak as E
from src.eval import learning_curve as C
from src.utils.config import load_config

DAY = 86400
KNOBS = {"normal_gap_days": 7, "min_quiet_closing": 5, "unusual_top_share": 0.2,
         "bootstrap": 200, "seed": 0, "ci": 0.95}


# -- earnings leak -----------------------------------------------------------

def test_partial_spearman_removes_what_the_control_explains():
    rng = np.random.default_rng(0)
    c = rng.normal(size=2000)
    x = c + 0.1 * rng.normal(size=2000)
    y = c + 0.1 * rng.normal(size=2000)
    assert E.spearman(x, y) > 0.9
    assert abs(E.partial_spearman(x, y, c[:, None])) < 0.1


def scored(ticker, quiet_scores, t0, event_score, closing=True, start=0):
    rows = [{"detector": "d", "ticker": ticker, "ts_utc": start + i * DAY,
             "score": s, "is_positive": False, "closing": True}
            for i, s in enumerate(quiet_scores)]
    rows.append({"detector": "d", "ticker": ticker, "ts_utc": t0 - 3600,
                 "score": event_score, "is_positive": True, "closing": closing})
    return rows


def test_score_percentile_is_against_the_stocks_own_normal_closing_bars():
    t0 = 100 * DAY
    scores = pd.DataFrame(scored("A", list(range(10)), t0, 7.0))
    events = pd.DataFrame({"event_id": ["e"], "ticker": ["A"], "t0_utc": [t0],
                           "score_ts": [t0 - 3600]})
    out = E.score_percentiles(scores, events, KNOBS)
    assert out["score_pct"].iloc[0] == pytest.approx((7 + 0.5) / 10)  # ties count half


def test_quiet_bars_just_before_earnings_are_not_the_normal():
    t0 = 20 * DAY
    # 20 daily quiet bars, the last six within a week of t0 and very high.
    quiet = [0.0] * 14 + [100.0] * 6
    scores = pd.DataFrame(scored("A", quiet, t0, 1.0))
    events = pd.DataFrame({"event_id": ["e"], "ticker": ["A"], "t0_utc": [t0],
                           "score_ts": [t0 - 3600]})
    out = E.score_percentiles(scores, events, KNOBS)
    assert out["score_pct"].iloc[0] == 1.0       # beats every bar of its normal


def test_a_non_closing_event_bar_is_dropped():
    t0 = 100 * DAY
    scores = pd.DataFrame(scored("A", list(range(10)), t0, 7.0, closing=False))
    events = pd.DataFrame({"event_id": ["e"], "ticker": ["A"], "t0_utc": [t0],
                           "score_ts": [t0 - 3600]})
    assert E.score_percentiles(scores, events, KNOBS).empty


def leak_fixture(leak: bool, n_stocks=60, seed=0):
    """Each stock: 30 quiet closing bars and 2 earnings. With `leak`, the
    event's score and the drift both track the coming move."""
    rng = np.random.default_rng(seed)
    rows, events = [], []
    for s in range(n_stocks):
        t = f"T{s}"
        for k in range(2):
            t0 = (200 + 100 * k) * DAY
            move = rng.normal(0, 0.05)
            sc = abs(move) * 40 + rng.normal() if leak else rng.normal()
            drift = move / 5 if leak else rng.normal(0, 0.01)
            rows += scored(t, rng.normal(size=30), t0, sc, start=k * 100 * DAY + 100 * DAY)
            events.append({"event_id": f"{t}{k}", "ticker": t, "t0_utc": t0,
                           "score_ts": t0 - 3600, "move": move, "pre_drift": drift,
                           "prior_abs_move": abs(rng.normal(0, 0.05)),
                           "volatility": abs(rng.normal())})
    return pd.DataFrame(rows), pd.DataFrame(events)


def test_a_planted_leak_is_found_and_noise_is_not():
    t = E.leak_table(*leak_fixture(True), KNOBS).iloc[0]
    assert t.rho_size_partial_lo > 0.3 and t.hit_unusual_lo > 0.5
    n = E.leak_table(*leak_fixture(False), KNOBS).iloc[0]
    assert n.rho_size_partial_lo < 0 < n.rho_size_partial_hi
    assert n.hit_all_lo < 0.5 < n.hit_all_hi


def test_attach_bars_reads_only_bars_before_t0():
    frame = pd.DataFrame({"ticker": "A", "ts_utc": np.arange(10) * 3600,
                          "ret_rel_1h": np.arange(10) * 1.0, "volatility": 1.0})
    ev = pd.DataFrame({"event_id": ["e"], "ticker": ["A"], "t0_utc": [7 * 3600]})
    out = R.attach_bars(ev, frame, 3)
    assert out["score_ts"].iloc[0] == 6 * 3600
    assert out["pre_drift"].iloc[0] == 4 + 5 + 6
    short = R.attach_bars(ev.assign(t0_utc=2 * 3600), frame, 3)
    assert short.empty


# -- item types ----------------------------------------------------------------

def test_holm_is_monotone_and_never_below_the_raw_p():
    p = np.array([0.01, 0.04, 0.03, 0.5])
    adj = B.holm(p)
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.5])
    assert (adj >= p).all()


def test_a_multi_item_filing_counts_under_each_item():
    codes = pd.Series(["1.01,2.03,9.01", "2.01", None])
    assert B.has_item(codes, "2.03").tolist() == [True, False, False]
    assert B.has_item(codes, "2.0").tolist() == [False, False, False]


def dump(detector, rng, n_tickers=30, item_signal=None):
    """Closing-bar dump: per ticker 40 quiet bars and four unscheduled
    positives, two of item 2.01 and two of 8.01."""
    rows = []
    for t in range(n_tickers):
        for i in range(40):
            rows.append({"detector": detector, "window_id": None, "ticker": f"T{t}",
                         "ts_utc": 1_760_000_000 + i * 3600 * 24, "score": rng.normal(),
                         "is_positive": False, "t0_utc": pd.NA, "is_scheduled": pd.NA,
                         "item_code": None, "closing": True})
        for j, item in enumerate(["2.01", "2.01", "8.01", "8.01"]):
            boost = 3.0 if item == item_signal else 0.0
            rows.append({"detector": detector, "window_id": f"E{t}{j}", "ticker": f"T{t}",
                         "ts_utc": 1_760_000_000 + (50 + j) * 3600 * 24,
                         "score": rng.normal() + boost, "is_positive": True,
                         "t0_utc": 1_760_000_000 + (50 + j) * 3600 * 24 + 3600,
                         "is_scheduled": False, "item_code": item, "closing": True})
    return pd.DataFrame(rows)


def test_a_footprint_in_one_item_type_is_found_only_there():
    cfg = load_config()
    cfg["eval"]["discovery"].update(bootstrap=300, min_item_events=10)
    cfg["items"]["unscheduled_focus"] = ["2.01", "8.01"]
    t = B.item_table(dump("d", np.random.default_rng(0), item_signal="2.01"), cfg)
    t = t.set_index("item")
    assert bool(t.loc["2.01", "footprint"]) and not bool(t.loc["8.01", "footprint"])
    assert t.loc["2.01", "auc"] > 0.9 and t.loc["2.01", "lift"] > 1


# -- learning curve --------------------------------------------------------------

def test_run_names_parse_to_share_and_seed():
    assert C.parse("footprint_scratch[s42]") == (1.0, 42)
    assert C.parse("footprint_scratch_f25[s43]") == (0.25, 43)
    assert C.parse("footprint_pretrained[s42]") is None


def test_curve_summary_ranges_over_seeds():
    rng = np.random.default_rng(0)
    scores = pd.concat([dump(n, rng) for n in
                        ("footprint_scratch[s1]", "footprint_scratch[s2]",
                         "footprint_scratch_f50[s1]", "volume_zscore")])
    scores.loc[scores["window_id"].fillna("").str.endswith("0"), "is_scheduled"] = True
    s = C.summary(C.curve_table(scores, load_config()))
    full = s[(s["fraction"] == 1.0) & (s["slice"] == "unscheduled")].iloc[0]
    assert full.n_seeds == 2 and full.lift_min <= full.lift_mean <= full.lift_max
    assert set(s["fraction"]) == {0.5, 1.0} and s["volume_zscore_lift"].notna().all()
