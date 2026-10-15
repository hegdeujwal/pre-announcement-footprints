"""Pattern templates, the template hybrid, and the template-shaped reward."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.baselines.hybrid import PriorTemplate
from src.baselines.template import PatternTemplate, lagged, last_bars
from src.rl.env import TEMPLATE_COL, FootprintEnv, observation_features
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts
from tests.test_leakage import assert_no_lookahead

HOUR = 3600
L = 6


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["baselines"]["pattern_template"].update(length=L, min_group_events=5)
    return c


def windows(spec, n: int = 12, start: str = "2025-10-01", seed: int = 0):
    """spec: list of (ticker, positive, item). Positives ramp up over their
    last three bars; everything else is N(0, 1) noise."""
    rng = np.random.default_rng(seed)
    base = date_str_to_ts(start)
    rows = []
    for w, (ticker, positive, item) in enumerate(spec):
        anchor = base + w * 200 * HOUR
        z = rng.normal(0, 1, n)
        if positive:
            z[-3:] += [1.0, 2.0, 3.0]
        for h in range(n):
            rows.append({"window_id": f"W{w}", "ticker": ticker,
                         "ts_utc": anchor - (n - h) * HOUR,
                         "t0_utc": anchor if positive else None,
                         "is_scheduled": (item == "2.02") if positive else None,
                         "item_code": item if positive else None,
                         "volume_z": z[h]})
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


def train_spec():
    return ([("HOT", True, "8.01")] * 30 + [("HOT", True, "2.02")] * 10
            + [("COLD", True, "8.01")] * 4
            + [(t, False, None) for t in ("HOT", "COLD") for _ in range(120)])


@pytest.fixture
def fitted(cfg):
    return PatternTemplate(cfg).fit(windows(train_spec()))


def test_last_bars_right_aligns_and_pads_left():
    f = windows([("A", True, "8.01")], n=3)
    wide = last_bars(f, "volume_z", 5)
    assert np.isnan(wide.iloc[0, :2]).all()
    np.testing.assert_allclose(wide.iloc[0, 2:].to_numpy(),
                               f["volume_z"].to_numpy())


def test_lagged_rows_end_at_the_current_bar():
    out = lagged(np.arange(5.0), 3)
    np.testing.assert_array_equal(out[4], [2, 3, 4])
    assert np.isnan(out[0, :2]).all() and out[0, 2] == 0


def test_templates_learn_the_ramp_and_weight_the_end(fitted):
    assert {"all", "scheduled", "unscheduled", "item 8.01", "item 2.02"} <= set(fitted.templates)
    w = fitted.template_table().query("group == 'all'")["weight"].to_numpy()
    assert w[-1] > w[-2] > w[-3] > abs(w[:-3]).max()


def test_positives_outscore_quiet_on_fresh_data(fitted):
    val = windows(train_spec(), seed=1)
    s = fitted.score(val)
    last = val.assign(s=s).sort_values("ts_utc").groupby("window_id").tail(1)
    pos = last[last.t0_utc.notna()]["s"].mean()
    neg = last[last.t0_utc.isna()]["s"].mean()
    assert pos > neg + 2


def test_quiet_scores_are_roughly_standard(fitted):
    quiet = windows([("Q", False, None)] * 400, seed=3)
    s = fitted.score(quiet)
    last = quiet.assign(s=s).sort_values("ts_utc").groupby("window_id").tail(1)["s"]
    # Max over several templates shifts the mean up a little; it must not drift far.
    assert -0.5 < last.mean() < 1.5


def test_score_never_reads_the_label_columns(fitted):
    val = windows(train_spec(), seed=1)
    a = fitted.score(val)
    scrubbed = val.assign(t0_utc=pd.NA, is_scheduled=pd.NA, item_code=pd.NA)
    pd.testing.assert_series_equal(a, fitted.score(scrubbed))


def test_score_has_no_lookahead(fitted):
    one = windows([("A", False, None)], n=60, seed=5)
    frame = one.set_index("ts_utc")[["volume_z"]].astype("float64")

    def fn(f):
        g = f.reset_index().assign(ticker="A", window_id="W")
        return pd.DataFrame({"s": fitted.score(g).to_numpy()}, index=f.index)

    assert_no_lookahead(fn, frame)


def test_window_grouping_does_not_borrow_bars(fitted):
    two = windows([("A", False, None), ("A", False, None)], n=L, seed=2)
    by_ticker = fitted.score(two, by="ticker")
    by_window = fitted.score(two, by="window_id")
    first_of_second = two.index[two.window_id == "W1"][0]
    assert by_ticker[first_of_second] != by_window[first_of_second]


def test_fit_refuses_a_single_class(cfg):
    with pytest.raises(ValueError, match="both"):
        PatternTemplate(cfg).fit(windows([("A", True, "8.01")] * 10))


def test_hybrid_orders_stocks_then_patterns(cfg):
    model = PriorTemplate(cfg).fit(windows(train_spec()))
    val = windows([("HOT", True, "8.01"), ("COLD", True, "8.01"),
                   ("HOT", False, None)], seed=4)
    s = model.score(val)
    last = val.assign(s=s).sort_values("ts_utc").groupby("window_id").tail(1).set_index("window_id")["s"]
    assert last["W0"] > last["W1"]          # same ramp, HOT files more often
    assert last["W0"] > last["W2"]          # same stock, ramp beats noise


# -- template-shaped reward ---------------------------------------------

def env_frame(cfg, score: float):
    feats = observation_features(cfg)
    f = windows([("A", True, "8.01")], n=4)
    for c in feats:
        f[c] = 0.0
    f[TEMPLATE_COL] = score
    return f


def flag_reward(cfg, frame):
    env = FootprintEnv(cfg, frame, seed=0)
    env.reset()
    return env.step(1)[1]


def test_bonus_is_off_by_default(cfg):
    assert flag_reward(cfg, env_frame(cfg, 3.0)) == flag_reward(cfg, env_frame(cfg, -3.0))


def test_bonus_scales_with_match_and_caps(cfg):
    on = copy.deepcopy(cfg)
    on["reward"]["r_template_bonus"] = 0.6
    base = flag_reward(cfg, env_frame(cfg, 0.0))
    assert flag_reward(on, env_frame(on, -2.0)) == pytest.approx(base)
    assert flag_reward(on, env_frame(on, 1.5)) == pytest.approx(base + 0.3)
    assert flag_reward(on, env_frame(on, 30.0)) == pytest.approx(base + 0.6)


def test_bonus_without_a_score_column_is_refused(cfg):
    on = copy.deepcopy(cfg)
    on["reward"]["r_template_bonus"] = 0.5
    with pytest.raises(ValueError, match="template"):
        flag_reward(on, env_frame(on, 0.0).drop(columns=[TEMPLATE_COL]))
