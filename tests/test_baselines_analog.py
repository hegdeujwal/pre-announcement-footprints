"""The analog memory and its item-guess evaluation."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.baselines.analog import AnalogMemory
from src.eval import analog_items as AI
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts
from tests.test_leakage import assert_no_lookahead

HOUR = 3600
L = 6
#: Two distinct pre-filing shapes, one per item, so a match can name the item.
SHAPES = {"8.01": [0, 0, 0, 1.5, 3.0, 4.5],      # ramps into the filing
          "5.02": [0, 4.0, 4.0, 0, 0, 0]}        # an early burst, then calm


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["baselines"]["analog"].update(length=L, k=5, chunk_rows=7)
    return c


def windows(spec, n: int = 10, start: str = "2025-10-01", seed: int = 0,
            channel: str = "volume_z"):
    """spec: list of (ticker, item or None). A positive's last L bars carry its
    item's shape on top of N(0, 0.3) noise; a quiet window is noise alone."""
    rng = np.random.default_rng(seed)
    base = date_str_to_ts(start)
    rows = []
    for w, (ticker, item) in enumerate(spec):
        anchor = base + w * 200 * HOUR
        z = rng.normal(0, 0.3, n)
        if item:
            z[-L:] += SHAPES[item]
        for h in range(n):
            rows.append({"window_id": f"W{w}", "ticker": ticker,
                         "ts_utc": anchor - (n - h) * HOUR,
                         "t0_utc": anchor if item else None,
                         "is_scheduled": False if item else None,
                         "item_code": f"{item},9.01" if item else None,
                         channel: z[h]})
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


def train_spec():
    return ([("A", "8.01")] * 15 + [("B", "5.02")] * 15 + [("C", "8.01")] * 5
            + [(t, None) for t in "ABC" for _ in range(40)])


@pytest.fixture
def fitted(cfg):
    return AnalogMemory(cfg).fit(windows(train_spec()))


def last_scores(model, frame):
    s = model.score(frame)
    return (frame.assign(s=s).sort_values("ts_utc").groupby("window_id")
            .tail(1).set_index("window_id"))


# -- detection ----------------------------------------------------------

def test_positives_outscore_quiet_on_fresh_data(fitted):
    last = last_scores(fitted, windows(train_spec(), seed=1))
    assert last[last.t0_utc.notna()]["s"].mean() > last[last.t0_utc.isna()]["s"].mean() + 0.5


def test_score_is_a_weighted_share_between_0_and_1(fitted):
    s = fitted.score(windows(train_spec(), seed=1)).dropna()
    assert len(s) and s.between(0, 1).all()


def test_score_never_reads_the_label_columns(fitted):
    val = windows(train_spec(), seed=1)
    scrubbed = val.assign(t0_utc=pd.NA, is_scheduled=pd.NA, item_code=pd.NA)
    pd.testing.assert_series_equal(fitted.score(val), fitted.score(scrubbed))


def test_score_has_no_lookahead(fitted):
    one = windows([("A", None)], n=60, seed=5)
    frame = one.set_index("ts_utc")[["volume_z"]].astype("float64")

    def fn(f):
        g = f.reset_index().assign(ticker="A", window_id="W")
        return pd.DataFrame({"s": fitted.score(g).to_numpy()}, index=f.index)

    assert_no_lookahead(fn, frame)


def test_fit_refuses_a_single_class(cfg):
    with pytest.raises(ValueError, match="both"):
        AnalogMemory(cfg).fit(windows([("A", "8.01")] * 10))


def test_score_before_fit_is_refused(cfg):
    with pytest.raises(RuntimeError, match="fit"):
        AnalogMemory(cfg).score(windows(train_spec()))


def test_a_missing_channel_is_named(cfg):
    with pytest.raises(ValueError, match="range_z"):
        AnalogMemory(cfg, channels=["volume_z", "range_z"]).fit(windows(train_spec()))


def test_memory_stores_one_case_per_training_window(fitted):
    assert len(fitted.cases) == len(train_spec())
    assert fitted.cases["is_prefiling"].sum() == 35


# -- explanation --------------------------------------------------------

def test_explain_names_the_item_whose_shape_matches(fitted):
    val = windows([("D", "8.01"), ("E", "5.02")], seed=2)
    rows = val.sort_values("ts_utc").groupby("window_id").tail(1)
    out = fitted.explain(val, rows)
    assert out["guess"].str.split(",").str[0].tolist() == ["8.01", "5.02"]
    assert (out["nearest_items"] == ["8.01", "5.02"]).all()


def test_explain_drops_the_exhibits_item_and_pads_to_top_n(fitted):
    val = windows([("D", "8.01")], seed=2)
    out = fitted.explain(val, val.tail(1))
    # Only two substantive items exist in train, so top_n=3 pads to both.
    assert out["guess"].iloc[0].split(",") == ["8.01", "5.02"]


def test_other_stocks_variant_never_matches_a_stock_to_itself(cfg):
    model = AnalogMemory(cfg, exclude_same_ticker=True).fit(windows(train_spec()))
    val = windows([("A", "8.01"), ("B", "5.02"), ("C", "8.01")], seed=3)
    rows = val.sort_values("ts_utc").groupby("window_id").tail(1)
    out = model.explain(val, rows)
    assert (out["nearest_ticker"] != out["ticker"]).all()
    # B's own filings are its only 5.02 cases, so excluding them must change B.
    same = AnalogMemory(cfg).fit(windows(train_spec())).explain(val, rows)
    assert same.loc[same.ticker == "B", "nearest_ticker"].item() == "B"


def test_explain_keeps_row_order_and_marks_unmatched_rows(fitted):
    val = windows([("D", "8.01"), ("E", "5.02")], seed=2)
    rows = pd.DataFrame({"ticker": ["E", "ZZ", "D"],
                         "ts_utc": [val.ts_utc.max(), 1, val[val.ticker == "D"].ts_utc.max()]})
    out = fitted.explain(val, rows)
    assert out["ticker"].tolist() == ["E", "ZZ", "D"]
    assert pd.isna(out.loc[1, "guess"])


# -- the item-guess evaluation -----------------------------------------

def test_substantive_items_keep_codes_as_strings():
    assert AI.substantive_items("1.10,9.01, 1.1", ["9.01"]) == ["1.10", "1.1"]
    assert AI.substantive_items(None, ["9.01"]) == []
    assert AI.substantive_items(pd.NA, ["9.01"]) == []


def test_hit_rates_count_any_substantive_item_and_drop_empty_events():
    t = pd.DataFrame({
        "item_code": ["2.02,9.01", "8.01,5.02", "9.01", "1.01"],
        "is_scheduled": [True, False, False, False],
        "guess:m": ["2.02,8.01", "5.02,1.01", "8.01", "8.01,1.01"],
    })
    r = AI.hit_rates(t, ["9.01"], top_n=2, n_boot=50, seed=0, ci=0.9).set_index("slice")
    assert r.loc["all", "n_events"] == 3 and r.loc["all", "n_dropped"] == 1
    assert r.loc["all", "hit_at_1"] == pytest.approx(2 / 3)
    assert r.loc["all", "hit_at_2"] == pytest.approx(1.0)
    assert r.loc["unscheduled", "hit_at_1"] == pytest.approx(0.5)


def test_baseline_guesses_fall_back_to_majority_for_unknown_stocks():
    train = pd.DataFrame({"ticker": ["A", "A", "B", "B", "B"],
                          "item_code": ["5.02", "5.02", "8.01", "8.01", "8.01,9.01"],
                          "is_scheduled": [False] * 5})
    events = pd.DataFrame({"ticker": ["A", "NEW"], "is_scheduled": [False, False]})
    g = AI.baseline_guesses(train, events, ["9.01"], top_n=2, seed=0)
    assert g["guess:majority"].tolist() == ["8.01,5.02"] * 2
    assert g["guess:company_history"].tolist() == ["5.02,8.01", "8.01,5.02"]
    assert set(g["guess:random"].iloc[0].split(",")) <= {"5.02", "8.01"}


def test_baseline_guesses_refuse_train_events_with_no_items():
    train = pd.DataFrame({"ticker": ["A"], "item_code": ["9.01"],
                          "is_scheduled": [False]})
    with pytest.raises(ValueError, match="no substantive"):
        AI.baseline_guesses(train, train, ["9.01"], top_n=2, seed=0)
