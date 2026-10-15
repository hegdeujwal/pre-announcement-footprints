"""Chart-reading features: no lookahead, correct patterns, off by default."""

import copy

import numpy as np
import pandas as pd
import pytest

from src.pipeline import features as F
from src.pipeline import technical as T
from src.utils.config import load_config
from tests.test_leakage import assert_no_lookahead

HOUR = 3600
DAY = 86400
START = 1_725_283_800  # 2024-09-02 13:30 UTC, a Monday open


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["features"]["technical"]["enabled"] = True
    # Small windows so a 400-bar fixture leaves most rows defined.
    c["features"]["volume_zscore_window_h"] = 60
    c["features"]["min_baseline_bars"] = 20
    c["features"]["technical"]["tod_window_sessions"] = 10
    c["features"]["technical"]["tod_min_sessions"] = 5
    c["features"]["technical"]["range_window"] = 30
    return c


def sessions(n_days: int, bars_per_day: int = 7) -> pd.Index:
    stamps = [START + d * DAY + b * HOUR for d in range(n_days)
              for b in range(bars_per_day)]
    return pd.Index(stamps, name="ts_utc")


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = sessions(60)
    n = len(idx)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.004, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.004, n))
    vol = rng.lognormal(12, 0.5, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": vol}, index=idx)


@pytest.mark.parametrize("builder", [T.volume_scores, T.candles, T.indicators,
                                     T.technical_features])
def test_no_builder_reads_the_future(builder, ohlcv, cfg):
    assert_no_lookahead(lambda f: builder(f, cfg), ohlcv)


def test_columns_match_the_declared_list(ohlcv, cfg):
    out = T.technical_features(ohlcv, cfg)
    assert tuple(out.columns) == T.technical_columns(cfg)
    assert not np.isinf(out.to_numpy(dtype="float64")).any()


def test_off_by_default_and_matrix_keeps_its_name():
    base = load_config()
    assert base["features"]["technical"]["enabled"] is False
    assert T.technical_columns(base) == ()
    assert F.matrix_path(base).name == "features.parquet"
    on = copy.deepcopy(base)
    on["features"]["technical"]["enabled"] = True
    assert F.matrix_path(on).name == "features-with-technical.parquet"
    on["features"]["include_news_coverage"] = True
    assert F.matrix_path(on).name == "features-with-news-with-technical.parquet"


def test_ticker_features_appends_only_when_enabled(ohlcv, cfg):
    bench = ohlcv[["close"]]
    empty = np.array([], dtype=np.int64)
    off = copy.deepcopy(cfg)
    off["features"]["technical"]["enabled"] = False
    a = F.ticker_features(ohlcv, bench, empty, empty, off)
    b = F.ticker_features(ohlcv, bench, empty, empty, cfg)
    assert set(b.columns) - set(a.columns) == set(T.technical_columns(cfg))
    pd.testing.assert_frame_equal(a, b[a.columns])


def one_bar(o, h, l, c):
    return pd.DataFrame({"open": [o], "high": [h], "low": [l], "close": [c],
                         "volume": [1.0]}, index=pd.Index([START], name="ts_utc"))


@pytest.mark.parametrize("bar,expected", [
    ((100, 101, 99, 100.05), "pat_doji"),
    ((100.6, 101.0, 97.0, 101.0), "pat_hammer"),          # long lower wick
    ((99.4, 103.0, 99.0, 99.0), "pat_shooting_star"),      # long upper wick
    ((99.0, 101.0, 99.0, 101.0), "pat_marubozu"),          # all body
])
def test_single_candle_patterns(bar, expected, cfg):
    out = T.candles(one_bar(*bar), cfg).iloc[0]
    fired = {p for p in T.PATTERNS if out[p] == 1.0}
    assert expected in fired
    assert "pat_engulf_bull" not in fired and "pat_engulf_bear" not in fired


def test_engulfing_needs_the_previous_bar(cfg):
    idx = pd.Index([START, START + HOUR], name="ts_utc")
    bull = pd.DataFrame({"open": [101, 99.5], "high": [101.2, 102.2],
                         "low": [99.8, 99.3], "close": [100, 102],
                         "volume": [1, 1]}, index=idx, dtype="float64")
    out = T.candles(bull, cfg)
    assert out["pat_engulf_bull"].tolist() == [0.0, 1.0]
    bear = bull.rename(columns={"open": "close", "close": "open"})
    assert T.candles(bear, cfg)["pat_engulf_bear"].tolist() == [0.0, 1.0]


def test_zero_range_bar_is_undefined_not_zero(cfg):
    out = T.candles(one_bar(100, 100, 100, 100), cfg).iloc[0]
    assert np.isnan(out["candle_body"]) and np.isnan(out["pat_doji"])


def test_time_of_day_score_ignores_the_daily_open_spike(cfg):
    """Every open is 10x busier. A plain log z-score calls each one unusual;
    the same-slot score sees an ordinary open."""
    idx = sessions(40)
    rng = np.random.default_rng(1)
    vol = rng.lognormal(10, 0.1, len(idx))
    vol[::7] *= 10                                # the opening bar, every day
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": vol}, index=idx)
    out = T.volume_scores(frame, cfg)
    opens = out.iloc[::7].iloc[-10:]
    # Typical values, not extremes: with a 10-session baseline a single
    # random open can still sit a few sigma out, and that is correct.
    assert opens["volume_logz"].median() > 2.0
    assert opens["volume_tod_z"].abs().median() < 1.5


def test_time_of_day_score_excludes_the_current_bar(cfg):
    idx = sessions(30)
    vol = np.full(len(idx), 1000.0)
    vol += np.arange(len(idx)) % 3                # some variance
    vol[-1] = 1e7                                 # a spike on the last bar
    frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                          "volume": vol}, index=idx)
    out = T.volume_scores(frame, cfg)
    assert out["volume_tod_z"].iloc[-1] > 10


def test_rsi_extremes(cfg):
    idx = sessions(10)
    up = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0,
                       "close": np.linspace(100, 150, len(idx)),
                       "volume": 1.0}, index=idx)
    assert T.indicators(up, cfg)["rsi"].iloc[-1] == pytest.approx(100.0)
    down = up.assign(close=up["close"][::-1].to_numpy())
    assert T.indicators(down, cfg)["rsi"].iloc[-1] == pytest.approx(0.0)


def test_missing_ohlc_is_refused(cfg):
    with pytest.raises(ValueError, match="OHLCV"):
        T.technical_features(pd.DataFrame({"close": [1.0], "volume": [1.0]}), cfg)
