"""Chart-reading features — candles, classic indicators, and two fixes to the volume score.

Added 2026-10-15 from the project guide's review list. Everything here obeys
the rule `features.py` opens with: a value at time t uses only bars at or
before t, through backward-looking windows. The leakage tests in
`tests/test_technical.py` perturb the future and demand nothing earlier moves.

Off by default (`features.technical.enabled`), for the same reason the news
channel is: switching it on is an experiment, and every number already
published has to stay reproducible with the switch in its default position.

Three groups:

**1. The volume score, repaired for fat tails.** `volume_z` assumes "3 sigma"
is rare, a bell-curve assumption. Measured on the feature matrix, z > 3 occurs
2.6% of the time against 0.13% for a bell curve, because raw volume is heavily
right-skewed: one block trade can be fifty times a normal hour. Two
alternatives, both standard:

* `volume_logz` — the z-score of log(1 + volume). Logs turn "five times
  normal" and "fifty times normal" into a difference of one unit rather than
  of ten, which pulls the tail in toward a bell shape.
* `volume_tod_z` — the same log z-score, measured against the **same time of
  day** in earlier sessions. Volume is U-shaped through a session: the open
  and the close are always busy. A plain z-score compares a closing bar with
  midday bars and calls the close unusual every day.

**2. Candles.** One hourly bar drawn as a candle has a body (open to close)
and wicks (the high and low beyond it). The shape fractions are scale-free so
a $5 stock and a $500 stock read the same way, and the named patterns are the
textbook ones: doji, hammer, shooting star, bullish/bearish engulfing,
marubozu. Pattern flags are 0/1 floats.

**3. Indicators.** RSI (who has been winning, buyers or sellers), Bollinger
%b and bandwidth (where price sits in its normal band, and how wide the band
is), on-balance-volume slope (is volume arriving on up-bars or down-bars —
quiet accumulation), and the gap to a rolling VWAP.

A bar with zero range (high == low) has undefined candle fractions and gets
NaN, never a fabricated 0. The frozen snapshot holds such bars, and the test
fixtures in this repo write open/high/low as 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline.features import _assert_sorted
from src.utils.config import load_config

DAY_S = 86400

#: Pattern columns, in output order. Kept in one place so the evaluation and
#: the pattern study cannot disagree about what exists.
PATTERNS: tuple[str, ...] = ("pat_doji", "pat_hammer", "pat_shooting_star",
                             "pat_engulf_bull", "pat_engulf_bear",
                             "pat_marubozu")


def technical_columns(cfg: dict) -> tuple[str, ...]:
    """Every column `technical_features` adds, or () when the group is off."""
    if not cfg["features"].get("technical", {}).get("enabled", False):
        return ()
    return ("volume_logz", "volume_tod_z",
            "candle_body", "candle_upper_wick", "candle_lower_wick",
            "range_z", "gap_open") + PATTERNS + (
            "rsi", "bb_pctb", "bb_width", "obv_slope", "vwap_gap")


def _tcfg(cfg: dict) -> dict:
    return cfg["features"]["technical"]


def _trailing_z(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """z of each value against the `window` values BEFORE it — never itself."""
    base = series.shift(1).rolling(window, min_periods=min_periods)
    std = base.std()
    return ((series - base.mean()) / std).where(std > 0)


def volume_scores(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """`volume_logz` and `volume_tod_z` — see the module docstring, group 1.

    The time-of-day slot is the bar's position within its UTC calendar date.
    A US session (13:30–21:00 UTC at the latest) never crosses UTC midnight, so
    the UTC date IS the session and position 0 is the opening bar — no
    timezone arithmetic, and so nothing for daylight-saving to break. A
    half-day session simply has fewer slots; its bars still line up from the
    open.
    """
    cfg = cfg or load_config()
    t, f = _tcfg(cfg), cfg["features"]
    _assert_sorted(frame.index, "frame index")
    logv = np.log1p(frame["volume"].clip(lower=0).astype("float64"))

    logz = _trailing_z(logv, f["volume_zscore_window_h"], f["min_baseline_bars"])

    stamps = np.asarray(frame.index, dtype=np.int64)
    day = stamps // DAY_S
    slot = pd.Series(day).groupby(day).cumcount().to_numpy()
    tod = pd.Series(np.nan, index=frame.index, dtype="float64")
    for s in np.unique(slot):
        mask = slot == s
        tod.iloc[np.flatnonzero(mask)] = _trailing_z(
            logv[mask], int(t["tod_window_sessions"]),
            int(t["tod_min_sessions"])).to_numpy()
    return pd.DataFrame({"volume_logz": logz, "volume_tod_z": tod},
                        index=frame.index)


def candles(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Candle shape fractions, range surprise, opening gap, and named patterns."""
    cfg = cfg or load_config()
    t = _tcfg(cfg)
    _assert_sorted(frame.index, "frame index")
    o, h, l, c = (frame[k].astype("float64") for k in ("open", "high", "low", "close"))
    rng = (h - l).where((h > l) & (l > 0))           # zero/absent range -> NaN

    body = (c - o) / rng                              # signed, in [-1, 1]
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng
    abs_body = body.abs()

    log_range = np.log(h / l).where(rng.notna())
    range_z = _trailing_z(log_range, int(t["range_window"]),
                          int(t["range_window"]) // 2)
    prev_c = c.shift(1)
    gap = (o / prev_c - 1).where((prev_c > 0) & (o > 0))

    doji_max = float(t["doji_body_max"])
    wick_ratio = float(t["hammer_wick_ratio"])
    small_wick = float(t["hammer_other_wick_max"])
    maru_min = float(t["marubozu_body_min"])

    def flag(cond: pd.Series) -> pd.Series:
        # NaN when the candle is undefined, so "no pattern" is never claimed
        # for a bar whose shape could not be measured.
        return cond.astype("float64").where(rng.notna())

    prev_o = o.shift(1)
    prev_bear = prev_c < prev_o
    prev_bull = prev_c > prev_o
    engulf_bull = prev_bear & (c > o) & (o <= prev_c) & (c >= prev_o)
    engulf_bear = prev_bull & (c < o) & (o >= prev_c) & (c <= prev_o)

    return pd.DataFrame({
        "candle_body": body,
        "candle_upper_wick": upper,
        "candle_lower_wick": lower,
        "range_z": range_z,
        "gap_open": gap,
        "pat_doji": flag(abs_body <= doji_max),
        "pat_hammer": flag((lower >= wick_ratio * abs_body) & (upper <= small_wick)
                           & (abs_body > doji_max)),
        "pat_shooting_star": flag((upper >= wick_ratio * abs_body)
                                  & (lower <= small_wick) & (abs_body > doji_max)),
        "pat_engulf_bull": flag(engulf_bull),
        "pat_engulf_bear": flag(engulf_bear),
        "pat_marubozu": flag(abs_body >= maru_min),
    }, index=frame.index)


def indicators(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """RSI, Bollinger %b and width, OBV slope, VWAP gap — all trailing.

    RSI uses Wilder's smoothing, an exponential average with alpha = 1/n.
    `ewm(adjust=False)` is a pure recursion over earlier values, so it is
    backward-looking by construction.

    OBV is a running sum whose LEVEL depends on where the series starts, so
    the level itself would differ between a live run and a backfill. Its
    change over `obv_window` bars does not, and is scaled by the trailing
    average volume so it reads in "bars' worth of volume".
    """
    cfg = cfg or load_config()
    t = _tcfg(cfg)
    _assert_sorted(frame.index, "frame index")
    c = frame["close"].astype("float64").where(frame["close"] > 0)
    v = frame["volume"].astype("float64").clip(lower=0)

    n = int(t["rsi_window"])
    diff = c.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    total = gain + loss
    rsi = (100 * gain / total).where(total > 0)

    b = int(t["bollinger_window"])
    k = float(t["bollinger_k"])
    mid = c.rolling(b, min_periods=b).mean()
    sd = c.rolling(b, min_periods=b).std()
    upper, lower = mid + k * sd, mid - k * sd
    width = upper - lower
    pctb = ((c - lower) / width).where(width > 0)
    bw = (width / mid).where(mid > 0)

    w = int(t["obv_window"])
    direction = np.sign(diff).fillna(0.0)
    obv = (direction * v).cumsum()
    avg_v = v.shift(1).rolling(w, min_periods=w).mean()
    obv_slope = ((obv - obv.shift(w)) / (w * avg_v)).where(avg_v > 0)

    vw = int(t["vwap_window"])
    pv = (c * v).rolling(vw, min_periods=vw).sum()
    vv = v.rolling(vw, min_periods=vw).sum()
    vwap = (pv / vv).where(vv > 0)
    vwap_gap = (c / vwap - 1).where(vwap > 0)

    return pd.DataFrame({"rsi": rsi, "bb_pctb": pctb, "bb_width": bw,
                         "obv_slope": obv_slope, "vwap_gap": vwap_gap},
                        index=frame.index)


def technical_features(frame: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """All three groups for one ticker. Needs open, high, low, close, volume."""
    cfg = cfg or load_config()
    missing = [k for k in ("open", "high", "low", "close", "volume")
               if k not in frame.columns]
    if missing:
        raise ValueError(f"technical features need OHLCV bars; missing {missing}")
    return pd.concat([volume_scores(frame, cfg), candles(frame, cfg),
                      indicators(frame, cfg)], axis=1)
