"""Do candle patterns and indicators look different before a filing?

Added 2026-10-15 for the guide's "explain statistical patterns" and "what are
the technical indicators like candle patterns" points. Like `event_study`, this
scores no detector: it lines every train/validation event up on its t0 and
compares the last session before it with a **matched control** — the same
stock, the same bars of the day, `event_study.control_shift_bars` earlier.

The control is not optional. Candles have a strong time-of-day shape too (the
opening bar has the widest range of the day), and most pre-filing sessions end
at the close, so against "all hours" the clock alone would pass for a pattern.

Two tables:

* `pattern_rates`  — per candle pattern and slice: the share of pre-filing
  bars showing it, the share of control bars showing it, their ratio, and a
  95% interval from a bootstrap over EVENTS (one event's seven bars are not
  independent draws).
* `indicator_shift` — per indicator and slice: mean before filings, mean in the
  control, the difference, and the same bootstrap interval.

Plus `pattern_paths` — per bar offset, the event and control means of a few
indicators, for the PDF's line charts.

Usage:
  python -m src.eval.pattern_study --out-dir data/processed/guide-review
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.pipeline.features import _ticker_frame
from src.pipeline.split import assert_not_test, boundaries
from src.pipeline.technical import PATTERNS, technical_features
from src.utils.config import load_config

INDICATORS = ("volume_tod_z", "range_z", "candle_body", "gap_open", "rsi",
              "bb_width", "obv_slope", "vwap_gap")
PATH_COLS = ("volume_tod_z", "range_z", "obv_slope", "pat_doji",
             "pat_marubozu")


def aligned(feats: pd.DataFrame, t0s: np.ndarray, pre: int, shift: int,
            avoid: np.ndarray | None) -> list[tuple[int, np.ndarray]]:
    """(event index, bar positions) for each t0, `pre` bars ending at bar -1.

    bar -1 is the last bar that ENDED at or before t0 — the decision point the
    detectors act on. With `shift` > 0 the window moves that many bars earlier
    and is dropped if any of the stock's events falls inside it.
    """
    ts = feats.index.to_numpy()
    out = []
    for k, t0 in enumerate(t0s):
        i0 = int(np.searchsorted(ts + 3600, t0, side="right")) - shift
        lo = i0 - pre
        if lo < 0 or i0 > len(ts):
            continue
        if shift and avoid is not None and (
                (avoid >= ts[lo]) & (avoid < ts[i0 - 1] + 3600)).any():
            continue
        out.append((k, np.arange(lo, i0)))
    return out


def collect(cfg: dict, conn) -> pd.DataFrame:
    """One row per (event, arm, bar offset) carrying every technical column."""
    tcfg = copy.deepcopy(cfg)
    tcfg["features"]["technical"]["enabled"] = True
    pre = int(cfg["pattern_study"]["pre_bars"])
    shift = int(cfg["event_study"]["control_shift_bars"])
    _, val_end = boundaries(cfg)
    interval = cfg["market"]["interval"]

    events = pd.read_sql_query(
        "SELECT event_id, ticker, t0_utc, is_scheduled FROM events "
        "WHERE usable = 1 AND t0_utc < ?", conn, params=(val_end,))
    all_t0 = {t: g["t0_utc"].to_numpy() for t, g in pd.read_sql_query(
        "SELECT ticker, t0_utc FROM events WHERE usable = 1", conn
    ).groupby("ticker")}
    cols = list(PATTERNS) + list(INDICATORS)

    rows = []
    for ticker, evs in events.groupby("ticker"):
        bars = _ticker_frame(conn, ticker, interval)
        bars = bars[bars.index < val_end]
        if bars.empty:
            continue
        assert_not_test(cfg, conn, bars.index, f"pattern study ({ticker})")
        feats = technical_features(bars, tcfg)[cols]
        values = feats.to_numpy(dtype="float64")
        t0s = evs["t0_utc"].to_numpy()
        for arm, sh, avoid in (("before filing", 0, None),
                               ("control", shift, all_t0.get(ticker))):
            for k, idx in aligned(feats, t0s, pre, sh, avoid):
                ev = evs.iloc[k]
                block = pd.DataFrame(values[idx], columns=cols)
                block["bar"] = np.arange(-pre, 0)
                block["event_id"] = ev["event_id"]
                block["arm"] = arm
                block["slice"] = "scheduled" if ev["is_scheduled"] else "unscheduled"
                rows.append(block)
    if not rows:
        raise SystemExit("pattern study: no event had enough bars before t0")
    return pd.concat(rows, ignore_index=True)


def _boot(event_means: pd.DataFrame, n_boot: int, seed: int,
          ratio: bool) -> tuple[float, float, float]:
    """Point estimate and 95% bootstrap interval over events.

    `event_means` has one row per (event, arm) with the event's mean value and
    bar count; events are resampled within each arm.
    """
    rng = np.random.default_rng(seed)
    a = event_means[event_means["arm"] == "before filing"]
    c = event_means[event_means["arm"] == "control"]

    def stat(x, y):
        mx = (x["sum"].sum() / x["n"].sum())
        my = (y["sum"].sum() / y["n"].sum())
        return (mx / my) if ratio else (mx - my)

    point = stat(a, c)
    draws = []
    for _ in range(n_boot):
        xa = a.iloc[rng.integers(0, len(a), len(a))]
        xc = c.iloc[rng.integers(0, len(c), len(c))]
        draws.append(stat(xa, xc))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def summarise(paths: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    knobs = cfg["pattern_study"]
    n_boot, seed = int(knobs["bootstrap"]), int(knobs["seed"])
    rates, shifts = [], []
    for s in ("scheduled", "unscheduled"):
        sl = paths[paths["slice"] == s]
        for col in list(PATTERNS) + list(INDICATORS):
            x = sl[["event_id", "arm", col]].dropna()
            per = (x.groupby(["event_id", "arm"])[col]
                   .agg(["sum", "count"]).rename(columns={"count": "n"})
                   .reset_index())
            before = x[x["arm"] == "before filing"][col]
            control = x[x["arm"] == "control"][col]
            if col in PATTERNS:
                est, lo, hi = _boot(per, n_boot, seed, ratio=True)
                rates.append({"slice": s, "pattern": col,
                              "share_before": before.mean(),
                              "share_control": control.mean(),
                              "ratio": est, "ci_lo": lo, "ci_hi": hi,
                              "n_events": per[per.arm == "before filing"].shape[0]})
            else:
                est, lo, hi = _boot(per, n_boot, seed, ratio=False)
                shifts.append({"slice": s, "indicator": col,
                               "mean_before": before.mean(),
                               "mean_control": control.mean(),
                               "difference": est, "ci_lo": lo, "ci_hi": hi,
                               "n_events": per[per.arm == "before filing"].shape[0]})
    path_tab = (paths.groupby(["slice", "arm", "bar"])[list(PATH_COLS)]
                .mean().reset_index())
    return {"pattern_rates": pd.DataFrame(rates),
            "indicator_shift": pd.DataFrame(shifts),
            "pattern_paths": path_tab}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"], readonly=True)
    out_dir = Path(args.out_dir or Path(cfg["paths"]["processed"]) / "guide-review")
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = summarise(collect(cfg, conn), cfg)
    for name, t in tables.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    with pd.option_context("display.width", 160):
        print(tables["pattern_rates"].round(4).to_string(index=False))
        print(tables["indicator_shift"].round(4).to_string(index=False))
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
