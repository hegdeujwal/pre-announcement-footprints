"""Event study — what trading looks like in the hours around t0.

Every other number in the project scores a detector. This one scores nothing:
it lines up every usable event on its own t0 and asks what the stock was
doing in the trading hours before and after, so a reader can see the
footprint the detectors are hunting rather than take a lift figure on trust.

For each event, the hourly bars of its stock are indexed in trading time
relative to t0:

    bar 0     the bar that contains t0, or the first one after it
    bar -k    k bars earlier; every one of them ends at or before t0
    bar +k    k bars later

Most 8-Ks land outside market hours, so bar -1 is usually the previous
session's last hour and bar 0 the next session's first: the axis is in hours
the market was open, never wall-clock (AGENTS rule 4).

Two measures per bar, both the project's own:

- `volume_z` from `features.volume_zscore` — the same function the detectors
  use, so the picture cannot drift from what they saw.
- `abs_rel_ret` — the absolute one-bar return minus the benchmark's, in
  percent. Absolute, because news moves prices both ways and the signed mean
  would cancel to nothing.

Reported per slice and bar: event count, median and interquartile range of
`volume_z`, the share of events whose bar clears the tuned volume z-score
threshold, and the median `abs_rel_ret`.

**Every slice has a matched control**, and it is the comparison to read.
Volume has a strong intraday shape — on these stocks about 10% of opening
bars clear the threshold against under 1% at midday — and bar 0 is usually an
opening bar, bar -1 a closing one. Against "all hours" that shape alone would
pass for a footprint. The control is the same stock, aligned the same way,
`control_shift_bars` earlier (whole sessions, so the same time of day), and is
dropped if a usable event of that stock falls inside it. A reference row
(`bar` empty) still gives the all-hours figures, for context only.

**Train and validation events only.** The test period is sealed (P4-14) and
its one opening is spent; an event whose post-window would reach into it is
dropped, and every bar read goes through `assert_not_test`. Looking past t0
is correct here for the same reason it is in `materiality`: this is the
answer-key side, and no detector or feature reads it.

Usage:
  python -m src.eval.event_study                 # writes paths.processed/<out>
  python -m src.eval.event_study --out study.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import db
from src.pipeline.features import volume_zscore
from src.pipeline.split import assert_not_test, boundaries
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso

log = logging.getLogger(__name__)

#: Slices reported, in reading order. AGENTS rule 7: never pooled. Each has
#: a "<slice> control" partner (see the module docstring).
SLICES = ("scheduled", "unscheduled")


def _bars(conn, ticker: str, interval: str, lo: int, hi: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT ts_utc, close, volume FROM bars WHERE ticker = ? AND "
        "interval = ? AND ts_utc >= ? AND ts_utc < ? ORDER BY ts_utc",
        conn, params=(ticker, interval, lo, hi)).set_index("ts_utc")


def event_paths(events: pd.DataFrame, bars: pd.DataFrame, bench_ret: pd.Series,
                pre: int, post: int, cfg: dict, shift: int = 0,
                avoid: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (event, bar offset) for a single stock's events.

    `bars` is that stock's hourly bars indexed by `ts_utc`; `bench_ret` the
    benchmark's one-bar return on the same index type. Bars are taken by
    position, so the offset counts trading hours, not clock hours. An event
    without `pre` bars before it or `post` bars after it is dropped rather
    than padded.

    `shift` > 0 builds the control instead: the same alignment `shift` bars
    earlier, skipped when any timestamp in `avoid` (the stock's own event
    t0s) falls inside the shifted window.
    """
    if bars.empty:
        return pd.DataFrame()
    z = volume_zscore(bars, cfg)["volume_z"].to_numpy()
    ret = bars["close"].pct_change()
    rel = (ret - bench_ret.reindex(bars.index)).abs().to_numpy() * 100
    ts = bars.index.to_numpy()
    bar_s = 3600
    out = []
    for ev in events.itertuples():
        # First bar that has not ENDED by t0: it contains t0 or follows it.
        i0 = int(np.searchsorted(ts + bar_s, ev.t0_utc, side="right")) - shift
        if i0 - pre < 0 or i0 + post >= len(ts):
            continue
        if shift and avoid is not None and (
                (avoid >= ts[i0 - pre]) & (avoid < ts[i0 + post] + bar_s)).any():
            continue
        idx = np.arange(i0 - pre, i0 + post + 1)
        out.append(pd.DataFrame({
            "event_id": ev.event_id,
            "slice": ("scheduled" if ev.is_scheduled else "unscheduled")
                     + (" control" if shift else ""),
            "bar": idx - i0,
            "volume_z": z[idx],
            "abs_rel_ret": rel[idx],
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarise(paths: pd.DataFrame, reference: pd.DataFrame,
              threshold: float) -> pd.DataFrame:
    """Per slice and bar offset; plus one reference row per slice."""
    def stats(g: pd.DataFrame) -> pd.Series:
        z = g["volume_z"].dropna()
        return pd.Series({
            "n_events": g["event_id"].nunique() if "event_id" in g else np.nan,
            "n_bars": len(z),
            "volume_z_median": z.median(),
            "volume_z_q25": z.quantile(0.25),
            "volume_z_q75": z.quantile(0.75),
            "share_above_threshold": (z > threshold).mean() if len(z) else np.nan,
            "abs_rel_ret_median_pct": g["abs_rel_ret"].median(),
        })

    rows = (paths.groupby(["slice", "bar"])[["event_id", "volume_z",
                                            "abs_rel_ret"]]
            .apply(stats).reset_index())
    ref = stats(reference)
    ref_rows = pd.DataFrame([{"slice": "all hours", "bar": np.nan, **ref}])
    out = pd.concat([rows, ref_rows], ignore_index=True)
    out["threshold"] = threshold
    return out


def run(cfg: dict, conn) -> pd.DataFrame:
    ecfg = cfg["event_study"]
    pre, post = int(ecfg["pre_bars"]), int(ecfg["post_bars"])
    shift = int(ecfg["control_shift_bars"])
    interval = cfg["market"]["interval"]
    threshold = float(cfg["baselines"]["volume_zscore"]["threshold"])
    _, val_end = boundaries(cfg)
    lookback = int(cfg["features"]["volume_zscore_window_h"])
    # Room for the z-score baseline before the first event, and for `post`
    # bars after the last. Seven bars a session; the margin is generous so an
    # event near the edge is dropped by the position check, not misread.
    lo = date_str_to_ts(cfg["study_window"]["start"]) - 86400 * (lookback // 7 + 14)
    margin = 86400 * (post // 7 + 7)

    events = pd.read_sql_query(
        "SELECT event_id, ticker, t0_utc, is_scheduled FROM events "
        "WHERE usable = 1 AND t0_utc < ?", conn, params=(val_end - margin,))
    if events.empty:
        raise SystemExit("event study: no usable train/validation events — "
                         "build the events table first (Phase 4).")

    all_t0 = {t: g["t0_utc"].to_numpy() for t, g in pd.read_sql_query(
        "SELECT ticker, t0_utc FROM events WHERE usable = 1", conn
    ).groupby("ticker")}

    bench = _bars(conn, cfg["market"]["benchmark"], interval, lo, val_end)
    assert_not_test(cfg, conn, bench.index, "event study (benchmark)")
    bench_ret = bench["close"].pct_change()

    paths, refs = [], []
    for ticker, evs in events.groupby("ticker"):
        bars = _bars(conn, ticker, interval, lo, val_end)
        assert_not_test(cfg, conn, bars.index, f"event study ({ticker})")
        p = event_paths(evs, bars, bench_ret, pre, post, cfg)
        if not p.empty:
            paths.append(p)
            # Avoid every usable event of the stock, test-period ones too:
            # their t0 is a date, not a test row, and is never scored here.
            mine = all_t0.get(ticker, np.array([], dtype=np.int64))
            paths.append(event_paths(evs, bars, bench_ret, pre, post, cfg,
                                     shift=shift, avoid=mine))
            ret = bars["close"].pct_change()
            refs.append(pd.DataFrame({
                "volume_z": volume_zscore(bars, cfg)["volume_z"],
                "abs_rel_ret": (ret - bench_ret.reindex(bars.index)).abs() * 100,
            }))
    if not paths:
        raise SystemExit("event study: no event had enough bars either side "
                         "of t0 — nothing to report.")
    paths = pd.concat(paths, ignore_index=True)
    reference = pd.concat(refs, ignore_index=True)
    out = summarise(paths, reference, threshold)
    log.info("event study: %d events (%s), bars %+d..%+d, train/val before %s",
             paths["event_id"].nunique(),
             paths.groupby("slice")["event_id"].nunique().to_dict(),
             -pre, post, ts_to_iso(val_end))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", help="CSV path (default: paths.processed/"
                                  "event_study.out)")
    args = ap.parse_args()
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"], readonly=True)
    table = run(cfg, conn)
    out = Path(args.out or Path(cfg["paths"]["processed"]) / cfg["event_study"]["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    ref = table[table["slice"] == "all hours"].iloc[0]
    for sl in [x for s in SLICES for x in (s, s + " control")]:
        t = table[table["slice"] == sl].set_index("bar")
        print(f"{sl:20s} events {int(t['n_events'].max()):,}  "
              f"median volume_z: bar -1 {t.loc[-1, 'volume_z_median']:+.2f}, "
              f"bar 0 {t.loc[0, 'volume_z_median']:+.2f}  "
              f"share > threshold: bar -1 {t.loc[-1, 'share_above_threshold']:.1%}, "
              f"bar 0 {t.loc[0, 'share_above_threshold']:.1%}")
    print(f"all hours    median volume_z {ref['volume_z_median']:+.2f}, "
          f"share > threshold {ref['share_above_threshold']:.1%}")
    print("wrote", out)


if __name__ == "__main__":
    main()
