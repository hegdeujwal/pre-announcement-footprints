"""Scores for the discovery questions — scored once per split, saved, analysed.

Added 2026-10-23. FootprintNet's earnings result finds a date the calendar
already knows, and its surprise-filing result is a null. Three questions
decide what, if anything, a learned detector DISCOVERS here:

1. **Earnings leak** (`eval.earnings_leak`). The date is public, the result
   is not. Does a pre-earnings score that is unusual for that stock predict
   the size or the direction of the move that follows?
2. **Learning curve** (`eval.learning_curve`, validation only). Would more
   labelled filings help? Scratch runs on nested shares of the train filings.
3. **Item types** (`eval.by_item`). Does one surprise type carry a footprint
   the ten-type mix hides?

One pass scores every detector on the split's frame and writes:

* `scores.parquet` — every scored bar (a positive's final bar before t0, a
  quiet window's only bar) per detector, with whether it is a closing bar;
* `earnings.parquet` — every earnings 8-K in the split with a measured move,
  material or not, with the signed move, the controls known before t0, and
  the bar its score is read at.

then runs the three analyses on those files. Re-running an analysis never
re-scores (`--analyse-only`).

The holdout is opened under the same record as `footprint_review`: a
`--reason` is required and the opening is appended to that review's
`holdout_opened.json`, the single history of who looked at it and why.

Usage:
  python -m src.baselines.discovery_review                     # validation
  python -m src.baselines.discovery_review --split holdout --reason "..."
  python -m src.baselines.discovery_review --analyse-only
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.compare import conform
from src.baselines.cusum import CUSUM
from src.baselines.footprint_net import FootprintDetector, knobs
from src.baselines.footprint_review import OPENED, inside_holdout, open_holdout
from src.baselines.footprint_train import run_name
from src.baselines.guide_review import configs
from src.baselines.volume_zscore import VolumeZScore
from src.eval.same_hour import scored_bars
from src.pipeline import holdout as H
from src.pipeline.evalset import build_eval_frame, split_bounds
from src.pipeline.split import boundaries
from src.utils.config import load_config
from src.utils.timeutils import get_market_calendar, seconds_to_session_close

DAY = 86400
SCORE_COLUMNS = ["detector", "window_id", "ticker", "ts_utc", "score",
                 "is_positive", "t0_utc", "is_scheduled", "item_code", "closing"]


def knobs_of(cfg: dict) -> dict:
    return cfg["eval"]["discovery"]


# -- detectors -------------------------------------------------------------

def detectors(cfg_off: dict, cfg_on: dict, seeds: list[int], split: str):
    """The classical pair, FootprintNet per seed, and (validation only) every
    learning-curve share. A missing run is reported, never silently dropped."""
    yield VolumeZScore(cfg_off)
    yield CUSUM(cfg_off)
    runs = Path(knobs(cfg_on)["runs_dir"])
    wanted = [(s, kind, 1.0) for s in seeds for kind in ("pretrained", "scratch")]
    if split == "val":
        lc = knobs(cfg_on)["learning_curve"]
        wanted += [(s, lc["init"], f) for s in seeds for f in lc["fractions"]]
    for seed, kind, fraction in wanted:
        path = runs / run_name(seed, kind, fraction)
        if not (path / "manifest.json").exists():
            print(f"  (missing {path}, skipped)")
            continue
        tag = kind if fraction == 1 else f"{kind}_f{round(fraction * 100):d}"
        yield FootprintDetector(cfg_on, path, "detect", f"footprint_{tag}[s{seed}]")


def closing_flags(ts: np.ndarray, cfg: dict) -> np.ndarray:
    """True where the bar starts within `eval.same_hour` of its session close —
    the same rule as `same_hour.closing_bar_windows`."""
    cal = get_market_calendar(cfg["market"]["calendar"])
    to_close = seconds_to_session_close(np.asarray(ts, dtype="int64"), cal, strict=False)
    return to_close <= int(cfg["eval"]["same_hour"]["max_seconds_to_close"])


def score_rows(name: str, pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per window at the bar the metrics score."""
    bars = scored_bars(pred)
    positive = bars["t0_utc"].notna()
    return pd.DataFrame({
        "detector": name,
        # Quiet ids are rebuilt from (ticker, ts) on read; storing 300k
        # strings per detector buys nothing.
        "window_id": bars["window_id"].where(positive, None).astype("string"),
        "ticker": bars["ticker"].astype("string"),
        "ts_utc": bars["ts_utc"].astype("int64"),
        "score": bars["score"].astype("float32"),
        "is_positive": positive.to_numpy(),
        "t0_utc": bars["t0_utc"].astype("Int64"),
        "is_scheduled": bars["is_scheduled"].astype("boolean"),
        "item_code": bars["item_code"].astype("string"),
        "closing": closing_flags(bars["ts_utc"].to_numpy(), cfg),
    }).reset_index(drop=True)


# -- earnings events -------------------------------------------------------

def study_events(conn, cfg: dict) -> pd.DataFrame:
    """Earnings with a measured move in TRAIN and VALIDATION only — the sealed
    test period is never read, not even for a control."""
    _, val_end = boundaries(cfg)
    return pd.read_sql_query(
        "SELECT event_id, ticker, t0_utc, is_scheduled, abs_return, exclude_reason "
        "FROM events WHERE abs_return IS NOT NULL AND t0_utc < ?", conn,
        params=(val_end,))


def earnings_events(cfg: dict, conn, split: str, lo: int, hi: int) -> pd.DataFrame:
    """Every earnings 8-K with t0 in [lo, hi) and a measured move, material or
    not, with the controls an analyst could know before t0:

    * `prior_abs_move` — median |move| of the same stock's EARLIER earnings
      (train, validation and, for the holdout, earlier holdout ones);
    * `n_prior` — how many it rests on.

    `move` is the signed benchmark-relative move `materiality.measure`
    returns; its absolute value must equal the stored `abs_return`, which is
    checked rather than assumed.
    """
    from src.pipeline.materiality import BarSeries, interval_seconds, measure

    past = study_events(conn, cfg)
    if split == "holdout":
        held = pd.read_parquet(H.out_path(cfg))
        held = held[held["abs_return"].notna()]
        past = pd.concat([past, held[past.columns]], ignore_index=True)
    earn = past[past["is_scheduled"].astype(int) == 1]
    earn = earn[earn["exclude_reason"].isna() | (earn["exclude_reason"] == "immaterial")]
    here = earn[(earn["t0_utc"] >= lo) & (earn["t0_utc"] < hi)].copy()
    if here.empty:
        raise SystemExit(f"no earnings events with a measured move in {split}")

    prior = []
    for r in here.itertuples():
        before = earn[(earn["ticker"] == r.ticker) & (earn["t0_utc"] < r.t0_utc)]
        prior.append((before["abs_return"].median() if len(before) else np.nan,
                      len(before)))
    here["prior_abs_move"] = [p[0] for p in prior]
    here["n_prior"] = [p[1] for p in prior]

    interval = cfg["market"]["interval"]
    names = sorted(set(here["ticker"]) | {cfg["market"]["benchmark"]})
    marks = ",".join("?" * len(names))
    rows: dict[str, list] = {}
    for t, ts, o, c in conn.execute(
            f"SELECT ticker, ts_utc, open, close FROM bars WHERE interval = ? "
            f"AND ticker IN ({marks}) ORDER BY ticker, ts_utc", (interval, *names)):
        rows.setdefault(t, []).append((ts, o, c))
    series = {t: BarSeries(v, interval_seconds(interval)) for t, v in rows.items()}
    here["move"] = [measure(cfg, series, r.ticker, int(r.t0_utc)).adjusted
                    for r in here.itertuples()]
    gap = (here["move"].abs() - here["abs_return"]).abs()
    if gap.max() > 1e-9:
        raise ValueError(f"signed move disagrees with stored abs_return for "
                         f"{int((gap > 1e-9).sum())} event(s)")
    here["is_material"] = here["exclude_reason"].isna()
    return here.drop(columns=["exclude_reason", "is_scheduled"]).reset_index(drop=True)


def attach_bars(events: pd.DataFrame, frame: pd.DataFrame, n_bars: int) -> pd.DataFrame:
    """Each event's last frame bar before t0 (`score_ts`) and the benchmark-
    relative drift over the `n_bars` bars ending there. Both read only bars
    strictly before t0. An event with fewer bars in the frame is dropped."""
    by_ticker = {t: g.sort_values("ts_utc") for t, g in
                 frame[["ticker", "ts_utc", "ret_rel_1h", "volatility"]]
                 .groupby("ticker", sort=False)}
    out = []
    for r in events.itertuples():
        g = by_ticker.get(r.ticker)
        if g is None:
            continue
        ts = g["ts_utc"].to_numpy(dtype="int64")
        end = int(np.searchsorted(ts, int(r.t0_utc), side="left"))
        if end < n_bars:
            continue
        last = g.iloc[end - n_bars:end]
        out.append({"event_id": r.event_id, "score_ts": int(ts[end - 1]),
                    "pre_drift": float(np.nansum(last["ret_rel_1h"].to_numpy(dtype="float64"))),
                    "volatility": float(last["volatility"].iloc[-1])})
    if not out:
        return events.iloc[:0].assign(score_ts=pd.Series(dtype="int64"),
                                      pre_drift=pd.Series(dtype="float64"),
                                      volatility=pd.Series(dtype="float64"))
    return events.merge(pd.DataFrame(out), on="event_id", how="inner")


# -- run -------------------------------------------------------------------

def out_dir(cfg: dict, split: str) -> Path:
    return Path(cfg["paths"]["processed"]) / knobs_of(cfg)["out"] / split


def score(cfg: dict, conn, split: str, seeds: list[int]) -> Path:
    cfg_off, cfg_on = configs(cfg)
    if split == "holdout":
        lo, hi = H.holdout_window(cfg)
        frame = build_eval_frame(cfg_on, conn, lo, hi, t0_variant="news_adjusted",
                                 events=pd.read_parquet(H.out_path(cfg)))
        frame = inside_holdout(frame, lo)
    else:
        lo, hi = split_bounds(cfg_on, split, conn=conn)
        frame = build_eval_frame(cfg_on, conn, lo, hi, t0_variant="news_adjusted")
    frame = conform(frame)

    out = out_dir(cfg, split)
    out.mkdir(parents=True, exist_ok=True)
    events = attach_bars(earnings_events(cfg, conn, split, lo, hi), frame,
                         int(knobs_of(cfg)["pre_drift_bars"]))
    events.to_parquet(out / "earnings.parquet", index=False)
    print(f"  {len(events):,} earnings events "
          f"({int(events['is_material'].sum()):,} material)")

    parts = []
    for model in detectors(cfg_off, cfg_on, seeds, split):
        started = time.time()
        pred = model.predict(frame, threshold=float("inf"), conn=conn,
                             context=f"discovery_review/{model.name}")
        parts.append(score_rows(model.name, pred, cfg))
        del pred
        print(f"  {model.name:32s} {time.time() - started:5.0f}s")
    pd.concat(parts, ignore_index=True).to_parquet(out / "scores.parquet", index=False)
    return out


def analyse(cfg: dict, split: str) -> None:
    from src.eval import by_item, earnings_leak, learning_curve

    out = out_dir(cfg, split)
    scores = pd.read_parquet(out / "scores.parquet")
    events = pd.read_parquet(out / "earnings.parquet")
    k = knobs_of(cfg)
    leak = earnings_leak.leak_table(scores, events, k)
    leak.to_csv(out / "earnings_leak.csv", index=False)
    items = by_item.item_table(scores, cfg)
    items.to_csv(out / "by_item.csv", index=False)
    print("\n-- earnings: does an unusual pre-earnings score predict the result? --")
    print(leak.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n-- surprise filings by item type, closing-bar windows --")
    print(items.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if split == "val":
        curve = learning_curve.curve_table(scores, cfg)
        curve.to_csv(out / "learning_curve.csv", index=False)
        print("\n-- learning curve (scratch, validation, closing-bar windows) --")
        print(learning_curve.summary(curve).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "holdout"])
    ap.add_argument("--reason", default=None, help="required for --split holdout")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--analyse-only", action="store_true",
                    help="re-run the analyses on the saved scores")
    args = ap.parse_args()

    cfg = load_config()
    if not args.analyse_only:
        conn = db.get_conn(cfg["paths"]["db"])
        if args.split == "holdout":
            marker = Path(cfg["paths"]["processed"]) / "footprint-net" / "holdout"
            if not (marker / OPENED).exists():
                raise SystemExit("the holdout has no opening record yet — run "
                                 "footprint_review --split holdout first")
            open_holdout(marker, args.reason, reopen=True)
        started = time.time()
        score(cfg, conn, args.split, args.seeds or list(knobs(cfg)["seeds"]))
        print(f"scored in {time.time() - started:.0f}s")
    analyse(cfg, args.split)


if __name__ == "__main__":
    main()
