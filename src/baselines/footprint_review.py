"""FootprintNet beside the classical detectors — validation, or the holdout once.

Every row reaches the metrics through `Baseline.predict` on one frame, and
every table is written twice: all hours, and closing-bar windows only
(`eval.same_hour`), where the clock is worth nothing. The closing-bar table
is the one to read; the all-hours table is kept so the clock's share stays
visible. The clock-only null sits in both.

Rows: always-quiet, random noise, clock-only, volume z-score, CUSUM, the
stock-selection control (`ticker_prior`: which stock, never which hour — a
detector that does not beat it on budgeted lift is mostly picking filing-prone
companies), and for every seed FootprintNet pre-trained, FootprintNet from
scratch, and the Stage 1 surprise score.

The holdout is opened ONCE
--------------------------
`--split holdout` evaluates `pipeline.holdout`'s events. It refuses to run
without `--reason`, records the opening in `holdout_opened.json` beside its
output, and refuses a second opening unless `--reopen` is passed — the same
discipline as the study's sealed test set, so the number it produces is a
number nobody tuned towards. Run validation as often as needed first.

Usage:
  python -m src.baselines.footprint_review                       # validation
  python -m src.baselines.footprint_review --split holdout --reason "final"
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.clock_only import ClockOnly
from src.baselines.compare import (comparison_table, conform, noise_seeds,
                                   train_positives)
from src.baselines.cusum import CUSUM
from src.baselines.footprint_net import FootprintDetector, knobs
from src.baselines.guide_review import (SLICES, configs, lead_rows,
                                        same_hour_rows, with_lift)
from src.baselines.random_noise import RandomNoise
from src.baselines.ticker_prior import TickerPrior
from src.baselines.volume_zscore import VolumeZScore
from src.eval.lift_ci import alerted_windows
from src.eval.same_hour import same_hour_view
from src.eval.within_stock import within_stock_table
from src.pipeline import holdout as H
from src.pipeline.evalset import build_eval_frame, split_bounds
from src.utils.config import load_config
from src.utils.timeutils import ts_to_iso, utc_now_ts

OPENED = "holdout_opened.json"


def footprint_rows(cfg: dict, runs_dir: Path, seeds: list[int]):
    for s in seeds:
        for kind, mode, run in (("pretrained", "detect", f"s{s}-pretrained"),
                                ("scratch", "detect", f"s{s}-scratch"),
                                ("surprise", "surprise", f"s{s}-pretrain")):
            path = runs_dir / run
            if (path / "manifest.json").exists():
                name = f"footprint_{kind}[s{s}]"
                yield f"FootprintNet ({kind})", FootprintDetector(cfg, path, mode, name)
            else:
                print(f"  (missing {path}, row skipped)")


def detectors(cfg_off: dict, cfg_on: dict, seeds: list[int], conn=None):
    yield "null", AlwaysQuiet(cfg_off)
    for s in noise_seeds(cfg_off):
        yield "null", RandomNoise(cfg_off, seed=s)
    yield "null", ClockOnly(cfg_off)
    yield "pure technical", VolumeZScore(cfg_off)
    yield "pure technical", CUSUM(cfg_off)
    yield "stock only", TickerPrior(cfg_off).fit(train_positives(cfg_off), conn=conn)
    yield from footprint_rows(cfg_on, Path(knobs(cfg_on)["runs_dir"]), seeds)


def inside_holdout(frame: pd.DataFrame, lo: int) -> pd.DataFrame:
    """Drop every window holding a bar before the holdout starts.

    An early-August filing's look-back reaches into late July, which is the
    study's sealed test period (`split.assert_not_test` refuses it). Dropping
    the window, not trimming it, keeps every scored window the same shape as
    on validation.
    """
    early = frame.loc[frame["ts_utc"].astype("int64") < lo, "window_id"].unique()
    if len(early):
        gone = frame[frame["window_id"].isin(early)]
        n_pos = gone.loc[gone["t0_utc"].notna(), "window_id"].nunique()
        print(f"  holdout: dropped {len(early):,} window(s) reaching before the "
              f"holdout start, {n_pos:,} of them positive")
    return frame[~frame["window_id"].isin(early)].reset_index(drop=True)


def open_holdout(out_dir: Path, reason: str | None, reopen: bool) -> None:
    marker = out_dir / OPENED
    if not reason:
        raise SystemExit("--split holdout needs --reason: the holdout is opened "
                         "once, and the reason is recorded")
    if marker.exists() and not reopen:
        prev = json.loads(marker.read_text())
        raise SystemExit(f"the holdout was already opened at {prev['opened_at']} "
                         f"({prev['reason']!r}). Pass --reopen only if you mean "
                         f"to spend it again, and say why in --reason")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = json.loads(marker.read_text())["history"] if marker.exists() else []
    entry = {"opened_at": ts_to_iso(utc_now_ts()), "reason": reason}
    marker.write_text(json.dumps({**entry, "history": history + [entry]}, indent=2))


def run(cfg: dict, conn, split: str, out_dir: Path, seeds: list[int],
        tickers: list[str] | None = None,
        save_windows: bool = False) -> dict[str, pd.DataFrame]:
    cfg_off, cfg_on = configs(cfg)
    variant = "news_adjusted"
    if split == "holdout":
        lo, hi = H.holdout_window(cfg)
        events = pd.read_parquet(H.out_path(cfg))
        frame = build_eval_frame(cfg_on, conn, lo, hi, tickers=tickers,
                                 t0_variant=variant, events=events)
        frame = inside_holdout(frame, lo)
    else:
        lo, hi = split_bounds(cfg_on, split, conn=conn)
        frame = build_eval_frame(cfg_on, conn, lo, hi, tickers=tickers,
                                 t0_variant=variant)
    frame = conform(frame)

    comp, comp_same, within, within_same, lead, windows = [], [], [], [], [], []
    for family, model in detectors(cfg_off, cfg_on, seeds, conn=conn):
        started = time.time()
        pred = model.predict(frame, threshold=float("inf"), conn=conn,
                             context=f"footprint_review/{model.name}")
        comp.append(comparison_table(cfg_off, {model.name: pred}, variant,
                                     models={model.name: model}).assign(family=family))
        within.append(within_stock_table(cfg, {model.name: pred}, variant)
                      .assign(family=family))
        c, w = same_hour_rows(cfg, cfg_off, model, pred, variant, family)
        comp_same.append(c)
        within_same.append(w)
        lead.extend(lead_rows(model.name, pred))
        if save_windows:
            windows.append(alerted_windows(model.name, same_hour_view(pred, cfg)))
        del pred
        print(f"  {model.name:32s} {time.time() - started:5.0f}s")

    out = {"comparison": with_lift(pd.concat(comp, ignore_index=True)),
           "comparison_same_hour": with_lift(pd.concat(comp_same, ignore_index=True)),
           "within_stock": pd.concat(within, ignore_index=True),
           "within_stock_same_hour": pd.concat(within_same, ignore_index=True),
           "lead_time": pd.DataFrame(lead)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    if windows:
        # Per-window operating points on closing-bar windows, for `eval.lift_ci`.
        pd.concat(windows, ignore_index=True).to_parquet(
            out_dir / "windows_same_hour.parquet", index=False)
    return out


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", choices=["val", "holdout"])
    ap.add_argument("--reason", default=None, help="required for --split holdout")
    ap.add_argument("--reopen", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-windows", action="store_true",
                    help="also write windows_same_hour.parquet for eval.lift_ci")
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="first N in-universe tickers only (smoke run)")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    seeds = args.seeds or list(knobs(cfg)["seeds"])
    out_dir = Path(args.out_dir or Path(cfg["paths"]["processed"])
                   / "footprint-net" / args.split)
    if args.split == "holdout" and not args.limit_tickers:
        open_holdout(out_dir, args.reason, args.reopen)
    tickers = None
    if args.limit_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (args.limit_tickers,))]
    started = time.time()
    out = run(cfg, conn, args.split, out_dir, seeds, tickers=tickers,
              save_windows=args.save_windows)

    for key, label in (("comparison_same_hour", "CLOSING-BAR WINDOWS (read this)"),
                       ("comparison", "all hours (clock included)")):
        comp = out[key]
        for s in SLICES[1:]:
            view = comp[comp["slice"] == s].sort_values("lift", ascending=False)
            print(f"\n-- {args.split}, {s}, {label} --")
            print(view[["baseline", "precision", "lift", "recall",
                        "median_lead_trading_h"]].to_string(
                            index=False, float_format=lambda v: f"{v:.4f}"))
    w = out["within_stock_same_hour"]
    print("\n-- within-stock AUC, closing-bar windows --")
    print(w[w["slice"] != "all"][["baseline", "slice", "auc", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_dir}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
