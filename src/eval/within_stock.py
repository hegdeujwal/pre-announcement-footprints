"""Within-stock AUC — does a detector know WHEN, once WHICH stock is removed?

The headline metric ranks every window in the frame against every other, so a
detector earns lift two ways: by picking the hours before a filing, and by
picking the companies that file often. Only the first is a footprint. The
`ticker_prior` baseline measured how much the second is worth on its own; this
measures the first on its own.

For each stock, its positive windows (the decision bar immediately before t0,
the same bar `metrics.window_summary` scores) are ranked against that SAME
stock's quiet windows. The pooled AUC is the share of (positive, quiet) pairs
from one stock that the detector orders correctly, ties counting half:

    AUC = sum over stocks of U_stock / sum over stocks of (n_pos * n_quiet)

0.5 means the detector carries no timing information beyond stock identity.
A stock with no positive or no quiet window contributes no pair and drops out
— it has nothing to say about timing.

The interval is a bootstrap over STOCKS, not windows: windows of one stock are
not independent draws, and resampling them would claim a precision the data
does not have.

AUC reads the whole ranking, the budget metric only its top ~3%. The two can
disagree, and both are reported for that reason.

Usage:
  python -m src.eval.within_stock                    # validation, news_adjusted
  python -m src.eval.within_stock --variant filing --out within.csv
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.eval.metrics import window_summary

#: Slices reported, in reading order. AGENTS.md rule 7.
SLICES = ("all", "scheduled", "unscheduled")


def stock_pairs(windows: pd.DataFrame, slice_name: str = "all") -> pd.DataFrame:
    """Mann-Whitney U and pair count per stock.

    `windows` is `metrics.window_summary` output plus an `is_scheduled` column
    (null for quiet windows). A slice filters the POSITIVES only and keeps every
    quiet window, for the reason `report.slice_frames` gives: a quiet window
    belongs to no event.
    """
    if slice_name not in SLICES:
        raise ValueError(f"unknown slice {slice_name!r}; expected one of {SLICES}")
    pos = windows["is_positive"].astype(bool)
    if slice_name != "all":
        want = slice_name == "scheduled"
        pos = pos & (windows["is_scheduled"].astype("boolean") == want).fillna(False)
    keep = pos | ~windows["is_positive"].astype(bool)

    rows = []
    for ticker, g in windows[keep].groupby("ticker", sort=True):
        p = pos.loc[g.index].to_numpy()
        n_pos, n_quiet = int(p.sum()), int((~p).sum())
        if n_pos == 0 or n_quiet == 0:
            continue
        ranks = rankdata(g["peak_score"].to_numpy())   # average ranks: ties = half
        u = float(ranks[p].sum() - n_pos * (n_pos + 1) / 2)
        rows.append((ticker, u, n_pos * n_quiet, n_pos))
    return pd.DataFrame(rows, columns=["ticker", "u", "pairs", "n_pos"])


def within_stock_auc(pairs: pd.DataFrame, n_boot: int, seed: int,
                     ci: float = 0.95) -> dict:
    """Pooled AUC with a bootstrap-over-stocks interval."""
    if pairs.empty:
        nan = float("nan")
        return {"auc": nan, "ci_lo": nan, "ci_hi": nan, "mean_stock_auc": nan,
                "n_stocks": 0, "n_positive": 0}
    u, n = pairs["u"].to_numpy(), pairs["pairs"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))
    boots = u[idx].sum(axis=1) / n[idx].sum(axis=1)
    tail = (1 - ci) / 2 * 100
    lo, hi = np.percentile(boots, [tail, 100 - tail])
    return {"auc": float(u.sum() / n.sum()), "ci_lo": float(lo),
            "ci_hi": float(hi),
            # Every stock weighted equally, beside the pair-weighted figure, so
            # a result carried by a few prolific filers is visible.
            "mean_stock_auc": float((u / n).mean()),
            "n_stocks": len(pairs), "n_positive": int(pairs["n_pos"].sum())}


def windows_with_schedule(predictions: pd.DataFrame) -> pd.DataFrame:
    """`window_summary` plus each window's `is_scheduled`."""
    windows = window_summary(predictions)
    sched = predictions.groupby("window_id", sort=False)["is_scheduled"].first()
    return windows.assign(is_scheduled=sched.reindex(windows.index).to_numpy())


def within_stock_table(cfg: dict, predictions: Mapping[str, pd.DataFrame],
                       variant: str) -> pd.DataFrame:
    """One row per (detector x slice)."""
    knobs = cfg["eval"]["within_stock"]
    rows = []
    for name, pred in predictions.items():
        windows = windows_with_schedule(pred)
        for slice_name in SLICES:
            rows.append({"baseline": name, "slice": slice_name,
                         "t0_variant": variant,
                         **within_stock_auc(stock_pairs(windows, slice_name),
                                            n_boot=int(knobs["bootstrap"]),
                                            seed=int(knobs["seed"]),
                                            ci=float(knobs["ci"]))})
    return pd.DataFrame(rows)


def main() -> None:
    from src import db
    from src.baselines.compare import conform, run_baselines
    from src.pipeline.evalset import build_eval_frame, split_bounds
    from src.utils.config import load_config

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val",
                    help="train | val | test. Defaults to val; test is sealed.")
    ap.add_argument("--variant", default="news_adjusted",
                    help="t0 variant to label the frame with")
    ap.add_argument("--out", default=None, help="write the table as CSV")
    ap.add_argument("--policy-run", action="append", default=None,
                    dest="policy_runs", help="a P6-03 run directory; repeatable")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    lo, hi = split_bounds(cfg, args.split, conn=conn)
    frame = conform(build_eval_frame(cfg, conn, lo, hi, t0_variant=args.variant))
    # Gradient boosting is skipped: it needs a model fit, and its budgeted
    # lift is already below the stock-only control.
    predictions, _ = run_baselines(cfg, conn, frame, skip_gb=True,
                                   policy_runs=args.policy_runs)
    table = within_stock_table(cfg, predictions, args.variant)
    print(table.sort_values(["slice", "auc"], ascending=[True, False])
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
