"""Is detector A's budgeted lift really above detector B's? A paired interval.

The lift tables give one number per detector. On a slice with ~100 positives,
6.6x against 4.6x can be luck, and the table cannot say. This answers it.

What is resampled
-----------------
STOCKS, with replacement — the same unit as `within_stock`, because one
stock's windows are not independent draws. Both detectors see the SAME
resampled stocks in every draw (paired), so the interval is on the gap between
them rather than on two noisy numbers compared by eye.

What is held fixed
------------------
Each detector's operating point: which windows it alerted at the budget on the
full frame (`metrics.precision_at_alert_budget`, the threshold the comparison
table used). A draw only re-weights stocks; it never re-ranks. Within one slice
both detectors share a floor (always-quiet's precision is the slice's base
rate), so the lift ratio IS the precision ratio, and is reported as such.

A slice keeps every quiet window, as `report.slice_frames` does: precision on
"scheduled" is scheduled hits over every alert the detector spent on quiet or
scheduled windows.

Several seeds of one model are pooled into a GROUP (`footprint_scratch[s42]`
.. `[s46]` -> `footprint_scratch`): its precision in a draw is the mean over its
seeds, so the group row answers "this model, whichever seed", and a single
lucky seed cannot carry it.

Usage:
  python -m src.eval.lift_ci data/processed/footprint-net/holdout/windows_same_hour.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import precision_at_alert_budget
from src.eval.within_stock import windows_with_schedule
from src.utils.config import load_config

#: Slices with a positive class of their own. AGENTS.md rule 7.
SLICES = ("scheduled", "unscheduled")
WINDOW_COLUMNS = ["detector", "window_id", "ticker", "is_positive",
                  "is_scheduled", "peak_score", "alerted"]


def group_of(detector: str) -> str:
    """`footprint_scratch[s42]` -> `footprint_scratch`; others unchanged."""
    return re.sub(r"\[s\d+\]$", "", detector)


def alerted_windows(name: str, pred: pd.DataFrame) -> pd.DataFrame:
    """One row per window: its scored bar's peak, and whether the budget alerted it."""
    threshold = precision_at_alert_budget(pred).threshold
    w = windows_with_schedule(pred)
    return pd.DataFrame({
        "detector": name, "window_id": w.index.astype("string"),
        "ticker": w["ticker"].astype("string"),
        "is_positive": w["is_positive"].astype(bool),
        "is_scheduled": w["is_scheduled"].astype("boolean"),
        "peak_score": w["peak_score"].astype("float64"),
        "alerted": (w["peak_score"] >= threshold).fillna(False).astype(bool),
    }, columns=WINDOW_COLUMNS).reset_index(drop=True)


def stock_counts(windows: pd.DataFrame, slice_name: str,
                 tickers: pd.Index) -> pd.DataFrame:
    """Per detector x stock: alerts spent in the slice, and hits among them."""
    if slice_name not in SLICES:
        raise ValueError(f"unknown slice {slice_name!r}; expected one of {SLICES}")
    want = slice_name == "scheduled"
    in_slice = (~windows["is_positive"]
                | (windows["is_scheduled"].astype("boolean") == want).fillna(False))
    w = windows[in_slice & windows["alerted"]]
    hits = w["is_positive"].astype(int)
    g = (w.assign(hits=hits, alerts=1)
          .groupby(["detector", "ticker"], observed=True)[["alerts", "hits"]].sum())
    full = pd.MultiIndex.from_product([windows["detector"].unique(), tickers],
                                      names=["detector", "ticker"])
    return g.reindex(full, fill_value=0)


def paired_intervals(windows: pd.DataFrame, references: list[str], n_boot: int,
                     seed: int, ci: float = 0.95) -> pd.DataFrame:
    """Every detector group against every reference group, per slice."""
    tickers = pd.Index(sorted(windows["ticker"].unique()))
    rng = np.random.default_rng(seed)
    # One weight matrix for every slice and pair: draw b counts stock i w[b, i] times.
    weights = rng.multinomial(len(tickers), np.full(len(tickers), 1 / len(tickers)),
                              size=n_boot).astype("float64")
    groups: dict[str, list[str]] = {}
    for d in windows["detector"].unique():
        groups.setdefault(group_of(d), []).append(d)
    missing = [r for r in references if r not in groups]
    if missing:
        raise ValueError(f"reference detector(s) {missing} not in the windows file")
    tail = (1 - ci) / 2 * 100

    rows = []
    for slice_name in SLICES:
        counts = stock_counts(windows, slice_name, tickers)

        def precision(group: str) -> tuple[float, np.ndarray]:
            point, boots = [], []
            for d in groups[group]:
                c = counts.loc[d]
                a, h = c["alerts"].to_numpy("float64"), c["hits"].to_numpy("float64")
                point.append(h.sum() / a.sum() if a.sum() else np.nan)
                with np.errstate(invalid="ignore", divide="ignore"):
                    boots.append((weights @ h) / (weights @ a))
            return float(np.mean(point)), np.mean(boots, axis=0)

        cache = {g: precision(g) for g in groups}
        for ref in references:
            p_ref, b_ref = cache[ref]
            for g in groups:
                if g == ref:
                    continue
                p, b = cache[g]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = b / b_ref
                ok = np.isfinite(ratio)
                lo, hi = (np.percentile(ratio[ok], [tail, 100 - tail])
                          if ok.any() else (np.nan, np.nan))
                rows.append({
                    "slice": slice_name, "detector": g, "reference": ref,
                    "n_seeds": len(groups[g]), "precision": p,
                    "reference_precision": p_ref,
                    "lift_ratio": p / p_ref if p_ref else np.nan,
                    "ci_lo": float(lo), "ci_hi": float(hi),
                    # Share of draws in which the detector beat the reference.
                    "p_better": float((b[ok] > b_ref[ok]).mean()) if ok.any() else np.nan,
                    "n_boot": int(ok.sum()),
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("windows", help="windows parquet written by a review runner")
    ap.add_argument("--out", default=None, help="default: lift_ci.csv beside it")
    args = ap.parse_args()

    knobs = load_config()["eval"]["lift_ci"]
    windows = pd.read_parquet(args.windows)
    table = paired_intervals(windows, list(knobs["references"]),
                             n_boot=int(knobs["bootstrap"]), seed=int(knobs["seed"]),
                             ci=float(knobs["ci"]))
    out = Path(args.out or Path(args.windows).with_name("lift_ci.csv"))
    table.to_csv(out, index=False)
    for s in SLICES:
        view = table[table["slice"] == s].sort_values(["reference", "lift_ratio"],
                                                      ascending=[True, False])
        print(f"\n-- {s}: lift of detector / lift of reference, "
              f"{knobs['ci']:.0%} interval over stocks --")
        print(view[["detector", "reference", "lift_ratio", "ci_lo", "ci_hi",
                    "p_better"]].to_string(index=False,
                                           float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
