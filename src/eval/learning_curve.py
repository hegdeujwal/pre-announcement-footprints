"""Would more labelled filings help? FootprintNet trained on nested shares.

Added 2026-10-23. `footprint_train --learning-curve` fine-tunes from scratch
on 10–75% of the train filings; the 100% point is the existing scratch run.
Each is scored on validation, closing-bar windows only, per slice:

* lift at the budget threshold of the closing-bar view, over the slice's base
  rate (the always-quiet row's precision);
* within-stock AUC (`eval.within_stock`) — whether the model knows WHEN.

Read the shape, not one point: a curve still climbing at 100% says more
filings would help; a flat one says the amount of labelled data is not what
limits it. Five seeds per share, every one reported.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.eval.by_item import as_predictions, slice_lift
from src.eval.metrics import precision_at_alert_budget
from src.eval.within_stock import stock_pairs

#: footprint_scratch[s42] -> 100%, footprint_scratch_f25[s42] -> 25%.
NAME = re.compile(r"^footprint_scratch(?:_f(\d+))?\[s(\d+)\]$")


def parse(name: str) -> tuple[float, int] | None:
    m = NAME.match(name)
    if not m:
        return None
    return (int(m.group(1)) / 100 if m.group(1) else 1.0), int(m.group(2))


def curve_table(scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    for name, d in scores.groupby("detector", sort=False):
        point = parse(name)
        if point is None and name != "volume_zscore":
            continue
        fraction, seed = point if point else (np.nan, -1)
        pred = as_predictions(d)
        threshold = precision_at_alert_budget(pred).threshold
        positive = pred["t0_utc"].notna()
        windows = pd.DataFrame({"ticker": pred["ticker"], "peak_score": pred["score"],
                                "is_positive": positive, "is_scheduled": pred["is_scheduled"]})
        for sl in ("scheduled", "unscheduled"):
            mask = positive & (pred["is_scheduled"] == (sl == "scheduled"))
            pairs = stock_pairs(windows, sl)
            rows.append({"detector": name, "fraction": fraction, "seed": seed,
                         "slice": sl, "lift": slice_lift(pred, mask, threshold),
                         "auc": float(pairs["u"].sum() / pairs["pairs"].sum())})
    return pd.DataFrame(rows)


def summary(curve: pd.DataFrame) -> pd.DataFrame:
    """Per share and slice: the seed mean and range."""
    fp = curve[curve["seed"] >= 0]
    out = (fp.groupby(["slice", "fraction"])
           .agg(lift_mean=("lift", "mean"), lift_min=("lift", "min"),
                lift_max=("lift", "max"), auc_mean=("auc", "mean"),
                auc_min=("auc", "min"), auc_max=("auc", "max"),
                n_seeds=("seed", "nunique"))
           .reset_index())
    ref = curve[curve["detector"] == "volume_zscore"].set_index("slice")
    out["volume_zscore_lift"] = out["slice"].map(ref["lift"])
    out["volume_zscore_auc"] = out["slice"].map(ref["auc"])
    return out
