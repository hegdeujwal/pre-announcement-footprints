"""Surprise filings one item type at a time — is a footprint hiding in the mix?

Added 2026-10-23. The unscheduled slice pools ten 8-K item types, from new
contracts (1.01) to bankruptcies (1.03). A type that leaks — completed
mergers are the usual suspect — could be diluted to nothing by nine that do
not. Each type is scored on closing-bar windows only (`eval.same_hour`), so
the clock is worth nothing:

* **within-stock AUC** — the type's positives against the same stock's quiet
  closing bars, stocks bootstrapped (`eval.within_stock`);
* **lift** at the detector's budget threshold on the closing-bar view, over
  that slice's own base rate (the always-quiet row's precision).

Ten types tested means about one false "footprint" at 95% by chance alone,
so a type counts only if its AUC clears 0.5 after a Holm correction across
the types tested for that detector. A filing with several items counts once
under each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.metrics import precision_at_alert_budget
from src.eval.report import evaluate
from src.eval.within_stock import stock_pairs

#: Learning-curve shares are scored on validation for their own question,
#: not this one.
SHARE_TAG = "_f"


def as_predictions(d: pd.DataFrame, closing_only: bool = True) -> pd.DataFrame:
    """One detector's saved scored bars back in the prediction contract."""
    if closing_only:
        d = d[d["closing"]]
    wid = d["window_id"].astype("object")
    quiet = ~d["is_positive"].to_numpy()
    wid = wid.where(~quiet, "bar:" + d["ticker"].astype(str) + ":" + d["ts_utc"].astype(str))
    return pd.DataFrame({
        "window_id": wid.astype("string"), "ticker": d["ticker"].astype("string"),
        "ts_utc": d["ts_utc"].astype("Int64"), "t0_utc": d["t0_utc"].astype("Int64"),
        "score": d["score"].astype("float64"), "action": "WAIT",
        "is_scheduled": d["is_scheduled"].astype("boolean"),
        "item_code": d["item_code"].astype("string")}).reset_index(drop=True)


def slice_lift(pred: pd.DataFrame, keep_positive: pd.Series, threshold: float) -> float:
    """Precision at `threshold` over the base rate, positives in `keep_positive`
    plus every quiet window."""
    sliced = pred[pred["t0_utc"].isna() | keep_positive]
    n_pos = int(sliced["t0_utc"].notna().sum())
    if n_pos == 0:
        return float("nan")
    m = evaluate(sliced, threshold=threshold)
    return float(m["precision"] / (n_pos / len(sliced)))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values (step-down, monotone)."""
    p = np.asarray(p, dtype="float64")
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def has_item(codes: pd.Series, item: str) -> pd.Series:
    return codes.fillna("").str.split(",").apply(
        lambda parts: item in [c.strip() for c in parts])


def item_table(scores: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    k = cfg["eval"]["discovery"]
    items = list(cfg["items"]["unscheduled_focus"])
    tail = (1 - float(k["ci"])) / 2 * 100
    rows = []
    for name, d in scores.groupby("detector", sort=False):
        if SHARE_TAG in name:
            continue
        pred = as_predictions(d)
        threshold = precision_at_alert_budget(pred).threshold
        positive = pred["t0_utc"].notna()
        surprise = positive & (pred["is_scheduled"] == False).fillna(False)  # noqa: E712
        windows = pd.DataFrame({"ticker": pred["ticker"], "peak_score": pred["score"],
                                "is_positive": positive, "is_scheduled": pred["is_scheduled"]})
        found = []
        for item in ["all unscheduled"] + items:
            mask = surprise if item == "all unscheduled" else surprise & has_item(
                pred["item_code"], item)
            n = int(mask.sum())
            row = {"detector": name, "item": item, "n_events": n}
            if n < int(k["min_item_events"]):
                rows.append({**row, "tested": False})
                continue
            pairs = stock_pairs(windows[mask | ~positive], "all")
            u, nn = pairs["u"].to_numpy(), pairs["pairs"].to_numpy()
            rng = np.random.default_rng(int(k["seed"]))
            idx = rng.integers(0, len(pairs), size=(int(k["bootstrap"]), len(pairs)))
            boots = u[idx].sum(axis=1) / nn[idx].sum(axis=1)
            lo, hi = np.percentile(boots, [tail, 100 - tail])
            p = 2 * min((boots <= 0.5).mean(), (boots >= 0.5).mean())
            row.update(tested=item != "all unscheduled", auc=float(u.sum() / nn.sum()),
                       auc_lo=float(lo), auc_hi=float(hi), p=float(min(p, 1.0)),
                       n_stocks=len(pairs),
                       lift=slice_lift(pred, mask, threshold))
            found.append(row) if row["tested"] else rows.append(row)
        if found:
            adj = holm(np.array([r["p"] for r in found]))
            for r, a in zip(found, adj):
                r["p_holm"] = float(a)
                r["footprint"] = bool(a < 1 - float(k["ci"]) and r["auc"] > 0.5)
            rows.extend(found)
    return pd.DataFrame(rows)
