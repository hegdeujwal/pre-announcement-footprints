"""Do the detectors catch the BIG filings? Results split by how much the stock moved.

Added 2026-10-15 for the guide's "flagging filter according to the impact"
point. Not every usable 8-K matters equally: one that moves the stock 20% is
the kind of filing a surveillance desk cares about, one that moves it 3% much
less so. Two questions, both answered on the answer-key side:

1. **Recall by impact.** Positives are split into low / mid / high impact by
   `events.abs_return` — the post-announcement move the materiality filter
   already stores. Each slice keeps every quiet window, as `report.slice_frames`
   does, so precision stays comparable to the headline. The cut points are the
   tercile edges of TRAINING events' moves, fixed before validation is read.
2. **Does a louder alert mean a bigger move?** The Spearman rank correlation
   between a detector's score on a positive's final bar and that event's move.
   If it is clearly above zero, the alert score itself is a usable impact
   filter: the top of the queue is where the big filings are.

`abs_return` is measured AFTER t0. That is correct here and nowhere else: like
`materiality.py` this is grading, not detecting, and no detector reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.eval.metrics import precision_at_alert_budget, window_summary
from src.eval.report import evaluate

#: Bucket names, low to high.
BUCKETS = ("impact low", "impact mid", "impact high")


def event_impacts(conn) -> pd.Series:
    """abs_return per usable event_id."""
    rows = conn.execute("SELECT event_id, abs_return FROM events "
                        "WHERE usable = 1 AND abs_return IS NOT NULL").fetchall()
    return pd.Series({r[0]: float(r[1]) for r in rows}, name="abs_return",
                     dtype="float64")


def impact_edges(cfg: dict, conn, impacts: pd.Series) -> np.ndarray:
    """Tercile edges from TRAIN events only, so validation cannot shape them."""
    from src.pipeline.split import boundaries

    train_end, _ = boundaries(cfg)
    ids = [r[0] for r in conn.execute(
        "SELECT event_id FROM events WHERE usable = 1 AND t0_utc < ?",
        (train_end,))]
    train = impacts.reindex(ids).dropna()
    if train.empty:
        raise ValueError("no training events with abs_return to set impact edges")
    q = cfg["eval"]["impact"]["quantiles"]
    return np.quantile(train.to_numpy(), q)


def bucket_of(values: pd.Series, edges: np.ndarray) -> pd.Series:
    idx = np.searchsorted(edges, values.to_numpy(), side="right")
    return pd.Series([BUCKETS[i] for i in idx], index=values.index)


def impact_slices(pred: pd.DataFrame, buckets: pd.Series) -> dict[str, pd.DataFrame]:
    """Each bucket's positives plus every quiet window."""
    quiet = pred["t0_utc"].isna()
    wid = pred["window_id"].astype("object")
    out = {}
    for b in BUCKETS:
        ids = set(buckets.index[buckets == b])
        out[b] = pred[quiet | wid.isin(ids)]
    return out


def impact_rows(name: str, pred: pd.DataFrame, impacts: pd.Series,
                edges: np.ndarray) -> list[dict]:
    """Per-bucket metrics for one detector, at its whole-frame operating point.

    Split three ways — all, scheduled, unscheduled — because AGENTS.md rule 7
    applies here as everywhere: a big earnings move and a big surprise move
    are different questions.
    """
    threshold = precision_at_alert_budget(pred).threshold
    positive = pred["t0_utc"].notna()
    sched = (pred.loc[positive].groupby("window_id", sort=False)["is_scheduled"]
             .first().astype(bool))
    windows = window_summary(pred)

    rows = []
    for schedule in ("all", "scheduled", "unscheduled"):
        ids = sched.index if schedule == "all" else sched.index[
            sched == (schedule == "scheduled")]
        part = pred[~positive | pred["window_id"].isin(set(ids))]
        present = impacts.reindex(ids.astype("object")).dropna()
        buckets = bucket_of(present, edges)
        for b, sliced in impact_slices(part, buckets).items():
            if not sliced["t0_utc"].notna().any():
                continue
            m = evaluate(sliced, threshold=threshold)
            rows.append({"baseline": name, "schedule": schedule, "slice": b,
                         "n_positive": m["n_positive"],
                         "precision": m["precision"], "recall": m["recall"],
                         "median_lead_trading_h": m["median_lead_trading_h"]})

        # Spearman over these positives: final-bar score vs the move after.
        pos = windows[windows["is_positive"]]
        pos = pos[pos.index.isin(set(ids))]
        joined = pos.join(impacts.rename("abs_return"), how="inner")
        rho, p = (spearmanr(joined["peak_score"], joined["abs_return"])
                  if len(joined) > 2 else (np.nan, np.nan))
        rows.append({"baseline": name, "schedule": schedule,
                     "slice": "spearman score~impact",
                     "n_positive": len(joined), "precision": np.nan,
                     "recall": np.nan, "median_lead_trading_h": np.nan,
                     "spearman_rho": float(rho), "spearman_p": float(p)})
    return rows


def add_lift(table: pd.DataFrame, floor_name: str = "always_quiet") -> pd.DataFrame:
    """Precision over the always-quiet floor of the SAME bucket and schedule."""
    floor = (table[table["baseline"] == floor_name]
             .set_index(["schedule", "slice"])["precision"])
    table = table.copy()
    table["lift"] = [p / floor[k] if k in floor.index and floor[k] else np.nan
                     for p, k in zip(table["precision"],
                                     zip(table["schedule"], table["slice"]))]
    return table
