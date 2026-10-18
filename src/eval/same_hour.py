"""The same-hour view — every comparison re-scored where the clock is worthless.

Added 2026-10-18. `metrics.window_summary` scores a positive window by its
final bar before t0 and a quiet window by its single bar. Most 8-Ks land
after the close, so 84% of positives are scored on the session's last bar
while quiet bars spread across the whole day. Every detector that responds to
the busy closing bar collects lift for it, whether or not it sees a footprint
(`baselines.clock_only` measures how much).

This view keeps only the windows whose SCORED bar is the session's last bar —
positives and quiet alike — and hands the result to the unchanged metrics.
Inside it every window is judged at the same time of day, so knowing the time
of day is worth nothing: the clock-only row must fall to the floor, and any
lift left on another row is lift the clock cannot explain.

What it costs, stated: positives whose final bar is not the closing bar
(15.6% of them — mostly filings made during the session) leave the view, and
the budget is still 2 alerts per stock per month over far fewer quiet windows,
so absolute precision and recall are not comparable with the all-hours table.
Lift — precision over the always-quiet row of the SAME view — is.

Pure function of a prediction frame: no model, no pipeline import.
"""

from __future__ import annotations

import pandas as pd

from src.utils.timeutils import get_market_calendar, seconds_to_session_close

VIEWS = ("all_hours", "closing_bar")


def scored_bars(pred: pd.DataFrame) -> pd.DataFrame:
    """The bar `window_summary` scores for each window: a positive's last bar,
    a quiet window's only bar (its last bar too, so one rule covers both)."""
    ordered = pred.sort_values(["window_id", "ts_utc"], kind="mergesort")
    return ordered[~ordered["window_id"].duplicated(keep="last")]


def closing_bar_windows(pred: pd.DataFrame, cfg: dict) -> pd.Index:
    """Window ids whose scored bar starts within the configured distance of
    its session's close."""
    knobs = cfg["eval"]["same_hour"]
    limit = int(knobs["max_seconds_to_close"])
    bars = scored_bars(pred)
    cal = get_market_calendar(cfg["market"]["calendar"])
    # Off-session bars (synthetic frames only) are NaN and so never closing bars.
    to_close = seconds_to_session_close(bars["ts_utc"].astype("int64").to_numpy(),
                                        cal, strict=False)
    return pd.Index(bars.loc[to_close <= limit, "window_id"].unique())


def same_hour_view(pred: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """`pred` restricted to closing-bar windows. Refuses an empty result — a
    view with no positive or no quiet window has no lift to report."""
    keep = closing_bar_windows(pred, cfg)
    out = pred[pred["window_id"].isin(keep)]
    positive = out["t0_utc"].notna()
    if out.empty or positive.all() or not positive.any():
        raise ValueError(
            f"same_hour_view: {len(keep)} closing-bar window(s) leave "
            f"{int(positive.sum())} positive and {int((~positive).sum())} quiet "
            f"rows — the view needs both")
    return out


def views(pred: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """{"all_hours": pred, "closing_bar": the same-hour view}."""
    return {"all_hours": pred, "closing_bar": same_hour_view(pred, cfg)}
