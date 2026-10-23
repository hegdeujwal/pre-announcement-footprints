"""Does unusual trading before earnings predict the RESULT? The date is public.

Added 2026-10-23. Every detector here "finds" earnings, and an earnings date
is published weeks ahead, so finding one discovers nothing. What is NOT
public before t0 is whether the result is good or bad. If it leaks, the
stocks whose pre-earnings trading is unusual should be the ones that then
move most, or their price should already be drifting the right way.

Per detector, per earnings 8-K (material or not — conditioning on a 3% move
would cut away exactly the small moves the size test needs):

* `score_pct` — the event's score on its last bar before t0, as a percentile
  of THAT stock's quiet closing bars in the same split. Only events whose last
  bar is a closing bar are kept, so the hour of day is the same on both
  sides. Quiet bars within `normal_gap_days` before any of the stock's
  earnings are left out of its normal, being the build-up itself.
* **Size.** Spearman between `score_pct` and |move|, raw and partial — after
  removing what the stock's own past earnings moves and its volatility
  already say, both known before t0. A raw correlation alone would only
  rediscover that volatile stocks move more.
* **Direction.** How often the benchmark-relative drift over the last session
  has the same sign as the move after: on every event, and on the `unusual`
  ones (top `unusual_top_share` by `score_pct`). A leak would show as a hit
  rate above 50% that is higher where the detector is unusual.

Intervals bootstrap STOCKS, as everywhere in this project. The move is
measured after t0, which is correct here and nowhere else: this grades, it
does not detect, and no detector reads it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

DAY = 86400


def score_percentiles(scores: pd.DataFrame, events: pd.DataFrame,
                      knobs: dict) -> pd.DataFrame:
    """One row per (detector, event) whose last bar is a closing bar and whose
    stock has enough normal closing bars."""
    gap = int(knobs["normal_gap_days"]) * DAY
    need = int(knobs["min_quiet_closing"])
    earn_t0 = events.groupby("ticker")["t0_utc"].apply(
        lambda s: np.sort(s.to_numpy(dtype="int64"))).to_dict()

    out = []
    for name, d in scores.groupby("detector", sort=False):
        quiet = d[~d["is_positive"] & d["closing"]]
        keyed = d.set_index(["ticker", "ts_utc"])[["score", "closing"]]
        keyed = keyed[~keyed.index.duplicated()]
        normal = {}
        for ticker, q in quiet.groupby("ticker", sort=False):
            ts = q["ts_utc"].to_numpy(dtype="int64")
            t0s = earn_t0.get(ticker, np.array([], "int64"))
            ahead = np.full(len(ts), np.inf)
            if len(t0s):
                nxt = np.searchsorted(t0s, ts, side="right")
                has = nxt < len(t0s)
                ahead[has] = t0s[nxt[has]] - ts[has]
            normal[ticker] = np.sort(q["score"].to_numpy(dtype="float64")[ahead > gap])
        for r in events.itertuples():
            key = (r.ticker, int(r.score_ts))
            if key not in keyed.index:
                continue
            s, closing = keyed.loc[key]
            ref = normal.get(r.ticker)
            if not closing or ref is None or len(ref) < need or not np.isfinite(s):
                continue
            below = np.searchsorted(ref, s, side="left")
            ties = np.searchsorted(ref, s, side="right") - below
            out.append({"detector": name, "event_id": r.event_id,
                        "score_pct": (below + 0.5 * ties) / len(ref)})
    if not out:
        return pd.DataFrame(columns=["detector", "event_id", "score_pct", *events.columns])
    return pd.DataFrame(out).merge(events, on="event_id", how="left")


def _residual(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Spearman of x and y after regressing both ranks on the control ranks."""
    rx, ry = rankdata(x), rankdata(y)
    rc = np.column_stack([rankdata(c) for c in controls.T])
    ex, ey = _residual(rx, rc), _residual(ry, rc)
    den = np.sqrt((ex ** 2).sum() * (ey ** 2).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def _hit(drift: np.ndarray, move: np.ndarray) -> float:
    keep = (drift != 0) & (move != 0)
    return float((np.sign(drift[keep]) == np.sign(move[keep])).mean()) if keep.any() \
        else float("nan")


def _stats(g: pd.DataFrame, cut: float) -> dict:
    move = g["move"].to_numpy(dtype="float64")
    size = np.abs(move)
    pct = g["score_pct"].to_numpy(dtype="float64")
    drift = g["pre_drift"].to_numpy(dtype="float64")
    controls = g[["prior_abs_move", "volatility"]].to_numpy(dtype="float64")
    unusual = pct >= cut
    return {"rho_size": spearman(pct, size),
            "rho_size_partial": partial_spearman(pct, size, controls),
            "rho_controls": spearman(controls[:, 0], size),
            "hit_all": _hit(drift, move),
            "hit_unusual": _hit(drift[unusual], move[unusual]),
            "hit_usual": _hit(drift[~unusual], move[~unusual]),
            "size_ratio": float(np.median(size[unusual]) / np.median(size[~unusual]))
            if unusual.any() and (~unusual).any() else float("nan")}


def leak_table(scores: pd.DataFrame, events: pd.DataFrame, knobs: dict) -> pd.DataFrame:
    """One row per detector: estimates and stock-bootstrap intervals."""
    rows = []
    tail = (1 - float(knobs["ci"])) / 2 * 100
    share = float(knobs["unusual_top_share"])
    table = score_percentiles(scores, events, knobs)
    if table.empty:
        raise ValueError("earnings_leak: no event survived the closing-bar and "
                         "normal-bar requirements")
    for name, g in table.groupby("detector", sort=False):
        g = g.dropna(subset=["prior_abs_move", "volatility", "move", "pre_drift"])
        g = g.reset_index(drop=True)
        cut = float(np.quantile(g["score_pct"], 1 - share))
        point = _stats(g, cut)
        rng = np.random.default_rng(int(knobs["seed"]))
        groups = list(g.groupby("ticker").indices.values())
        boots = []
        for _ in range(int(knobs["bootstrap"])):
            pick = rng.integers(0, len(groups), len(groups))
            boots.append(_stats(g.iloc[np.concatenate([groups[i] for i in pick])], cut))
        boots = pd.DataFrame(boots)
        row = {"detector": name, "n_events": len(g), "n_stocks": len(groups),
               "n_unusual": int((g["score_pct"] >= cut).sum()),
               "unusual_cut_pct": cut}
        for key, value in point.items():
            lo, hi = np.nanpercentile(boots[key], [tail, 100 - tail])
            row.update({key: value, f"{key}_lo": lo, f"{key}_hi": hi})
        rows.append(row)
    return pd.DataFrame(rows)
