"""Pattern templates — "does the last stretch of hours look like a pre-filing stretch?"

Added 2026-10-15 from the guide's review list ("create a template of every
pattern"). Volume z-score asks whether THIS hour is loud; CUSUM asks whether
loudness has piled up. Both treat every hour of the run-up alike. A template
asks a sharper question: the hours before a filing have a typical SHAPE —
quiet, then a rise in the last session, strongest in the final hour — so
weight each recent hour by how much it usually matters.

How a template is built (train split only)
------------------------------------------
For every training window take the last `length` bars of `channel` (bar -L
.. bar -1, aligned on t0). Then, per position j:

    mu_g[j]     mean over pre-filing windows of group g
    mu_0[j]     mean over quiet windows
    sd_0[j]     standard deviation over quiet windows

and the score of the most recent `length` bars x is

    score_g = sum_j w_g[j] * (x[j] - mu_0[j]) / norm_g
    w_g[j]  = (mu_g[j] - mu_0[j]) / sd_0[j]^2
    norm_g  = sqrt( sum_j w_g[j]^2 * sd_0[j]^2 )

This is the **matched filter**, the textbook detector for a known signal
shape buried in noise: under quiet conditions score_g has mean 0 and
standard deviation 1, and it is largest when the recent bars deviate from
quiet in exactly the way pre-filing bars do. Positions where filings differ
little from quiet get almost no weight.

One template per group — every filing, scheduled, unscheduled, and each item
code with at least `min_group_events` training events. At scoring time the
filing type is unknown (it is the future), so the hour's score is the
**maximum over templates**: "the best-matching pattern", the generalised
likelihood-ratio rule. A test pins that the score never uses the window's own
label columns.

Missing bars (NaN) count as "exactly quiet" — zero deviation — so they add no
evidence either way. An hour with no defined bar at all in its stretch is NaN
and goes to `Baseline`'s unscoreable handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline
from src.baselines.prior_volume import final_bars


def last_bars(frame: pd.DataFrame, channel: str, length: int) -> pd.DataFrame:
    """The last `length` values of `channel` per window, right-aligned.

    Column `length - 1` is the window's final bar (bar -1 before t0). Short
    windows are padded with NaN on the LEFT, so position always means "this
    many bars before the end".
    """
    ordered = frame.sort_values(["window_id", "ts_utc"], kind="mergesort")
    pos = ordered.groupby("window_id", sort=False).cumcount(ascending=False)
    keep = ordered[pos < length]
    col = (length - 1) - pos[pos < length]
    wide = (keep.assign(_col=col.to_numpy())
            .pivot_table(index="window_id", columns="_col", values=channel,
                         aggfunc="first", dropna=False))
    return wide.reindex(columns=range(length))


def lagged(x: np.ndarray, length: int) -> np.ndarray:
    """Row t holds x[t-length+1 .. t]; positions before the start are NaN."""
    n = len(x)
    out = np.full((n, length), np.nan)
    for j in range(length):
        shift = length - 1 - j
        if shift == 0:
            out[:, j] = x
        elif shift < n:
            out[shift:, j] = x[:-shift]
    return out


class PatternTemplate(Baseline):
    """Matched-filter score against the best-matching learned pre-filing shape."""

    name = "pattern_template"

    def __init__(self, cfg: dict | None = None, channel: str | None = None,
                 name: str | None = None) -> None:
        super().__init__(cfg)
        knobs = self.cfg.get("baselines", {}).get("pattern_template", {})
        self.channel = channel or knobs.get("channel", "volume_z")
        self.length = int(knobs.get("length", 14))
        self.min_group_events = int(knobs.get("min_group_events", 150))
        if name:
            self.name = name
        self.mu0: np.ndarray | None = None
        self.sd0: np.ndarray | None = None
        #: group name -> (mean path of that group, n training events)
        self.templates: dict[str, tuple[np.ndarray, int]] = {}
        self._weights: dict[str, np.ndarray] = {}

    # -- fitting ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, conn=None) -> "PatternTemplate":
        if conn is not None:
            from src.pipeline import split
            split.assert_not_test(self.cfg, conn, train["ts_utc"],
                                  f"{self.name}.fit")
        if self.channel not in train.columns:
            raise ValueError(f"{self.name}: training frame has no "
                             f"{self.channel!r} column")
        paths = last_bars(train, self.channel, self.length)
        meta = final_bars(train).set_index("window_id")
        meta = meta.reindex(paths.index)
        positive = meta["t0_utc"].notna().to_numpy()
        if positive.all() or not positive.any():
            raise ValueError(f"{self.name}: training frame needs both "
                             f"pre-filing and quiet windows")

        quiet = paths[~positive].to_numpy(dtype="float64")
        self.mu0 = np.nanmean(quiet, axis=0)
        self.sd0 = np.nanstd(quiet, axis=0)
        # A constant position (sd 0) carries no usable contrast; give it a
        # huge variance so its weight is ~0 rather than dividing by zero.
        self.sd0 = np.where(self.sd0 > 0, self.sd0, np.inf)

        pos_paths = paths[positive]
        pos_meta = meta[positive]
        groups: dict[str, np.ndarray] = {
            "all": np.ones(len(pos_paths), dtype=bool),
            "scheduled": pos_meta["is_scheduled"].astype(bool).to_numpy(),
            "unscheduled": ~pos_meta["is_scheduled"].astype(bool).to_numpy(),
        }
        items = pos_meta["item_code"].fillna("").astype(str).str.split(",")
        for code in sorted({c.strip() for row in items for c in row if c.strip()}):
            mask = items.map(lambda r, c=code: c in [x.strip() for x in r]).to_numpy()
            if mask.sum() >= self.min_group_events:
                groups[f"item {code}"] = mask

        self.templates, self._weights = {}, {}
        for g, mask in groups.items():
            n = int(mask.sum())
            if n < self.min_group_events and g != "all":
                continue
            mu = np.nanmean(pos_paths.to_numpy(dtype="float64")[mask], axis=0)
            w = (mu - self.mu0) / self.sd0 ** 2
            w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
            norm = np.sqrt(np.sum(w ** 2 * np.where(np.isfinite(self.sd0),
                                                    self.sd0, 0.0) ** 2))
            if norm <= 0:
                continue
            self.templates[g] = (mu, n)
            self._weights[g] = w / norm
        if not self._weights:
            raise ValueError(f"{self.name}: no template had any contrast "
                             f"with quiet windows")
        return self

    # -- scoring ---------------------------------------------------------
    def score(self, frame: pd.DataFrame, by: str = "ticker") -> pd.Series:
        """Score every row. `by="ticker"` is the live/evaluation shape: an
        hour's stretch is the stock's own preceding bars. `by="window_id"`
        keeps each stretch inside its own window — what the RL reward needs,
        because sampled training windows of one stock can sit weeks apart and
        must not borrow each other's bars."""
        if self.mu0 is None:
            raise RuntimeError(f"{self.name}: call fit() before score()")
        if self.channel not in frame.columns:
            raise ValueError(f"{self.name}: frame has no {self.channel!r} column")
        W = np.column_stack(list(self._weights.values()))       # (L, groups)

        # One series per ticker over its distinct timestamps, in time order:
        # the stretch before an hour is the stock's own preceding bars, the
        # same continuity CUSUM relies on. Overlapping episodes repeat a
        # timestamp; it is scored once and mapped back to every row.
        keys = frame[[by, "ts_utc"]].copy()
        keys["x"] = frame[self.channel].to_numpy(dtype="float64")
        uniq = (keys.drop_duplicates([by, "ts_utc"])
                .sort_values([by, "ts_utc"], kind="stable"))
        scored = []
        for _, block in uniq.groupby(by, sort=False):
            X = lagged(block["x"].to_numpy(), self.length)
            undefined = np.isnan(X).all(axis=1)
            D = np.nan_to_num(X - self.mu0, nan=0.0)
            s = (D @ W).max(axis=1)
            s[undefined] = np.nan
            scored.append(pd.Series(s, index=pd.MultiIndex.from_arrays(
                [block[by], block["ts_utc"]])))
        lookup = pd.concat(scored)
        idx = pd.MultiIndex.from_arrays([keys[by], keys["ts_utc"]])
        return pd.Series(lookup.reindex(idx).to_numpy(), index=frame.index,
                         dtype="float64")

    def template_table(self) -> pd.DataFrame:
        """One row per (group, bar before t0) — what the PDF charts."""
        rows = []
        for g, (mu, n) in self.templates.items():
            for j in range(self.length):
                rows.append({"group": g, "n_events": n,
                             "bar": j - self.length, "mean": mu[j],
                             "quiet_mean": self.mu0[j],
                             "quiet_sd": self.sd0[j],
                             "weight": self._weights[g][j]})
        return pd.DataFrame(rows)
