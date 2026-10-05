"""Which stock, never which hour — the control for ticker selection.

Every other row of the comparison table claims to detect a *moment*: the hours
before a filing look different from the stock's other hours. But the
evaluation frame draws its quiet windows from every stock in the universe, so a
detector can also earn lift by learning *which stocks* file often — volatile,
newsy names whose hours are disproportionately pre-filing hours whatever they
look like. That is a property of the company, not a footprint.

This baseline does only that. It scores a stock by how many positive episodes
it had in the TRAIN split and reads nothing about the hour at all. If it scores
near the timing detectors, their lift is mostly stock selection; if they beat
it clearly, the timing claim stands on its own.

Why it was added
----------------
A live check (2026-10-05) graded the alert log against EDGAR: CUSUM's alerts
were followed by a filing 4.4% of the time, against 1.4% for a random stock at
the same hour — but 5.5% for the SAME stock at a random hour. The sealed test
frame would not tell those apart, so the table needs a row that can.

Ties, and why the jitter carries no information
-----------------------------------------------
Every hour of one stock gets the same prior, and `precision_at_alert_budget`
admits ties at the cut together — a constant per-stock score would spend
hundreds of alerts past the budget on the boundary stock and be compared at a
different operating point from every other row. So each row gets a seeded
uniform jitter strictly smaller than the gap between two distinct priors. It
reorders hours *within* a stock and never across stocks, which makes this row
exactly "the same stocks, at a random hour" — the control the live check used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline


class TickerPrior(Baseline):
    """Scores a stock by its count of train-split positive episodes."""

    name = "ticker_prior"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        knobs = self.cfg.get("baselines", {}).get("ticker_prior", {})
        self.seed = int(knobs.get("seed", 0))
        self.prior: pd.Series | None = None

    def fit(self, train: pd.DataFrame, conn=None) -> "TickerPrior":
        """Count distinct positive episodes per ticker in `train`.

        Only positive rows are read, so a positives-only frame and the full
        gradient-boosting training frame give the same prior. `conn` is passed
        to `assert_not_test` for the same reason gradient boosting does it:
        fitting on sealed data is the one leak no later check could catch.
        """
        if conn is not None:
            from src.pipeline import split
            split.assert_not_test(self.cfg, conn, train["ts_utc"],
                                  f"{self.name}.fit")
        positives = train[train["t0_utc"].notna()]
        if positives.empty:
            raise ValueError(
                f"{self.name}: training frame holds no positive episodes, so "
                f"every stock would score zero and this row would silently "
                f"become always-quiet under another name.")
        self.prior = (positives.groupby("ticker")["window_id"].nunique()
                      .astype("float64"))
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.prior is None:
            raise RuntimeError(f"{self.name}: call fit() before score()")
        # A stock never seen positive in train scores 0: the prior's floor,
        # not NaN — it is a real answer ("this company did not file"), and
        # flooring it as unscoreable would rank it below stocks it ties with.
        base = (frame["ticker"].astype("object").map(self.prior)
                .fillna(0.0).astype("float64"))
        # Counts are integers, so distinct priors differ by at least 1 and a
        # jitter in [0, 0.5) can never move a row past another stock.
        rng = np.random.default_rng(self.seed)
        return pd.Series(base.to_numpy() + 0.5 * rng.random(len(frame)),
                         index=frame.index, dtype="float64")
