"""Which stock AND which hour — the two signals the diagnostics found, combined.

Two measurements on validation (2026-10-05/06) pulled the detectors apart:

* `ticker_prior` — which stock, never which hour — beat every timing detector
  on unscheduled filings at the alert budget (6.05x against CUSUM's 3.15x);
* `eval.within_stock` showed volume z-score DOES carry timing signal once stock
  selection is removed (within-stock AUC 0.734 on unscheduled filings).

So each detector was spending its alerts on half the evidence: volume z-score
on spikes in stocks that rarely file, the prior on random hours of stocks that
file often. This combines them the simplest defensible way, as a naive-Bayes
log-odds sum:

    score = log(train episodes of this stock + alpha)
          + log( p(volume_z | pre-filing bar) / p(volume_z | quiet bar) )

The second term is a binned likelihood ratio. Both terms are estimated on the
TRAIN split only, and the two knobs (`alpha`, `n_bins`) are set in config
before validation is read, so nothing here is tuned on the frame it is scored
on. The naive-Bayes assumption — that how a pre-filing hour's volume looks does
not depend on which stock it is — is an approximation, stated rather than
hidden; `volume_z` is already standardised per stock, which is what makes it
tolerable.

Why the final bar
-----------------
`metrics.window_summary` scores a positive by its final bar — the decision
point immediately before t0 — and a quiet window by its single bar. The
likelihood ratio is fitted on exactly those bars (the final bar of every train
window, positive and quiet), so it models the quantity the metric reads rather
than an average over hours the metric never looks at.

An hour with no `volume_z` (too little history) gets a log-ratio of 0 — no
timing evidence — and keeps its stock term. Flooring it as unscoreable would
throw away the half of the evidence that does exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline

FEATURE = "volume_z"
#: Tie-break magnitudes: |z term| < 1e-6, jitter < 1e-9.
TIE_Z, TIE_JITTER = 1e-6, 1e-9


def final_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """The last bar of every window — the bar the budget metric scores."""
    ordered = frame.sort_values(["window_id", "ts_utc"], kind="mergesort")
    return ordered[~ordered["window_id"].duplicated(keep="last")]


class PriorVolume(Baseline):
    """log stock prior + log likelihood ratio of the hour's volume z-score."""

    name = "prior_volume"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        knobs = self.cfg.get("baselines", {}).get("prior_volume", {})
        self.alpha = float(knobs.get("alpha", 1.0))
        self.n_bins = int(knobs.get("n_bins", 20))
        self.seed = int(knobs.get("seed", 0))
        if self.alpha <= 0:
            raise ValueError(f"{self.name}: alpha must be > 0, got {self.alpha}")
        self.log_prior: pd.Series | None = None
        self.edges: np.ndarray | None = None
        self.log_lr: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, conn=None) -> "PriorVolume":
        """Estimate both terms from a train frame holding both classes.

        `train` is `compare.build_training_frame` output: positive episodes plus
        sampled quiet windows, all from the train split.
        """
        if conn is not None:
            from src.pipeline import split
            split.assert_not_test(self.cfg, conn, train["ts_utc"],
                                  f"{self.name}.fit")
        last = final_bars(train)
        positive = last["t0_utc"].notna().to_numpy()
        if positive.all() or not positive.any():
            raise ValueError(
                f"{self.name}: the training frame must hold both positive and "
                f"quiet windows to estimate a likelihood ratio; it holds "
                f"{int(positive.sum())} positive and {int((~positive).sum())} "
                f"quiet.")

        counts = last[positive].groupby("ticker")["window_id"].nunique()
        self.log_prior = np.log(counts.astype("float64") + self.alpha)

        z = last[FEATURE].to_numpy(dtype="float64")
        finite = np.isfinite(z)
        quiet_z = z[finite & ~positive]
        # Quantile edges from the quiet bars: equal-mass bins where the bulk of
        # the data is, so the spike tail is resolved rather than lumped. The
        # outer edges are open so a validation value beyond anything seen in
        # train still lands in a bin.
        inner = np.unique(np.quantile(quiet_z, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
        self.edges = inner
        pos_hist = np.bincount(np.searchsorted(inner, z[finite & positive], side="right"),
                               minlength=len(inner) + 1)
        quiet_hist = np.bincount(np.searchsorted(inner, quiet_z, side="right"),
                                 minlength=len(inner) + 1)
        # Add-one smoothing: an empty bin on either side would otherwise be
        # +/-inf, and the contract rejects infinities.
        p_pos = (pos_hist + 1) / (pos_hist.sum() + len(pos_hist))
        p_quiet = (quiet_hist + 1) / (quiet_hist.sum() + len(quiet_hist))
        self.log_lr = np.log(p_pos / p_quiet)
        return self

    def score(self, frame: pd.DataFrame) -> pd.Series:
        if self.log_prior is None:
            raise RuntimeError(f"{self.name}: call fit() before score()")
        # A stock with no train episode still gets a finite prior, log(alpha):
        # "this company did not file" is an answer, not a missing value.
        prior = (frame["ticker"].astype("object").map(self.log_prior)
                 .fillna(np.log(self.alpha)).to_numpy(dtype="float64"))
        z = frame[FEATURE].to_numpy(dtype="float64")
        lr = np.zeros(len(frame))
        ok = np.isfinite(z)
        lr[ok] = self.log_lr[np.searchsorted(self.edges, z[ok], side="right")]
        # Tie-breaking. Both terms are discrete — a few distinct counts, n_bins
        # ratios — and `precision_at_alert_budget` admits ties at the cut
        # together, so without this the row would overspend its budget and sit
        # at a different operating point from every other row. Within a bin the
        # larger raw z goes first; what still ties (no z at all) is ordered by
        # seeded jitter. Both are bounded far below the gap between two
        # distinct (prior + ratio) values, so they never reorder those.
        rng = np.random.default_rng(self.seed)
        tie = (TIE_Z * np.tanh(np.where(ok, z, 0.0) / 10.0)
               + TIE_JITTER * rng.random(len(frame)))
        return pd.Series(prior + lr + tie, index=frame.index, dtype="float64")
