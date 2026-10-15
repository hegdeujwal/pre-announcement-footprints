"""Hybrid detector — which stock (the prior) AND does the run-up match a template.

Added 2026-10-15 for the guide's "hybrid systems vs pure technical" point.
The project already holds one hybrid, `prior_volume`: a stock prior plus the
hour's volume z-score, combined as a naive-Bayes log-odds sum. This is the same
combination with the timing half swapped for the pattern-template score:

    score = log(train episodes of this stock + alpha)
          + log( p(template score | pre-filing bar) / p(template score | quiet bar) )

Everything that makes `prior_volume` defensible carries over unchanged
because the code is shared — both halves fitted on the TRAIN split only, the
likelihood ratio fitted on the final bar of each window (the bar the budget
metric scores), and the same tie-breaking. Only the evidence column differs.
"""

from __future__ import annotations

import pandas as pd

from src.baselines.prior_volume import PriorVolume
from src.baselines.template import PatternTemplate

SCORE_COL = "_template_score"


class PriorTemplate(PriorVolume):
    """Stock prior + likelihood ratio of a fitted pattern-template score."""

    name = "prior_template"

    def __init__(self, cfg: dict | None = None,
                 template: PatternTemplate | None = None) -> None:
        super().__init__(cfg)
        self.template = template
        self.feature = SCORE_COL

    def _with_score(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[SCORE_COL] = self.template.score(frame).to_numpy()
        return out

    def fit(self, train: pd.DataFrame, conn=None) -> "PriorTemplate":
        if self.template is None:
            self.template = PatternTemplate(self.cfg).fit(train, conn=conn)
        return super().fit(self._with_score(train), conn=conn)

    def score(self, frame: pd.DataFrame) -> pd.Series:
        return super().score(self._with_score(frame))
