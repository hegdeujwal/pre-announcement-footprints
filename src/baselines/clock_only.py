"""Clock-only — the null that knows the time of day and nothing else.

Added 2026-10-18. The budget metric scores a positive window by its final bar
before t0, and most 8-Ks are filed after the close: on the feature matrix
84.4% of positive windows end on the session's last bar (89.5% of unscheduled
ones), against roughly one bar in seven for a quiet hour. A detector that
learns only "this is the closing bar" therefore earns lift without detecting
anything. That was measured once already — `trading_hours_to_close` alone
scored 5.67x and was removed from the learned models' inputs — and it came
back in the 2026-10-18 analog check, where the price-shape rows beat CUSUM on
unscheduled filings and the time-of-day-adjusted row fell to chance.

So the clock gets its own row, beside always-quiet and random noise:

    score = -seconds to the session close  +  seeded jitter

The jitter (smaller than any gap between two bars' scores) breaks ties among
bars at the same distance from the close, the way `ticker_prior` breaks ties
within a stock. A bar outside every session (real bars never are; synthetic
test frames sometimes) has no clock reading and is returned as NaN, which
`Baseline` ranks last like any unscoreable hour.

Any detector that does not clearly beat this row is, as far as this frame can
tell, reading the clock. `eval.same_hour` is the other half of the fix: it
re-scores every row on closing bars only, where this row is worth nothing by
construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.baselines.base import Baseline
from src.utils.timeutils import get_market_calendar, seconds_to_session_close

#: Jitter magnitude. Bars sit whole half-hours apart in seconds-to-close, so
#: anything below 1 can never reorder two bars at different distances.
TIE_JITTER = 1e-3


class ClockOnly(Baseline):
    """Ranks every hour by how close it is to the session close."""

    name = "clock_only"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        knobs = self.cfg.get("baselines", {}).get("clock_only", {})
        self.seed = int(knobs.get("seed", 0))

    def score(self, frame: pd.DataFrame) -> pd.Series:
        cal = get_market_calendar(self.cfg["market"]["calendar"])
        to_close = seconds_to_session_close(
            frame["ts_utc"].astype("int64").to_numpy(), cal, strict=False)
        rng = np.random.default_rng(self.seed)
        jitter = rng.uniform(0.0, TIE_JITTER, len(frame))
        return pd.Series(-to_close + jitter, index=frame.index)
