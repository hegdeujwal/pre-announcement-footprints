# Phase 10 — which files are the results, and which are void

## THE RESULTS
`FINAL-test-evaluation-r2.csv` + `FINAL-test-run-r2.log` — run **2026-09-10**,
once, on the sealed test set. **These are the numbers.**

## VOID — do not quote
`FINAL-test-evaluation.csv`, `FINAL-test-evaluation-corrected.csv` and
`FINAL-test-run.log` — run 2026-09-08. Kept as a record, not as results.

They were produced by an evaluation frame that gave a positive episode 48 bars
and a quiet window 1, and scored a window by its maximum — so a positive had 48
chances to cross the threshold against a quiet bar's one, at the same one-alert
cost. On that exact frame a **pure random-noise scorer reached 29.6x lift**,
beating every tuned detector in the table. Those numbers rank on window length,
not on detection, and no caveat repairs them. Fixed in `9ada635`; the decision
to re-run is in `context/progress-tracker.md`'s decision log, dated and with its
reason, and the seal carries the same reason.

## The 2026-09-10 numbers (news_adjusted)

| detector | all | scheduled | unscheduled | median lead |
|---|---|---|---|---|
| cusum | 9.84x | 14.63x | **5.58x** | 21.7 trading h |
| volume_zscore | 9.73x | 14.74x | **5.27x** | 20.0 trading h |
| rl_policy[s43] | 9.16x | 13.32x | 5.46x | 20.0 trading h |
| gradient_boosting | 2.20x | 1.52x | 2.83x | 38.5 trading h |
| random_noise (5 seeds) | 0.78–1.26 | — | 0.40–1.21 | — |

**Read the unscheduled column, not the pooled one.** Scheduled events are
earnings, whose dates are public weeks ahead.

**The pooled figure rose against validation (7.01 -> 9.84) mostly by
arithmetic, not by improvement**: precision barely moved (0.0323 -> 0.0314)
while the base rate fell (0.00461 -> 0.00319). Same detection, rarer events,
bigger multiple. The unscheduled slice did genuinely improve
(precision 0.0061 -> 0.0092). Quote ~5x, and be able to explain this.

**The simple threshold ties the change detector** (9.73 vs 9.84). Plan §5 says
to report that.

**The learned policy is not usable.** Seed 43 scores 9.16x while flagging
99.98% of hours — it ranks well and decides terribly. Across seeds: 9.16, 1.82,
1.41, 0.40, 0.00. A result that swings that far on the seed is not a result.

**`always_quiet` is correctly flagged degenerate** (tie_spill_ratio 52.9) while
every real detector sits at 1.00.

## 2026-10-18 — read before quoting the UNSCHEDULED column

**The unscheduled lifts above (cusum 5.58x, volume_zscore 5.27x, rl 5.46x)
are mostly time of day, not a footprint.** Do not quote them as detection.

Why: a positive window is scored on its last bar before t0, and 89.5% of
unscheduled positives' last bars are the session's closing bar (15:30 ET) —
filings land after the close. A quiet window is one random bar, a closing bar
only ~1 time in 7. The closing bar is also the busiest hour of the day, so any
volume score ranks it high. A detector that reads **nothing but the clock**
(`clock_only`: rank by nearness to the close) scores **6.69x** on unscheduled
validation filings, above every detector in this table; its within-stock AUC
is 0.877.

The fair comparison scores only closing-bar windows, where the clock is worth
nothing (`eval.same_hour`, clock_only falls to ~1x):

| | scheduled | unscheduled |
|---|---|---|
| volume_zscore (validation) | 3.07x | 1.19x |
| cusum (validation) | 3.07x | 1.19x |
| volume_zscore (fresh Aug–Oct 2026 holdout) | 4.56x | 1.00x |
| cusum (fresh holdout) | 4.49x | 0.98x |
| FootprintNet, 10 runs (fresh holdout) | 6.4–7.0x | 1.65–1.93x |
| stock-only prior (fresh holdout) | 1.06x | **2.44x** |

**What stands:** a footprint before SCHEDULED filings (earnings), at the same
hour of day — a real effect, and dates are public, so it is anticipation, not
leakage. **What does not:** a timing footprint before UNSCHEDULED filings.
At the same hour, every detector's within-stock AUC is ~0.53 with intervals
touching 0.5, and what lift remains comes from choosing filing-prone stocks.

The sealed test set was NOT re-opened for this; the corrected figures are from
validation and from a fresh post-study holdout (`footprint-net/holdout/`). The
2026-09-10 table above is kept unchanged as the record of what was run.
