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
