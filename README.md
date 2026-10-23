# Pre-Announcement Footprints — Detecting Material Corporate News Before It Is Disclosed

**Project Code 31 · NMAM Institute of Technology · Dept. of ISE**

A **same-day surveillance system** that reads a stock's hourly price and volume
and raises a hand before the company files: *"something is coming for this
company."* Every US listed company must disclose material events on an SEC
**8-K**, timestamped to the second — which gives a free, exact answer key to
grade against.

**On "same-day", precisely.** The detector is *online*: at each hourly bar it
decides WAIT or FLAG using only that bar and earlier ones, and it is tested by
tampering with the future and demanding the output not move. The deployed
monitor, however, runs **once per trading day, after the close** — it replays
that day's bars in order rather than polling hourly. Because every feature is
computed point-in-time, the alerts are identical to what an hour-by-hour
deployment would have raised; only the moment they are *noticed* differs, and
the log stores bar time and notice time in separate columns so the gap is
visible. This is deliberately **not** live intraday alerting, and nothing here
claims it is. The benchmark it is measured against is current compliance
practice, which investigates months later, by hand, usually only after a
complaint.

The headline result is **how many trading hours of advance warning public
market data gives, at a controlled false-alarm rate**, split by event type and
by scheduled vs unscheduled announcements.

## The framing

This is **sequential change detection / optimal stopping**, not per-hour
classification. At each hour the system decides WAIT (accumulate more
evidence) or FLAG (declare a regime change), trading detection delay against
false-alarm rate. That framing buys a principled reason for hour-by-hour
decisions, a strong classical baseline (**CUSUM**), and existing literature to
position against.

**Terminology, held strictly:** *footprint*, never *insider trading*. Volume
spikes have innocent causes — index rebalancing, an analyst note, a fund
unwinding. We detect a footprint, not a culprit.

## What is defensible here

Pre-announcement volume anomalies have been documented in finance since ~1981,
and ML has been applied to insider-trading detection — but on **regulator-held
investor-level trading records**. Work on 8-K filings predicts returns *from
filing text, after the filing*; their clock starts where ours ends. Our four
restrictions, each genuinely excluding that prior work: **forward not
retrospective · public data only · a stopping decision, not post-hoc detection ·
detection delay at a fixed alert budget.**

## The two things that decide whether the numbers are real

**1. t₀ is not the filing time.** Companies wire a press release first, then
file the 8-K with that release attached minutes-to-hours later. Acceptance
times cluster at 20:00–21:00 UTC, but earnings releases hit the wire at ~16:05
ET. Using acceptance time alone silently counts time when the market already
knew as "warning". So:

```
t₀ = min(8-K acceptanceDateTime, earliest news article for that ticker/event)
```

Both variants are reported, with the gap between them. This makes news
collection **core label infrastructure from week 1**, not an optional channel.

**2. Never report plain accuracy.** The base rate is ~0.3% positive — "nothing
coming" scores 99.7%. The headline metric is precision at a fixed alert budget
(2 alerts per stock per month), decided before the model was built.

## What the study found

Every number below is on **closing-hour windows only**: most filings land
after the close, so a detector that knows nothing but the time of day scored
6.7x on surprise filings until that comparison was fixed. Tables are in
[`results/`](results/MANIFEST.md); the holdout is Aug 1 – Oct 17 2026, after
the study ended, and every opening of it is recorded in
`results/footprint-net/holdout/holdout_opened.json`.

| Question | Answer | Evidence |
|---|---|---|
| Does a deep model beat the classical detectors? | **On earnings filings, yes** — FootprintNet (a GRU over the last 48 bars) ~1.45x the correct alerts of volume z-score on the holdout, error bar above 1x. | `footprint-net/holdout/lift_ci.csv` |
| Is there a timing footprint before **surprise** filings? | **No**, for any detector: within-stock AUC ~0.53. The lift that remains comes from picking filing-prone stocks, which a stock-only rule does better. | `footprint-net/holdout/within_stock_same_hour.csv` |
| Does a footprint hide in one surprise item type? | **No.** Ten item types tested with a Holm correction; none on validation. On the holdout only Reg FD disclosures (7.01) passed, for 2 of 10 FootprintNet runs and no classical detector — not a finding. | `discovery/*/by_item.csv` |
| Would more labelled filings help on surprise filings? | **No.** From 10% to 100% of the train filings, within-stock AUC stays 0.51–0.53, never above volume z-score's 0.535. Earnings gains flatten by 50%. | `discovery/val/learning_curve.csv` |
| Does pre-earnings trading reveal the **result**? | **Its size, not its direction.** Unusual trading before earnings predicts a bigger move after, beyond what the stock's past earnings moves and volatility already say (partial Spearman 0.11–0.17 validation, 0.24–0.36 holdout, every run's interval above 0). The top fifth moves ~1.4x (validation) to ~2x (holdout) as much. Which way it moves is not shown: the pre-earnings drift matches the move's sign 51–63% of the time, with intervals reaching 50% for most runs. | `discovery/*/earnings_leak.csv` |

The earnings-size result passed the rule fixed in `eval.discovery` before the
holdout was opened (both splits, whole seed range); direction did not. It
reads as the market anticipating a *big* surprise, not knowing which way —
and options-implied volatility, which this project does not collect, is the
control that would test that properly.

Reproduce: `python -m src.baselines.footprint_train` (and `--learning-curve`),
`python -m src.baselines.footprint_review`, then
`python -m src.baselines.discovery_review [--split holdout --reason "..."]`.

## Repository layout

```
├── config/config.yaml         # every knob — no hardcoded tickers/dates/thresholds in code
├── data/
│   ├── raw/edgar/             # cached EDGAR JSON responses (gitignored)
│   ├── db/footprints.db       # SQLite: companies, filings, events, bars, news, meta
│   ├── processed/             # events.parquet + feature matrices
├── src/
│   ├── collectors/            # edgar, market, news
│   ├── pipeline/              # universe, coverage, t0, events, materiality,
│   │                          #   features, sampling, split, evalset
│   ├── baselines/             # always-quiet, volume z-score, CUSUM, gradient boosting
│   ├── rl/                    # Gymnasium env + SB3 learned stopping policy
│   ├── live/                  # the same-day monitor: alert log, outcomes, catch-up
│   ├── eval/                  # precision @ alert budget, detection delay, Brier, ECE
│   └── utils/                 # rate limiting, UTC + market-hours helpers, config
├── app/                       # Streamlit dashboard — every alert shows its reasons
├── live-log/                  # COMMITTED: the live alert log, its grades, quarantined sessions
├── results/                   # COMMITTED: the final result tables the dashboard reads
├── scripts/                   # browse any generated dataset; build the CI bootstrap DB
├── .github/workflows/         # live-monitor.yml — the scheduled same-day run
├── tests/                     # pytest — leakage tests are mandatory
└── implementation_plan.md     # the authoritative plan — read first
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add your free Finnhub key — needed on day one
```

Then set `SEC_USER_AGENT` in that `.env` to a project name and a contact
address you actually read. SEC requires it, and it is the entire terms of
service alongside a 10 req/s cap. It lives in `.env` rather than in
`config/config.yaml` because this repository is public: a committed address
gets scraped, and worse, anyone who forked the project would identify to the
SEC as its author, so their rate-limit violations would land on that author.
The config ships a placeholder and the EDGAR client refuses to make a live
request while that placeholder is all it has.

## Running the dashboard on a fresh clone

Everything the dashboard needs to show the results is committed, so a teammate
needs no database, no API key and no data download:

```bash
git clone https://github.com/hegdeujwal/pre-announcement-footprints.git
cd pre-announcement-footprints && git checkout dev-new
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

| Screen | Reads | On a fresh clone |
|---|---|---|
| Today's alerts | `live-log/alerts.csv`, `live-log/outcomes.csv` | full |
| Live monitor log | the same, plus `live-log/quarantine.csv` | full |
| Evaluation | `results/` (Phase 10 test set, Phase 8 news pair) | full |
| Ticker detail | the alert, plus price bars from the database | alert and its reasons; price panels empty |

The price panels need the SQLite database, which is never committed (about
1 GB). For them, download the slim copy attached to the `bootstrap-2026-09-07`
release (`bootstrap.db.gz`, 31 MB, prices to 2026-09-07), gunzip it to
`data/db/footprints.db`, and restart the dashboard. `git pull` picks up each
night's alerts and grades, which the scheduled monitor commits.

To refresh `results/` after rebuilding a table: `python scripts/export_results.py`
copies the files listed under `results_export` in `config/config.yaml` and
writes their checksums to `results/MANIFEST.md`.

## Running the collectors

```bash
# EDGAR — the study universe, then one submissions request per company.
# Every raw response is cached under data/raw/edgar/ and never re-fetched.
python -m src.collectors.edgar --build-universe
python -m src.collectors.edgar --universe --resume
python -m src.collectors.edgar --report

# Market data — yfinance hourly and daily bars, cached incrementally.
python -m src.collectors.market --tickers TSLA,AAPL --start 2025-09-01 --end 2026-08-01
python -m src.collectors.market --universe --stamp-snapshot

# News — Finnhub primary, GDELT for breadth. Label infrastructure, not an
# optional channel: it runs from week 1 because t0 depends on it.
python -m src.collectors.news --ticker TSLA --start 2025-01-01 --end 2025-01-08
```

Any collector cycle that parses zero records fails loudly rather than passing
quietly — HTTP 200 responses carrying redirect HTML or empty JSON are the
failure mode that lets a pipeline look healthy while writing nothing for days.

## Tests

```bash
pytest
```

## Documents

- [`implementation_plan.md`](implementation_plan.md) — the authoritative plan
- [`AGENTS.md`](AGENTS.md) — operational contract for agents and contributors
- `docs/project-ideas-review.pdf` — the five-direction review this project came from
- `docs/context.md` — corrections and additions to that review; where they
  disagree, `context.md` wins
