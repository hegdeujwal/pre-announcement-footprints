"""Live alerts against chance — did a filing follow more often than it would anyway?

`live.outcomes` answers "was this alert followed by an 8-K within 48 hours?",
and the hit rate it produces cannot be read on its own: some stocks file every
few weeks, so any alert on them is "followed by a filing" fairly often. This
compares the live hit rate with three controls, each on the same alerts, the
same filings and the same window:

  A. random stock, same hour  — every universe stock at the alert's bar. What
     an alert is worth over picking a stock blind.
  B. same stock, random hour  — the alert's stock at `random_hours` hours drawn
     from the hours the monitor actually alerted on. The TIMING control: a
     ratio above 1 here is evidence of a footprint, not of stock selection.
  C. stock-only ranking       — at each alert hour, the top-n universe stocks
     by train-split episode count, n being that detector's alerts in that hour.
     The live counterpart of the `ticker_prior` baseline.

Every rate is split by what followed: any 8-K (what `live.outcomes` grades),
scheduled (an item in `items.scheduled`, i.e. earnings) and unscheduled (a
substantive item and no scheduled one). A filing whose items are all in
`items.exclude` is routine and counts only under "any".

Grading follows `live.outcomes.first_filing_after`: t0 is the event's
corrected instant when one exists, else acceptance; the filing must be strictly
after the alert bar and within `live.outcome_window_hours` wall-clock hours;
and an alert is graded only if its whole window ends before the filing horizon
less `news.t0_lookback_hours`, so missing data is never scored as a miss. For a
kind slice the question is "did a filing of this kind follow", not "was the
first filing of this kind".

Intervals bootstrap alert DAYS, not alerts: one stock often alerts for hours
running and every alert in a day shares one market, so alerts are not
independent draws. Ratio intervals resample the same days for the numerator and
the control, so they are paired.

Usage:
  python -m src.eval.live_vs_chance                  # print the table
  python -m src.eval.live_vs_chance --out live.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

#: Slices of "what followed", in reading order. AGENTS.md rule 7.
KINDS = ("any", "scheduled", "unscheduled")

#: The controls, keyed by the column prefix used throughout.
CONTROLS = {"A": "random_stock_same_hour", "B": "same_stock_random_hour",
            "C": "stock_only_ranking"}


def filing_kind(items: str | None, excluded: set[str], scheduled: set[str]) -> str:
    """'scheduled', 'unscheduled' or 'routine', from a filing's item codes."""
    codes = {c.strip() for c in str(items or "").split(",") if c.strip()} - excluded
    if not codes:
        return "routine"
    return "scheduled" if codes & scheduled else "unscheduled"


class FilingIndex:
    """Sorted t0s per (kind, ticker), for fast 'did a filing follow' lookups."""

    def __init__(self, filings: pd.DataFrame, span_s: int) -> None:
        """`filings` has columns ticker, t0, kind."""
        self.span = int(span_s)
        self._by: dict[str, dict[str, np.ndarray]] = {}
        for kind in KINDS:
            rows = filings if kind == "any" else filings[filings["kind"] == kind]
            self._by[kind] = {t: np.sort(g["t0"].to_numpy(dtype="int64"))
                              for t, g in rows.groupby("ticker")}

    def followed(self, kind: str, ticker: str, ts: int) -> bool:
        """A filing of `kind` in (ts, ts + span]."""
        t0s = self._by[kind].get(ticker)
        if t0s is None:
            return False
        i = np.searchsorted(t0s, ts, side="right")
        return bool(i < len(t0s) and t0s[i] <= ts + self.span)


def grade(alerts: pd.DataFrame, index: FilingIndex, universe: list[str],
          ranked_universe: list[str], random_hours: int,
          rng: np.random.Generator) -> pd.DataFrame:
    """Per alert: `hit_<kind>` and each control's expected hit rate `<X>_<kind>`.

    `alerts` needs detector, ticker, ts_utc. `ranked_universe` is the universe
    in stock-only order, best first.
    """
    out = alerts.copy()
    out["day"] = pd.to_datetime(out["ts_utc"], unit="s").dt.date
    hours = np.sort(out["ts_utc"].unique())
    draws = rng.choice(hours, size=(len(out), random_hours))
    per_det_hour = out.groupby(["detector", "ts_utc"]).size()

    for kind in KINDS:
        f = index.followed
        out[f"hit_{kind}"] = [f(kind, t, ts) for t, ts in zip(out.ticker, out.ts_utc)]
        same_hour = {ts: np.mean([f(kind, t, ts) for t in universe]) for ts in hours}
        out[f"A_{kind}"] = out["ts_utc"].map(same_hour)
        out[f"B_{kind}"] = [np.mean([f(kind, t, h) for h in hs])
                            for t, hs in zip(out.ticker, draws)]
        top = {(d, ts): np.mean([f(kind, t, ts) for t in ranked_universe[:n]])
               for (d, ts), n in per_det_hour.items()}
        out[f"C_{kind}"] = [top[(d, ts)] for d, ts in zip(out.detector, out.ts_utc)]
    return out


def summarise(graded: pd.DataFrame, n_boot: int, seed: int,
              ci: float = 0.95) -> pd.DataFrame:
    """One row per (detector, kind), plus a pooled 'ALL' detector."""
    rng = np.random.default_rng(seed)
    tail = (1 - ci) / 2 * 100
    groups = list(graded.groupby("detector")) + [("ALL", graded)]
    rows = []
    for det, g in groups:
        cols = [f"{p}_{k}" for p in ("hit", *CONTROLS) for k in KINDS]
        by_day = g.groupby("day")[cols].sum().astype("float64")
        counts = g.groupby("day").size().to_numpy()
        days = np.arange(len(by_day))
        picks = rng.choice(days, size=(n_boot, len(days)))
        sums = {c: by_day[c].to_numpy()[picks].sum(axis=1) for c in cols}
        n = counts[picks].sum(axis=1)
        for kind in KINDS:
            hit = sums[f"hit_{kind}"]
            row = {"detector": det, "followed_by": kind, "n_alerts": len(g),
                   "n_ticker_days": len(g[["ticker", "day"]].drop_duplicates()),
                   "n_days": len(by_day), "hit_rate": g[f"hit_{kind}"].mean()}
            row["hit_lo"], row["hit_hi"] = np.percentile(hit / n, [tail, 100 - tail])
            for p, name in CONTROLS.items():
                control = g[f"{p}_{kind}"].mean()
                row[name] = control
                row[f"{p}_ratio"] = row["hit_rate"] / control if control else np.nan
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = hit / sums[f"{p}_{kind}"]
                ratios = ratios[np.isfinite(ratios)]
                lo, hi = (np.percentile(ratios, [tail, 100 - tail])
                          if len(ratios) else (np.nan, np.nan))
                row[f"{p}_ratio_lo"], row[f"{p}_ratio_hi"] = lo, hi
            rows.append(row)
    return pd.DataFrame(rows)


def load_inputs(cfg: dict, conn, alerts_csv: str):
    """Alerts that can be graded, the filing index, and the two universe orders."""
    from src.baselines.compare import train_positives

    span = int(cfg["live"]["outcome_window_hours"]) * 3600
    lookback = int((cfg.get("news") or {}).get("t0_lookback_hours", 0)) * 3600
    horizon = conn.execute("SELECT MAX(acceptance_utc) FROM filings").fetchone()[0]
    if horizon is None:
        raise SystemExit("no filings stored — nothing can be graded.")

    alerts = pd.read_csv(alerts_csv)
    alerts = alerts[alerts["ts_utc"] + span <= horizon - lookback]
    if alerts.empty:
        raise SystemExit("no alert has a closed outcome window yet — fetch "
                         "newer filings first (see live.monitor.fetch_recent_filings).")

    forms = tuple(cfg["edgar"]["forms"])
    marks = ",".join("?" * len(forms))
    filings = pd.read_sql(
        f"SELECT f.ticker, COALESCE(e.t0_utc, f.acceptance_utc) AS t0, f.items "
        f"FROM filings f LEFT JOIN events e ON e.accession_no = f.accession_no "
        f"WHERE f.form IN ({marks}) AND COALESCE(e.t0_utc, f.acceptance_utc) > ?",
        conn, params=(*forms, int(alerts["ts_utc"].min())))
    excluded = set(cfg["items"]["exclude"])
    scheduled = set(cfg["items"]["scheduled"])
    filings["kind"] = filings["items"].map(
        lambda s: filing_kind(s, excluded, scheduled))

    universe = [r[0] for r in conn.execute(
        "SELECT ticker FROM companies WHERE in_universe = 1 ORDER BY ticker")]
    return alerts, FilingIndex(filings, span), universe, horizon, train_positives(cfg)


def stock_only_order(train_positives: pd.DataFrame, universe: list[str],
                     rng: np.random.Generator) -> list[str]:
    """Universe by train-split episode count, ties broken by seeded jitter."""
    counts = (train_positives.groupby("ticker")["window_id"].nunique()
              .reindex(universe).fillna(0.0))
    jitter = pd.Series(0.5 * rng.random(len(counts)), index=counts.index)
    return list((counts + jitter).sort_values(ascending=False).index)


def main() -> None:
    from src import db
    from src.live.catchup import DEFAULT_LOG_CSV
    from src.utils.config import load_config
    from src.utils.timeutils import ts_to_iso

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alerts", default=DEFAULT_LOG_CSV)
    ap.add_argument("--out", default=None, help="write the table as CSV")
    args = ap.parse_args()

    cfg = load_config()
    knobs = cfg["eval"]["live_vs_chance"]
    conn = db.get_conn(cfg["paths"]["db"])
    alerts, index, universe, horizon, positives = load_inputs(cfg, conn, args.alerts)
    rng = np.random.default_rng(int(knobs["seed"]))
    graded = grade(alerts, index, universe,
                   stock_only_order(positives, universe, rng),
                   int(knobs["random_hours"]), rng)
    table = summarise(graded, int(knobs["bootstrap"]), int(knobs["seed"]),
                      float(knobs["ci"]))

    print(f"{len(graded):,} alerts graded, {ts_to_iso(int(graded.ts_utc.min()))} "
          f"to {ts_to_iso(int(graded.ts_utc.max()))}; filing horizon "
          f"{ts_to_iso(int(horizon))}\n")
    show = ["detector", "followed_by", "n_alerts", "n_days", "hit_rate",
            *CONTROLS.values(), "B_ratio", "B_ratio_lo", "B_ratio_hi",
            "C_ratio", "C_ratio_lo", "C_ratio_hi"]
    with pd.option_context("display.width", 250):
        print(table[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
