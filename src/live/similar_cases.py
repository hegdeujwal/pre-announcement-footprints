"""Similar past cases for every live alert — the mentor's "pattern memory", as context.

Added 2026-10-19. The mentor's idea: remember the stretch before every past
filing, and when a stock looks like one now, say so — "this looks like XYZ
before its 8-K for item 1.01". `baselines.analog.AnalogMemory` is that memory;
this runs it over the live alert log and writes, for each alerted hour, the
`n_cases` nearest past pre-filing stretches in OTHER companies.

What it is NOT
--------------
A forecast of what will be filed. On validation the nearest case's item codes
named the filing that followed an unscheduled alert 6–17% of the time, below
guessing the most common item (37%) — `analog_review`, decision log
2026-10-18. The dashboard says so beside every list. What a past look-alike
does give an analyst is a concrete precedent to compare the chart with.

Leakage and the seal
--------------------
The memory holds TRAIN-split windows only (`build_training_frame`), every one
of whose filings was public months before the live period. A query stretch is
the alerted stock's own bars at or before the alerted hour (`template.lagged`
inside the memory), built through the monitor's own feature path.

Read-only against the alert log: it reads `live-log/alerts.csv` and writes a
separate file, so the hash-chained log is never touched.

Usage:
  python -m src.live.similar_cases               # every alerted hour
  python -m src.live.similar_cases --limit-tickers 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.baselines.analog import AnalogMemory
from src.baselines.compare import build_training_frame
from src.baselines.guide_review import configs
from src.live.monitor import latest_bar_frame
from src.utils.config import load_config

DAY = 86400


def knobs(cfg: dict) -> dict:
    return cfg["live"]["similar_cases"]


def memory(cfg_on: dict, conn) -> AnalogMemory:
    """The configured analog variant, fitted on the train split."""
    name = knobs(cfg_on)["variant"]
    found = [v for v in cfg_on["baselines"]["analog"]["variants"] if v["name"] == name]
    if not found:
        raise ValueError(f"live.similar_cases.variant {name!r} is not one of "
                         f"baselines.analog.variants")
    v = found[0]
    model = AnalogMemory(cfg_on, channels=v["channels"],
                         exclude_same_ticker=v.get("exclude_same_ticker", False),
                         name=v["name"])
    return model.fit(build_training_frame(cfg_on, conn), conn=conn)


def alerted_hours(path: Path) -> pd.DataFrame:
    """Distinct (ticker, ts_utc) in the alert log — one lookup per hour, not per detector."""
    log = pd.read_csv(path, usecols=["ticker", "ts_utc"])
    if log.empty:
        raise SystemExit(f"{path}: the alert log is empty — nothing to match")
    return log.drop_duplicates().sort_values(["ticker", "ts_utc"]).reset_index(drop=True)


def build(cfg: dict, conn, hours: pd.DataFrame, model: AnalogMemory,
          log=print) -> pd.DataFrame:
    """Nearest past cases for every row of `hours`, one stock at a time."""
    _, cfg_on = configs(cfg)
    n = int(knobs(cfg)["n_cases"])
    # Bars kept before a stock's first alert: the memory reads `length` bars,
    # padded generously so a weekend or a halt never leaves a stretch short.
    margin = int(knobs(cfg)["margin_days"]) * DAY
    out = []
    tickers = hours["ticker"].unique()
    for i, ticker in enumerate(tickers):
        want = hours[hours["ticker"] == ticker]
        frame = latest_bar_frame(cfg_on, conn, [ticker], as_of=int(want["ts_utc"].max()),
                                 lookback_bars=10**6)
        frame = frame[frame["ts_utc"] >= int(want["ts_utc"].min()) - margin]
        out.append(model.nearest_cases(frame, want, n))
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(tickers)} stocks")
    return pd.concat(out, ignore_index=True)


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="first N alerted tickers only (smoke run)")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    k = knobs(cfg)
    hours = alerted_hours(Path(k["alert_log"]))
    if args.limit_tickers:
        hours = hours[hours["ticker"].isin(hours["ticker"].unique()[:args.limit_tickers])]
    started = time.time()
    print(f"fitting {k['variant']} on the train split...")
    model = memory(configs(cfg)[1], conn)
    print(f"matching {len(hours):,} alerted hours across "
          f"{hours['ticker'].nunique():,} stocks...")
    table = build(cfg, conn, hours, model)
    matched = table[["ticker", "ts_utc"]].drop_duplicates()
    if matched.empty:
        raise SystemExit("no alerted hour got a match — check the bars and channels")
    out = Path(cfg["paths"]["processed"]) / k["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    print(f"{len(matched):,} of {len(hours):,} hours matched; wrote {out} "
          f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
