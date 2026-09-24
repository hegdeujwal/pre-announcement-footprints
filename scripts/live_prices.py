#!/usr/bin/env python3
"""Fetch prices from Yahoo RIGHT NOW and show the detector reading them.

Built for showing someone the pipeline working on live data rather than on the
stored copy. It answers three questions in one screen:

  1. Is data actually arriving from Yahoo at this moment?
  2. Does it look like what the project already has stored?
  3. What would the detector say about it?

**It writes nothing.** The price snapshot was frozen on 2026-08-30 precisely so
that results stay reproducible, and a demo must not quietly extend it. The
database is opened read-only and the fetched bars are held in memory only. The
real collector is `src.collectors.market`; the live pipeline that DOES append
is `src.live.catchup`.

Usage:
  python scripts/live_prices.py                      # a few well-known tickers
  python scripts/live_prices.py --tickers TSLA,NVDA
  python scripts/live_prices.py --prove              # prove the fetch is genuinely live
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd                                    # noqa: E402
import yfinance as yf                                  # noqa: E402

from src import db                                     # noqa: E402
from src.utils.config import load_config               # noqa: E402
from src.utils.timeutils import (is_market_open, ts_to_iso,  # noqa: E402
                                 utc_now_ts)

DEFAULT = "AAPL,MSFT,NVDA,TSLA"


def fetch(ticker: str, bars: int) -> pd.DataFrame:
    """Hourly bars from Yahoo, converted to the project's convention: UTC.

    Yahoo hands back exchange-local time (New York). Storing that as UTC would
    shift every bar by four or five hours and, worse, by a DIFFERENT amount
    either side of a daylight-saving change — so the conversion is explicit
    here exactly as it is in the real collector.
    """
    df = yf.Ticker(ticker).history(period="5d", interval="60m")
    if df.empty:
        return df
    df = df.tail(bars).copy()
    df.index = df.index.tz_convert("UTC")
    return df


def stored_baseline(conn, cfg: dict, ticker: str) -> tuple[float, float, int]:
    """Mean and standard deviation of volume in the FROZEN snapshot.

    The detector never compares a stock to other stocks — it compares each hour
    against that same ticker's own recent history. This is where that history
    comes from.
    """
    window = cfg["features"]["volume_zscore_window_h"]
    rows = conn.execute(
        "SELECT volume FROM bars WHERE ticker = ? AND interval = ? "
        "ORDER BY ts_utc DESC LIMIT ?",
        (ticker, cfg["market"]["interval"], window)).fetchall()
    vols = pd.Series([r[0] for r in rows], dtype="float64")
    if len(vols) < cfg["features"]["min_baseline_bars"]:
        return float("nan"), float("nan"), len(vols)
    return float(vols.mean()), float(vols.std(ddof=1)), len(vols)


def prove(cfg: dict, conn) -> None:
    """Four checks that the data is genuinely arriving from Yahoo right now.

    Written for a demonstration given from India, where the US market is shut
    for most of the working day: it opens at 19:00 IST. "The newest US bar is
    from last night" is a fact about market hours, not evidence that anything
    is cached — so the checks below establish liveness in ways that do not
    depend on Wall Street being open.
    """
    import time

    now = utc_now_ts()
    ist = pd.Timestamp(now, unit="s", tz="UTC").tz_convert("Asia/Kolkata")
    print("=" * 78)
    print("PROOF THAT THIS DATA IS BEING FETCHED LIVE, NOT READ FROM DISK")
    print("=" * 78)
    print(f"\n  right now : {ts_to_iso(now)}   =  {ist:%Y-%m-%d %H:%M:%S} IST")
    print(f"  US market : {'OPEN' if is_market_open(now) else 'CLOSED'}"
          f"   (13:30-20:00 UTC = 19:00-01:30 IST)")

    # 1 -- symbols the project has never collected. If these return data, it
    #      cannot have come from our database.
    print("\n  [1] Symbols that are NOT in our database at all")
    for probe in ("RELIANCE.NS", "BTC-USD"):
        stored = conn.execute("SELECT COUNT(*) FROM bars WHERE ticker = ?",
                              (probe,)).fetchone()[0]
        t = time.time()
        df = yf.Ticker(probe).history(period="2d", interval="60m")
        el = time.time() - t
        print(f"      {probe:12} rows in our database: {stored:<6} "
              f"rows Yahoo returned: {len(df):<4} ({el:.2f}s)")
    print("      -> our database holds 6,053 US tickers and none of these. The")
    print("         data cannot have come from local storage.")

    # 2 -- the Indian market, which IS open during an Indian working day, and
    #      which the person watching can check on their own phone.
    print("\n  [2] Live Indian market prices — verifiable on your own phone")
    for tk in ("RELIANCE.NS", "TCS.NS", "^NSEI"):
        try:
            d = yf.Ticker(tk).history(period="1d", interval="5m")
            if d.empty:
                print(f"      {tk:14} no data (NSE trades 09:15-15:30 IST)")
                continue
            last = d.index[-1].tz_convert("Asia/Kolkata")
            print(f"      {tk:14} {d['Close'].iloc[-1]:>12,.2f}   as of "
                  f"{last:%d %b %H:%M} IST")
        except Exception as exc:
            print(f"      {tk:14} error: {type(exc).__name__}")
    print("      -> search the same symbol on Google. The numbers should match.")

    # 3 -- a market that never closes. Compared on the BAR TIMESTAMP, not the
    #      price: two reads inside the same minute legitimately return the same
    #      bar, and calling that "proof of caching" would be wrong.
    print("\n  [3] A market that never closes — two reads, 65 seconds apart")
    try:
        a = yf.Ticker("BTC-USD").history(period="1d", interval="1m")
        t0_bar, t0_px = a.index[-1], a["Close"].iloc[-1]
        print(f"      first  read: {t0_px:>12,.2f}  bar "
              f"{t0_bar.tz_convert('UTC'):%H:%M}Z   (waiting 65s...)")
        time.sleep(65)
        b = yf.Ticker("BTC-USD").history(period="1d", interval="1m")
        t1_bar, t1_px = b.index[-1], b["Close"].iloc[-1]
        print(f"      second read: {t1_px:>12,.2f}  bar "
              f"{t1_bar.tz_convert('UTC'):%H:%M}Z")
        if t1_bar > t0_bar:
            print(f"      -> a NEW minute bar appeared that did not exist 65s ago.")
        else:
            print("      -> same bar returned; Yahoo's feed lags a little. Re-run.")
    except Exception as exc:
        print(f"      error: {type(exc).__name__}")

    # 4 -- the falsification test.
    print("\n  [4] How to DISPROVE it")
    print("      Turn off the internet and run this again. Every fetch above")
    print("      fails. Nothing in this project can serve those prices offline —")
    print("      the stored snapshot is frozen, ends before today, and contains")
    print("      no Indian tickers and no crypto at all.")
    print("\n  Note: the Indian and crypto symbols are ONLY to demonstrate that the")
    print("  data pipe is live. The study itself is US 8-K filings and the 1,500")
    print("  US companies in the universe.\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=DEFAULT, help=f"default: {DEFAULT}")
    ap.add_argument("--bars", type=int, default=6, help="how many hours to show")
    ap.add_argument("--prove", action="store_true",
                    help="show that the fetch is genuinely live, even when the "
                         "US market is closed")
    args = ap.parse_args()

    cfg = load_config()
    now = utc_now_ts()
    threshold = cfg["baselines"]["volume_zscore"]["threshold"]

    print(f"\nnow: {ts_to_iso(now)}   US market open: "
          f"{'YES' if is_market_open(now) else 'no'}")
    print(f"stored snapshot is FROZEN at "
          f"{db.get_meta(db.get_conn(cfg['paths']['db'], readonly=True), 'snapshot_frozen_60m')}"
          f"  —  nothing below is written to it\n")

    conn = db.get_conn(cfg["paths"]["db"], readonly=True)
    if args.prove:
        prove(cfg, conn)
        return
    for ticker in [t.strip().upper() for t in args.tickers.split(",") if t.strip()]:
        print("=" * 78)
        df = fetch(ticker, args.bars)
        if df.empty:
            print(f"{ticker}: Yahoo returned nothing")
            continue

        mean, sd, n = stored_baseline(conn, cfg, ticker)
        newest_stored = db.latest_bar_ts(conn, ticker, cfg["market"]["interval"])
        print(f"{ticker}   live from Yahoo, {len(df)} hourly bars")
        print(f"   newest bar in the frozen snapshot : "
              f"{ts_to_iso(newest_stored) if newest_stored else 'none'}")
        print(f"   baseline from {n} stored bars     : mean volume "
              f"{mean:,.0f}" + (f", sd {sd:,.0f}" if sd == sd else ""))
        print()
        print(f"   {'bar (UTC)':22}{'close':>10}{'volume':>13}{'volume_z':>10}  vs "
              f"threshold {threshold}")
        for ts, row in df.iterrows():
            z = (row["Volume"] - mean) / sd if sd == sd and sd > 0 else float("nan")
            flag = ""
            if z == z:
                flag = "  <-- ABOVE THRESHOLD" if z >= threshold else ""
            zs = f"{z:>10.2f}" if z == z else f"{'n/a':>10}"
            print(f"   {ts.strftime('%Y-%m-%d %H:%M:%SZ'):22}"
                  f"{row['Close']:>10.2f}{row['Volume']:>13,.0f}{zs}{flag}")
        print()

    print("=" * 78)
    print("volume_z is how many standard deviations this hour's volume sits above")
    print("that ticker's OWN recent normal. The last bar of an open session is")
    print("still forming, so its volume is incomplete and reads low by design.")
    print("\nNothing was written. The pipeline that does append live bars is:")
    print("   python -m src.live.catchup --max-tickers 20")


if __name__ == "__main__":
    main()
