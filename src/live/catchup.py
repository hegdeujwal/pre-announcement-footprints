"""One command, the whole cycle: fetch, scan, log, score, export.

This is what the scheduled job runs. It exists as a single entry point rather
than four so an unattended run has one exit code and one log to read — a
four-step shell pipeline fails in four ways, and three of them are silent.

**It does not need to run continuously.** yfinance serves roughly two years of
hourly history, so a run that happens once a day, or after a gap of a week,
fetches every hour it missed. The alert log records the bar (`ts_utc`) and the
moment the monitor noticed (`raised_utc`) in separate columns, so a catch-up
run is self-describing: nothing is disguised as real-time.

What the evidence is, stated precisely
--------------------------------------
This is **not** live alerting, and the report should not say it is. What it is,
and what actually matters for defending a result: a **pre-registered
out-of-sample test**. The detectors, their tuned thresholds and this code were
committed to git before these bars existed, so there is no route by which they
could have been tuned to them. Git's timestamps make that ordering checkable by
someone who does not trust the author, and the alert log's hash chain makes the
record internally consistent. Neither depends on the monitor having run at any
particular instant.

Usage:
  python -m src.live.catchup                    # the full cycle
  python -m src.live.catchup --max-tickers 300  # stay inside a CI budget
  python -m src.live.catchup --no-fetch         # score what is already stored
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts, ts_to_iso, utc_now_ts

#: Where the durable log lives. Committed to the repository, because the
#: database is a rebuildable cache and this is not.
#:
#: Deliberately NOT under `data/`. The code standards say never commit anything
#: there, and they are right — `data/` holds a 1 GB database, raw API caches
#: and parquet files, all regenerable bulk. This file is the opposite: small,
#: append-only, and the one artefact of Phase 7 that cannot be rebuilt from any
#: source. Putting it at the top level keeps the standard intact rather than
#: carving an exception into it, and `.gitignore`'s `data/**` would otherwise
#: have silently dropped every commit the scheduled job made.
DEFAULT_LOG_CSV = "live-log/alerts.csv"

#: The grades for that log — did a filing follow each alert? Committed beside
#: it so any copy of the project sees the same answers the scheduled job found,
#: not just whatever its own database can grade.
DEFAULT_OUTCOMES_CSV = "live-log/outcomes.csv"

#: Sessions the volume check took out of `bars` instead of scoring. Committed
#: for the same reason as the two files above: the database is a cache.
DEFAULT_QUARANTINE_CSV = "live-log/quarantine.csv"


def run(cfg: dict, conn, max_tickers: int | None = None,
        fetch: bool = True, log_csv: str | None = DEFAULT_LOG_CSV,
        as_of: int | None = None,
        outcomes_csv: str | None = DEFAULT_OUTCOMES_CSV,
        quarantine_csv: str | None = DEFAULT_QUARANTINE_CSV,
        restate_from: int | None = None) -> dict:
    """Fetch, scan, log, backfill outcomes, export. Returns a summary dict.

    Ordering matters. Outcomes are backfilled **after** the new alerts are
    logged, so an alert raised today and a filing that lands within the same
    run are both accounted for. And the export happens last, so the committed
    CSV reflects everything this run learned.
    """
    from src.live.alertlog import append, export_csv, summary, verify_chain
    from src.live.monitor import (check_session_volumes, conform,
                                  export_quarantine_csv, fetch_latest,
                                  fetch_recent_filings, latest_bar_frame,
                                  latest_stored_bar, restate_start,
                                  retime_filings)
    from src.live.outcomes import backfill, export_outcomes_csv, regrade

    started = time.time()
    result: dict = {"started_utc": utc_now_ts()}

    tickers = None
    if max_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (max_tickers,))]

    if fetch:
        # Where the restate will begin, read BEFORE fetching: afterwards the
        # newest bar has moved on and this would name the wrong session.
        newest = latest_stored_bar(conn, cfg["market"]["interval"])
        since = (restate_start(cfg, conn, newest, restate_from)
                 if newest is not None else None)
        result["bars_appended"] = fetch_latest(cfg, conn, tickers=tickers,
                                               restate_from=restate_from)
        # Before anything is scored: a session that fails is removed from
        # `bars`, so neither its alerts nor its volumes reach the frame.
        result["volume_check"] = (check_session_volumes(cfg, conn, since,
                                                        tickers=tickers)
                                  if since is not None else None)
        # Filings BEFORE the frame is built, for two separate reasons. One
        # feature — days_since_last_8k — reads this table, so a stale filings
        # table would score today's bars against yesterday's idea of when the
        # company last filed. And `backfill` below can only answer an alert
        # whose window ends at or before the newest filing on record, so
        # without this the horizon never advances and every alert the monitor
        # ever raises stays deferred forever.
        result["filings"] = fetch_recent_filings(cfg, conn, tickers=tickers)
        # Filings stored before acceptance times came from headers.
        # Self-limiting: once every live-period filing is re-timed this
        # finds nothing and makes no request.
        result["retime"] = retime_filings(cfg, conn)
    else:
        result["bars_appended"] = 0
        result["volume_check"] = None
        result["filings"] = None
        result["retime"] = None

    # A grade is only as good as the filing times behind it: when any time
    # moved, every grade is recomputed; otherwise only rows graded before the
    # kind columns existed are filled in.
    moved = bool(result["retime"] and result["retime"]["moved"])
    result["regrade"] = regrade(cfg, conn, only_missing_kinds=not moved)

    frame = conform(latest_bar_frame(cfg, conn, tickers, as_of=as_of))
    result["bars_scored"] = len(frame)
    result["tickers"] = int(frame["ticker"].nunique())
    result["newest_bar_utc"] = int(frame["ts_utc"].max())

    alerts = scan_alerts(cfg, conn, frame)
    result["alerts_found"] = len(alerts)
    result["alerts_new"] = append(conn, alerts)

    # After logging, so a filing that lands in the same run is picked up.
    result["outcomes"] = backfill(cfg, conn)

    # The chain is verified on every run rather than on demand: a break found
    # weeks later is a break nobody can date.
    result["chain"] = {k: v["ok"] for k, v in verify_chain(conn).items()}
    result["log"] = summary(conn)

    if log_csv:
        result["exported_rows"] = export_csv(conn, log_csv)
        result["log_csv"] = str(log_csv)
    if outcomes_csv:
        result["exported_outcomes"] = export_outcomes_csv(conn, outcomes_csv)
        result["outcomes_csv"] = str(outcomes_csv)
    if quarantine_csv:
        result["exported_quarantine"] = export_quarantine_csv(conn, quarantine_csv)
        result["quarantine_csv"] = str(quarantine_csv)

    result["elapsed_s"] = round(time.time() - started, 1)
    return result


def scan_alerts(cfg: dict, conn, frame):
    """Score the frame with the rule detectors and every configured policy."""
    from src.live.monitor import build_detectors, live_policies, scan
    runs = [p["run"] for p in live_policies(cfg)]
    return scan(cfg, conn, frame, build_detectors(cfg, policy_runs=runs))


def _volume_line(check: dict | None) -> str:
    if not check:
        return "volume check  : skipped (nothing fetched)"
    line = (f"volume check  : {check['sessions']:,} sessions, "
            f"{check['quarantined']} quarantined, "
            f"{check['unverified']} unverifiable (no daily bar)")
    if check["rows"]:
        line += "\n                " + ", ".join(
            f"{r['ticker']} {r['session']} ({r['ratio']}x)" for r in check["rows"][:20])
    return line


def render(result: dict) -> str:
    """A summary an unattended run can be read from, days later."""
    f = result.get("filings")
    lines = [
        f"bars appended : {result['bars_appended']:,}",
        (f"filings       : {f['new_filings']:,} new from {f['records']:,} "
         f"records across {f['companies']:,} companies"
         + (f", {f['failed']} failed" if f["failed"] else "")
         if f else "filings       : skipped (--no-fetch)"),
        _volume_line(result.get("volume_check")),
        f"bars scored   : {result['bars_scored']:,} "
        f"across {result['tickers']:,} tickers",
        f"newest bar    : {ts_to_iso(result['newest_bar_utc'])}",
        f"alerts        : {result['alerts_found']} found, "
        f"{result['alerts_new']} new",
    ]
    rt, rg = result.get("retime"), result.get("regrade")
    if f:
        lines.append(f"filing times  : {f.get('header_timed', 0)} new from headers"
                     + (f", {f['header_failed']} held back" if f.get("header_failed") else "")
                     + (f"; JSON drift (h) {f['json_drift_h']}" if f.get("json_drift_h") else ""))
    if rt:
        lines.append(f"re-timed      : {rt['checked']} filings, {rt['moved']} moved"
                     + (f", {rt['failed']} unreadable" if rt["failed"] else "")
                     + (f"; drift (h) {rt['drift_h']}" if rt["drift_h"] else ""))
    if rg:
        lines.append(f"regraded      : {rg['regraded']} outcomes, "
                     f"{rg['filed_changed']} changed filed/not filed")
    o = result["outcomes"]
    lines.append(f"outcomes      : scored {o['scored']}, filed {o['filed']}, "
                 f"deferred {o['pending']} "
                 f"(filing horizon {ts_to_iso(o['horizon_utc'])})")
    broken = [k for k, ok in result["chain"].items() if not ok]
    lines.append(f"chain         : {'ok' if not broken else 'BROKEN: ' + ', '.join(broken)}")
    for name, s in result["log"].items():
        lines.append(f"  {name:20s} {s['alerts']:6,d} alerts total")
    if "exported_rows" in result:
        lines.append(f"exported      : {result['exported_rows']:,} rows -> "
                     f"{result['log_csv']}")
    if "exported_outcomes" in result:
        lines.append(f"              : {result['exported_outcomes']:,} outcomes "
                     f"-> {result['outcomes_csv']}")
    if "exported_quarantine" in result:
        lines.append(f"              : {result['exported_quarantine']:,} quarantined "
                     f"sessions -> {result['quarantine_csv']}")
    lines.append(f"elapsed       : {result['elapsed_s']}s")
    return "\n".join(lines)


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tickers", type=int, default=None,
                    help="cap the universe, to stay inside a CI minute budget")
    ap.add_argument("--no-fetch", action="store_true",
                    help="score what is already stored; download nothing")
    ap.add_argument("--log-csv", default=DEFAULT_LOG_CSV,
                    help="where to export the durable log")
    ap.add_argument("--outcomes-csv", default=DEFAULT_OUTCOMES_CSV,
                    help="where to export the log's graded outcomes")
    ap.add_argument("--quarantine-csv", default=DEFAULT_QUARANTINE_CSV,
                    help="where to export the volume check's quarantine record")
    ap.add_argument("--as-of", type=int, default=None)
    ap.add_argument("--restate-from", default=None, metavar="YYYY-MM-DD",
                    help="one-off repair: re-fetch and overwrite every live bar "
                         "from this UTC date (never before the snapshot stamp)")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    result = run(cfg, conn, max_tickers=args.max_tickers,
                 fetch=not args.no_fetch, log_csv=args.log_csv,
                 as_of=args.as_of, outcomes_csv=args.outcomes_csv,
                 quarantine_csv=args.quarantine_csv,
                 restate_from=(date_str_to_ts(args.restate_from)
                               if args.restate_from else None))
    print(render(result))

    # A broken chain is the one condition that must fail the job rather than
    # be noticed in a log nobody reads.
    if any(not ok for ok in result["chain"].values()):
        raise SystemExit("alert log chain is BROKEN — see the report above")


if __name__ == "__main__":
    main()
