"""The fresh holdout — events after the study window, built by the study's rules.

Added 2026-10-18 (the user's decision, "option A"). The study's sealed test set
was opened once, on 2026-09-10, and is spent. A model built after that date —
FootprintNet — needs a final test nobody has looked at, and the period after
the study window is exactly that: the live monitor stored its bars, filings and
news, but no detector was ever tuned on it.

Why a separate builder and not a wider `study_window`
-----------------------------------------------------
`t0.write_events` PRUNES every event outside `study_window`, and the events
table feeds the feature matrix and the sampler. Widening the window would put
holdout events where training code can reach them, and narrowing it back would
delete them again. So the holdout never touches `events`: it is written to its
own parquet file, and `evalset.build_eval_frame` takes it through an explicit
`events=` argument.

Same rules, called — not copied
-------------------------------
Each gate is the study's own function, run per filing:

* scope        in-universe ticker (the same `in_universe` flag, dated at the
               study start — a stated limitation, not re-derived), acceptance
               inside the holdout window, a form in `edgar.forms`
* items        `events.classify_items` — drop 9.01-only filings, tag scheduled
* coverage     `coverage.audit`, run read-only on a copy of the config whose
               `study_window` is the holdout window
* t0           `t0.match_filing` — min(acceptance, earliest article)
* materiality  `materiality.measure` on the news-adjusted t0, as in the study

A filing too close to the window's end to have its post-t0 bar is excluded by
materiality as `no_post_t0_bar`, the same reason the study uses.

Usage:
  python -m src.pipeline.holdout            # build, write, report
"""

from __future__ import annotations

import argparse
import copy
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from src import db
from src.pipeline import coverage, events as ev, materiality, t0
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

log = logging.getLogger(__name__)

COLUMNS = ["event_id", "accession_no", "ticker", "items", "t0_filing_utc",
           "t0_news_utc", "t0_utc", "t0_source", "is_scheduled", "abs_return",
           "is_material", "usable", "exclude_reason"]


def holdout_window(cfg: dict) -> tuple[int, int]:
    h = cfg["holdout"]
    lo, hi = date_str_to_ts(h["start"]), date_str_to_ts(h["end"])
    study_end = date_str_to_ts(cfg["study_window"]["end"])
    if lo < study_end:
        raise ValueError(f"holdout.start {h['start']} is inside the study window "
                         f"(ends {cfg['study_window']['end']}) — a holdout must "
                         f"never overlap data the study was tuned on")
    if hi <= lo:
        raise ValueError(f"holdout window is empty: {h['start']} .. {h['end']}")
    return lo, hi


def out_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["processed"]) / cfg["holdout"]["out"]


def build(cfg: dict, conn) -> pd.DataFrame:
    """One row per in-window, in-universe filing, every gate's verdict on it."""
    lo, hi = holdout_window(cfg)
    universe = set(db.universe_tickers(conn))
    filings = [f for f in db.filings_in_window(conn, lo, hi, cfg["edgar"]["forms"])
               if f["ticker"] in universe]
    if not filings:
        raise SystemExit(f"holdout: no in-universe filings between "
                         f"{cfg['holdout']['start']} and {cfg['holdout']['end']} — "
                         f"run the live filings fetch first")

    window_cfg = copy.deepcopy(cfg)
    window_cfg["study_window"] = {"start": cfg["holdout"]["start"],
                                  "end": cfg["holdout"]["end"]}
    cover = {v.accession_no: v.outcome for v in coverage.audit(window_cfg, conn)}
    series = materiality.load_series(conn, cfg["market"]["interval"])
    mcfg = cfg["materiality"]

    rows = []
    for f in filings:
        label = ev.classify_items(cfg, f["items"])
        match = t0.match_filing(cfg, conn, f["ticker"], f["acceptance_utc"])
        reason = label.exclude_reason
        if reason is None and cover.get(f["accession_no"], "no_bars") != "ok":
            reason = f"{ev.COVERAGE_REASON_PREFIX}{cover.get(f['accession_no'], 'no_bars')}"
        abs_ret, material = None, 0
        if reason is None:
            move = materiality.measure(cfg, series, f["ticker"], match.t0_utc)
            if move.reason:
                reason = move.reason
            else:
                abs_ret = abs(move.adjusted)
                material = int(abs_ret >= mcfg["min_abs_return"])
                reason = None if material else "immaterial"
        rows.append({
            "event_id": t0.event_id_for(f["accession_no"]),
            "accession_no": f["accession_no"], "ticker": f["ticker"],
            "items": f["items"], "t0_filing_utc": f["acceptance_utc"],
            "t0_news_utc": match.news_utc, "t0_utc": match.t0_utc,
            "t0_source": match.source, "is_scheduled": label.is_scheduled,
            "abs_return": abs_ret, "is_material": material,
            "usable": int(reason is None), "exclude_reason": reason,
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    if not out["usable"].any():
        top = Counter(out["exclude_reason"]).most_common(3)
        raise SystemExit(f"holdout: every one of {len(out)} filings was excluded "
                         f"({top}) — do not treat this as a test set")
    return out


def report(frame: pd.DataFrame) -> str:
    use = frame[frame["usable"] == 1]
    lines = [f"holdout filings: {len(frame):,}   usable events: {len(use):,} "
             f"(scheduled {int(use['is_scheduled'].sum()):,}, "
             f"unscheduled {int((use['is_scheduled'] == 0).sum()):,})",
             f"t0 from news: {int((use['t0_source'] == 'news').sum()):,} of {len(use):,}",
             "excluded:"]
    for reason, n in Counter(frame["exclude_reason"].dropna()).most_common():
        lines.append(f"  {reason:<28} {n:>6,}")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter
                            ).parse_args()
    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    frame = build(cfg, conn)
    path = out_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    print(report(frame))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
