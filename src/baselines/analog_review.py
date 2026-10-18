"""The analog-memory feasibility check — two questions, validation split only.

Added 2026-10-18. Before the mentor's "remember patterns, match them in other
companies" idea becomes a phase, measure whether it has any signal:

1. **Detection.** Does an analog row beat CUSUM and volume z-score at the same
   alert budget? `comparison.csv`, `within_stock.csv`, `lead_time.csv` — the
   same tables, produced the same way, as `guide_review`.
2. **Explanation.** When a real filing follows, does the nearest past case
   name the right 8-K item more often than guesses that never look at the
   bars? `items.csv`, on two sets of filings:
     * `alerted`  — filings the analog row caught at the budget, guessed at
                    the hour it flagged (what an analyst would actually see)
     * `last_bar` — every validation filing, guessed at its last bar before
                    t0 (more events, so a tighter interval)
   `examples.csv` lists caught filings beside their nearest past case, so the
   output can be read, not only counted.

Reference rows (always-quiet, random noise, volume z-score, CUSUM, the pattern
template) are re-scored here on the identical frame so the comparison never
mixes runs. The test split stays sealed: it was opened once, on 2026-09-10.

Lives beside `compare.py` rather than in `src/eval/`, because it wires
detectors together and `eval` never imports models.

Usage:
  python -m src.baselines.analog_review
  python -m src.baselines.analog_review --limit-tickers 40   # smoke run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.clock_only import ClockOnly
from src.baselines.analog import AnalogMemory
from src.baselines.compare import (build_training_frame, comparison_table,
                                   conform, noise_seeds)
from src.baselines.cusum import CUSUM
from src.baselines.guide_review import (SLICES, configs, lead_rows,
                                       same_hour_rows, with_lift)
from src.baselines.prior_volume import final_bars
from src.baselines.random_noise import RandomNoise
from src.baselines.template import PatternTemplate
from src.baselines.volume_zscore import VolumeZScore
from src.eval import analog_items as AI
from src.eval.contract import FLAG, actions_from_scores
from src.eval.metrics import precision_at_alert_budget
from src.eval.within_stock import within_stock_table
from src.pipeline.evalset import build_eval_frame, split_bounds
from src.utils.config import load_config
from src.utils.timeutils import ts_to_dt


def analog_variants(cfg: dict) -> list[AnalogMemory]:
    variants = cfg["baselines"]["analog"].get("variants") or []
    if not variants:
        raise ValueError("baselines.analog.variants is empty — nothing to check")
    return [AnalogMemory(cfg, channels=v["channels"],
                         exclude_same_ticker=v.get("exclude_same_ticker", False),
                         name=v["name"]) for v in variants]


def references(cfg: dict, conn, train: pd.DataFrame):
    yield "null", AlwaysQuiet(cfg)
    for s in noise_seeds(cfg):
        yield "null", RandomNoise(cfg, seed=s)
    yield "null", ClockOnly(cfg)
    yield "pure technical", VolumeZScore(cfg)
    yield "pure technical", CUSUM(cfg)
    yield "pure technical", PatternTemplate(cfg).fit(train, conn=conn)


def alerted_rows(pred: pd.DataFrame) -> pd.DataFrame:
    """The flag hour of every filing caught at the budget's operating point."""
    threshold = precision_at_alert_budget(pred).threshold
    acts = actions_from_scores(pred, threshold)
    hit = acts[(acts["action"] == FLAG) & acts["t0_utc"].notna()]
    hit = hit.sort_values(["window_id", "ts_utc"]).drop_duplicates("window_id")
    return hit[["window_id", "ticker", "ts_utc", "t0_utc", "is_scheduled",
                "item_code", "score"]].reset_index(drop=True)


def last_bar_rows(frame: pd.DataFrame) -> pd.DataFrame:
    last = final_bars(frame)
    last = last[last["t0_utc"].notna()]
    return last[["window_id", "ticker", "ts_utc", "t0_utc", "is_scheduled",
                 "item_code"]].reset_index(drop=True)


def item_table(cfg: dict, model: AnalogMemory, frame: pd.DataFrame,
               rows: pd.DataFrame, train_events: pd.DataFrame,
               label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(hit-rate rows, the guessed events) for one set of filings."""
    knobs = cfg["eval"]["analog_items"]
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    expl = model.explain(frame, rows)
    events = rows.reset_index(drop=True).copy()
    events[AI.GUESS_PREFIX + "analog"] = expl["guess"].fillna("").to_numpy()
    for c in ("nearest_ticker", "nearest_t0_utc", "nearest_items",
              "nearest_distance"):
        events[c] = expl[c].to_numpy()
    events = AI.baseline_guesses(train_events, events, model.drop_items,
                                 model.top_n, int(knobs["seed"]))
    rates = AI.hit_rates(events, model.drop_items, model.top_n,
                         int(knobs["bootstrap"]), int(knobs["seed"]),
                         float(knobs["ci"]), label=label)
    return rates.assign(detector=model.name), events.assign(detector=model.name)


def examples(events: pd.DataFrame, n: int) -> pd.DataFrame:
    """The strongest caught filings beside their nearest past case, readable."""
    if events.empty:
        return events
    top = events.sort_values("score", ascending=False).head(n)

    def iso(v):
        return "" if pd.isna(v) else ts_to_dt(int(v)).strftime("%Y-%m-%d %H:%M UTC")

    return pd.DataFrame({
        "detector": top["detector"],
        "ticker": top["ticker"],
        "flagged_at": top["ts_utc"].map(iso),
        "filed_at_t0": top["t0_utc"].map(iso),
        "actual_items": top["item_code"].map(
            lambda c: ",".join(AI.substantive_items(c, ["9.01"]))),
        "analog_guess": top[AI.GUESS_PREFIX + "analog"],
        "looked_like": top["nearest_ticker"],
        "that_filing_t0": top["nearest_t0_utc"].map(iso),
        "that_filing_items": top["nearest_items"],
        "distance": top["nearest_distance"],
    })


def run(cfg: dict, conn, split: str, out_dir: Path,
        tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    cfg_off, cfg_on = configs(cfg)
    lo, hi = split_bounds(cfg_on, split, conn=conn)
    variant = "news_adjusted"

    # Technical columns on for BOTH frames: the price variants read range_z and
    # candle_body. Reference detectors read only their own columns, so the
    # extra columns change none of their numbers.
    print("building the evaluation frame (technical columns on)...")
    frame = conform(build_eval_frame(cfg_on, conn, lo, hi, tickers=tickers,
                                     t0_variant=variant))
    print("building the 3:1 train frame (technical columns on)...")
    train = build_training_frame(cfg_on, conn)
    tf = final_bars(train)
    train_events = tf[tf["t0_utc"].notna()][["ticker", "item_code", "is_scheduled"]]
    last_rows = last_bar_rows(frame)

    comp, lead, within, items, guessed = [], [], [], [], []
    comp_same, within_same = [], []

    def measure(family, model, pred):
        table = comparison_table(cfg_off, {model.name: pred}, variant,
                                 models={model.name: model})
        comp.append(table.assign(family=family))
        lead.extend(lead_rows(model.name, pred))
        within.append(within_stock_table(cfg, {model.name: pred}, variant)
                      .assign(family=family))
        c, w = same_hour_rows(cfg, cfg_off, model, pred, variant, family)
        comp_same.append(c)
        within_same.append(w)

    for family, model in references(cfg_off, conn, train):
        started = time.time()
        pred = model.predict(frame, threshold=float("inf"), conn=conn,
                             context=f"analog_review/{model.name}")
        measure(family, model, pred)
        del pred
        print(f"  {model.name:40s} {time.time() - started:5.0f}s")

    for model in analog_variants(cfg_on):
        started = time.time()
        model.fit(train, conn=conn)
        pred = model.predict(frame, threshold=float("inf"), conn=conn,
                             context=f"analog_review/{model.name}")
        measure("analog memory", model, pred)
        caught = alerted_rows(pred)
        del pred
        for label, rows in (("alerted", caught), ("last_bar", last_rows)):
            r, e = item_table(cfg, model, frame, rows, train_events, label)
            items.append(r)
            if label == "alerted":
                guessed.append(e)
        print(f"  {model.name:40s} {time.time() - started:5.0f}s")

    comp = with_lift(pd.concat(comp, ignore_index=True))
    guessed = pd.concat(guessed, ignore_index=True) if guessed else pd.DataFrame()
    out = {
        "comparison": comp,
        "comparison_same_hour": with_lift(pd.concat(comp_same, ignore_index=True)),
        "within_stock_same_hour": pd.concat(within_same, ignore_index=True),
        "lead_time": pd.DataFrame(lead),
        "within_stock": pd.concat(within, ignore_index=True),
        "items": pd.concat(items, ignore_index=True),
        "examples": examples(guessed, int(cfg["eval"]["analog_items"]["examples"])),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    return out


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", help="val (default) or train")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="evaluate only the first N in-universe tickers (smoke run)")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    out_dir = Path(args.out_dir or Path(cfg["paths"]["processed"]) / "analog-check")
    tickers = None
    if args.limit_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (args.limit_tickers,))]
    started = time.time()
    out = run(cfg, conn, args.split, out_dir, tickers=tickers)

    for key, label in (("comparison", "all hours"),
                       ("comparison_same_hour", "closing-bar windows only")):
        comp = out[key]
        for s in SLICES:
            view = comp[comp["slice"] == s].sort_values("lift", ascending=False)
            print(f"\n-- detection, {s}, {label} --")
            print(view[["baseline", "precision", "lift", "recall",
                        "median_lead_trading_h"]].to_string(
                            index=False, float_format=lambda v: f"{v:.4f}"))
    items = out["items"]
    top_n = cfg["baselines"]["analog"]["top_n"]
    for label in ("alerted", "last_bar"):
        view = items[(items["set"] == label) & (items["slice"] != "all")]
        print(f"\n-- item guess, {label} --")
        print(view[["detector", "method", "slice", "n_events", "hit_at_1",
                    f"hit_at_{top_n}"]].to_string(
                        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {out_dir}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
