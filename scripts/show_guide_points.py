#!/usr/bin/env python3
"""Show, point by point, that the guide's review points (2026-10-15) are done.

For each point it prints where the code is, the test that checks it, and the
measured result — read from the COMMITTED tables in results/, so it works on a
fresh clone with no database and no network. Points 2 and 5 also run the real
code on a tiny made-up example, so the method can be seen working.

    .venv/bin/python scripts/show_guide_points.py          # every point
    .venv/bin/python scripts/show_guide_points.py 4        # one point
    .venv/bin/python scripts/show_guide_points.py 1 9 10   # several

Point 7 (the 20 papers) was skipped on purpose and has nothing to show.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402

GR = ROOT / "results" / "guide-review"
FINAL = ROOT / "results" / "phase10" / "FINAL-test-evaluation-r2.csv"
SPLITS = ("all", "scheduled", "unscheduled")

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def head(n: str, title: str) -> None:
    print(f"\n{'=' * 78}\nPOINT {n} — {title}\n{'=' * 78}")


def where(code: list[str], tests: list[str], extra: list[str] = ()) -> None:
    print("Code:  " + "\n       ".join(code))
    print("Test:  .venv/bin/python -m pytest -q " + " ".join(tests))
    for line in extra:
        print(line)
    print()


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(GR / name)


def lifts(names: list[str]) -> pd.DataFrame:
    """Lift at the alert budget, one row per detector, split three ways."""
    c = csv("comparison.csv")
    c = c[c["baseline"].isin(names) & c["slice"].isin(SPLITS)]
    t = c.pivot(index="baseline", columns="slice", values="lift")[list(SPLITS)]
    return t.reindex([n for n in names if n in t.index]).round(2)


def p1() -> None:
    head("1", "Gaussian distribution: checked, and two repaired volume scores")
    where(["src/pipeline/technical.py  volume_scores()  -> volume_logz, volume_tod_z",
           "src/baselines/guide_review.py  distribution_table()"],
          ["tests/test_technical.py"])
    d = csv("distribution.csv").set_index("score")
    d["P(z>3) vs bell curve"] = (d["p_above_3"] / d["gaussian_p_above_3"]).round(1)
    print("How far each score is from a bell curve (a bell curve has skew 0, "
          "excess kurtosis 0, ratio 1):")
    print(d[["sd", "skew", "excess_kurtosis", "p_above_3",
             "P(z>3) vs bell curve"]].round(3).to_string(), "\n")
    print("Lift at the alert budget (higher is better):")
    print(lifts(["volume_zscore", "volume_logz", "volume_tod_z"]).to_string(), "\n")
    w = csv("within_stock.csv")
    w = w[w["baseline"].isin(["volume_zscore", "volume_tod_z"])
          & w["slice"].isin(["scheduled", "unscheduled"])]
    print("Within-stock AUC (same stock, other hours; 0.5 = coin flip):")
    print(w.pivot(index="baseline", columns="slice", values="auc").round(3).to_string())


def p2() -> None:
    from src.baselines.cusum import cusum_statistic

    head("2", "Cumulative sum (CUSUM): the detector already in the project")
    where(["src/baselines/cusum.py  cusum_statistic()  (built Phase 5)",
           "config/config.yaml  baselines.cusum  (drift k, threshold h)"],
          ["tests/test_baselines_cusum.py"])
    cfg = load_config()["baselines"]["cusum"]
    k, h = float(cfg["drift"]), float(cfg["threshold"])
    z = np.array([0.3, 1.0, 1.2, 0.8, 2.0, 2.2, 2.4, 0.5, -0.4, 4.0, 0.2, 0.9])
    s = cusum_statistic(z, k=k, h=h)
    print(f"The real function on a made-up run of hourly volume z-scores "
          f"(k = {k}, h = {h}):")
    print(pd.DataFrame({"hour": range(1, len(z) + 1), "volume z": z,
                        "running total S": s.round(2),
                        "flag": np.where(s >= h, "FLAG", "")}).to_string(index=False))
    print("\nS = max(0, S + z - k); a FLAG when S reaches h, then S restarts.\n")
    print("Lift at the alert budget:")
    print(lifts(["cusum", "volume_zscore"]).to_string())


def p3() -> None:
    head("3", "How early, on average: already reported, mean added")
    where(["results/phase10/FINAL-test-evaluation-r2.csv  column median_lead_trading_h",
           "app/screens.py  evaluation()  table column 'median lead'",
           "src/baselines/guide_review.py  lead_rows()  (adds the mean)"],
          ["tests/test_metrics_delay.py"],
          ["Dashboard: .venv/bin/streamlit run app/dashboard.py -> 'Evaluation' "
           "-> 'median lead' column"])
    f = pd.read_csv(FINAL)
    f = f[(f["t0_variant"] == "news_adjusted")
          & f["baseline"].isin(["cusum", "volume_zscore"]) & f["slice"].isin(SPLITS)]
    print("Already in the final TEST results (trading hours before t0):")
    print(f.pivot(index="baseline", columns="slice",
                  values="median_lead_trading_h")[list(SPLITS)].round(1).to_string(), "\n")
    lt = csv("lead_time.csv")
    lt = lt[lt["baseline"].isin(["cusum", "volume_zscore"]) & lt["slice"].isin(SPLITS)]
    print("Added on VALIDATION — mean beside median:")
    print(lt[["baseline", "slice", "n_detected", "mean_lead_trading_h",
              "median_lead_trading_h"]].round(1).to_string(index=False))


def p4() -> None:
    head("4", "Flagging filter according to the impact")
    where(["src/eval/impact.py  impact_edges(), impact_rows()"],
          ["tests/test_eval_impact.py"])
    e = csv("impact_edges.csv")
    print("Impact buckets, cut at the TRAIN events' terciles of the move after "
          "t0: low < {:.1%} <= mid < {:.1%} <= high\n".format(*e["bucket_upper_edge"]))
    t = csv("impact.csv")
    c = t[(t["baseline"] == "cusum") & t["slice"].str.startswith("impact")]
    print("CUSUM recall by impact bucket:")
    print(c.pivot(index="schedule", columns="slice", values="recall")
           .reindex(list(SPLITS)).round(3).to_string(), "\n")
    rho = t[(t["baseline"] == "cusum") & (t["slice"] == "spearman score~impact")]
    print("Does a louder alert mean a bigger move? Spearman rho (0 = no):")
    print(rho[["schedule", "n_positive", "spearman_rho", "spearman_p"]]
          .round(3).to_string(index=False))


def p5() -> None:
    from src.pipeline.technical import candles, technical_columns

    head("5", "Technical indicators and candle patterns")
    where(["src/pipeline/technical.py  candles(), indicators(), technical_features()",
           "config/config.yaml  features.technical  (enabled: false by default)"],
          ["tests/test_technical.py"])
    cfg = load_config()
    cfg["features"]["technical"]["enabled"] = True
    cols = technical_columns(cfg)
    print(f"{len(cols)} new columns: {', '.join(cols)}\n")
    bars = pd.DataFrame({"open": [10.0, 10.0, 10.0], "high": [10.5, 10.32, 11.0],
                         "low": [9.5, 9.3, 10.0], "close": [10.02, 10.3, 11.0],
                         "volume": [1e5, 1e5, 1e5]})
    out = candles(bars, cfg)
    pats = [c for c in out.columns if c.startswith("pat_")]
    print("The real code on three made-up candles (a doji, a hammer, a marubozu):")
    print(pd.concat([bars[["open", "high", "low", "close"]], out[pats]], axis=1)
          .to_string(index=False), "\n")
    print("Lift at the alert budget, gradient boosting without and with them:")
    print(lifts(["gradient_boosting", "gradient_boosting[technical]", "cusum"]).to_string())


def p6() -> None:
    head("6", "LLM summary of the 8-K (built as a display, not a reward)")
    where(["src/live/summaries.py  generate(), call_groq()",
           "app/screens.py  _llm_summary()  — the 'Generate summary' button",
           "config/config.yaml  live.summaries;  GROQ_API_KEY in .env"],
          ["tests/test_live_summaries.py",
           "tests/test_dashboard.py::test_a_filing_summary_is_made_only_on_click"],
          ["Dashboard: 'Ticker detail' -> 'Filing history' -> pick a filing -> "
           "'Generate summary'",
           "Command:   .venv/bin/python -m src.live.summaries <accession_no>"])
    cfg = load_config()
    saved = sorted(Path(ROOT / cfg["live"]["summaries"]["dir"]).glob("*.json"))
    print(f"Summaries saved on this machine: {len(saved)} "
          f"(in {cfg['live']['summaries']['dir']}/)")
    if saved:
        rec = json.loads(saved[0].read_text())
        print(f"\nExample — {rec['accession_no']} ({rec.get('ticker')}, items "
              f"{', '.join(rec['items'])}), model {rec['model']}:\n")
        print(rec["summary"])


def p8() -> None:
    head("8", "The statistical patterns before a filing")
    where(["src/eval/pattern_study.py  (new)",
           "src/eval/event_study.py   (already there: the footprint chart)"],
          ["tests/test_event_study.py tests/test_technical.py"],
          ["Dashboard: 'Evaluation' -> 'The footprint itself — trading around t0'",
           "Re-run:    .venv/bin/python -m src.eval.pattern_study"])
    s = csv("indicator_shift.csv")
    print("Last 7 hours before t0 vs the same stock at the same time 10 sessions "
          "earlier (difference in the indicator's own units — SD for the two "
          "z-scores, RSI points for rsi — with its 95% interval):")
    print(s[["slice", "indicator", "difference", "ci_lo", "ci_hi", "n_events"]]
          .round(3).to_string(index=False), "\n")
    r = csv("pattern_rates.csv")
    print("How often each candle pattern appears, before vs control (ratio 1 = no change):")
    print(r[["slice", "pattern", "share_before", "share_control", "ratio"]]
          .round(3).to_string(index=False))


def p9() -> None:
    head("9", "A template of every pattern, then reward accordingly")
    where(["src/baselines/template.py  PatternTemplate  (the detector)",
           "src/rl/env.py  _template_bonus()  (the reward)",
           "config/config.yaml  baselines.pattern_template, reward.r_template_bonus"],
          ["tests/test_baselines_template.py"],
          ["Train:  .venv/bin/python -m src.rl.train --seed 43 --template-bonus 0.5 "
           "--run-dir data/runs/tmpl-r1-s43"])
    t = csv("templates.csv")
    t = t[t["detector"] == "pattern_template"]
    print("Templates learned from TRAIN events (one per group):")
    print(t.groupby("group")["n_events"].first().sort_values(ascending=False)
          .to_string(), "\n")
    print("Lift at the alert budget:")
    print(lifts(["pattern_template", "pattern_template[tod]", "cusum",
                 "rl_policy[s43]", "rl_template[s44]"]).to_string())


def p10() -> None:
    head("10", "Hybrid systems vs pure technical")
    where(["src/baselines/prior_volume.py  PriorVolume  (stock prior x volume)",
           "src/baselines/hybrid.py  PriorTemplate  (stock prior x template)"],
          ["tests/test_baselines_prior_volume.py tests/test_baselines_template.py"])
    print("pure technical = volume_zscore, cusum, pattern_template; "
          "pure stock knowledge = ticker_prior; hybrid = prior_*\n")
    print("Lift at the alert budget:")
    print(lifts(["volume_zscore", "cusum", "pattern_template", "ticker_prior",
                 "prior_volume", "prior_template"]).to_string())


POINTS = {"1": p1, "2": p2, "3": p3, "4": p4, "5": p5, "6": p6, "8": p8,
          "9": p9, "10": p10}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("points", nargs="*", choices=[*POINTS, "7"], metavar="POINT",
                    help="1-10 (default: all)")
    args = ap.parse_args()
    for p in args.points or list(POINTS):
        if p == "7":
            print("\nPOINT 7 — the 20 papers were skipped on purpose.")
            continue
        POINTS[p]()


if __name__ == "__main__":
    main()
