"""The guide's review list, measured — every new detector beside the old ones.

Added 2026-10-15. Answers, on the VALIDATION split only (the test split's one
opening was spent on 2026-09-10 and is not reopened for exploratory work):

* point 1  — is volume bell-shaped? `distribution.csv`, and two repaired
             volume scores (`volume_logz`, `volume_tod_z`) as detectors
* point 3  — how early, on average? `lead_time.csv`: mean AND median lead,
             with quartiles, per detector and slice
* point 4  — impact: `impact.csv`, recall per impact tercile and the
             score-vs-move rank correlation
* point 5  — candles and indicators: `gradient_boosting[technical]`
* point 9  — templates: `pattern_template*` rows, `templates.csv`, and the
             template-shaped RL reward (`--template-run`)
* point 10 — pure technical vs hybrid: the `family` column of
             `comparison.csv`

Lives beside `compare.py` rather than in `src/eval/`, because it wires
detectors together and `eval` never imports models (architecture rule).

Every detector reaches the metrics through the identical `Baseline.predict`
path and one shared evaluation frame, as in `baselines.compare`. Detectors
are scored one at a time and their predictions dropped, so twenty of them fit
in memory beside a 2.3M-row frame.

Usage:
  python -m src.baselines.guide_review
  python -m src.baselines.guide_review --policy-run data/runs/p6-r2-s43 \
      --template-run data/runs/tmpl-r1-s43
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.baselines.always_quiet import AlwaysQuiet
from src.baselines.compare import (build_training_frame, comparison_table,
                                   conform, noise_seeds, train_positives)
from src.baselines.cusum import CUSUM
from src.baselines.gradient_boosting import GradientBoosting
from src.baselines.hybrid import PriorTemplate
from src.baselines.prior_volume import PriorVolume
from src.baselines.random_noise import RandomNoise
from src.baselines.template import PatternTemplate
from src.baselines.ticker_prior import TickerPrior
from src.baselines.volume_zscore import VolumeZScore
from src.eval import impact as I
from src.eval.contract import actions_from_scores
from src.eval.metrics import detection_delays, precision_at_alert_budget
from src.eval.within_stock import within_stock_table
from src.pipeline.evalset import build_eval_frame, split_bounds
from src.utils.config import load_config

SLICES = ("all", "scheduled", "unscheduled")
VOLUME_SCORES = ("volume_z", "volume_logz", "volume_tod_z")


def configs(cfg: dict) -> tuple[dict, dict]:
    """(technical off, technical on). Off is the published configuration."""
    off = copy.deepcopy(cfg)
    off["features"]["technical"]["enabled"] = False
    on = copy.deepcopy(cfg)
    on["features"]["technical"]["enabled"] = True
    return off, on


def distribution_table(frame: pd.DataFrame) -> pd.DataFrame:
    """How far each volume score is from a bell curve, on every scored hour."""
    rows = []
    for col in VOLUME_SCORES:
        x = frame[col].to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        rows.append({
            "score": col, "n": len(x), "mean": x.mean(), "sd": x.std(),
            "skew": stats.skew(x), "excess_kurtosis": stats.kurtosis(x),
            "p_above_2": (x > 2).mean(), "p_above_3": (x > 3).mean(),
            "p_above_5": (x > 5).mean(),
            "gaussian_p_above_2": stats.norm.sf(2),
            "gaussian_p_above_3": stats.norm.sf(3),
            "gaussian_p_above_5": stats.norm.sf(5),
        })
    return pd.DataFrame(rows)


def histogram_table(frame: pd.DataFrame, lo: float = -4, hi: float = 10,
                    width: float = 0.25) -> pd.DataFrame:
    """Density per bin for each volume score — what the PDF's histogram draws."""
    edges = np.arange(lo, hi + width, width)
    rows = []
    for col in VOLUME_SCORES:
        x = frame[col].to_numpy(dtype="float64")
        x = x[np.isfinite(x)]
        dens, _ = np.histogram(np.clip(x, lo, hi), bins=edges, density=True)
        rows += [{"score": col, "bin_lo": a, "bin_hi": b, "density": d}
                 for a, b, d in zip(edges[:-1], edges[1:], dens)]
    return pd.DataFrame(rows)


def lead_rows(name: str, pred: pd.DataFrame) -> list[dict]:
    """Mean and median lead per slice, at the budget's operating point."""
    threshold = precision_at_alert_budget(pred).threshold
    delays = detection_delays(actions_from_scores(pred, threshold))
    rows = []
    for s in SLICES:
        d = delays
        if s != "all":
            d = delays[delays["is_scheduled"].astype(bool) == (s == "scheduled")]
        lead = d["lead_trading_hours"].astype("float64")
        rows.append({
            "baseline": name, "slice": s, "n_detected": len(lead),
            "mean_lead_trading_h": lead.mean() if len(lead) else np.nan,
            "median_lead_trading_h": lead.median() if len(lead) else np.nan,
            "p25_lead_trading_h": lead.quantile(.25) if len(lead) else np.nan,
            "p75_lead_trading_h": lead.quantile(.75) if len(lead) else np.nan,
            "share_within_7h": (lead <= 7).mean() if len(lead) else np.nan,
            "mean_lead_wall_h": d["lead_wall_hours"].mean() if len(d) else np.nan,
        })
    return rows


def detectors(cfg_off: dict, cfg_on: dict, conn, train: pd.DataFrame,
              policy_runs: list[str], template_runs: list[str]):
    """Yield (family, model) lazily, so each is built only when scored."""
    yield "null", AlwaysQuiet(cfg_off)
    for s in noise_seeds(cfg_off):
        yield "null", RandomNoise(cfg_off, seed=s)
    yield "pure technical", VolumeZScore(cfg_off)
    yield "pure technical", VolumeZScore(cfg_off, feature="volume_logz",
                                         name="volume_logz")
    yield "pure technical", VolumeZScore(cfg_off, feature="volume_tod_z",
                                         name="volume_tod_z")
    yield "pure technical", CUSUM(cfg_off)
    yield "pure technical", CUSUM(cfg_off, feature="volume_logz",
                                  name="cusum[logz]")
    yield "pure technical", CUSUM(cfg_off, feature="volume_tod_z",
                                  name="cusum[tod]")
    template = PatternTemplate(cfg_off).fit(train, conn=conn)
    yield "pure technical", template
    yield "pure technical", PatternTemplate(
        cfg_off, channel="volume_tod_z",
        name="pattern_template[tod]").fit(train, conn=conn)

    gb = GradientBoosting(cfg_off).fit(train, conn=conn)
    yield "learned (market features)", gb
    gbt = GradientBoosting(cfg_on)
    gbt.name = "gradient_boosting[technical]"
    yield "learned (market + candles/indicators)", gbt.fit(train, conn=conn)

    from src.rl import load_policy
    for run in policy_runs:
        p = load_policy(cfg_off, run)
        p.name = f"rl_policy[{Path(run).name.split('-')[-1]}]"
        yield "learned (RL, base reward)", p
    for run in template_runs:
        p = load_policy(cfg_off, run)
        p.name = f"rl_template[{Path(run).name.split('-')[-1]}]"
        yield "learned (RL, template reward)", p

    yield "stock only", TickerPrior(cfg_off).fit(train_positives(cfg_off), conn=conn)
    yield "hybrid", PriorVolume(cfg_off).fit(train, conn=conn)
    yield "hybrid", PriorTemplate(cfg_off, template=template).fit(train, conn=conn)


def run(cfg: dict, conn, split: str, policy_runs: list[str],
        template_runs: list[str], out_dir: Path,
        tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    cfg_off, cfg_on = configs(cfg)
    lo, hi = split_bounds(cfg_on, split, conn=conn)
    variant = "news_adjusted"

    print("building the evaluation frame (technical columns on)...")
    frame = conform(build_eval_frame(cfg_on, conn, lo, hi, tickers=tickers,
                                     t0_variant=variant))
    print("building the 3:1 train frame (technical columns on)...")
    train = build_training_frame(cfg_on, conn)

    impacts = I.event_impacts(conn)
    edges = I.impact_edges(cfg, conn, impacts)

    out = {"distribution": distribution_table(frame),
           "histogram": histogram_table(frame)}
    comp, imp, lead, within, templates = [], [], [], [], []
    for family, model in detectors(cfg_off, cfg_on, conn, train,
                                   policy_runs, template_runs):
        started = time.time()
        pred = model.predict(frame, threshold=float("inf"), conn=conn,
                             context=f"guide_review/{model.name}")
        table = comparison_table(cfg_off, {model.name: pred}, variant,
                                 models={model.name: model})
        comp.append(table.assign(family=family))
        imp += I.impact_rows(model.name, pred, impacts, edges)
        lead += lead_rows(model.name, pred)
        within.append(within_stock_table(cfg, {model.name: pred}, variant)
                      .assign(family=family))
        if isinstance(model, PatternTemplate):
            templates.append(model.template_table().assign(detector=model.name))
        del pred
        print(f"  {model.name:32s} {time.time() - started:5.0f}s")

    comp = pd.concat(comp, ignore_index=True)
    floor = comp[comp["baseline"] == "always_quiet"].set_index("slice")["precision"]
    comp["lift"] = comp["precision"] / comp["slice"].map(floor)
    out["comparison"] = comp
    out["impact"] = I.add_lift(pd.DataFrame(imp))
    out["impact_edges"] = pd.DataFrame({"bucket_upper_edge": edges,
                                        "quantile": cfg["eval"]["impact"]["quantiles"]})
    out["lead_time"] = pd.DataFrame(lead)
    out["within_stock"] = pd.concat(within, ignore_index=True)
    out["templates"] = pd.concat(templates, ignore_index=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(out_dir / f"{name}.csv", index=False)
    return out


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val", help="val (default) or train")
    ap.add_argument("--policy-run", action="append", default=[], dest="policy_runs")
    ap.add_argument("--template-run", action="append", default=[],
                    dest="template_runs",
                    help="an RL run trained with reward.r_template_bonus > 0")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit-tickers", type=int, default=None,
                    help="evaluate only the first N in-universe tickers (smoke run)")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    out_dir = Path(args.out_dir or Path(cfg["paths"]["processed"]) / "guide-review")
    tickers = None
    if args.limit_tickers:
        tickers = [r[0] for r in conn.execute(
            "SELECT ticker FROM companies WHERE in_universe = 1 "
            "ORDER BY ticker LIMIT ?", (args.limit_tickers,))]
    started = time.time()
    out = run(cfg, conn, args.split, args.policy_runs, args.template_runs,
              out_dir, tickers=tickers)

    comp = out["comparison"]
    for s in SLICES:
        view = comp[comp["slice"] == s].sort_values("lift", ascending=False)
        print(f"\n-- {s} --")
        print(view[["baseline", "family", "precision", "lift", "recall",
                    "median_lead_trading_h"]].to_string(
                        index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {out_dir}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
