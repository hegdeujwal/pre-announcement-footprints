"""Training FootprintNet: Stage 1 (forecast, unlabelled), Stage 2 (detect).

Everything is fitted on the TRAIN split. The last `inner_val_fraction` of the
train period is held back for early stopping and nothing else; the validation
split is never read here, so no stopping or tuning decision sees it.

Stage 1 — learn what normal looks like
--------------------------------------
Windows of `length` bars are drawn at random from every train-period bar of
every stock. At each bar the network forecasts the NEXT bar's standardised
`pretrain.targets`; the first `warmup` bars of a window carry no loss. No
label is read. A bar's target is read from the following row only while
training — at scoring time nothing after the hour is ever touched.

Stage 2 — learn what comes before a filing
------------------------------------------
Positives: every train filing's final bar before t0, the bar the budget metric
scores. Negatives: quiet bars, `negatives_per_positive` per positive, drawn so
their bar-of-session mix is EXACTLY the positives' (84% closing bars). The
clock is then worth nothing to the loss: a model that only learned the hour
would score every pair alike. The loss weights positives by the same ratio,
and the stopping rule is inner-split ROC AUC.

`init="pretrained"` starts Stage 2 from that seed's Stage 1 weights;
`init="scratch"` starts from random weights with the same seed. The gap
between the two is the test of whether pre-training helps.

Learning curve
--------------
`fraction` < 1 fine-tunes on that share of the fitting period's filings only,
each with its usual quota of clock-matched negatives; the inner split used for
stopping is untouched. The shares are nested — one fixed shuffle per seed, the
first `fraction` of it kept — so a 50% run holds every filing of the 25% run.
If validation keeps improving as the share grows, more filings would help; if
it is flat, the bottleneck is not the amount of labelled data
(`src.eval.learning_curve`).

Usage:
  python -m src.baselines.footprint_train                 # every seed, all stages
  python -m src.baselines.footprint_train --seeds 42 --stage pretrain
  python -m src.baselines.footprint_train --learning-curve     # scratch, every share
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import footprint_net as F
from src.baselines.prior_volume import final_bars
from src.utils.config import load_config


# -- data ----------------------------------------------------------------

class Prepared:
    """Every train-period bar of every stock, laid out for window gathering.

    Each stock's rows sit after `length - 1` zero rows, so a window ending on
    any row never reaches into the previous stock. Arrays are aligned on that
    padded layout; `rows` holds the padded position of each real bar.
    """

    def __init__(self, frame: pd.DataFrame, cfg: dict, cutoff_utc: int) -> None:
        k = F.knobs(cfg)
        L = int(k["length"])
        targets = list(k["pretrain"]["targets"])
        last = final_bars(frame)
        final_pos = set(zip(last.loc[last["t0_utc"].notna(), "ticker"].astype(str),
                            last.loc[last["t0_utc"].notna(), "ts_utc"].astype("int64")))
        quiet_keys = set(zip(frame.loc[frame["t0_utc"].isna(), "ticker"].astype(str),
                             frame.loc[frame["t0_utc"].isna(), "ts_utc"].astype("int64")))

        X, Yn, ts_all, pos_flag, quiet_flag, btc, rows = [], [], [], [], [], [], []
        offset = 0
        for ticker, block in F.ticker_series(frame):
            n = len(block)
            x = F.bar_inputs(block, cfg)
            y = block[targets].to_numpy(dtype="float64")
            y_next = np.vstack([y[1:], np.full((1, len(targets)), np.nan)])
            t = block["ts_utc"].astype("int64").to_numpy()
            keys = list(zip([str(ticker)] * n, t))
            X += [np.zeros((L - 1, x.shape[1]), "float32"), x]
            Yn += [np.full((L - 1, len(targets)), np.nan), y_next]
            ts_all += [np.zeros(L - 1, "int64"), t]
            pos_flag += [np.zeros(L - 1, bool), np.array([kk in final_pos for kk in keys])]
            quiet_flag += [np.zeros(L - 1, bool), np.array([kk in quiet_keys for kk in keys])]
            btc += [np.full(L - 1, -1.0), np.nan_to_num(F.bars_to_close(t, cfg), nan=-1.0)]
            rows.append(offset + L - 1 + np.arange(n))
            offset += L - 1 + n
        if not rows:
            raise ValueError("footprint_train: the training frame holds no bars")
        self.L = L
        self.targets = targets
        self.X = np.concatenate(X)
        self.y_next = np.concatenate(Yn)
        self.ts = np.concatenate(ts_all)
        self.is_final_pos = np.concatenate(pos_flag)
        self.is_quiet = np.concatenate(quiet_flag)
        self.bar_of_session = np.concatenate(btc)
        self.rows = np.concatenate(rows)
        real = np.zeros(len(self.X), bool)
        real[self.rows] = True
        self.real = real
        self.inner = real & (self.ts >= cutoff_utc)
        self.fit = real & (self.ts < cutoff_utc)
        # Standardise targets on the fitting rows only.
        fit_targets = self.y_next[self.fit]
        self.target_mean = np.nanmean(fit_targets, axis=0)
        sd = np.nanstd(fit_targets, axis=0)
        self.target_sd = np.where(sd > 0, sd, 1.0)
        self.y_std = (self.y_next - self.target_mean) / self.target_sd
        if not self.is_final_pos[self.fit].any():
            raise ValueError("footprint_train: no filing in the fitting period")

    def window(self, ends: np.ndarray) -> np.ndarray:
        idx = ends[:, None] - self.L + 1 + np.arange(self.L)[None, :]
        return self.X[idx]

    def window_targets(self, ends: np.ndarray) -> np.ndarray:
        idx = ends[:, None] - self.L + 1 + np.arange(self.L)[None, :]
        return self.y_std[idx]


def matched_negatives(data: Prepared, region: np.ndarray, ratio: int,
                      rng: np.random.Generator,
                      pos: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(positive rows, negative rows) in `region`, negatives matched on
    bar-of-session to the positives' exact mix. `pos` overrides which
    positives are used (a learning-curve share)."""
    if pos is None:
        pos = np.flatnonzero(region & data.is_final_pos)
    neg = []
    for b in np.unique(data.bar_of_session[pos]):
        want = ratio * int((data.bar_of_session[pos] == b).sum())
        pool = np.flatnonzero(region & data.is_quiet & (data.bar_of_session == b))
        if len(pool) == 0:
            raise ValueError(f"footprint_train: no quiet bar at bar-of-session {b}")
        neg.append(rng.choice(pool, size=want, replace=len(pool) < want))
    return pos, np.concatenate(neg)


def training_frame(cfg: dict, conn, refresh: bool = False) -> pd.DataFrame:
    """Every TRAIN-split bar with features (technical on), cached under runs_dir."""
    from src.baselines.compare import conform
    from src.baselines.guide_review import configs
    from src.pipeline.evalset import build_eval_frame, split_bounds

    path = Path(F.knobs(cfg)["runs_dir"]) / "train_frame.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    _, cfg_on = configs(cfg)
    lo, hi = split_bounds(cfg_on, "train", conn=conn)
    frame = conform(build_eval_frame(cfg_on, conn, lo, hi))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def share_of_positives(data: Prepared, fraction: float, seed: int) -> np.ndarray:
    """The first `fraction` of one fixed shuffle of the fitting filings, so
    shares are nested. Its own generator: the share never moves the training
    draws that follow it."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    pos = np.flatnonzero(data.fit & data.is_final_pos)
    order = np.random.default_rng(seed).permutation(len(pos))
    keep = max(1, int(round(fraction * len(pos))))
    return np.sort(pos[order[:keep]])


def run_name(seed: int, init: str, fraction: float = 1.0) -> str:
    """`s42-scratch`, or `s42-scratch-f25` for a learning-curve share."""
    base = f"s{seed}-{init}"
    return base if fraction == 1 else f"{base}-f{round(fraction * 100):d}"


def inner_cutoff(cfg: dict, conn) -> int:
    from src.pipeline.evalset import split_bounds
    lo, hi = split_bounds(cfg, "train", conn=conn)
    return int(hi - F.knobs(cfg)["inner_val_fraction"] * (hi - lo))


# -- training ------------------------------------------------------------

def _seed_all(seed: int) -> np.random.Generator:
    import torch
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def pretrain(data: Prepared, cfg: dict, seed: int, run_dir: Path,
             log=print) -> dict:
    import torch
    k = F.knobs(cfg)
    p = k["pretrain"]
    torch.set_num_threads(int(k["threads"]))
    rng = _seed_all(seed)
    model = F.build_model(data.X.shape[1], cfg, len(data.targets))
    opt = torch.optim.Adam(model.parameters(), lr=float(p["lr"]))
    has_next = ~np.isnan(data.y_next).all(axis=1)
    fit_rows = np.flatnonzero(data.fit & has_next)
    inner_rows = np.flatnonzero(data.inner & has_next)
    inner_eval = rng.choice(inner_rows, size=min(20000, len(inner_rows)), replace=False)
    warm = int(p["warmup"])

    def loss_on(ends):
        x = torch.from_numpy(data.window(ends))
        y = torch.from_numpy(data.window_targets(ends)[:, warm:]).float()
        pred = model.forecast(model.states(x))[:, warm:]
        mask = ~torch.isnan(y)
        return ((pred - torch.nan_to_num(y)) ** 2 * mask).sum() / mask.sum().clamp(min=1)

    best, best_state, bad, history = np.inf, None, 0, []
    started = time.time()
    for epoch in range(int(p["epochs"])):
        model.train()
        ends = rng.choice(fit_rows, size=int(p["windows_per_epoch"]), replace=False)
        total = 0.0
        for a in range(0, len(ends), int(p["batch"])):
            opt.zero_grad()
            loss = loss_on(ends[a:a + int(p["batch"])])
            loss.backward()
            opt.step()
            total += loss.item() * len(ends[a:a + int(p["batch"])])
        model.eval()
        with torch.no_grad():
            inner = np.mean([float(loss_on(inner_eval[a:a + 2048]))
                             for a in range(0, len(inner_eval), 2048)])
        history.append({"epoch": epoch, "train_mse": total / len(ends), "inner_mse": inner})
        log(f"  pretrain s{seed} epoch {epoch}: train {total / len(ends):.4f} "
            f"inner {inner:.4f} ({time.time() - started:.0f}s)")
        if inner < best - 1e-4:
            best, best_state, bad = inner, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(p["patience"]):
                break
    model.load_state_dict(best_state)
    return F.save_run(run_dir, model, _manifest(cfg, data, seed, "pretrain", None,
                                                history, time.time() - started,
                                                {"best_inner_mse": best}))


def finetune(data: Prepared, cfg: dict, seed: int, run_dir: Path,
             init_dir: Path | None, log=print, fraction: float = 1.0) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score
    k = F.knobs(cfg)
    f = k["finetune"]
    torch.set_num_threads(int(k["threads"]))
    rng = _seed_all(seed)
    model = F.build_model(data.X.shape[1], cfg, len(data.targets))
    if init_dir is not None:
        pre, _ = F.load_run(init_dir, cfg)
        model.load_state_dict(pre.state_dict())
    ratio = int(f["negatives_per_positive"])
    share = None if fraction == 1 else share_of_positives(data, fraction, seed)
    pos, neg = matched_negatives(data, data.fit, ratio, rng, pos=share)
    ipos, ineg = matched_negatives(data, data.inner, ratio, rng)
    ends = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype("float32")
    iends = np.concatenate([ipos, ineg])
    ilabels = np.concatenate([np.ones(len(ipos)), np.zeros(len(ineg))])
    opt = torch.optim.Adam(model.parameters(), lr=float(f["lr"]),
                           weight_decay=float(f["weight_decay"]))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(ratio)))

    def predict(rows):
        model.eval()
        with torch.no_grad():
            return np.concatenate([model(torch.from_numpy(data.window(rows[a:a + 4096]))).numpy()
                                   for a in range(0, len(rows), 4096)])

    best, best_state, bad, history = -np.inf, None, 0, []
    started = time.time()
    for epoch in range(int(f["epochs"])):
        model.train()
        order = rng.permutation(len(ends))
        total = 0.0
        for a in range(0, len(order), int(f["batch"])):
            b = order[a:a + int(f["batch"])]
            opt.zero_grad()
            loss = lossf(model(torch.from_numpy(data.window(ends[b]))),
                         torch.from_numpy(labels[b]))
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        auc = float(roc_auc_score(ilabels, predict(iends)))
        history.append({"epoch": epoch, "train_bce": total / len(ends), "inner_auc": auc})
        log(f"  finetune s{seed} {'pretrained' if init_dir else 'scratch'} "
            f"epoch {epoch}: bce {total / len(ends):.4f} inner AUC {auc:.4f} "
            f"({time.time() - started:.0f}s)")
        if auc > best + 1e-4:
            best, best_state, bad = auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= int(f["patience"]):
                break
    model.load_state_dict(best_state)
    extra = {"best_inner_auc": best, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
             "fraction": fraction,
             "n_inner_pos": int(len(ipos)), "n_inner_neg": int(len(ineg))}
    return F.save_run(run_dir, model, _manifest(
        cfg, data, seed, "finetune", "pretrained" if init_dir else "scratch",
        history, time.time() - started, extra))


def _manifest(cfg, data, seed, stage, init, history, elapsed, extra) -> dict:
    import torch
    k = F.knobs(cfg)
    return {"stage": stage, "init": init, "seed": seed,
            "channels": list(k["channels"]), "n_inputs": int(data.X.shape[1]),
            "targets": data.targets, "length": data.L,
            "target_mean": [float(v) for v in data.target_mean],
            "target_sd": [float(v) for v in data.target_sd],
            "config": copy.deepcopy(k), "history": history,
            "elapsed_s": round(elapsed, 1), "torch": torch.__version__,
            "n_fit_bars": int(data.fit.sum()), "n_inner_bars": int(data.inner.sum()),
            **extra}


def main() -> None:
    from src import db

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--stage", choices=["pretrain", "finetune", "all"], default="all")
    ap.add_argument("--refresh-frame", action="store_true",
                    help="rebuild the cached train-split frame")
    ap.add_argument("--learning-curve", action="store_true",
                    help="fine-tune from scratch on every footprint_net."
                         "learning_curve.fractions share instead")
    args = ap.parse_args()

    cfg = load_config()
    conn = db.get_conn(cfg["paths"]["db"])
    k = F.knobs(cfg)
    seeds = args.seeds or list(k["seeds"])
    runs = Path(k["runs_dir"])
    started = time.time()
    print("loading the train-split frame...")
    frame = training_frame(cfg, conn, refresh=args.refresh_frame)
    data = Prepared(frame, cfg, inner_cutoff(cfg, conn))
    del frame
    print(f"  {int(data.real.sum()):,} bars, {int(data.is_final_pos.sum()):,} filings, "
          f"{int(data.inner.sum()):,} inner-split bars ({time.time() - started:.0f}s)")
    if args.learning_curve:
        lc = k["learning_curve"]
        for seed in seeds:
            for fraction in lc["fractions"]:
                run_dir = runs / run_name(seed, lc["init"], fraction)
                if (run_dir / "manifest.json").exists():
                    print(f"  {run_dir.name} exists, skipped")
                    continue
                init = runs / f"s{seed}-pretrain" if lc["init"] == "pretrained" else None
                finetune(data, cfg, seed, run_dir, init, fraction=fraction)
        print(f"done ({time.time() - started:.0f}s)")
        return
    for seed in seeds:
        pre_dir = runs / f"s{seed}-pretrain"
        if args.stage in ("pretrain", "all"):
            pretrain(data, cfg, seed, pre_dir)
        if args.stage in ("finetune", "all"):
            finetune(data, cfg, seed, runs / f"s{seed}-pretrained", pre_dir)
            finetune(data, cfg, seed, runs / f"s{seed}-scratch", None)
    print(f"done ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
