"""FootprintNet — a small pre-trained sequence model as a detector.

Added 2026-10-18 on the user's decision, after the clock fix (`clock_only`,
`eval.same_hour`). Its job, stated before it was built: be the strongest
challenger the classical detectors have faced. On scheduled filings, where a
footprint survives the same-hour score, can a learned model beat volume
z-score and CUSUM? On unscheduled filings, where nothing has survived it, does
a model that reads 48 hours and learned from ~1.6M unlabelled hours also find
nothing? Either answer is a result; a null only counts once a strong model
has failed too (plan §5).

What it reads
-------------
For one stock at one hour: the last `length` bars of that stock, oldest first,
each described by the configured `channels` (backward-looking features
already in the frame) plus two clock inputs — bars left to the session close,
and weekday. Nothing after the hour, no news, no label column. Missing bars at
the start of a series are zero rows, "nothing unusual", as in the template.

The network
-----------
A linear layer into a one-layer GRU — it reads the bars in order and keeps a
running summary — then two heads on that summary:

* `forecast`: next bar's standardised `pretrain.targets` (Stage 1, trained on
  every train-split bar, no labels needed);
* `detect`:   one logit, "a filing is coming" (Stage 2, fine-tuned on filings).

Stage 1 also yields a second detector for free: **surprise**, how far this
bar's volume sits above what the forecaster expected one bar earlier — a
learned replacement for the rolling-mean baseline inside volume z-score.

Training lives in `footprint_train.py`; this module holds what both training
and scoring need, so the two can never prepare inputs differently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines.base import Baseline
from src.utils.timeutils import get_market_calendar, seconds_to_session_close

HOUR = 3600
#: Bars in an XNYS session; the clock input is scaled into [0, 1] by it.
SESSION_BARS = 7
#: Rows scored per forward pass. Memory only, moves no number.
SCORE_BATCH = 4096


def knobs(cfg: dict) -> dict:
    return cfg["baselines"]["footprint_net"]


def bars_to_close(ts: np.ndarray, cfg: dict) -> np.ndarray:
    """0 for the session's last bar, 1 for the one before, ... NaN off-session."""
    cal = get_market_calendar(cfg["market"]["calendar"])
    secs = seconds_to_session_close(np.asarray(ts, dtype="int64"), cal, strict=False)
    return np.floor(secs / HOUR)


def bar_inputs(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    """(n_rows, n_channels + 2) float32 inputs for every row, row by row.

    Each row reads only its own bar's features, so a window assembled from
    these rows contains nothing later than its last bar.
    """
    k = knobs(cfg)
    missing = [c for c in k["channels"] if c not in frame.columns]
    if missing:
        raise ValueError(f"footprint_net: frame has no {missing} column(s) — "
                         f"build it with features.technical.enabled")
    X = frame[k["channels"]].to_numpy(dtype="float64")
    X = np.clip(np.nan_to_num(X, nan=0.0), -k["clip"], k["clip"])
    ts = frame["ts_utc"].astype("int64").to_numpy()
    close = np.nan_to_num(bars_to_close(ts, cfg), nan=0.0) / (SESSION_BARS - 1)
    weekday = pd.to_datetime(ts, unit="s", utc=True).dayofweek.to_numpy() / 4.0
    return np.column_stack([X, close, weekday]).astype("float32")


def ticker_series(frame: pd.DataFrame, by: str = "ticker"):
    """Yield (key, block) per stock: distinct timestamps in time order.

    Overlapping episodes repeat a bar; it appears once here, as in the
    template and CUSUM, so a stretch is the stock's own preceding bars.
    """
    keys = frame.drop_duplicates([by, "ts_utc"]).sort_values([by, "ts_utc"],
                                                             kind="stable")
    for key, block in keys.groupby(by, sort=False):
        yield key, block


def windows(X: np.ndarray, ends: np.ndarray, length: int) -> np.ndarray:
    """(len(ends), length, F): rows ends-length+1 .. ends, zero-padded on the left."""
    pad = np.zeros((length - 1, X.shape[1]), dtype=X.dtype)
    Xp = np.concatenate([pad, X])
    idx = ends[:, None] + np.arange(length)[None, :]
    return Xp[idx]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_model(n_inputs: int, cfg: dict, n_targets: int):
    """The network. Imported lazily so the rest of the project never needs torch."""
    import torch
    from torch import nn

    k = knobs(cfg)

    class FootprintGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            h = int(k["hidden"])
            self.inp = nn.Linear(n_inputs, h)
            self.gru = nn.GRU(h, h, batch_first=True)
            self.forecast = nn.Linear(h, n_targets)
            self.detect = nn.Sequential(nn.Dropout(float(k["dropout"])),
                                        nn.Linear(h, 1))

        def states(self, x: "torch.Tensor") -> "torch.Tensor":
            """Summary after every bar: (batch, length, hidden)."""
            return self.gru(torch.relu(self.inp(x)))[0]

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """Detection logit from the summary after the LAST bar."""
            return self.detect(self.states(x)[:, -1]).squeeze(-1)

    return FootprintGRU()


def save_run(run_dir: Path, model, manifest: dict) -> dict:
    import torch
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = run_dir / "model.pt"
    torch.save(model.state_dict(), weights)
    manifest = {**manifest, "model_sha256": sha256(weights)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load_run(run_dir: Path, cfg: dict):
    """(model, manifest). Refuses weights whose checksum or inputs have moved."""
    import torch
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    weights = run_dir / "model.pt"
    if sha256(weights) != manifest["model_sha256"]:
        raise ValueError(f"{run_dir}: model.pt does not match its manifest checksum")
    if manifest["channels"] != knobs(cfg)["channels"]:
        raise ValueError(f"{run_dir}: trained on channels {manifest['channels']}, "
                         f"config now says {knobs(cfg)['channels']}")
    model = build_model(manifest["n_inputs"], cfg, len(manifest["targets"]))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model, manifest


class FootprintDetector(Baseline):
    """A trained run as a detector: `mode="detect"` (Stage 2's logit) or
    `mode="surprise"` (Stage 1's forecast error on this bar's volume)."""

    def __init__(self, cfg: dict | None, run_dir: str | Path,
                 mode: str = "detect", name: str | None = None) -> None:
        super().__init__(cfg)
        if mode not in ("detect", "surprise"):
            raise ValueError(f"unknown FootprintDetector mode {mode!r}")
        self.mode = mode
        self.model, self.manifest = load_run(Path(run_dir), self.cfg)
        if mode == "detect" and self.manifest.get("stage") != "finetune":
            raise ValueError(f"{run_dir}: a detect head needs a fine-tuned run, "
                             f"this is stage {self.manifest.get('stage')!r}")
        self.name = name or f"footprint_{mode}[{Path(run_dir).name}]"

    def score(self, frame: pd.DataFrame, by: str = "ticker") -> pd.Series:
        import torch
        torch.set_num_threads(int(knobs(self.cfg)["threads"]))
        L = int(self.manifest["length"])
        vol = self.manifest["targets"].index("volume_logz") if self.mode == "surprise" else None
        scored = []
        with torch.no_grad():
            for _, block in ticker_series(frame, by):
                X = bar_inputs(block, self.cfg)
                out = np.full(len(block), np.nan)
                if self.mode == "detect":
                    ends = np.arange(len(block))
                else:
                    ends = np.arange(len(block) - 1)    # forecast made one bar earlier
                for a in range(0, len(ends), SCORE_BATCH):
                    e = ends[a:a + SCORE_BATCH]
                    xb = torch.from_numpy(windows(X, e, L))
                    if self.mode == "detect":
                        out[e] = self.model(xb).numpy()
                    else:
                        pred = self.model.forecast(self.model.states(xb)[:, -1])[:, vol].numpy()
                        actual = block["volume_logz"].to_numpy(dtype="float64")[e + 1]
                        mu, sd = self.manifest["target_mean"][vol], self.manifest["target_sd"][vol]
                        out[e + 1] = (actual - mu) / sd - pred
                scored.append(pd.Series(out, index=pd.MultiIndex.from_arrays(
                    [block[by], block["ts_utc"]])))
        lookup = pd.concat(scored)
        idx = pd.MultiIndex.from_arrays([frame[by], frame["ts_utc"]])
        return pd.Series(lookup.reindex(idx).to_numpy(), index=frame.index,
                         dtype="float64")
