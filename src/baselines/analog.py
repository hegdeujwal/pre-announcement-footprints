"""Analog memory — "which past pre-filing run-up does this stock look like now?"

Added 2026-10-18 on the mentor's suggestion: remember the price and volume
path before every past filing, look for the same path in other stocks later,
and when one turns up say which past case it resembles and what that filing
was about.

How it differs from `template.PatternTemplate`
----------------------------------------------
The template AVERAGES every pre-filing path of a group into one shape, then
asks how well the recent hours match that average. The memory keeps every
TRAIN case SEPARATELY — every pre-filing window and every quiet window — and
asks which individual cases the recent hours are closest to. Two things follow
that an average cannot give:

* a run-up shape that only some filings share is not washed out by the rest;
* the match has a name: the nearest pre-filing case's stock, date and item
  codes, which is what `explain()` returns.

How a case is stored (train split only)
---------------------------------------
For each training window take the last `length` bars of each channel, aligned
on the window's end (bar -1 is the last bar before t0). Standardise each
(channel, position) by the QUIET windows' mean and spread, so every number
reads "how unusual compared with a normal stretch", then join the channels
into one vector. A missing bar counts as exactly quiet (0), the same rule as
the template, so it adds no evidence either way.

How an hour is scored
---------------------
Build the same vector from the stock's own last `length` bars, find its `k`
nearest stored cases by Euclidean distance, and score

    score = sum(w_i * is_prefiling_i) / sum(w_i),   w_i = 1 / (d_i + eps)

— a distance-weighted vote between pre-filing and quiet neighbours. The train
frame holds about three quiet windows per filing, so the score is not a
probability; the budget metric only ranks it.

`exclude_same_ticker` drops a stock's own past cases from its neighbours. That
is the strict form of the mentor's claim, "the pattern turns up in OTHER
companies", and it runs as its own named row rather than being chosen after
the fact.

Leakage
-------
The memory holds only train-split windows, whose filings were all public
before validation starts, and a query vector uses only bars at or before its
own hour (`template.lagged`). A test pins both, and another that the score
never reads the label columns.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from src.baselines.base import Baseline
from src.baselines.prior_volume import final_bars
from src.baselines.template import lagged, last_bars
from src.eval.analog_items import ranked, substantive_items


class AnalogMemory(Baseline):
    """k-nearest past cases, by the shape of the last `length` hours."""

    name = "analog"

    def __init__(self, cfg: dict | None = None, channels: list[str] | None = None,
                 exclude_same_ticker: bool | None = None,
                 name: str | None = None) -> None:
        super().__init__(cfg)
        knobs = self.cfg.get("baselines", {}).get("analog", {})
        self.channels = list(channels or knobs.get("channels", ["volume_z"]))
        self.length = int(knobs.get("length", 14))
        self.k = int(knobs.get("k", 25))
        self.eps = float(knobs.get("eps", 1e-6))
        self.chunk_rows = int(knobs.get("chunk_rows", 4096))
        self.top_n = int(knobs.get("top_n", 3))
        self.drop_items = list(knobs.get("non_substantive_items", ["9.01"]))
        self.exclude_same_ticker = bool(
            knobs.get("exclude_same_ticker", False)
            if exclude_same_ticker is None else exclude_same_ticker)
        if name:
            self.name = name
        if self.k < 1 or self.length < 1:
            raise ValueError(f"{self.name}: k and length must be >= 1")
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None
        self.memory: np.ndarray | None = None
        self.cases: pd.DataFrame | None = None

    # -- vectors ---------------------------------------------------------
    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num((X - self.mu) / self.sd, nan=0.0).astype("float32")

    # -- fitting ---------------------------------------------------------
    def fit(self, train: pd.DataFrame, conn=None) -> "AnalogMemory":
        if conn is not None:
            from src.pipeline import split
            split.assert_not_test(self.cfg, conn, train["ts_utc"],
                                  f"{self.name}.fit")
        missing = [c for c in self.channels if c not in train.columns]
        if missing:
            raise ValueError(f"{self.name}: training frame has no {missing} column(s)")

        paths = [last_bars(train, c, self.length) for c in self.channels]
        index = paths[0].index
        X = np.hstack([p.reindex(index).to_numpy(dtype="float64") for p in paths])
        meta = final_bars(train).set_index("window_id").reindex(index)
        positive = meta["t0_utc"].notna().to_numpy()
        if positive.all() or not positive.any():
            raise ValueError(f"{self.name}: training frame needs both "
                             f"pre-filing and quiet windows")

        quiet = X[~positive]
        self.mu = np.nanmean(quiet, axis=0)
        sd = np.nanstd(quiet, axis=0)
        # A constant position carries no contrast: infinite spread makes its
        # standardised value 0 instead of a division by zero.
        self.sd = np.where(sd > 0, sd, np.inf)
        self.memory = self._standardise(X)
        self._sq = (self.memory.astype("float64") ** 2).sum(axis=1)
        self.cases = pd.DataFrame({
            "window_id": index.astype(str),
            "ticker": meta["ticker"].astype(str).to_numpy(),
            "t0_utc": meta["t0_utc"].to_numpy(),
            "item_code": meta["item_code"].to_numpy(),
            "is_prefiling": positive,
        })
        self._items = [substantive_items(c, self.drop_items) if p else []
                       for c, p in zip(self.cases["item_code"], positive)]
        overall = Counter(i for row in self._items for i in row)
        self._pad = ranked(overall, len(overall))
        return self

    # -- neighbour search ------------------------------------------------
    def _neighbours(self, Q: np.ndarray, ticker: str,
                    pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(indices into memory, distances), each (len(Q), k'), nearest first.

        Brute force in row chunks: the memory is ~20k cases, small enough that
        an exact search is cheap, and exact means no approximate-index knob
        that could change a reported number.
        """
        if self.exclude_same_ticker:
            pool = pool & (self.cases["ticker"].to_numpy() != ticker)
        cand = np.flatnonzero(pool)
        if len(cand) == 0:
            raise ValueError(f"{self.name}: no stored case is eligible for {ticker}")
        k = min(self.k, len(cand))
        M = self.memory[cand]
        msq = self._sq[cand]
        idx = np.empty((len(Q), k), dtype=np.int64)
        dist = np.empty((len(Q), k), dtype="float64")
        for a in range(0, len(Q), self.chunk_rows):
            q = Q[a:a + self.chunk_rows]
            d2 = (q.astype("float64") ** 2).sum(1)[:, None] + msq[None, :] \
                - 2.0 * (q @ M.T).astype("float64")
            np.maximum(d2, 0.0, out=d2)
            part = np.argpartition(d2, k - 1, axis=1)[:, :k]
            pd2 = np.take_along_axis(d2, part, axis=1)
            order = np.argsort(pd2, axis=1, kind="stable")
            idx[a:a + len(q)] = cand[np.take_along_axis(part, order, axis=1)]
            dist[a:a + len(q)] = np.sqrt(np.take_along_axis(pd2, order, axis=1))
        return idx, dist

    def _blocks(self, frame: pd.DataFrame, by: str):
        """Yield (block keys, standardised vectors, undefined mask) per stock.

        One series per `by` over its distinct timestamps in time order, as in
        the template: an hour's stretch is the stock's own preceding bars, and
        a timestamp repeated by overlapping episodes is scored once.
        """
        self._require_fit()
        missing = [c for c in self.channels if c not in frame.columns]
        if missing:
            raise ValueError(f"{self.name}: frame has no {missing} column(s)")
        cols = [by, "ts_utc"] + (["ticker"] if by != "ticker" else [])
        keys = frame[cols + self.channels].drop_duplicates([by, "ts_utc"])
        keys = keys.sort_values([by, "ts_utc"], kind="stable")
        for _, block in keys.groupby(by, sort=False):
            X = np.hstack([lagged(block[c].to_numpy(dtype="float64"), self.length)
                           for c in self.channels])
            undefined = np.isnan(X).all(axis=1)
            yield block, self._standardise(X), undefined

    # -- scoring ---------------------------------------------------------
    def _require_fit(self) -> None:
        if self.memory is None:
            raise RuntimeError(f"{self.name}: call fit() before scoring")

    def score(self, frame: pd.DataFrame, by: str = "ticker") -> pd.Series:
        self._require_fit()
        everyone = np.ones(len(self.cases), dtype=bool)
        is_pre = self.cases["is_prefiling"].to_numpy(dtype="float64")
        scored = []
        for block, Q, undefined in self._blocks(frame, by):
            idx, dist = self._neighbours(Q, str(block["ticker"].iloc[0]), everyone)
            w = 1.0 / (dist + self.eps)
            s = (w * is_pre[idx]).sum(axis=1) / w.sum(axis=1)
            s[undefined] = np.nan
            scored.append(pd.Series(s, index=pd.MultiIndex.from_arrays(
                [block[by], block["ts_utc"]])))
        lookup = pd.concat(scored)
        idx = pd.MultiIndex.from_arrays([frame[by], frame["ts_utc"]])
        return pd.Series(lookup.reindex(idx).to_numpy(), index=frame.index,
                         dtype="float64")

    def explain(self, frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """For each (ticker, ts_utc) in `rows`: the nearest PRE-FILING cases.

        Searches pre-filing cases only — the question is "which past filing
        does this look like", not "filing or quiet". Returns one row per input
        row with the ranked item guess (`guess`, comma-joined, padded to
        `top_n` from the overall train order), the single nearest case
        (`nearest_ticker`, `nearest_t0_utc`, `nearest_items`,
        `nearest_distance`) and `n_neighbours`. The stretch is built from
        `frame`, so `frame` must hold each stock's preceding bars.
        """
        self._require_fit()
        want = rows[["ticker", "ts_utc"]].astype({"ticker": str, "ts_utc": "int64"})
        sub = frame[frame["ticker"].astype(str).isin(set(want["ticker"]))]
        pool = self.cases["is_prefiling"].to_numpy(dtype=bool)
        found = []
        for block, Q, undefined in self._blocks(sub, "ticker"):
            ticker = str(block["ticker"].iloc[0])
            ts = set(want.loc[want["ticker"] == ticker, "ts_utc"])
            take = block["ts_utc"].isin(ts).to_numpy() & ~undefined
            if not take.any():
                continue
            idx, dist = self._neighbours(Q[take], ticker, pool)
            w = 1.0 / (dist + self.eps)
            for t, ii, dd, ww in zip(block["ts_utc"].to_numpy()[take], idx, dist, w):
                votes = Counter()
                for i, wi in zip(ii, ww):
                    for item in self._items[i]:
                        votes[item] += wi
                near = self.cases.iloc[ii[0]]
                found.append({
                    "ticker": ticker, "ts_utc": t,
                    "guess": ",".join(ranked(votes, self.top_n, self._pad)),
                    "nearest_ticker": near["ticker"],
                    "nearest_t0_utc": near["t0_utc"],
                    "nearest_items": ",".join(self._items[ii[0]]),
                    "nearest_distance": float(dd[0]),
                    "n_neighbours": len(ii),
                })
        cols = ["ticker", "ts_utc", "guess", "nearest_ticker", "nearest_t0_utc",
                "nearest_items", "nearest_distance", "n_neighbours"]
        out = pd.DataFrame(found, columns=cols).astype({"ts_utc": "int64"})
        return want.merge(out, on=["ticker", "ts_utc"], how="left")
