"""Can a matched past case say WHAT KIND of filing is coming?

Added 2026-10-18 with `baselines.analog`. A detector that only says "something
is coming" gives an analyst nothing to check. The analog memory can say more:
"this stock looks like XYZ before its 8-K Item 2.01". That claim is testable,
because every validation filing carries its real item codes, and this module
tests it.

The question is never "how often is the guess right" on its own — a constant
guess of the commonest item is right surprisingly often. It is "how often is
the guess right compared with guesses that know nothing about the bars":

* `majority`         the commonest item among TRAIN filings, every time
* `majority[slice]`  the commonest item within the event's own slice. It reads
                     the scheduled/unscheduled label, which the analog vote
                     does not see, so it is a generous reference, not a rival
* `company_history`  the stock's own commonest item in TRAIN, falling back to
                     `majority` for a stock with no train filing
* `random`           a seeded draw from the train item distribution

Scoring
-------
An 8-K often carries several items. A guess HITS when its top item is one of
the event's substantive items (`hit_at_1`), or when any of its top `top_n`
items is (`hit_at_n`). Items in `non_substantive_items` (9.01, "financial
statements and exhibits") are removed first — they ride along on most filings
and are never the reason for one. An event left with no substantive item is
dropped from the table and counted in `n_dropped`, never scored as a miss.

The interval bootstraps EVENTS. Every number is split scheduled vs
unscheduled.

Plain data in, plain data out: this module never imports a model (the
architecture rule for `src/eval`), so the guesses arrive as columns.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

SLICES = ("all", "scheduled", "unscheduled")
GUESS_PREFIX = "guess:"


def substantive_items(code: object, drop: Iterable[str]) -> list[str]:
    """Item codes of one filing as strings, minus the non-substantive ones.

    Codes stay strings throughout: "1.10" and "1.1" are different items and a
    float would merge them.
    """
    if code is None or (isinstance(code, float) and np.isnan(code)) or code is pd.NA:
        return []
    drop = set(drop)
    out = []
    for part in str(code).split(","):
        part = part.strip()
        if part and part not in drop and part not in out:
            out.append(part)
    return out


def ranked(counts: Counter, top_n: int, pad: Sequence[str] = ()) -> list[str]:
    """Top `top_n` items by count (ties by code, for determinism), padded from
    `pad` so every method guesses the same number of items."""
    order = [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    for p in pad:
        if len(order) >= top_n:
            break
        if p not in order:
            order.append(p)
    return order[:top_n]


def baseline_guesses(train_events: pd.DataFrame, events: pd.DataFrame,
                     drop: Iterable[str], top_n: int, seed: int) -> pd.DataFrame:
    """The four bar-blind guesses for every row of `events`.

    `train_events`: one row per TRAIN filing with `ticker`, `item_code`,
    `is_scheduled`. `events`: one row per evaluated filing with `ticker` and
    `is_scheduled`. Returns `events` with one `guess:<method>` column each,
    holding a comma-joined ranked list.
    """
    drop = list(drop)
    items = train_events["item_code"].map(lambda c: substantive_items(c, drop))
    overall = Counter(i for row in items for i in row)
    if not overall:
        raise ValueError("baseline_guesses: train events carry no substantive "
                         "item at all — nothing to guess from")
    majority = ranked(overall, top_n)
    pad = ranked(overall, len(overall))

    sched = train_events["is_scheduled"].astype(bool).to_numpy()
    by_slice = {flag: ranked(Counter(i for row in items[sched == flag] for i in row),
                             top_n, pad)
                for flag in (True, False)}
    by_stock = {t: ranked(Counter(i for row in grp for i in row), top_n, pad)
                for t, grp in items.groupby(train_events["ticker"].astype(str))}

    rng = np.random.default_rng(seed)
    codes = np.array(sorted(overall))
    p = np.array([overall[c] for c in codes], dtype="float64")
    p /= p.sum()
    k = min(top_n, len(codes))

    out = events.copy()
    out[GUESS_PREFIX + "majority"] = ",".join(majority)
    out[GUESS_PREFIX + "majority[slice]"] = [
        ",".join(by_slice[bool(s)]) for s in out["is_scheduled"]]
    out[GUESS_PREFIX + "company_history"] = [
        ",".join(by_stock.get(str(t), majority)) for t in out["ticker"]]
    out[GUESS_PREFIX + "random"] = [
        ",".join(rng.choice(codes, size=k, replace=False, p=p)) for _ in range(len(out))]
    return out


def _hits(actual: pd.Series, guess: pd.Series, n: int) -> np.ndarray:
    return np.array([bool(set(g.split(",")[:n]) & set(a)) if g else False
                     for a, g in zip(actual, guess)], dtype=bool)


def _interval(x: np.ndarray, n_boot: int, rng: np.random.Generator,
              ci: float) -> tuple[float, float]:
    if len(x) == 0:
        return np.nan, np.nan
    draws = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[draws].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def hit_rates(table: pd.DataFrame, drop: Iterable[str], top_n: int,
              n_boot: int, seed: int, ci: float,
              label: str = "") -> pd.DataFrame:
    """One row per (method, slice): hit@1 and hit@top_n with event intervals.

    `table`: one row per evaluated filing with `item_code`, `is_scheduled`,
    and one `guess:<method>` column per method (comma-joined ranked items).
    """
    methods = [c for c in table.columns if c.startswith(GUESS_PREFIX)]
    if not methods:
        raise ValueError("hit_rates: no 'guess:' columns to score")
    actual = table["item_code"].map(lambda c: substantive_items(c, drop))
    keep = actual.map(len) > 0
    n_dropped = int((~keep).sum())
    t, actual = table[keep], actual[keep]
    sched = t["is_scheduled"].astype(bool).to_numpy()

    rows = []
    for m in methods:
        rng = np.random.default_rng(seed)
        guess = t[m].fillna("").astype(str)
        h1 = _hits(actual, guess, 1)
        hn = _hits(actual, guess, top_n)
        for s in SLICES:
            mask = np.ones(len(t), bool) if s == "all" else (sched == (s == "scheduled"))
            lo1, hi1 = _interval(h1[mask].astype(float), n_boot, rng, ci)
            lon, hin = _interval(hn[mask].astype(float), n_boot, rng, ci)
            rows.append({
                "set": label, "method": m[len(GUESS_PREFIX):], "slice": s,
                "n_events": int(mask.sum()), "n_dropped": n_dropped,
                "hit_at_1": h1[mask].mean() if mask.any() else np.nan,
                "hit_at_1_lo": lo1, "hit_at_1_hi": hi1,
                f"hit_at_{top_n}": hn[mask].mean() if mask.any() else np.nan,
                f"hit_at_{top_n}_lo": lon, f"hit_at_{top_n}_hi": hin,
            })
    return pd.DataFrame(rows)
