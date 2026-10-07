"""Stretches of the live alert log scored on data known to be bad.

The alert log is append-only and hash-chained, so a day the monitor scored on
corrupt bars cannot be removed from it, and should not be: what the monitor
said, and when, is part of the record. What changes is what gets COUNTED.
`live.data_incidents` in config names each stretch, with its reason, and every
live rate — the outcome hit rates, the chance comparison, the dashboard —
leaves those alerts out through this one module, so the exclusion is the same
everywhere and visible in one place.

An incident names alerts one of two ways, or both:
  - by BAR date (`first_bar_date`/`last_bar_date`, UTC dates, inclusive):
    every alert on a bar in that stretch. An XNYS session never crosses a UTC
    midnight, so a UTC date is the session.
  - by RAISE time (`raised_from`/`raised_to`, ISO UTC, inclusive): every alert
    a particular run raised, whatever bar it sits on.
"""

from __future__ import annotations

import numpy as np

from src.utils.timeutils import date_str_to_ts, iso_utc_to_ts

DAY_S = 86400
_OPEN = (-(2 ** 62), 2 ** 62)


def incident_ranges(cfg: dict) -> list[dict]:
    """Each incident as {bar: (lo, hi), raised: (lo, hi), reason, label}.

    Half-open ranges in epoch seconds; an absent side is unbounded.
    """
    out = []
    for entry in (cfg.get("live") or {}).get("data_incidents") or []:
        bar, raised = _OPEN, _OPEN
        if "first_bar_date" in entry:
            bar = (date_str_to_ts(str(entry["first_bar_date"])),
                   date_str_to_ts(str(entry["last_bar_date"])) + DAY_S)
        if "raised_from" in entry:
            raised = (iso_utc_to_ts(str(entry["raised_from"])),
                      iso_utc_to_ts(str(entry["raised_to"])) + 1)
        if bar == _OPEN and raised == _OPEN:
            raise ValueError(f"data incident names no alerts: {entry}")
        if bar[1] <= bar[0] or raised[1] <= raised[0]:
            raise ValueError(f"data incident ends before it starts: {entry}")
        label = (f"bars {entry['first_bar_date']} to {entry['last_bar_date']}"
                 if bar != _OPEN else f"alerts raised {entry['raised_from']}")
        out.append({"bar": bar, "raised": raised, "label": label,
                    "reason": str(entry.get("reason", "")).strip()})
    return out


def in_incident(cfg: dict, ts, raised=None) -> np.ndarray:
    """Boolean per alert: does it fall inside a recorded incident?

    `raised` is required whenever an incident is defined by raise time; an
    alert whose raise time is unknown cannot be cleared of one.
    """
    stamps = np.asarray(ts, dtype="int64")
    hit = np.zeros(stamps.shape, dtype=bool)
    for inc in incident_ranges(cfg):
        (blo, bhi), (rlo, rhi) = inc["bar"], inc["raised"]
        mask = (stamps >= blo) & (stamps < bhi)
        if (rlo, rhi) != _OPEN:
            if raised is None:
                raise ValueError(f"incident '{inc['label']}' is defined by raise "
                                 f"time; pass the alerts' raised_utc")
            r = np.asarray(raised, dtype="int64")
            mask &= (r >= rlo) & (r < rhi)
        hit |= mask
    return hit


def sql_exclusion(cfg: dict, table: str = "a") -> tuple[str, list[int]]:
    """A WHERE fragment (and its params) leaving incident alerts out, or ('', [])."""
    parts, params = [], []
    for inc in incident_ranges(cfg):
        (blo, bhi), (rlo, rhi) = inc["bar"], inc["raised"]
        parts.append(f"NOT ({table}.ts_utc >= ? AND {table}.ts_utc < ? AND "
                     f"{table}.raised_utc >= ? AND {table}.raised_utc < ?)")
        params += [blo, bhi, rlo, rhi]
    return " AND ".join(parts), params
