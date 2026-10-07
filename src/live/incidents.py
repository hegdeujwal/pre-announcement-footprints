"""Stretches of the live alert log scored on data known to be bad.

The alert log is append-only and hash-chained, so a day the monitor scored on
corrupt bars cannot be removed from it, and should not be: what the monitor
said, and when, is part of the record. What changes is what gets COUNTED.
`live.data_incidents` in config names each stretch by bar date, with its
reason, and every live rate — the outcome hit rates, the chance comparison,
the dashboard — leaves those alerts out through this one function, so the
exclusion is the same everywhere and visible in one place.

Dates are bar dates in UTC, inclusive at both ends. An XNYS session never
crosses a UTC midnight, so a UTC date is the session.
"""

from __future__ import annotations

import numpy as np

from src.utils.timeutils import date_str_to_ts

DAY_S = 86400


def incident_ranges(cfg: dict) -> list[tuple[int, int, str]]:
    """[(start_ts, end_ts_exclusive, reason)] from `live.data_incidents`."""
    out = []
    for entry in (cfg.get("live") or {}).get("data_incidents") or []:
        lo = date_str_to_ts(str(entry["first_bar_date"]))
        hi = date_str_to_ts(str(entry["last_bar_date"])) + DAY_S
        if hi <= lo:
            raise ValueError(f"data incident ends before it starts: {entry}")
        out.append((lo, hi, str(entry.get("reason", "")).strip()))
    return out


def in_incident(cfg: dict, ts) -> np.ndarray:
    """Boolean per bar timestamp: does it fall inside a recorded incident?"""
    stamps = np.asarray(ts, dtype="int64")
    hit = np.zeros(stamps.shape, dtype=bool)
    for lo, hi, _ in incident_ranges(cfg):
        hit |= (stamps >= lo) & (stamps < hi)
    return hit


def sql_exclusion(cfg: dict, column: str = "a.ts_utc") -> tuple[str, list[int]]:
    """A WHERE fragment (and its params) leaving incident bars out, or ('', [])."""
    ranges = incident_ranges(cfg)
    if not ranges:
        return "", []
    clause = " AND ".join(f"NOT ({column} >= ? AND {column} < ?)" for _ in ranges)
    return clause, [v for lo, hi, _ in ranges for v in (lo, hi)]
