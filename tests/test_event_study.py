"""The event study lines every event up on t0 in trading time.

Its one easy mistake is an off-by-one at t0: a bar that straddles t0 counted
as "before" would put post-announcement trading on the pre-announcement side
of the chart, and the footprint would look larger than it is. These tests pin
the alignment, the matched control, and the per-slice summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.event_study import event_paths, summarise
from src.utils.config import load_config

H = 3600


def _bars(n: int = 60, start: int = 1_700_000_000) -> pd.DataFrame:
    idx = pd.Index([start + i * H for i in range(n)], name="ts_utc")
    # A little noise: a perfectly flat baseline has zero spread, and the
    # z-score is then undefined by design (`volume_zscore` returns NaN).
    vol = 100.0 + np.random.default_rng(0).normal(0, 5, n)
    vol[40] = 1000.0                                  # one loud bar
    return pd.DataFrame({"close": np.linspace(10, 11, n), "volume": vol},
                        index=idx)


def _cfg() -> dict:
    cfg = load_config()
    cfg = {**cfg, "features": {**cfg["features"], "volume_zscore_window_h": 10,
                               "min_baseline_bars": 5}}
    return cfg


def _event(t0: int, scheduled: bool = False, eid: str = "e1") -> pd.DataFrame:
    return pd.DataFrame([{"event_id": eid, "t0_utc": t0,
                          "is_scheduled": scheduled}])


def test_bar_zero_follows_a_t0_that_falls_between_bars():
    bars = _bars()
    ts = bars.index.to_numpy()
    # t0 exactly at the END of bar 39 (= start of bar 40): bar 39 is wholly
    # before, so it is -1, and the loud bar 40 is bar 0.
    p = event_paths(_event(int(ts[40])), bars, pd.Series(0.0, index=bars.index),
                    pre=5, post=3, cfg=_cfg())
    z = p.set_index("bar")["volume_z"]
    assert z.idxmax() == 0 and z[0] > 3 * z[-1]


def test_a_bar_that_straddles_t0_is_bar_zero_never_before():
    bars = _bars()
    ts = bars.index.to_numpy()
    # t0 half-way through bar 40: that bar is not wholly before t0.
    p = event_paths(_event(int(ts[40]) + H // 2), bars,
                    pd.Series(0.0, index=bars.index), pre=5, post=3, cfg=_cfg())
    assert p.loc[p["bar"] == 0, "volume_z"].iloc[0] > 3
    assert (p["bar"] < 0).sum() == 5


def test_an_event_without_room_either_side_is_dropped_not_padded():
    bars = _bars()
    ts = bars.index.to_numpy()
    zero = pd.Series(0.0, index=bars.index)
    assert event_paths(_event(int(ts[2])), bars, zero, 5, 3, _cfg()).empty
    assert event_paths(_event(int(ts[-2])), bars, zero, 5, 3, _cfg()).empty


def test_the_control_is_the_same_alignment_earlier_and_skips_other_events():
    bars = _bars()
    ts = bars.index.to_numpy()
    zero = pd.Series(0.0, index=bars.index)
    ev = _event(int(ts[40]))
    ctl = event_paths(ev, bars, zero, 5, 3, _cfg(), shift=14)
    assert set(ctl["slice"]) == {"unscheduled control"}
    assert ctl.loc[ctl["bar"] == 0, "volume_z"].iloc[0] < 1   # not the loud bar
    # Another event of the stock inside the shifted window: no control.
    clash = np.array([int(ts[26])])
    assert event_paths(ev, bars, zero, 5, 3, _cfg(), shift=14,
                       avoid=clash).empty


def test_the_summary_keeps_slices_apart_and_counts_events():
    paths = pd.DataFrame({
        "event_id": ["a", "a", "b", "b"],
        "slice": ["scheduled", "scheduled", "unscheduled", "unscheduled"],
        "bar": [-1, 0, -1, 0],
        "volume_z": [1.0, 5.0, 0.0, 3.0],
        "abs_rel_ret": [0.1, 2.0, 0.1, 1.0],
    })
    ref = pd.DataFrame({"volume_z": [0.0, 0.0, 4.0], "abs_rel_ret": [0, 0, 0]})
    out = summarise(paths, ref, threshold=2.0)
    s0 = out[(out["slice"] == "scheduled") & (out["bar"] == 0)].iloc[0]
    assert s0["n_events"] == 1 and s0["share_above_threshold"] == 1.0
    assert set(out["slice"]) == {"scheduled", "unscheduled", "all hours"}
    assert out.loc[out["slice"] == "all hours", "share_above_threshold"].iloc[0] \
        == 1 / 3
