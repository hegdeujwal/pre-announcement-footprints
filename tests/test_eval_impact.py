"""Impact split: buckets and slices."""

import numpy as np
import pandas as pd

from src.eval import impact as I
from tests.test_baselines_template import windows


def test_buckets_cover_every_value_once():
    edges = np.array([0.05, 0.10])
    b = I.bucket_of(pd.Series([0.01, 0.05, 0.07, 0.5]), edges)
    assert b.tolist() == ["impact low", "impact mid", "impact mid", "impact high"]


def test_impact_slices_keep_every_quiet_window():
    pred = windows([("A", True, "8.01"), ("B", True, "8.01"),
                    ("C", False, None)], n=3)
    buckets = pd.Series({"W0": "impact low", "W1": "impact high"})
    out = I.impact_slices(pred, buckets)
    assert set(out["impact low"].window_id) == {"W0", "W2"}
    assert set(out["impact high"].window_id) == {"W1", "W2"}
    assert out["impact mid"]["t0_utc"].notna().sum() == 0


def test_impact_rows_split_by_schedule():
    pred = windows([("A", True, "2.02"), ("B", True, "8.01"),
                    ("C", False, None), ("D", False, None)], n=3)
    pred["score"] = pred["volume_z"]
    pred["action"] = "WAIT"
    impacts = pd.Series({"W0": 0.01, "W1": 0.2})
    rows = pd.DataFrame(I.impact_rows("x", pred, impacts, np.array([0.05, 0.1])))
    got = set(zip(rows["schedule"], rows["slice"]))
    assert ("scheduled", "impact low") in got and ("unscheduled", "impact high") in got
    assert ("scheduled", "impact high") not in got
    assert ("unscheduled", "impact low") not in got
