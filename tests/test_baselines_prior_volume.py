"""Which stock AND which hour — the naive-Bayes combination.

The properties that matter: each term moves the score the way its evidence
says, neither term reads the evaluation frame (both are fitted on train), a
missing volume reading keeps the stock term instead of flooring the window,
and the discrete score does not overspend the alert budget through ties.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.baselines import PriorVolume
from src.eval.metrics import precision_at_alert_budget
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


def make_frame(spec, hours: int = 3, start: str = "2025-10-01") -> pd.DataFrame:
    """spec: list of (ticker, is_positive, final_volume_z)."""
    base = date_str_to_ts(start)
    rows = []
    for w, (ticker, positive, z) in enumerate(spec):
        anchor = base + w * 100 * HOUR
        n = hours if positive else 1
        for h in range(n):
            rows.append({
                "window_id": f"W{w}", "ticker": ticker,
                "ts_utc": anchor - (n - h) * HOUR,
                "t0_utc": anchor if positive else None,
                "is_scheduled": False if positive else None,
                "item_code": "8.01" if positive else None,
                # Only the final bar carries the signal; earlier bars are 0.
                "volume_z": z if h == n - 1 else 0.0,
            })
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


def train_frame(rng) -> pd.DataFrame:
    """HOT files often, COLD rarely; pre-filing bars run hot on volume."""
    spec = ([("HOT", True, rng.normal(3, 1)) for _ in range(40)]
            + [("COLD", True, rng.normal(3, 1)) for _ in range(4)]
            + [(t, False, rng.normal(0, 1)) for t in ("HOT", "COLD") for _ in range(400)])
    return make_frame(spec)


@pytest.fixture
def model(cfg):
    return PriorVolume(cfg).fit(train_frame(np.random.default_rng(0)))


def test_the_stock_term_orders_stocks_at_equal_volume(model):
    s = model.score(make_frame([("HOT", False, 0.0), ("COLD", False, 0.0),
                                ("NEVER", False, 0.0)]))
    assert s.iloc[0] > s.iloc[1] > s.iloc[2]


def test_the_volume_term_orders_hours_within_a_stock(model):
    s = model.score(make_frame([("COLD", False, 0.0), ("COLD", False, 4.0)]))
    assert s.iloc[1] > s.iloc[0]


def test_a_spike_can_outrank_a_frequent_filer(model):
    """The whole point: neither signal alone decides."""
    s = model.score(make_frame([("HOT", False, -1.0), ("COLD", False, 5.0)]))
    assert s.iloc[1] > s.iloc[0]


def test_missing_volume_keeps_the_stock_term(model):
    s = model.score(make_frame([("HOT", False, np.nan), ("COLD", False, np.nan)]))
    assert s.notna().all() and s.iloc[0] > s.iloc[1]


def test_scores_are_finite_far_outside_the_train_range(model):
    s = model.score(make_frame([("HOT", False, 1e6), ("HOT", False, -1e6)]))
    assert np.isfinite(s).all()


def test_ties_do_not_overspend_the_alert_budget(model):
    frame = make_frame([("HOT", False, 0.0)] * 300 + [("COLD", True, 0.0)] * 5)
    result = precision_at_alert_budget(model.predict(frame, threshold=0.0),
                                       max_alerts=10)
    assert result.realised_alerts == 10


def test_a_one_class_train_frame_is_refused(cfg):
    with pytest.raises(ValueError, match="both positive and quiet"):
        PriorVolume(cfg).fit(make_frame([("A", False, 0.0)] * 5))


def test_score_before_fit_is_refused(cfg):
    with pytest.raises(RuntimeError, match="fit"):
        PriorVolume(cfg).score(make_frame([("A", False, 0.0)]))


def test_non_positive_alpha_is_refused(cfg):
    bad = {**cfg, "baselines": {**cfg["baselines"],
                                "prior_volume": {"alpha": 0, "n_bins": 5}}}
    with pytest.raises(ValueError, match="alpha"):
        PriorVolume(bad)


def test_fit_refuses_the_sealed_test_split(cfg, tmp_path):
    conn = db.get_conn(tmp_path / "pv.db")
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    sealed = make_frame([("A", True, 1.0), ("A", False, 0.0)])
    sealed["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(sealed))],
                                dtype="Int64")
    sealed.loc[sealed.t0_utc.notna(), "t0_utc"] = val_end + 99 * HOUR
    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        PriorVolume(cfg).fit(sealed, conn=conn)
