"""The stock-selection control — which stock, never which hour.

What matters is not that a lookup returns a number. It is that the row reads
nothing about the hour (so it cannot detect a moment), that its tie-breaking
jitter can never carry one stock past another, and that it spends the same
alert budget as every other row instead of overspending on ties.
"""

import pandas as pd
import pytest

from src import db
from src.baselines import TickerPrior
from src.eval.metrics import precision_at_alert_budget
from src.pipeline.split import boundaries, seal
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


def make_frame(tickers: dict[str, tuple[int, int]], hours: int = 3) -> pd.DataFrame:
    """`tickers` maps name -> (positive windows, quiet windows)."""
    base = date_str_to_ts("2025-10-01")
    rows, w = [], 0
    for ticker, (n_pos, n_quiet) in tickers.items():
        for i in range(n_pos + n_quiet):
            anchor = base + w * 100 * HOUR
            positive = i < n_pos
            for h in range(hours if positive else 1):
                rows.append({
                    "window_id": f"W{w}", "ticker": ticker,
                    "ts_utc": anchor - (hours - h) * HOUR,
                    "t0_utc": anchor if positive else None,
                    "is_scheduled": False if positive else None,
                    "item_code": "8.01" if positive else None,
                })
            w += 1
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


def test_prior_counts_episodes_not_rows(cfg):
    """A positive episode spans many hourly rows; it is one filing."""
    train = make_frame({"AAA": (3, 0), "BBB": (1, 0)}, hours=5)
    model = TickerPrior(cfg).fit(train)
    assert model.prior.to_dict() == {"AAA": 3.0, "BBB": 1.0}


def test_jitter_never_moves_a_row_past_another_stock(cfg):
    train = make_frame({"HOT": (2, 0), "WARM": (1, 0)})
    model = TickerPrior(cfg).fit(train)
    scores = model.score(make_frame({"HOT": (0, 200), "WARM": (0, 200),
                                     "COLD": (0, 200)}))
    tick = make_frame({"HOT": (0, 200), "WARM": (0, 200),
                       "COLD": (0, 200)})["ticker"]
    by = scores.groupby(tick.to_numpy())
    assert by.min()["HOT"] > by.max()["WARM"] > by.min()["WARM"]
    assert by.min()["WARM"] > by.max()["COLD"]


def test_it_reads_nothing_about_the_hour(cfg):
    """Shifting every timestamp must not change a single score."""
    model = TickerPrior(cfg).fit(make_frame({"AAA": (2, 0)}))
    frame = make_frame({"AAA": (1, 5), "BBB": (0, 5)})
    shifted = frame.assign(ts_utc=frame["ts_utc"] + 37 * HOUR)
    pd.testing.assert_series_equal(model.score(frame), model.score(shifted))


def test_ties_do_not_overspend_the_alert_budget(cfg):
    """Without jitter every hour of HOT ties, and the cut admits all of them."""
    model = TickerPrior(cfg).fit(make_frame({"HOT": (1, 0)}))
    frame = make_frame({"HOT": (2, 300), "COLD": (1, 300)})
    result = precision_at_alert_budget(model.predict(frame, threshold=0.0),
                                       max_alerts=10)
    assert result.realised_alerts == 10


def test_an_unseen_stock_scores_the_floor_not_nan(cfg):
    model = TickerPrior(cfg).fit(make_frame({"AAA": (1, 0)}))
    scores = model.score(make_frame({"ZZZ": (0, 3)}))
    assert scores.notna().all() and (scores < 0.5).all()


def test_a_train_frame_without_positives_is_refused(cfg):
    with pytest.raises(ValueError, match="no positive episodes"):
        TickerPrior(cfg).fit(make_frame({"AAA": (0, 4)}))


def test_score_before_fit_is_refused(cfg):
    with pytest.raises(RuntimeError, match="fit"):
        TickerPrior(cfg).score(make_frame({"AAA": (1, 0)}))


def test_fit_refuses_the_sealed_test_split(cfg, tmp_path):
    conn = db.get_conn(tmp_path / "prior.db")
    seal(cfg, conn)
    _, val_end = boundaries(cfg)
    sealed = make_frame({"AAA": (1, 0)})
    sealed["ts_utc"] = pd.array([val_end + i * HOUR for i in range(len(sealed))],
                                dtype="Int64")
    sealed["t0_utc"] = pd.array([val_end + 99 * HOUR] * len(sealed), dtype="Int64")
    with pytest.raises(SystemExit, match="SEALED TEST SET"):
        TickerPrior(cfg).fit(sealed, conn=conn)
