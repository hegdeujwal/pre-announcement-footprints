"""FootprintNet: inputs, leakage, the clock-matched negatives, and a tiny run."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.baselines import footprint_net as F  # noqa: E402
from src.baselines import footprint_train as T  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.timeutils import get_market_calendar  # noqa: E402
from tests.test_leakage import assert_no_lookahead  # noqa: E402

HOUR = 3600
L = 8


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    k = c["baselines"]["footprint_net"]
    k.update(length=L, hidden=8, threads=1, runs_dir=str(tmp_path / "runs"))
    k["pretrain"].update(windows_per_epoch=256, epochs=2, batch=64, warmup=2)
    k["finetune"].update(epochs=3, batch=64, negatives_per_positive=3, patience=3)
    return c


def session_stamps(cfg, start="2025-10-01", end="2025-12-31"):
    cal = get_market_calendar(cfg["market"]["calendar"])
    out = []
    for s in cal.sessions_in_range(start, end):
        o, c = int(cal.session_open(s).timestamp()), int(cal.session_close(s).timestamp())
        out += list(range(o, c, HOUR))
    return np.array(out, dtype="int64")


def frame(cfg, tickers=("A", "B", "C", "D"), every=5, seed=0):
    """Every bar a quiet single-bar window, except an L-bar positive episode
    ending on the closing bar every `every`-th session, whose volume ramps."""
    rng = np.random.default_rng(seed)
    stamps = session_stamps(cfg)
    closing = np.flatnonzero(F.bars_to_close(stamps, cfg) == 0)
    rows = []
    for t in tickers:
        vol = rng.normal(0, 1, len(stamps))
        claimed = np.zeros(len(stamps), bool)
        episodes = closing[L::every]
        for end in episodes:
            vol[end - 2:end + 1] += [1.5, 3.0, 4.5]
            claimed[end - L + 1:end + 1] = True
        for i, ts in enumerate(stamps):
            base = {c: rng.normal(0, 1) for c in F.knobs(cfg)["channels"]}
            base.update(volume_logz=vol[i], ticker=t, ts_utc=ts)
            if not claimed[i]:
                rows.append({**base, "window_id": f"bar:{t}:{ts}", "t0_utc": None,
                             "is_scheduled": None, "item_code": None})
        for e_i, end in enumerate(episodes):
            for i in range(end - L + 1, end + 1):
                base = {c: rng.normal(0, 1) for c in F.knobs(cfg)["channels"]}
                base.update(volume_logz=vol[i], ticker=t, ts_utc=stamps[i])
                rows.append({**base, "window_id": f"E{t}{e_i}",
                             "t0_utc": int(stamps[end]) + HOUR,
                             "is_scheduled": False, "item_code": "8.01"})
    return pd.DataFrame(rows).astype({
        "window_id": "string", "ticker": "string", "ts_utc": "Int64",
        "t0_utc": "Int64", "is_scheduled": "boolean", "item_code": "string"})


@pytest.fixture
def trained(cfg, tmp_path):
    f = frame(cfg)
    cutoff = int(np.quantile(f["ts_utc"].astype("int64"), 0.75))
    data = T.Prepared(f, cfg, cutoff)
    runs = tmp_path / "runs"
    T.pretrain(data, cfg, 0, runs / "pre", log=lambda *_: None)
    T.finetune(data, cfg, 0, runs / "ft", runs / "pre", log=lambda *_: None)
    return f, data, runs


# -- inputs --------------------------------------------------------------

def test_inputs_never_read_label_columns(cfg):
    f = frame(cfg)
    scrubbed = f.assign(t0_utc=pd.NA, is_scheduled=pd.NA, item_code=pd.NA)
    np.testing.assert_array_equal(F.bar_inputs(f, cfg), F.bar_inputs(scrubbed, cfg))


def test_inputs_are_clipped_and_carry_the_two_clock_columns(cfg):
    f = frame(cfg).iloc[:50].copy()
    f["volume_logz"] = 1e6
    X = F.bar_inputs(f, cfg)
    assert X.shape[1] == len(F.knobs(cfg)["channels"]) + 2
    assert X[:, 0].max() == F.knobs(cfg)["clip"]
    assert 0 <= X[:, -2].min() and X[:, -2].max() <= 1


def test_windows_end_at_the_row_and_pad_on_the_left():
    X = np.arange(10, dtype="float32")[:, None]
    w = F.windows(X, np.array([1, 9]), 4)
    np.testing.assert_array_equal(w[0, :, 0], [0, 0, 0, 1])
    np.testing.assert_array_equal(w[1, :, 0], [6, 7, 8, 9])


ALL_FIT = 2**62      # an inner-split cutoff after every bar: everything fits


def test_prepared_windows_never_cross_into_another_stock(cfg):
    f = frame(cfg, tickers=("A", "B"))
    n_a = f.loc[f.ticker == "A", "ts_utc"].nunique()
    data = T.Prepared(f, cfg, ALL_FIT)
    first_b = data.rows[n_a]
    w = data.window(np.array([first_b]))
    # B's first window holds B's first bar and zero padding — none of A's bars.
    assert (w[0, :-1] == 0).all() and (w[0, -1] != 0).any()


# -- the clock-matched negatives ---------------------------------------------

def test_negatives_match_the_positives_bar_of_session_mix(cfg):
    data = T.Prepared(frame(cfg), cfg, ALL_FIT)
    region = data.real
    pos, neg = T.matched_negatives(data, region, 3, np.random.default_rng(0))
    p = pd.Series(data.bar_of_session[pos]).value_counts(normalize=True).sort_index()
    n = pd.Series(data.bar_of_session[neg]).value_counts(normalize=True).sort_index()
    pd.testing.assert_series_equal(p, n, check_names=False)
    assert len(neg) == 3 * len(pos)
    assert data.is_quiet[neg].all() and data.is_final_pos[pos].all()


# -- training and scoring -----------------------------------------------------

def test_a_tiny_run_trains_saves_and_scores(cfg, trained):
    f, data, runs = trained
    manifest = json.loads((runs / "ft" / "manifest.json").read_text())
    assert manifest["stage"] == "finetune" and manifest["init"] == "pretrained"
    assert manifest["n_pos"] > 0 and manifest["n_neg"] == 3 * manifest["n_pos"]
    det = F.FootprintDetector(cfg, runs / "ft")
    s = det.score(f)
    assert s.notna().all() and len(s) == len(f)
    sur = F.FootprintDetector(cfg, runs / "pre", mode="surprise").score(f)
    assert sur.notna().mean() > 0.9           # only each stock's first bar is NaN


def test_detector_score_has_no_lookahead(cfg, trained):
    _, _, runs = trained
    det = F.FootprintDetector(cfg, runs / "ft")
    one = frame(cfg, tickers=("A",)).drop_duplicates("ts_utc").sort_values("ts_utc").head(60)
    cols = F.knobs(cfg)["channels"]
    base = one.set_index("ts_utc")[cols].astype("float64")

    def fn(fr):
        g = one.set_index("ts_utc").copy()
        g[cols] = fr[cols]
        g = g.reset_index()
        return pd.DataFrame({"s": det.score(g).to_numpy()}, index=fr.index)

    assert_no_lookahead(fn, base)


def test_detector_score_never_reads_labels(cfg, trained):
    f, _, runs = trained
    det = F.FootprintDetector(cfg, runs / "ft")
    scrubbed = f.assign(t0_utc=pd.NA, is_scheduled=pd.NA, item_code=pd.NA)
    pd.testing.assert_series_equal(det.score(f), det.score(scrubbed))


def test_a_detect_head_needs_a_finetuned_run(cfg, trained):
    _, _, runs = trained
    with pytest.raises(ValueError, match="fine-tuned"):
        F.FootprintDetector(cfg, runs / "pre")


def test_tampered_weights_and_moved_channels_are_refused(cfg, trained):
    _, _, runs = trained
    other = copy.deepcopy(cfg)
    other["baselines"]["footprint_net"]["channels"] = ["volume_logz"]
    with pytest.raises(ValueError, match="channels"):
        F.load_run(runs / "ft", other)
    (runs / "ft" / "model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        F.load_run(runs / "ft", cfg)


# -- learning curve -------------------------------------------------------------

def test_learning_curve_shares_are_nested_and_sized(cfg):
    data = T.Prepared(frame(cfg), cfg, ALL_FIT)
    n = int((data.fit & data.is_final_pos).sum())
    quarter, half = (T.share_of_positives(data, f, 7) for f in (0.25, 0.5))
    assert len(quarter) == round(0.25 * n) and len(half) == round(0.5 * n)
    assert set(quarter) <= set(half)
    assert data.is_final_pos[half].all()
    np.testing.assert_array_equal(T.share_of_positives(data, 1.0, 7),
                                  np.flatnonzero(data.fit & data.is_final_pos))
    with pytest.raises(ValueError, match="fraction"):
        T.share_of_positives(data, 0, 7)


def test_a_share_run_uses_only_its_filings_with_their_negatives(cfg, tmp_path):
    f = frame(cfg)
    data = T.Prepared(f, cfg, int(np.quantile(f["ts_utc"].astype("int64"), 0.75)))
    m = T.finetune(data, cfg, 0, tmp_path / "half", None, log=lambda *_: None,
                   fraction=0.5)
    full = int((data.fit & data.is_final_pos).sum())
    assert m["fraction"] == 0.5 and m["n_pos"] == round(0.5 * full)
    assert m["n_neg"] == 3 * m["n_pos"]
    assert T.run_name(42, "scratch", 0.25) == "s42-scratch-f25"
    assert T.run_name(42, "scratch") == "s42-scratch"
