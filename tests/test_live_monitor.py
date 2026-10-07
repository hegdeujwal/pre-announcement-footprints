"""P7-01 — the live loop, and the one property that makes its claim worth making.

`test_live_features_match_the_training_code_path` is the done-when. If the live
frame were built by a second implementation of the features, the live result
would measure the drift between the two rather than the market, and "of the N
alerts it raised live, M were followed by a filing" would be a statement about
a bug. So the live path calls `features.ticker_features` — the same function
`build_matrix` and `build_eval_frame` call — and this asserts the values agree
column for column.
"""

import numpy as np
import pandas as pd
import pytest

from src import db
from src.eval import contract
from src.live import (Alert, build_detectors, conform, default_thresholds,
                      latest_bar_frame, latest_stored_bar, scan)
from src.pipeline.features import _event_times, _ticker_frame, ticker_features
from src.pipeline.split import LIVE, seal, split_of
from src.utils.config import load_config
from src.utils.timeutils import date_str_to_ts

HOUR = 3600


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def conn(tmp_path, cfg):
    """Two tickers with enough history for volume_z, running past the window."""
    c = db.get_conn(tmp_path / "live.db")
    iv = cfg["market"]["interval"]
    start = date_str_to_ts("2025-09-01")
    n = 1000                                # past volume_zscore_window_h

    for ticker in ("AAA", "BBB", cfg["market"]["benchmark"]):
        db.upsert_bars(c, [
            (ticker, start + i * HOUR, 100.0, 101.0, 99.0,
             100.0 + (i % 7) * 0.1,
             1_000_000 + (i % 13) * 1000 + (500_000 if i == n - 1 else 0), iv)
            for i in range(n)])

    db.upsert_companies(c, [{"cik": "C1", "ticker": "AAA", "in_universe": 1},
                            {"cik": "C2", "ticker": "BBB", "in_universe": 1}])
    # The freeze stamp sits at the start of these bars, as the real one sits
    # before every live bar; the restate must never reach behind it.
    db.set_meta(c, f"snapshot_frozen_{iv}", "2025-09-01 00:00:00Z", start)
    return c


@pytest.fixture
def as_of():
    return date_str_to_ts("2025-09-01") + 999 * HOUR


# --------------------------------------------------------------------------
# THE done-when
# --------------------------------------------------------------------------
def test_live_features_match_the_training_code_path(cfg, conn, as_of):
    """A second implementation would drift from what the detectors were tuned
    on, and the live numbers would measure the drift rather than the market."""
    live = latest_bar_frame(cfg, conn, ["AAA"], as_of=as_of, lookback_bars=10)

    # Rebuild the same rows straight from the training builders.
    interval = cfg["market"]["interval"]
    bench = _ticker_frame(conn, cfg["market"]["benchmark"], interval)
    filings, earnings = _event_times(conn, cfg, "AAA")
    expected = ticker_features(_ticker_frame(conn, "AAA", interval), bench,
                               filings, earnings, cfg)

    feature_cols = [c for c in live.columns
                    if c not in ("window_id", "ticker", "ts_utc", "t0_utc",
                                 "is_scheduled", "item_code")]
    for _, row in live.iterrows():
        ref = expected.loc[int(row["ts_utc"])]
        for col in feature_cols:
            a, b = row[col], ref[col]
            if pd.isna(a) and pd.isna(b):
                continue
            assert a == pytest.approx(b), f"{col} at {row['ts_utc']} differs"


def test_features_are_computed_on_full_history_not_the_slice(cfg, conn, as_of):
    """volume_z needs min_baseline_bars of prior bars. Computing on a short
    live slice would return NaN for every row and look entirely reasonable."""
    live = latest_bar_frame(cfg, conn, ["AAA"], as_of=as_of, lookback_bars=5)
    assert np.isfinite(live["volume_z"]).any()


# --------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------
def test_it_scores_only_bars_at_or_before_now(cfg, conn):
    """The live monitor must never see a bar from the future — the same rule
    every offline builder follows, here enforced against the clock."""
    cutoff = date_str_to_ts("2025-09-01") + 500 * HOUR
    live = latest_bar_frame(cfg, conn, ["AAA"], as_of=cutoff)
    assert (live["ts_utc"] <= cutoff).all()


def test_each_live_bar_is_its_own_decision_point(cfg, conn, as_of):
    """Matching how the evaluation population treats a quiet hour, so a live
    alert means what a validation alert meant."""
    live = latest_bar_frame(cfg, conn, ["AAA"], as_of=as_of, lookback_bars=6)
    assert live["window_id"].nunique() == len(live)
    assert live["window_id"].str.startswith("live:").all()


def test_label_columns_are_null_because_nobody_knows_yet(cfg, conn, as_of):
    """There is no t0 on a live bar. That is the whole point: whether news is
    coming is unknowable until P7-03 backfills the outcome."""
    live = latest_bar_frame(cfg, conn, ["AAA"], as_of=as_of)
    assert live["t0_utc"].isna().all()
    assert live["is_scheduled"].isna().all()
    assert live["item_code"].isna().all()


def test_the_frame_satisfies_the_contract(cfg, conn, as_of):
    from src.baselines import CUSUM

    live = conform(latest_bar_frame(cfg, conn, ["AAA", "BBB"], as_of=as_of))
    out = CUSUM(cfg).predict(live, threshold=2.0)
    pd.testing.assert_frame_equal(out, contract.validate_predictions(out))


def test_an_empty_universe_is_refused(cfg, conn, as_of):
    with pytest.raises(SystemExit, match="no in-universe companies"):
        latest_bar_frame(cfg, conn, [], as_of=as_of)


def test_a_time_before_any_bar_is_refused(cfg, conn):
    with pytest.raises(SystemExit, match="no bars at or before"):
        latest_bar_frame(cfg, conn, ["AAA"],
                         as_of=date_str_to_ts("2020-01-01"))


# --------------------------------------------------------------------------
# The seal must not block live data
# --------------------------------------------------------------------------
def test_bars_after_the_study_window_are_live_not_sealed(cfg, conn):
    """P7-01's other prerequisite. `split_of` used to call everything after
    val_end TEST, unbounded, so the monitor would have been refused its own
    inputs by a seal that was protecting nothing."""
    from src.baselines import CUSUM

    seal(cfg, conn)
    after = date_str_to_ts(cfg["study_window"]["end"]) + 5 * HOUR
    assert split_of(cfg, after) == LIVE

    iv = cfg["market"]["interval"]
    db.upsert_bars(conn, [(t, after + i * HOUR, 100.0, 101.0, 99.0,
                           100.0 + i * 0.1, 1_000_000 + i * 900, iv)
                          for t in ("AAA", cfg["market"]["benchmark"])
                          for i in range(60)])
    live = conform(latest_bar_frame(cfg, conn, ["AAA"], as_of=after + 59 * HOUR,
                                    lookback_bars=5))
    # Would raise SystemExit("SEALED TEST SET") before the fix.
    CUSUM(cfg).predict(live, threshold=2.0, conn=conn, context="live test")


# --------------------------------------------------------------------------
# Detectors and alerts
# --------------------------------------------------------------------------
def test_every_detector_runs_not_only_the_winner(cfg):
    """Running all of them turns the live period into a forward-looking
    replication of the Phase 5/6 comparison rather than a one-detector demo."""
    detectors = build_detectors(cfg)
    assert "cusum" in detectors and "volume_zscore" in detectors


def test_always_quiet_is_not_among_them(cfg):
    """It never alerts, so it would contribute nothing to an alert log."""
    assert "always_quiet" not in build_detectors(cfg)


def test_thresholds_come_from_the_tuned_operating_points(cfg):
    """Without this, "it raised N alerts" would be a claim about an arbitrary
    cut rather than about the detector that was evaluated."""
    t = default_thresholds(cfg)
    assert t["cusum"] == cfg["baselines"]["cusum"]["threshold"]
    assert t["volume_zscore"] == cfg["baselines"]["volume_zscore"]["threshold"]


def test_alerts_carry_the_features_that_triggered_them(cfg, conn, as_of):
    """P7-02 logs these. An alert without its inputs cannot be audited later."""
    live = conform(latest_bar_frame(cfg, conn, ["AAA", "BBB"], as_of=as_of))
    alerts = scan(cfg, conn, live, build_detectors(cfg),
                  thresholds={"cusum": -1e9, "volume_zscore": -1e9})
    assert alerts
    a = alerts[0]
    assert "volume_z" in a.features
    row = a.as_row()
    assert row["ticker"] == a.ticker and row["detector"] == a.detector
    assert row["raised_utc"] > 0


def test_alert_rows_are_json_safe(cfg, conn, as_of):
    """NaN is not valid JSON and would break the append-only log P7-02 writes."""
    import json

    live = conform(latest_bar_frame(cfg, conn, ["AAA"], as_of=as_of))
    alerts = scan(cfg, conn, live, build_detectors(cfg),
                  thresholds={"cusum": -1e9, "volume_zscore": -1e9})
    json.dumps([a.as_row() for a in alerts])          # must not raise


def test_a_quiet_market_raises_nothing(cfg, conn, as_of):
    """The common case, and it must not error."""
    live = conform(latest_bar_frame(cfg, conn, ["AAA"], as_of=as_of))
    assert scan(cfg, conn, live, build_detectors(cfg),
                thresholds={"cusum": 1e9, "volume_zscore": 1e9}) == []


def test_alerts_are_ordered_deterministically(cfg, conn, as_of):
    live = conform(latest_bar_frame(cfg, conn, ["AAA", "BBB"], as_of=as_of))
    kw = dict(thresholds={"cusum": -1e9, "volume_zscore": -1e9})
    a = scan(cfg, conn, live, build_detectors(cfg), **kw)
    b = scan(cfg, conn, live, build_detectors(cfg), **kw)
    assert [(x.ts_utc, x.detector, x.ticker) for x in a] == \
           [(x.ts_utc, x.detector, x.ticker) for x in b]


# --------------------------------------------------------------------------
# Fetching appends, never re-downloads
# --------------------------------------------------------------------------
def test_latest_stored_bar_finds_the_newest(cfg, conn):
    newest = latest_stored_bar(conn, cfg["market"]["interval"])
    assert newest == date_str_to_ts("2025-09-01") + 999 * HOUR


def test_fetch_is_a_noop_when_nothing_is_missing(cfg, conn):
    """With restating off, a fetch at the newest bar has nothing to do."""
    from src.live.monitor import fetch_latest

    off = {**cfg, "live": {**cfg["live"], "restate_sessions": 0}}
    newest = latest_stored_bar(conn, cfg["market"]["interval"])
    assert fetch_latest(off, conn, tickers=["AAA"], now_ts=newest) == 0


def test_the_latest_stored_session_is_fetched_again_and_overwritten(
        cfg, conn, monkeypatch):
    """Found 2026-10-07: a vendor correction never reached an append-only cache.

    The run starts at the OPEN of the session holding the newest stored bar —
    a whole session, never mid-bar — and passes `force` so `collect_ticker`
    does not skip what it already holds.
    """
    from src.collectors import market
    from src.live.monitor import fetch_latest
    from src.utils.timeutils import session_open_before

    calls = []

    def spy(cfg_, conn_, tickers, start_ts, end_ts, iv, resume=False, force=False):
        calls.append((tuple(tickers), start_ts, force))
        return 0

    monkeypatch.setattr(market, "collect_many", spy)
    newest = latest_stored_bar(conn, cfg["market"]["interval"])
    fetch_latest(cfg, conn, tickers=["AAA"], now_ts=newest + 10 * HOUR)

    n = cfg["live"]["restate_sessions"]
    assert calls[0] == (("AAA",), session_open_before(newest, n - 1), True)


def test_a_restate_reaching_into_the_frozen_snapshot_is_refused(cfg, conn):
    """The one-off repair may widen the restate, never past the stamp."""
    from src.live.monitor import fetch_latest

    with pytest.raises(SystemExit, match="frozen"):
        fetch_latest(cfg, conn, tickers=["AAA"],
                     restate_from=date_str_to_ts("2025-08-01"))


def test_fetching_without_a_snapshot_is_refused(cfg, tmp_path):
    """This appends to a snapshot; it does not create one."""
    from src.live.monitor import fetch_latest

    empty = db.get_conn(tmp_path / "empty.db")
    with pytest.raises(SystemExit, match="no bars stored"):
        fetch_latest(cfg, empty, tickers=["AAA"])


# --------------------------------------------------------------------------
# Two ways the live fetch went quiet without erroring (found 2026-09-07)
# --------------------------------------------------------------------------
def test_a_warm_fetch_state_does_not_silence_the_fetch(cfg, conn, monkeypatch):
    """`resume=True` would have skipped every ticker on the first warm-cache run.

    `fetch_state` rows are keyed on the ticker alone and carry no window, so
    once a run marked a ticker 'ok', a resuming run skipped it for ever. Every
    run so far began from the bootstrap, whose `fetch_state` is empty, which is
    the only reason this never fired.
    """
    from src.collectors import market
    from src.live.monitor import fetch_latest, latest_stored_bar

    interval = cfg["market"]["interval"]
    db.set_fetch_state(conn, f"market:{interval}", "AAA", "ok",
                       records=1, rows_written=1)

    seen: list[bool] = []

    def spy(cfg_, conn_, tickers, start_ts, end_ts, iv, resume=False, force=False):
        seen.append(resume)
        return 0

    monkeypatch.setattr(market, "collect_many", spy)
    fetch_latest(cfg, conn, tickers=["AAA"],
                 now_ts=latest_stored_bar(conn, interval) + 10 * HOUR)

    assert seen, "collect_many was never called"
    assert not any(seen), (
        "the live fetch must not resume: a warm fetch_state would skip every "
        "ticker and append nothing while reporting success")


def test_the_benchmark_is_fetched_even_though_it_is_not_in_the_universe(cfg, conn,
                                                                       monkeypatch):
    """Otherwise every ret_rel_* feature decays to NaN as the universe moves on.

    The benchmark is not `in_universe`, so it never appears in the ticker list,
    and it is fetched from ITS OWN newest bar — starting from the universe's
    would step over the gap that has already opened and never close it.
    """
    from src.collectors import market
    from src.live.monitor import fetch_latest, latest_stored_bar

    interval = cfg["market"]["interval"]
    benchmark = cfg["market"]["benchmark"]
    universe_newest = latest_stored_bar(conn, interval)

    # Make the benchmark lag the universe by ten bars, which is the production
    # state: it is not in_universe, so the live fetch never advanced it.
    bench_newest = universe_newest - 10 * HOUR
    conn.execute("DELETE FROM bars WHERE ticker = ? AND interval = ? AND ts_utc > ?",
                 (benchmark, interval, bench_newest))
    conn.commit()
    assert latest_stored_bar(conn, interval, ticker=benchmark) == bench_newest

    calls: list[tuple] = []

    def spy(cfg_, conn_, tickers, start_ts, end_ts, iv, resume=False, force=False):
        calls.append((tuple(tickers), start_ts))
        return 0

    monkeypatch.setattr(market, "collect_many", spy)
    fetch_latest(cfg, conn, tickers=["AAA"], now_ts=universe_newest + 10 * HOUR)

    bench_calls = [c for c in calls if c[0] == (benchmark,)]
    assert bench_calls, f"{benchmark} was never fetched; ret_rel_* would go NaN"
    from src.live.monitor import restate_start
    assert bench_calls[0][1] == restate_start(cfg, conn, bench_newest), (
        "the benchmark must resume from its own newest bar, or the gap between "
        "it and the universe is stepped over and never filled")


# --------------------------------------------------------------------------
# The learned policy in the live run
# --------------------------------------------------------------------------
def _with_policy(cfg, run, sha):
    return {**cfg, "live": {**cfg.get("live", {}),
                            "policies": [{"run": str(run), "threshold": 0.99,
                                          "sha256": sha}]}}


def test_the_configured_policy_is_frozen_in_git_and_matches_its_hash(cfg):
    """The committed file IS the pre-registration: if it and config disagree,
    the live RL series would be scored by something nobody registered."""
    from src.live.monitor import live_policies
    policies = live_policies(cfg)
    assert policies, "config names no live policy"
    for p in policies:
        assert p["run"].startswith("models/live/")
        assert p["name"].startswith("rl_policy[")


def test_a_changed_policy_file_fails_the_run(cfg, tmp_path):
    from src.live.monitor import live_policies
    (tmp_path / "p6-x-s1").mkdir()
    (tmp_path / "p6-x-s1" / "policy.zip").write_bytes(b"not the frozen model")
    with pytest.raises(SystemExit, match="has been changed"):
        live_policies(_with_policy(cfg, tmp_path / "p6-x-s1", "0" * 64))


def test_a_missing_policy_file_fails_the_run_not_silently(cfg, tmp_path):
    """A policy that dropped out would leave a gap that looks like a quiet
    market."""
    from src.live.monitor import live_policies
    with pytest.raises(SystemExit, match="must not drop out"):
        live_policies(_with_policy(cfg, tmp_path / "p6-x-s1", "0" * 64))


def test_the_policy_uses_its_budget_cut_not_its_own_half_rule(cfg):
    """P(FLAG) >= 0.5 flags almost every hour; the live cut must be the
    validation budget cut recorded in config."""
    from src.live.monitor import policy_name
    cuts = default_thresholds(cfg)
    for entry in cfg["live"]["policies"]:
        assert cuts[policy_name(entry["run"])] == float(entry["threshold"])
        assert cuts[policy_name(entry["run"])] > 0.9


# --------------------------------------------------------------------------
# The session volume check (2026-10-07)
# --------------------------------------------------------------------------
def _session_bars(conn, cfg, ticker, day, volumes):
    iv = cfg["market"]["interval"]
    open_ = date_str_to_ts(day) + 13 * HOUR + 1800                 # 13:30Z
    db.upsert_bars(conn, [(ticker, open_ + i * HOUR, 1, 1, 1, 1, float(v), iv)
                          for i, v in enumerate(volumes)])
    return open_


def test_a_session_far_above_its_daily_total_is_quarantined(cfg, tmp_path):
    """QQQ's 2026-10-06 hourly bars summed to ~8x its day; nothing scores them."""
    from src.live.monitor import check_session_volumes

    c = db.get_conn(tmp_path / "q.db")
    since = _session_bars(c, cfg, "QQQ", "2026-10-06", [22e6, 62e6, 61e6, 26e6])
    _session_bars(c, cfg, "AAA", "2026-10-06", [1e6, 1e6, 1e6, 1e6])
    daily = pd.DataFrame({"QQQ": [24e6], "AAA": [5e6]}, index=["2026-10-06"])

    out = check_session_volumes(cfg, c, since, now_ts=since + 10 * HOUR,
                                daily=daily)

    assert out["quarantined"] == 1 and out["rows"][0]["ticker"] == "QQQ"
    left = dict(c.execute("SELECT ticker, COUNT(*) FROM bars GROUP BY ticker"))
    assert "QQQ" not in left and left["AAA"] == 4
    row = c.execute("SELECT session_date, bars_removed FROM bar_quarantine").fetchone()
    assert tuple(row) == ("2026-10-06", 4)


def test_a_session_with_no_daily_bar_is_left_alone_and_counted(cfg, tmp_path):
    from src.live.monitor import check_session_volumes

    c = db.get_conn(tmp_path / "u.db")
    since = _session_bars(c, cfg, "AAA", "2026-10-06", [1e6, 1e6])
    _session_bars(c, cfg, "BBB", "2026-10-06", [1e6, 1e6])
    daily = pd.DataFrame({"BBB": [5e6]}, index=["2026-10-06"])
    out = check_session_volumes(cfg, c, since, now_ts=since + 10 * HOUR,
                                daily=daily)
    assert out == {"sessions": 2, "quarantined": 0, "unverified": 1, "rows": []}


def test_a_check_that_sees_no_daily_bars_at_all_fails_the_run(cfg, tmp_path):
    """AGENTS rule 8: a blind check must not report a clean bill of health."""
    from src.live.monitor import check_session_volumes

    c = db.get_conn(tmp_path / "z.db")
    since = _session_bars(c, cfg, "AAA", "2026-10-06", [1e6])
    with pytest.raises(SystemExit, match="ZERO daily bars"):
        check_session_volumes(cfg, c, since, now_ts=since + HOUR,
                              daily=pd.DataFrame())


def test_the_quarantine_record_survives_a_rebuilt_cache(cfg, tmp_path):
    """Merged with the committed file, never replaced by a shorter one."""
    from src.live.monitor import check_session_volumes, export_quarantine_csv

    path = tmp_path / "quarantine.csv"
    first = db.get_conn(tmp_path / "a.db")
    since = _session_bars(first, cfg, "QQQ", "2026-10-06", [9e6, 9e6])
    check_session_volumes(cfg, first, since, now_ts=since + 5 * HOUR,
                          daily=pd.DataFrame({"QQQ": [1e6]}, index=["2026-10-06"]))
    assert export_quarantine_csv(first, path) == 1

    rebuilt = db.get_conn(tmp_path / "b.db")                 # knows nothing
    assert export_quarantine_csv(rebuilt, path) == 1
