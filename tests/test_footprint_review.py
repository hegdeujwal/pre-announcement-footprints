"""The holdout opens once, with a recorded reason."""

import json

import pytest

pytest.importorskip("torch")

from src.baselines.footprint_review import OPENED, open_holdout  # noqa: E402


def test_opening_needs_a_reason(tmp_path):
    with pytest.raises(SystemExit, match="--reason"):
        open_holdout(tmp_path, None, reopen=False)
    assert not (tmp_path / OPENED).exists()


def test_a_second_opening_is_refused_without_reopen(tmp_path):
    open_holdout(tmp_path, "final FootprintNet number", reopen=False)
    with pytest.raises(SystemExit, match="already opened"):
        open_holdout(tmp_path, "peek again", reopen=False)


def test_reopening_keeps_the_whole_history(tmp_path):
    open_holdout(tmp_path, "first", reopen=False)
    open_holdout(tmp_path, "second, because X", reopen=True)
    rec = json.loads((tmp_path / OPENED).read_text())
    assert [h["reason"] for h in rec["history"]] == ["first", "second, because X"]
    assert rec["reason"] == "second, because X"


def test_windows_reaching_before_the_holdout_are_dropped_whole():
    import pandas as pd

    from src.baselines.footprint_review import inside_holdout
    f = pd.DataFrame({"window_id": ["E1", "E1", "E2", "E2", "bar:x"],
                      "ts_utc": [90, 110, 120, 130, 140],
                      "t0_utc": [200, 200, 300, 300, None]})
    out = inside_holdout(f, lo=100)
    assert out["window_id"].tolist() == ["E2", "E2", "bar:x"]
