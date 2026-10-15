"""scripts/show_guide_points.py — the viva demo must keep running.

It reads only committed tables, so it is checked on every run: a renamed
result column or detector would otherwise surface in front of the examiner.
"""
import importlib.util
import re
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "show_guide_points.py"
spec = importlib.util.spec_from_file_location("show_guide_points", SCRIPT)
show = importlib.util.module_from_spec(spec)
spec.loader.exec_module(show)


@pytest.mark.parametrize("point", sorted(show.POINTS, key=int))
def test_every_point_prints(point, capsys):
    show.POINTS[point]()
    out = capsys.readouterr().out
    assert f"POINT {point} " in out and "Test:" in out
    assert not re.search(r"\bnan\b", out, re.I)       # a missing number


def test_the_candle_demo_shows_each_named_pattern(capsys):
    show.p5()
    out = capsys.readouterr().out
    assert "a doji, a hammer, a marubozu" in out
