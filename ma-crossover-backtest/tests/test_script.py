from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest

from tests.conftest import yahoo_like_frame


def test_run_ma_crossover_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import quantbt.data as qd

    monkeypatch.setattr(
        qd, "yahoo_downloader", lambda symbol: yahoo_like_frame(n=800, start="2018-01-01")
    )
    monkeypatch.setattr(qd, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    out = tmp_path / "chart.png"
    argv = [
        "run_ma_crossover.py",
        "--start",
        "2019-06-03",
        "--end",
        "2020-12-31",
        "--short",
        "10",
        "--long",
        "40",
        "--out",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(Path("scripts/run_ma_crossover.py")), run_name="__main__")
    captured = capsys.readouterr().out
    assert "sharpe" in captured
    assert out.exists()
