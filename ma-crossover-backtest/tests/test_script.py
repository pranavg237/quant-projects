from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pytest


def test_run_ma_crossover_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """By default the script reads the committed snapshot and never touches the network."""
    import quantbt.data as qd

    def no_network(symbol: str) -> None:
        raise AssertionError("the script tried to download")

    monkeypatch.setattr(qd, "yahoo_downloader", no_network)
    monkeypatch.setattr(qd, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv(qd.LIVE_DATA_ENV, raising=False)
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
