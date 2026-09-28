"""The committed data snapshot: complete, hash-checked, and read by default without a network.

Every number in README.md and RESULTS.md is produced from data/snapshot-2026-09-28/, so
these tests pin down that it is intact and that loading it cannot silently fall back to
a download or return a shorter series than asked for.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantbt.data as qd
from quantbt.data import (
    SNAPSHOT_DIR,
    DataError,
    load_snapshot,
    load_yahoo,
    sha256_file,
    snapshot_manifest,
)
from quantbt.factors import load_factors
from quantbt.factors.french import FACTOR_FILES
from quantbt.portfolio import Portfolio
from quantbt.research.universes import ALL_SYMBOLS
from quantbt.strategy import Context


@pytest.fixture(autouse=True)
def _snapshot_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(qd.LIVE_DATA_ENV, raising=False)
    monkeypatch.setitem(qd._LIVE, "enabled", False)


def test_manifest_covers_every_input_and_every_hash_matches() -> None:
    manifest = snapshot_manifest()
    files = manifest["files"]
    assert manifest["download_date"] == "2026-09-28"
    assert set(manifest["symbols"]) == set(ALL_SYMBOLS)
    for symbol in ALL_SYMBOLS:
        assert f"prices/{symbol}.csv.gz" in files
    for (_, frequency), name in FACTOR_FILES.items():
        if frequency == "daily":
            assert f"french/{name}_CSV.zip" in files
    on_disk = {str(p.relative_to(SNAPSHOT_DIR)) for p in SNAPSHOT_DIR.rglob("*") if p.is_file()}
    assert on_disk == set(files) | {"MANIFEST.json"}
    for rel, meta in files.items():
        assert sha256_file(SNAPSHOT_DIR / rel) == meta["sha256"], rel


def test_default_load_reads_the_snapshot_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_network(symbol: str) -> pd.DataFrame:
        raise AssertionError("tried to download")

    monkeypatch.setattr(qd, "yahoo_downloader", no_network)
    data = load_yahoo("SPY", start="2005-01-03", end="2005-12-30")
    assert data.index[0] == pd.Timestamp("2005-01-03")
    assert data.index[-1] == pd.Timestamp("2005-12-30")
    assert len(data) == 252
    assert np.isfinite(data.close["SPY"]).all() and np.isfinite(data.open["SPY"]).all()
    assert data.high["SPY"].isna().all()  # not stored; see data/README.md
    same = load_snapshot(["SPY"], "2005-01-03", "2005-12-30")
    pd.testing.assert_frame_equal(same.close, data.close)
    # the French risk-free rate comes from the snapshot too
    rf = load_factors("ff3", "daily", start="2005-01-03", end="2005-12-30")["RF"]
    assert len(rf) == 252 and rf.between(0, 0.001).all()


def test_snapshot_refuses_what_it_does_not_hold() -> None:
    with pytest.raises(DataError, match="not in the snapshot"):
        load_yahoo("NOT_A_SYMBOL", end="2025-08-29")
    with pytest.raises(DataError, match="snapshot ends"):
        load_yahoo("SPY", end="2030-01-01")
    with pytest.raises(DataError, match="snapshot starts"):
        load_yahoo("AAPL", start="2002-12-31", end="2025-08-29")  # trimmed to 2003
    # a start on the trim date itself (a holiday here) is complete, so it is allowed
    assert load_yahoo("AAPL", start="2003-01-01", end="2003-01-31").index[0] == pd.Timestamp(
        "2003-01-02"
    )
    assert load_yahoo("SPY", start="1998-01-01", end="1998-01-31").index[0] == pd.Timestamp(
        "1998-01-02"
    )
    with pytest.raises(DataError, match="not in the snapshot"):
        load_factors("ff3", "monthly")  # only the daily files are frozen


def test_a_tampered_snapshot_file_is_rejected(tmp_path: Path) -> None:
    rel = "prices/SPY.csv.gz"
    (tmp_path / "prices").mkdir()
    shutil.copyfile(SNAPSHOT_DIR / rel, tmp_path / rel)
    meta = dict(snapshot_manifest()["files"][rel])
    (tmp_path / "MANIFEST.json").write_text(json.dumps({"files": {rel: meta}}))
    assert load_snapshot(["SPY"], "2020-01-02", "2020-01-31", snapshot_dir=tmp_path).symbols == [
        "SPY"
    ]
    (tmp_path / rel).write_bytes((tmp_path / rel).read_bytes() + b"\0")
    with pytest.raises(DataError, match="sha256"):
        load_snapshot(["SPY"], snapshot_dir=tmp_path)
    (tmp_path / rel).unlink()
    with pytest.raises(DataError, match="missing"):
        load_snapshot(["SPY"], snapshot_dir=tmp_path)
    with pytest.raises(DataError, match="no snapshot manifest"):
        load_snapshot(["SPY"], snapshot_dir=tmp_path / "nowhere")


def test_strategies_cannot_read_high_low_that_the_snapshot_omits() -> None:
    data = load_snapshot(["SPY"], "2020-01-02", "2020-03-31")
    ctx = Context(
        data=data, portfolio=Portfolio(cash=1.0), now=data.index[30], position_index=30,
        symbols=["SPY"],
    )  # fmt: skip
    assert len(ctx.history("close", 5)) == 5
    with pytest.raises(DataError, match="high/low"):
        ctx.history("high", 5)
