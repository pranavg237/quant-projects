"""Data handling: partial months, risk-free timing, return construction and caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from portopt import data


def _daily(start: str, end: str) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    return pd.DataFrame({"A": np.linspace(100, 110, len(index))}, index=index)


def test_partial_final_month_is_dropped() -> None:
    prices = _daily("2026-07-01", "2026-09-18")
    trimmed = data.drop_partial_last_month(prices)
    assert trimmed.index[-1] == pd.Timestamp("2026-08-31")


def test_complete_final_month_is_kept() -> None:
    prices = _daily("2026-07-01", "2026-08-31")
    assert data.drop_partial_last_month(prices).equals(prices)


def test_month_ending_on_a_weekend_counts_as_complete() -> None:
    prices = _daily("2026-01-01", "2026-05-29")  # 2026-05-31 is a Sunday
    assert data.drop_partial_last_month(prices).index[-1] == pd.Timestamp("2026-05-29")


def test_data_ending_on_the_first_drops_only_that_month() -> None:
    prices = _daily("2026-06-01", "2026-09-01")
    assert data.drop_partial_last_month(prices).index[-1] == pd.Timestamp("2026-08-31")


def test_risk_free_uses_the_previous_month_end_yield() -> None:
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    yields = pd.Series(
        np.where(index.month == 1, 3.0, np.where(index.month == 2, 6.0, 9.0)), index=index
    )
    rf = data.tbill_yield_to_monthly_rf(yields)
    # February's return is set by January's closing yield, so no future rate leaks in.
    assert rf.loc["2024-02-29"] == pytest.approx(0.03 / 12)
    assert rf.loc["2024-03-31"] == pytest.approx(0.06 / 12)
    assert pd.Timestamp("2024-01-31") not in rf.index


def test_monthly_returns_are_simple_returns_of_month_end_prices() -> None:
    index = pd.to_datetime(["2024-01-15", "2024-01-31", "2024-02-15", "2024-02-29"])
    prices = pd.DataFrame({"A": [100.0, 110.0, 50.0, 121.0]}, index=index)
    returns = data.to_returns(prices, "monthly")
    assert returns["A"].iloc[0] == pytest.approx(0.10)


def test_unknown_frequency_raises() -> None:
    with pytest.raises(ValueError, match="unknown frequency"):
        data.to_returns(_daily("2024-01-01", "2024-02-01"), "hourly")


def test_load_prices_reads_a_snapshot_without_network(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "prices_2026-01-01.csv"
    snapshot.parent.mkdir()
    frame = pd.DataFrame(
        {"X": [1.0, 2.0], "Y": [3.0, 4.0]}, index=pd.to_datetime(["2025-12-30", "2025-12-31"])
    )
    frame.to_csv(snapshot)
    history = data.load_prices(("X", "Y"), cache_dir=tmp_path)
    assert history.source == "cache"
    assert history.tickers == ["X", "Y"]


def test_risk_free_and_benchmark_read_snapshots_without_network(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    pd.Series(4.8, index=index, name="^IRX").to_csv(snapshots / "tbill_13w_2024-04-01.csv")
    pd.Series(np.arange(len(index)) + 100.0, index=index, name="SPY").to_csv(
        snapshots / "benchmark_SPY_2024-04-01.csv"
    )
    rf = data.load_risk_free(cache_dir=tmp_path)
    assert rf.iloc[-1] == pytest.approx(0.048 / 12)
    spy = data.load_benchmark("SPY", cache_dir=tmp_path)
    assert spy.name == "SPY"
    assert len(spy) == len(index)


def test_download_starts_the_universe_when_its_last_member_lists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fake yfinance: "NEW" only has prices from the third day. The panel must start there
    # rather than silently growing as assets appear.
    import sys
    import types

    index = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame(
        {"OLD": [1.0, 2.0, 3.0, 4.0, 5.0], "NEW": [np.nan, np.nan, 7.0, 8.0, 9.0]}, index=index
    )
    raw = pd.concat({"Close": close}, axis=1)
    fake = types.SimpleNamespace(download=lambda *args, **kwargs: raw)
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    history = data.download_prices(("OLD", "NEW"), cache_dir=tmp_path)
    assert history.start == index[2]
    assert history.binding_ticker == "NEW"
    assert not history.prices.isna().any().any()
    assert list((tmp_path / "raw").glob("prices_*.csv"))


def test_download_failure_falls_back_to_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    import types

    snapshot = tmp_path / "snapshots" / "prices_2026-01-01.csv"
    snapshot.parent.mkdir()
    pd.DataFrame({"X": [1.0, 2.0]}, index=pd.to_datetime(["2025-12-30", "2025-12-31"])).to_csv(
        snapshot
    )

    def offline(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ConnectionError("offline")

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=offline))
    with pytest.warns(UserWarning, match="download failed"):
        history = data.load_prices(("X",), cache_dir=tmp_path, force_refresh=True)
    assert history.tickers == ["X"]
