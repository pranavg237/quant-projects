from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbt.data import (
    DataError,
    PriceData,
    dividend_adjustment_factor,
    load_csv,
    load_yahoo,
    normalize_index,
    synthetic_prices,
)
from tests.conftest import yahoo_like_frame


def test_normalize_index_rejects_tz_aware() -> None:
    idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    with pytest.raises(DataError, match="tz-aware"):
        normalize_index(idx)


def test_normalize_index_rejects_duplicates_and_unsorted() -> None:
    with pytest.raises(DataError, match="duplicate"):
        normalize_index(pd.DatetimeIndex(["2020-01-01", "2020-01-01"]))
    with pytest.raises(DataError, match="sorted"):
        normalize_index(pd.DatetimeIndex(["2020-01-02", "2020-01-01"]))
    with pytest.raises(DataError, match="datetime"):
        normalize_index(pd.Index(["not", "dates"]))


def test_normalize_index_strips_time_of_day() -> None:
    idx = pd.DatetimeIndex(["2020-01-01 09:30", "2020-01-02 16:00"])
    out = normalize_index(idx)
    assert list(out) == list(pd.DatetimeIndex(["2020-01-01", "2020-01-02"]))
    assert out.name == "date"


def test_from_frames_adjustment_matches_yahoo_adj_close() -> None:
    raw = yahoo_like_frame(dividends={100: 0.5, 200: 0.6}, splits={150: 2.0})
    data = PriceData.from_frames({"X": raw})
    # Adj Close taken directly
    np.testing.assert_allclose(data.close["X"].to_numpy(), raw["Adj Close"].to_numpy())
    # OHL scaled by the same factor as close
    factor = raw["Adj Close"] / raw["Close"]
    np.testing.assert_allclose(data.open["X"].to_numpy(), (raw["Open"] * factor).to_numpy())
    # split_adj_close is Yahoo's Close
    np.testing.assert_allclose(data.split_adj_close["X"].to_numpy(), raw["Close"].to_numpy())
    # raw_close undoes the 2:1 split for all bars strictly before the split day
    ratio = data.raw_close["X"] / data.split_adj_close["X"]
    assert np.allclose(ratio.iloc[:150], 2.0)
    assert np.allclose(ratio.iloc[150:], 1.0)


def test_dividend_adjustment_factor_reproduces_adj_close_when_missing() -> None:
    raw = yahoo_like_frame(dividends={50: 1.0, 120: 1.5})
    expected = raw["Adj Close"].to_numpy()
    without = raw.drop(columns=["Adj Close"])
    data = PriceData.from_frames({"X": without})
    np.testing.assert_allclose(data.close["X"].to_numpy(), expected, rtol=1e-12)
    f = dividend_adjustment_factor(raw["Close"], raw["Dividends"])
    assert f.iloc[-1] == 1.0
    assert (f <= 1.0).all()


def test_no_actions_means_all_series_agree() -> None:
    raw = yahoo_like_frame().drop(columns=["Adj Close", "Dividends", "Stock Splits"])
    data = PriceData.from_frames({"X": raw})
    np.testing.assert_allclose(data.close["X"], data.raw_close["X"])
    np.testing.assert_allclose(data.close["X"], data.split_adj_close["X"])
    assert (data.dividends["X"] == 0).all()


def test_scale_invariant_signal_is_immune_to_future_dividends() -> None:
    """Back-adjusting for a dividend paid after t scales every price before t by one constant,
    so an MA-ratio signal computed live equals the one computed on the adjusted series."""
    from quantbt.vectorized import ma_crossover_signal

    raw = yahoo_like_frame(n=400, dividends={350: 2.0})
    live = raw["Close"].iloc[:300]  # what a trader saw before the dividend existed
    adjusted = raw["Adj Close"].iloc[:300]  # same window, as seen from the future
    assert not np.allclose(live, adjusted)  # the series differ ...
    s_live = ma_crossover_signal(live, 10, 30)
    s_adj = ma_crossover_signal(adjusted, 10, 30)
    assert s_live.equals(s_adj)  # ... but the signal does not


def test_multi_symbol_outer_join_leaves_nan_before_listing() -> None:
    a = yahoo_like_frame(n=100, start="2020-01-01")
    b = yahoo_like_frame(n=50, start="2020-03-01", seed=2)
    data = PriceData.from_frames({"A": a, "B": b})
    assert data.symbols == ["A", "B"]
    assert data.close["B"].iloc[0] != data.close["B"].iloc[0]  # NaN
    assert data.close["A"].notna().all()
    assert data.dividends["B"].iloc[0] == 0.0
    single = data.select(["B"])
    assert single.symbols == ["B"]
    with pytest.raises(KeyError):
        data.select(["C"])
    with pytest.raises(DataError):
        data.single()


def test_validation_errors() -> None:
    raw = yahoo_like_frame(n=20)
    with pytest.raises(DataError, match="missing columns"):
        PriceData.from_frames({"X": raw.drop(columns=["Open"])})
    bad = raw.copy()
    bad.loc[bad.index[3], "Low"] = 1e9
    with pytest.raises(DataError, match="high < low"):
        PriceData.from_frames({"X": bad})
    neg = raw.copy()
    neg.loc[neg.index[3], "Volume"] = -1
    with pytest.raises(DataError, match="negative"):
        PriceData.from_frames({"X": neg})
    with pytest.raises(DataError, match="no symbols"):
        PriceData.from_frames({})
    data = PriceData.from_frames({"X": raw})
    with pytest.raises(KeyError):
        data.field("nope")
    assert len(data) == 20
    assert data.index.name == "date"


def test_slice_is_inclusive_and_roundtrip() -> None:
    data = PriceData.from_frames({"X": yahoo_like_frame(n=30)})
    sub = data.slice("2020-01-06", "2020-01-10")
    assert len(sub) == 5
    frames = data.to_frames()
    again = PriceData.from_frames(frames)
    np.testing.assert_allclose(again.close["X"], data.close["X"])
    np.testing.assert_allclose(again.raw_close["X"], data.raw_close["X"])


def test_load_yahoo_uses_cache_and_manifest(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(symbol: str) -> pd.DataFrame:
        calls.append(symbol)
        return yahoo_like_frame(n=120)

    d1 = load_yahoo(
        ["X", "Y"], start="2020-02-01", end="2020-03-01", cache_dir=tmp_path, downloader=fake
    )
    assert calls == ["X", "Y"]
    assert d1.symbols == ["X", "Y"]
    assert d1.index[0] >= pd.Timestamp("2020-02-01")
    assert d1.index[-1] <= pd.Timestamp("2020-03-01")
    manifest = (tmp_path / "MANIFEST.json").read_text()
    assert '"X"' in manifest and "downloaded_at" in manifest

    d2 = load_yahoo("X", cache_dir=tmp_path, downloader=fake)  # cached, no call
    assert calls == ["X", "Y"]
    assert len(d2) == 120

    load_yahoo("X", cache_dir=tmp_path, downloader=fake, refresh=True)
    assert calls == ["X", "Y", "X"]

    # asking for an end date beyond the cache re-downloads with a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_yahoo("Y", end="2030-01-01", cache_dir=tmp_path, downloader=fake)
    assert any("re-downloading" in str(x.message) for x in w)
    assert calls == ["X", "Y", "X", "Y"]

    with pytest.raises(DataError):
        load_yahoo([], cache_dir=tmp_path, downloader=fake)


def test_yahoo_downloader_flattens_and_localizes(monkeypatch: pytest.MonkeyPatch) -> None:
    import types

    raw = yahoo_like_frame(n=10)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize("America/New_York")
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["X"]])

    fake_yf = types.SimpleNamespace(download=lambda *a, **k: raw)
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_yf)
    from quantbt.data import yahoo_downloader

    out = yahoo_downloader("X")
    assert pd.DatetimeIndex(out.index).tz is None
    assert "Adj Close" in out.columns and not isinstance(out.columns, pd.MultiIndex)

    fake_empty = types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame())
    monkeypatch.setitem(__import__("sys").modules, "yfinance", fake_empty)
    with pytest.raises(DataError, match="no data"):
        yahoo_downloader("X")


def test_load_csv(tmp_path: Path) -> None:
    raw = yahoo_like_frame(n=15)
    p = tmp_path / "ABC.csv"
    raw.to_csv(p)
    data = load_csv(p)
    assert data.symbols == ["ABC"]
    assert load_csv(p, symbol="Z").symbols == ["Z"]


def test_synthetic_prices_fields_consistent() -> None:
    data = synthetic_prices(50, symbols=("A", "B"), seed=3, gap_vol=0.01)
    assert data.symbols == ["A", "B"]
    assert (data.high >= data.low).all().all()
    assert (data.high >= data.close).all().all()
    assert (data.low <= data.open).all().all()
