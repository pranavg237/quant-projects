import sys
import types

import numpy as np
import pandas as pd
import pytest

from ffmodel.assets import build_portfolio, download_returns, load_returns_csv, prices_to_returns


def daily_prices(start, end, columns=("A",)):
    index = pd.bdate_range(start, end)
    return pd.DataFrame({c: np.linspace(100, 120, len(index)) for c in columns}, index=index)


def test_monthly_returns_use_month_end_prices():
    index = pd.to_datetime(["2024-01-10", "2024-01-31", "2024-02-15", "2024-02-29"])
    prices = pd.DataFrame({"A": [90.0, 100.0, 50.0, 110.0]}, index=index)
    returns = prices_to_returns(prices)
    assert list(returns.index) == [pd.Timestamp("2024-02-29")]
    assert returns["A"].iloc[0] == pytest.approx(0.10)


def test_trailing_partial_month_is_dropped():
    returns = prices_to_returns(daily_prices("2024-01-01", "2024-04-12"))
    assert returns.index[-1] == pd.Timestamp("2024-03-31")


def test_month_ending_on_a_weekend_is_kept():
    returns = prices_to_returns(daily_prices("2024-01-01", "2024-03-29"))  # 31 Mar is a Sunday
    assert returns.index[-1] == pd.Timestamp("2024-03-31")


def test_daily_returns_and_bad_frequency():
    prices = daily_prices("2024-01-01", "2024-01-10")
    assert len(prices_to_returns(prices, "daily")) == len(prices) - 1
    with pytest.raises(ValueError):
        prices_to_returns(prices, "weekly")


def test_csv_percent_returns_are_moved_to_month_end(tmp_path):
    path = tmp_path / "fund.csv"
    path.write_text("date,FUND\n2024-01-01,1.5\n2024-02-01,-2.0\n")
    frame = load_returns_csv(str(path), percent=True)
    assert list(frame.index) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert frame["FUND"].tolist() == pytest.approx([0.015, -0.02])


def test_csv_prices_become_returns(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("date,X\n2024-01-31,100\n2024-02-29,110\n2024-03-29,99\n")
    frame = load_returns_csv(str(path), prices=True)
    assert frame["X"].tolist() == pytest.approx([0.10, -0.10])


def test_daily_data_read_as_monthly_is_rejected(tmp_path):
    path = tmp_path / "daily.csv"
    path.write_text("date,X\n2024-01-02,0.01\n2024-01-03,0.02\n")
    with pytest.raises(ValueError, match="several rows per month"):
        load_returns_csv(str(path))


def test_portfolio_is_the_weighted_sum_each_period():
    index = pd.date_range("2024-01-31", periods=3, freq="ME")
    returns = pd.DataFrame({"A": [0.1, 0.0, -0.1], "B": [0.0, 0.2, np.nan]}, index=index)
    portfolio = build_portfolio(returns, {"A": 0.6, "B": 0.4}, "60/40")
    assert portfolio.name == "60/40"
    assert portfolio.iloc[:2].tolist() == pytest.approx([0.06, 0.08])
    assert np.isnan(portfolio.iloc[2])  # a missing holding makes the period unknown, not zero


def test_portfolio_with_unknown_holding_raises():
    returns = pd.DataFrame({"A": [0.1]}, index=pd.date_range("2024-01-31", periods=1, freq="ME"))
    with pytest.raises(KeyError):
        build_portfolio(returns, {"A": 0.5, "Z": 0.5})


def test_download_pads_the_start_and_drops_empty_tickers(monkeypatch):
    calls = {}
    index = pd.bdate_range("2023-12-01", "2024-03-29")
    close = pd.DataFrame({"AAA": np.linspace(10, 13, len(index)), "BBB": np.nan}, index=index)
    raw = pd.concat({"Close": close}, axis=1)

    def fake_download(tickers, start=None, **kwargs):
        calls["start"] = start
        return raw

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    with pytest.warns(UserWarning, match="no price data"):
        returns = download_returns(["aaa", "bbb"], start="2024-01")
    # One month of padding so January has a base price, and the output starts at `start`.
    assert calls["start"] == "2023-12-01"
    assert list(returns.columns) == ["AAA"]
    assert returns.index[0] == pd.Timestamp("2024-01-31")
