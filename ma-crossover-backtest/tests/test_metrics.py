from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics as m


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_periods_per_year_inference() -> None:
    daily = pd.bdate_range("2020-01-01", periods=50)
    assert m.periods_per_year(daily) == 252
    assert m.periods_per_year(pd.date_range("2020-01-01", periods=30, freq="W")) == 52
    assert m.periods_per_year(pd.date_range("2020-01-01", periods=30, freq="ME")) == 12
    assert m.periods_per_year(pd.date_range("2020-01-01", periods=10, freq="QE")) == 4
    assert m.periods_per_year(pd.date_range("2020-01-01", periods=5, freq="YE")) == 1
    assert m.periods_per_year(pd.DatetimeIndex(["2020-01-01"])) == 252


def test_cagr_of_constant_growth() -> None:
    # 1% per bar for 252 bars ~ one calendar year
    r = pd.Series(0.01, index=pd.bdate_range("2020-01-01", periods=253))
    eq = m.equity_curve(r)
    years = m.years_spanned(eq.index)
    # curve-based: growth between the first and last point of the curve
    expected = (1.01**252) ** (1 / years) - 1
    assert m.cagr(eq) == pytest.approx(expected)
    assert m.total_return(eq) == pytest.approx(1.01**252 - 1)
    assert m.annualized_return(r) == pytest.approx(1.01**252 - 1)
    # returns-based summary counts every period, including the first
    s = m.summary(r)
    assert s["total_return"] == pytest.approx(1.01**253 - 1)
    assert s["cagr"] == pytest.approx((1.01**253) ** (1 / (years + 1 / 252)) - 1)


def test_vol_and_sharpe_known_values() -> None:
    rng = np.random.default_rng(0)
    r = _series(list(rng.normal(0.001, 0.01, 5000)))
    assert m.annualized_vol(r) == pytest.approx(r.std() * np.sqrt(252))
    assert m.sharpe(r) == pytest.approx(r.mean() / r.std() * np.sqrt(252))
    # constant rf lowers Sharpe
    assert m.sharpe(r, rf=0.05) < m.sharpe(r)
    # rf as an aligned series
    rf = pd.Series(0.0, index=r.index)
    assert m.sharpe(r, rf=rf) == pytest.approx(m.sharpe(r))
    with pytest.raises(ValueError):
        m.sharpe(r, rf=rf.iloc[:10])
    assert np.isnan(m.sharpe(_series([0.0, 0.0, 0.0])))
    assert np.isnan(m.sharpe(_series([0.01])))
    assert np.isnan(m.annualized_vol(_series([0.01])))
    assert np.isnan(m.annualized_return(_series([])))


def test_sortino() -> None:
    r = _series([0.02, -0.01, 0.03, -0.02, 0.01])
    excess = r.to_numpy()
    dd = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    assert m.sortino(r) == pytest.approx(excess.mean() / dd * np.sqrt(252))
    assert m.sortino(_series([0.01, 0.02])) == float("inf")
    assert np.isnan(m.sortino(_series([0.0, 0.0])))
    assert np.isnan(m.sortino(_series([0.1])))


def test_drawdown_depth_and_duration() -> None:
    # peak at 120 (bar 2), trough 60 (bar 4), recovery to 121 at bar 8, then a second shallower dip
    eq = _series([100, 110, 120, 90, 60, 80, 100, 110, 121, 121, 110, 121])
    assert m.max_drawdown(eq) == pytest.approx(-0.5)
    info = m.max_drawdown_info(eq)
    assert info.depth == pytest.approx(-0.5)
    assert info.peak == eq.index[2]
    assert info.trough == eq.index[4]
    assert info.recovery == eq.index[8]
    assert info.duration_bars == 6
    assert info.recovered
    assert m.calmar(eq) == pytest.approx(m.cagr(eq) / 0.5)
    dd = m.drawdown_series(eq)
    assert dd.iloc[0] == 0 and dd.iloc[4] == pytest.approx(-0.5)


def test_drawdown_unrecovered_and_flat() -> None:
    eq = _series([100, 120, 90, 80, 85])
    info = m.max_drawdown_info(eq)
    assert not info.recovered and info.recovery is None
    assert info.duration_bars == 3
    flat = _series([100, 100, 100])
    info = m.max_drawdown_info(flat)
    assert info.depth == 0 and info.duration_bars == 0 and info.recovered
    assert np.isnan(m.calmar(flat))
    assert np.isnan(m.cagr(_series([100])))


def test_turnover_exposure_time_in_market() -> None:
    idx = pd.bdate_range("2020-01-01", periods=252)
    w = pd.Series(0.0, index=idx)
    w.iloc[10:110] = 1.0  # in for 100 bars: one buy and one sell = one round trip
    assert m.turnover(w) == pytest.approx(1.0, rel=0.02)
    assert m.exposure(w) == pytest.approx(100 / 252)
    assert m.time_in_market(w) == pytest.approx(100 / 252)
    df = pd.DataFrame({"A": w, "B": -w})
    assert m.exposure(df) == pytest.approx(200 / 252)
    assert m.turnover(df) == pytest.approx(2.0, rel=0.02)


def test_hit_rate_and_profit_factor() -> None:
    assert m.hit_rate([1, -1, 2, np.nan]) == pytest.approx(2 / 3)
    assert np.isnan(m.hit_rate([]))
    assert m.profit_factor([2, -1]) == 2.0
    assert m.profit_factor([1, 2]) == float("inf")
    assert np.isnan(m.profit_factor([0.0]))


def test_summary_and_format() -> None:
    rng = np.random.default_rng(1)
    r = _series(list(rng.normal(0.0005, 0.01, 500)))
    b = _series(list(rng.normal(0.0004, 0.01, 500)))
    w = pd.Series(1.0, index=r.index)
    s = m.summary(r, rf=0.01, weights=w, trade_pnls=[0.1, -0.05], benchmark=b)
    for key in (
        "cagr",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "turnover",
        "hit_rate",
        "exposure",
        "information_ratio",
        "benchmark_sharpe",
    ):
        assert key in s.index
    assert s["trades"] == 2
    text = m.format_summary(s)
    assert "sharpe" in text and "%" in text
    s2 = m.summary(r, benchmark=r)  # zero tracking error -> nan IR
    assert np.isnan(s2["information_ratio"])
