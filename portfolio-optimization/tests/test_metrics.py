"""Performance metrics: excess-return Sharpe, drawdown and the paired Sharpe test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt import backtest as bt
from portopt import metrics as mx


def _result(returns: pd.Series) -> bt.BacktestResult:
    return bt.buy_and_hold(returns, "r")


@pytest.fixture
def monthly(returns: pd.DataFrame) -> pd.Series:
    return returns["A3"]


def test_sharpe_is_on_excess_returns(monthly: pd.Series) -> None:
    rf = pd.Series(0.002, index=monthly.index)
    expected = (monthly - rf).mean() / (monthly - rf).std(ddof=1) * np.sqrt(12)
    assert mx.evaluate(_result(monthly), rf).sharpe == pytest.approx(expected)


def test_positive_rates_lower_sharpe(monthly: pd.Series) -> None:
    # The bug this project had: Sharpe computed with rf = 0 overstates it whenever rates
    # are positive.
    assert mx.evaluate(_result(monthly), 0.03).sharpe < mx.evaluate(_result(monthly), 0.0).sharpe


def test_annual_rate_and_equivalent_series_agree(monthly: pd.Series) -> None:
    annual = 0.04
    series = pd.Series((1 + annual) ** (1 / 12) - 1, index=monthly.index)
    a = mx.evaluate(_result(monthly), annual)
    b = mx.evaluate(_result(monthly), series)
    assert a.sharpe == pytest.approx(b.sharpe)
    assert a.sortino == pytest.approx(b.sortino)


def test_risk_free_series_must_cover_every_date(monthly: pd.Series) -> None:
    short = pd.Series(0.001, index=monthly.index[:-1])
    with pytest.raises(ValueError, match="missing 1 dates"):
        mx.evaluate(_result(monthly), short)


def test_max_drawdown_on_a_known_path() -> None:
    index = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.Series([0.10, -0.50, 0.20, 0.10], index=index)
    depth, peak, trough = mx.max_drawdown(returns)
    assert depth == pytest.approx(-0.50)
    assert peak == index[0]
    assert trough == index[1]


def test_annual_return_is_geometric() -> None:
    index = pd.date_range("2020-01-31", periods=24, freq="ME")
    returns = pd.Series([0.10, -0.0909090909] * 12, index=index)  # round trips to 1.0
    assert mx.evaluate(_result(returns)).annual_return == pytest.approx(0.0, abs=1e-9)


def test_sharpe_difference_of_a_series_with_itself(monthly: pd.Series) -> None:
    result = mx.sharpe_difference_test(monthly, monthly)
    assert result.difference == 0.0
    assert result.p_value == pytest.approx(1.0)


def test_sharpe_difference_is_antisymmetric(returns: pd.DataFrame) -> None:
    ab = mx.sharpe_difference_test(returns["A0"], returns["A5"], 0.01)
    ba = mx.sharpe_difference_test(returns["A5"], returns["A0"], 0.01)
    assert ab.z == pytest.approx(-ba.z)
    assert ab.p_value == pytest.approx(ba.p_value)


def test_sharpe_difference_detects_a_real_edge() -> None:
    rng = np.random.default_rng(11)
    index = pd.date_range("2000-01-31", periods=600, freq="ME")
    noise = rng.normal(0, 0.04, size=600)
    good = pd.Series(0.015 + noise, index=index)
    bad = pd.Series(-0.005 + noise + rng.normal(0, 0.01, size=600), index=index)
    assert mx.sharpe_difference_test(good, bad).p_value < 1e-6


def test_sharpe_difference_is_calibrated_under_the_null() -> None:
    # Two independent strategies with the same true Sharpe: about 5% of tests should reject.
    rng = np.random.default_rng(5)
    index = pd.date_range("2000-01-31", periods=200, freq="ME")
    p_values = [
        mx.sharpe_difference_test(
            pd.Series(rng.normal(0.005, 0.04, 200), index=index),
            pd.Series(rng.normal(0.005, 0.04, 200), index=index),
        ).p_value
        for _ in range(400)
    ]
    assert 0.02 < np.mean(np.array(p_values) < 0.05) < 0.09


def test_sharpe_interval_contains_the_estimate(monthly: pd.Series) -> None:
    lower, upper = mx.sharpe_confidence_interval(monthly, 12.0)
    point = monthly.mean() / monthly.std(ddof=1) * np.sqrt(12)
    assert lower < point < upper


def test_trading_cost_table_reconciles_with_the_cost_rate(returns: pd.DataFrame) -> None:
    from portopt import strategies as st

    builders = {
        "1/N": bt.equal_weight_builder,
        "max-Sharpe": st.make_max_sharpe(12.0, st.sample_estimator),
    }
    runs = bt.compare_strategies(returns, builders, lookback=36, cost_bps=25.0)
    table = mx.trading_cost_table(runs, 0.0).set_index("strategy")
    for name, run in runs.items():
        row = table.loc[name]
        years = len(run.returns) / 12.0
        # Linear costs: drag in bp/yr is exactly the one-way rate times annual turnover.
        assert row["cost_drag_bps"] == pytest.approx(25.0 * row["annual_turnover"])
        assert row["annual_turnover"] == pytest.approx(run.turnover.sum() / years)
        assert row["mean_turnover"] == pytest.approx(run.turnover.iloc[1:].mean())
        assert row["borrow_drag_bps"] == 0.0
        assert row["gross_sharpe"] >= row["net_sharpe"]
        assert row["sharpe_lost"] == pytest.approx(row["gross_sharpe"] - row["net_sharpe"])
    # The optimiser trades more than 1/N, so it loses more to costs.
    assert table.loc["max-Sharpe", "cost_drag_bps"] > table.loc["1/N", "cost_drag_bps"]
