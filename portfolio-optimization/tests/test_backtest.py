"""Walk-forward engine: no lookahead, correct drift, costs and borrow accounting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt import backtest as bt
from portopt import strategies as st
from portopt.types import Constraints


def _panel(values: list[list[float]], columns: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    index = pd.date_range("2020-01-31", periods=len(values), freq="ME")
    return pd.DataFrame(values, index=index, columns=list(columns))


def test_builder_never_sees_the_period_it_is_scored_on(returns: pd.DataFrame) -> None:
    seen: list[pd.Timestamp] = []

    def spy(window: pd.DataFrame) -> pd.Series:
        seen.append(window.index[-1])
        return bt.equal_weight_builder(window)

    result = bt.walk_forward(returns, spy, "spy", lookback=24)
    # The window used for the weights applied to period t ends exactly one period earlier.
    for last_seen, scored in zip(seen, result.returns.index, strict=True):
        assert last_seen < scored
        assert returns.index.get_loc(scored) - returns.index.get_loc(last_seen) == 1


def test_a_lookahead_strategy_cannot_profit() -> None:
    # Each period one random asset returns +10% and the other -10%. A builder that picks
    # the most recent winner would earn +10% every period *with* lookahead. Without it, the
    # future is independent of the past and it earns about zero.
    rng = np.random.default_rng(7)
    winners = rng.integers(0, 2, size=400)
    values = [[0.10, -0.10] if w == 0 else [-0.10, 0.10] for w in winners]
    returns = _panel(values)

    def chase_last_winner(window: pd.DataFrame) -> pd.Series:
        best = window.iloc[-1].idxmax()
        return pd.Series({c: 1.0 if c == best else 0.0 for c in window.columns})

    result = bt.walk_forward(returns, chase_last_winner, "chaser", lookback=1, cost_bps=0.0)
    assert abs(result.returns.mean()) < 0.02


def test_weights_drift_and_turnover_is_measured_against_drifted_weights() -> None:
    returns = _panel([[0.0, 0.0], [0.0, 0.0], [0.10, -0.10], [0.0, 0.0]])
    result = bt.walk_forward(returns, bt.equal_weight_builder, "1/N", lookback=2, cost_bps=100.0)
    # Period 3 (index 2): held 50/50 and earned 0. After it the weights drift to
    # 0.55/0.45, so re-balancing to 50/50 at the next decision trades 0.05 + 0.05.
    turnover = result.turnover.to_numpy()
    assert turnover[0] == pytest.approx(1.0)  # initial purchase from cash
    assert turnover[1] == pytest.approx(0.1)
    assert result.costs.iloc[1] == pytest.approx(0.01 * 0.1)
    assert result.returns.iloc[-1] == pytest.approx(0.0 - 0.01 * 0.1)


def test_rebalance_every_n_lets_weights_drift_for_free() -> None:
    returns = _panel([[0.0, 0.0]] * 2 + [[0.10, -0.10]] * 6)
    result = bt.walk_forward(
        returns, bt.equal_weight_builder, "1/N", lookback=2, rebalance_every=3, cost_bps=10.0
    )
    assert len(result.weights) == 2  # six periods, rebalanced at the first and fourth
    assert (result.costs > 0).all()


def test_borrow_fee_is_charged_on_short_exposure() -> None:
    returns = _panel([[0.0, 0.0]] * 14)

    def long_short(window: pd.DataFrame) -> pd.Series:
        return pd.Series({"A": 1.5, "B": -0.5})

    result = bt.walk_forward(returns, long_short, "ls", lookback=2, cost_bps=0.0, borrow_bps=120)
    # 120 bp a year on 0.5 of short exposure is 5 bp a month; weights never drift on zero
    # returns, so it is the same every month.
    np.testing.assert_allclose(result.borrow_costs, 0.0120 / 12 * 0.5)
    np.testing.assert_allclose(result.returns, -0.0120 / 12 * 0.5)


def test_long_only_pays_no_borrow(returns: pd.DataFrame) -> None:
    result = bt.walk_forward(returns, bt.equal_weight_builder, "1/N", lookback=12, borrow_bps=500)
    assert result.borrow_costs.abs().max() == 0.0


def test_net_return_is_gross_minus_costs(returns: pd.DataFrame) -> None:
    builder = st.make_max_sharpe(
        12.0, st.sample_estimator, Constraints(long_only=False, min_weight=-1e18, max_leverage=2.0)
    )
    result = bt.walk_forward(returns, builder, "ms", lookback=36, cost_bps=10.0, borrow_bps=50.0)
    # Costs are booked on the decision date, which is the previous period's index.
    trading = result.costs.copy()
    trading.index = [returns.index[returns.index.get_loc(d) + 1] for d in trading.index]
    trading = trading.reindex(result.returns.index, fill_value=0.0)
    np.testing.assert_allclose(
        result.returns, result.gross_returns - trading - result.borrow_costs, atol=1e-15
    )


def test_buy_and_hold_has_no_turnover(returns: pd.DataFrame) -> None:
    result = bt.buy_and_hold(returns["A0"], "A0")
    assert result.turnover.sum() == 0.0
    assert result.costs.sum() == 0.0
    pd.testing.assert_series_equal(result.returns, returns["A0"].rename("A0"))


def test_too_little_history_raises(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="more than"):
        bt.walk_forward(returns.iloc[:10], bt.equal_weight_builder, "1/N", lookback=12)


def test_ruin_is_detected() -> None:
    returns = _panel([[0.0, 0.0]] * 3 + [[-0.9, 0.9]])

    def levered(window: pd.DataFrame) -> pd.Series:
        return pd.Series({"A": 3.0, "B": -2.0})

    assert bt.walk_forward(returns, levered, "lev", lookback=2, cost_bps=0.0).is_ruined


def test_rebalance_dates_use_last_trading_day() -> None:
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    dates = bt.rebalance_dates(index)
    assert list(dates) == [
        pd.Timestamp("2024-01-31"),
        pd.Timestamp("2024-02-29"),
        pd.Timestamp("2024-03-29"),
    ]


def test_default_strategies_all_produce_valid_weights(returns: pd.DataFrame) -> None:
    window = returns.iloc[:60]
    for name, builder in st.default_strategies(12.0).items():
        weights = builder(window)
        assert weights.sum() == pytest.approx(1.0, abs=1e-6), name
        assert (weights >= -1e-8).all(), name
