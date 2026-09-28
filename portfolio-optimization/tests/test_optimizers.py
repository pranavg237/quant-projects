"""Mean-variance optimisers against their closed forms."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt.optimizers import (
    frontier_scalars,
    max_sharpe,
    min_variance,
    portfolio_volatility,
    unconstrained_min_variance,
    unconstrained_tangency,
)
from portopt.types import Constraints

UNCONSTRAINED = Constraints(long_only=False, min_weight=-1e18)


def test_gmv_variance_is_one_over_a(covariance: pd.DataFrame) -> None:
    mu = pd.Series(0.05, index=covariance.index)
    scalars = frontier_scalars(mu + np.arange(len(mu)) * 0.01, covariance)
    weights = unconstrained_min_variance(covariance)
    assert portfolio_volatility(weights, covariance) ** 2 == pytest.approx(1.0 / scalars.a)
    assert weights.sum() == pytest.approx(1.0)


def test_frontier_parabola_minimum_is_the_gmv(
    expected_returns: pd.Series, covariance: pd.DataFrame
) -> None:
    scalars = frontier_scalars(expected_returns, covariance)
    assert float(scalars.variance_at(scalars.gmv_return)) == pytest.approx(scalars.gmv_variance)
    assert scalars.d > 0


def test_solver_matches_closed_form_min_variance(covariance: pd.DataFrame) -> None:
    solved = min_variance(covariance, UNCONSTRAINED)
    np.testing.assert_allclose(solved.weights, unconstrained_min_variance(covariance), atol=1e-6)


def test_long_only_min_variance_respects_bounds(covariance: pd.DataFrame) -> None:
    result = min_variance(covariance, Constraints())
    assert (result.weights >= -1e-10).all()
    assert result.weights.sum() == pytest.approx(1.0)
    # Adding a constraint can only raise the minimum variance.
    unconstrained = min_variance(covariance, UNCONSTRAINED)
    assert result.volatility >= unconstrained.volatility - 1e-12


def test_unconstrained_max_sharpe_is_the_tangency_portfolio(
    expected_returns: pd.Series, covariance: pd.DataFrame
) -> None:
    solved = max_sharpe(expected_returns, covariance, UNCONSTRAINED, risk_free=0.02)
    closed = unconstrained_tangency(expected_returns, covariance, risk_free=0.02)
    np.testing.assert_allclose(solved.weights, closed, atol=1e-6)


def test_long_only_max_sharpe_beats_random_portfolios(
    expected_returns: pd.Series, covariance: pd.DataFrame
) -> None:
    best = max_sharpe(expected_returns, covariance, Constraints())
    assert (best.weights >= -1e-10).all()
    assert best.weights.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    for _ in range(500):
        w = pd.Series(rng.dirichlet(np.ones(len(expected_returns))), index=covariance.index)
        sharpe = float(w @ expected_returns) / portfolio_volatility(w, covariance)
        assert sharpe <= best.sharpe + 1e-9


@pytest.mark.parametrize("cap", [1.5, 2.0, 3.0])
def test_leverage_cap_binds(
    expected_returns: pd.Series, covariance: pd.DataFrame, cap: float
) -> None:
    # Regression test: the cap was once silently ignored by the max-Sharpe solver.
    constraints = Constraints(long_only=False, min_weight=-1e18, max_leverage=cap)
    result = max_sharpe(expected_returns, covariance, constraints)
    assert result.leverage <= cap + 1e-6
    assert result.weights.sum() == pytest.approx(1.0, abs=1e-8)


def test_infeasible_constraints_raise(covariance: pd.DataFrame) -> None:
    too_tight = Constraints(max_weight=0.1)  # six assets at most 10% each cannot sum to 1
    with pytest.raises(ValueError, match="infeasible"):
        min_variance(covariance, too_tight)
