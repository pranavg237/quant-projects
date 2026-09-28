"""Risk parity, HRP and Black-Litterman: the identities each method must satisfy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from portopt.blacklitterman import View, black_litterman
from portopt.hrp import correlation_distance, hierarchical_risk_parity, quasi_diagonal_order
from portopt.optimizers import unconstrained_tangency
from portopt.riskparity import equal_risk_contribution, inverse_volatility, risk_contributions


def _constant_correlation(vols: list[float], rho: float) -> pd.DataFrame:
    n = len(vols)
    corr = np.full((n, n), rho)
    np.fill_diagonal(corr, 1.0)
    names = [f"A{i}" for i in range(n)]
    return pd.DataFrame(corr * np.outer(vols, vols), index=names, columns=names)


# -- risk parity ---------------------------------------------------------------------------


def test_erc_contributions_are_equal(covariance: pd.DataFrame) -> None:
    result = equal_risk_contribution(covariance)
    contributions = risk_contributions(result.weights, covariance)
    assert contributions.sum() == pytest.approx(result.volatility)
    # Equal to the solver's tolerance, not to machine precision.
    np.testing.assert_allclose(
        contributions / contributions.sum(), 1.0 / len(covariance), atol=1e-6
    )
    assert result.weights.sum() == pytest.approx(1.0)
    assert (result.weights > 0).all()


def test_erc_equals_inverse_vol_when_correlations_are_equal() -> None:
    covariance = _constant_correlation([0.1, 0.15, 0.2, 0.3], rho=0.4)
    np.testing.assert_allclose(
        equal_risk_contribution(covariance).weights,
        inverse_volatility(covariance).weights,
        atol=1e-8,
    )


def test_risk_budget_is_honoured(covariance: pd.DataFrame) -> None:
    budget = pd.Series(np.linspace(1, 2, len(covariance)), index=covariance.index)
    budget /= budget.sum()
    result = equal_risk_contribution(covariance, budget=budget)
    contributions = risk_contributions(result.weights, covariance)
    np.testing.assert_allclose(contributions / contributions.sum(), budget, atol=1e-6)


# -- HRP -----------------------------------------------------------------------------------


def test_quasi_diagonal_order_matches_scipy(returns: pd.DataFrame) -> None:
    distance = correlation_distance(returns.corr())
    link = linkage(squareform(distance.to_numpy(), checks=False), "single")
    assert quasi_diagonal_order(np.asarray(link)) == leaves_list(link).tolist()


def test_hrp_weights_are_a_long_only_portfolio(covariance: pd.DataFrame) -> None:
    weights = hierarchical_risk_parity(covariance).weights
    assert weights.sum() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_hrp_on_uncorrelated_assets_is_inverse_variance() -> None:
    # With zero correlation every bisection allocates by inverse variance, so the whole
    # portfolio does too.
    vols = np.array([0.1, 0.2, 0.3, 0.4])
    covariance = _constant_correlation(list(vols), rho=0.0)
    inverse_variance = (1 / vols**2) / (1 / vols**2).sum()
    np.testing.assert_allclose(
        hierarchical_risk_parity(covariance).weights.to_numpy(), inverse_variance, atol=1e-12
    )


# -- Black-Litterman -----------------------------------------------------------------------


@pytest.fixture
def market(covariance: pd.DataFrame) -> pd.Series:
    raw = pd.Series(np.linspace(1, 3, len(covariance)), index=covariance.index)
    return raw / raw.sum()


def test_no_views_returns_the_prior_and_the_market(
    covariance: pd.DataFrame, market: pd.Series
) -> None:
    result = black_litterman(covariance, market)
    np.testing.assert_allclose(result.posterior_returns, result.prior_returns)
    assert result.tilt.abs().max() == 0.0
    # Reverse optimisation: the tangency portfolio of the equilibrium returns is the market.
    tangency = unconstrained_tangency(result.posterior_returns, covariance)
    np.testing.assert_allclose(tangency, market, atol=1e-10)


def test_view_moves_posterior_towards_it(covariance: pd.DataFrame, market: pd.Series) -> None:
    prior = black_litterman(covariance, market).prior_returns
    target = float(prior["A0"]) + 0.05
    weak = black_litterman(covariance, market, [View({"A0": 1.0}, target, 0.1)])
    strong = black_litterman(covariance, market, [View({"A0": 1.0}, target, 1.0)])
    assert prior["A0"] < weak.posterior_returns["A0"] < strong.posterior_returns["A0"] < target


def test_posterior_is_invariant_to_tau(covariance: pd.DataFrame, market: pd.Series) -> None:
    views = [View({"A0": 1.0, "A1": -1.0}, 0.03, 0.5)]
    a = black_litterman(covariance, market, views, tau=0.01).posterior_returns
    b = black_litterman(covariance, market, views, tau=0.5).posterior_returns
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_view_on_unknown_asset_raises(covariance: pd.DataFrame, market: pd.Series) -> None:
    with pytest.raises(ValueError, match="unknown asset"):
        black_litterman(covariance, market, [View({"ZZZ": 1.0}, 0.05)])
