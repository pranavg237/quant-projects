"""The portfolio construction methods, packaged as walk-forward builders.

Each builder is a closure over its hyper-parameters that takes the *visible* return history
and returns target weights. That signature is the only thing the backtester knows about a
strategy, which makes it impossible for a strategy to see data it should not.

The set deliberately spans the spectrum from "uses nothing" to "uses everything":

| Strategy | Uses expected returns | Uses covariance | Inverts the covariance |
|---|---|---|---|
| Equal weight (1/N) | no | no | no |
| Inverse volatility | no | variances only | no |
| Risk parity (ERC) | no | yes | no |
| Hierarchical Risk Parity | no | yes | no |
| Minimum variance | no | yes | **yes** |
| Markowitz max-Sharpe | **yes** | yes | **yes** |
| Black-Litterman | shrunk towards equilibrium | yes | **yes** |

That last column is the one that predicts out-of-sample performance, and the backtest is
designed to show it.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from . import covariance as cov_module
from .backtest import PortfolioBuilder, equal_weight_builder
from .blacklitterman import View, black_litterman
from .data import MARKET_WEIGHTS
from .hrp import hierarchical_risk_parity
from .optimizers import max_sharpe, min_variance
from .riskparity import equal_risk_contribution, inverse_volatility
from .types import Constraints, Weights

__all__ = [
    "CovarianceEstimator",
    "default_strategies",
    "make_black_litterman",
    "make_equal_weight",
    "make_hrp",
    "make_inverse_volatility",
    "make_max_sharpe",
    "make_min_variance",
    "make_risk_parity",
    "sample_estimator",
    "shrinkage_estimator",
]

#: Maps a return window and an annualisation factor to a covariance matrix.
CovarianceEstimator = Callable[[pd.DataFrame, float], pd.DataFrame]


def sample_estimator(returns: pd.DataFrame, annualise: float) -> pd.DataFrame:
    """Plain sample covariance."""
    return cov_module.sample_covariance(returns, annualise)


def shrinkage_estimator(
    returns: pd.DataFrame, annualise: float, target: str = "constant_correlation"
) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance."""
    matrix, _ = cov_module.ledoit_wolf(returns, target, annualise)
    return matrix


def _annualised_mean(returns: pd.DataFrame, periods_per_year: float) -> pd.Series:
    """Geometrically annualised sample mean return per asset."""
    return (1.0 + returns.mean()) ** periods_per_year - 1.0


def make_equal_weight() -> PortfolioBuilder:
    """1/N. The benchmark everything else has to beat."""
    return equal_weight_builder


def make_inverse_volatility(periods_per_year: float = 12.0) -> PortfolioBuilder:
    """Naive risk parity: weights proportional to the reciprocal of each asset's volatility."""

    def build(returns: pd.DataFrame) -> Weights:
        covariance = cov_module.sample_covariance(returns, periods_per_year)
        return inverse_volatility(covariance).weights

    return build


def make_risk_parity(
    periods_per_year: float = 12.0, estimator: CovarianceEstimator = shrinkage_estimator
) -> PortfolioBuilder:
    """Equal risk contribution."""

    def build(returns: pd.DataFrame) -> Weights:
        covariance = estimator(returns, periods_per_year)
        return equal_risk_contribution(covariance).weights

    return build


def make_hrp(
    periods_per_year: float = 12.0,
    estimator: CovarianceEstimator = sample_estimator,
    linkage_method: str = "single",
) -> PortfolioBuilder:
    """Hierarchical Risk Parity.

    Defaults to the **sample** covariance, because HRP's whole claim is that it does not
    need a well-conditioned matrix -- it never inverts one. Handing it a shrunk matrix would
    quietly do some of its work for it and muddy the comparison.
    """

    def build(returns: pd.DataFrame) -> Weights:
        covariance = estimator(returns, periods_per_year)
        return hierarchical_risk_parity(covariance, linkage_method=linkage_method).weights

    return build


def make_min_variance(
    periods_per_year: float = 12.0,
    estimator: CovarianceEstimator = shrinkage_estimator,
    constraints: Constraints | None = None,
) -> PortfolioBuilder:
    """Minimum variance. Uses the covariance matrix but never expected returns."""
    bounds = constraints or Constraints()

    def build(returns: pd.DataFrame) -> Weights:
        covariance = estimator(returns, periods_per_year)
        return min_variance(covariance, bounds).weights

    return build


def make_max_sharpe(
    periods_per_year: float = 12.0,
    estimator: CovarianceEstimator = sample_estimator,
    constraints: Constraints | None = None,
    risk_free: float = 0.0,
) -> PortfolioBuilder:
    """Markowitz maximum-Sharpe on sample moments -- the strategy this project indicts.

    Defaults to the sample covariance and the sample mean on purpose: that combination is
    what "Markowitz" means in practice, and showing what it does out of sample is the point.
    Pass ``estimator=shrinkage_estimator`` to see how much of the damage the covariance
    estimate is responsible for, versus the expected-return estimate.
    """
    bounds = constraints or Constraints()

    def build(returns: pd.DataFrame) -> Weights:
        covariance = estimator(returns, periods_per_year)
        mu = _annualised_mean(returns, periods_per_year)
        try:
            return max_sharpe(mu, covariance, bounds, risk_free).weights
        except ValueError:
            # No asset has a positive excess return in this window: nothing a long-only
            # tangency portfolio can do, so hold the minimum-variance portfolio instead.
            return min_variance(covariance, bounds).weights

    return build


def make_black_litterman(
    periods_per_year: float = 12.0,
    estimator: CovarianceEstimator = shrinkage_estimator,
    views: list[View] | None = None,
    market_weights: pd.Series | None = None,
    tau: float = 0.05,
    market_excess_return: float = 0.05,
    constraints: Constraints | None = None,
) -> PortfolioBuilder:
    """Black-Litterman, optimised on the posterior moments.

    With **no views** this is equilibrium investing: the posterior equals the prior, and
    optimising it reproduces the market portfolio exactly. That is the interesting
    configuration for a backtest, because it isolates what the *prior* buys you -- which is
    stability -- from what the views buy you, which depends entirely on whether the views
    are any good. Views are supported, and by default there are none, because inventing
    views in a backtest is indistinguishable from fitting them.
    """
    bounds = constraints or Constraints()

    def build(returns: pd.DataFrame) -> Weights:
        covariance = estimator(returns, periods_per_year)
        if market_weights is not None:
            weights = market_weights
        else:
            default = pd.Series(MARKET_WEIGHTS).reindex(covariance.index)
            weights = (
                default / default.sum()
                if default.notna().all()
                else pd.Series(1.0 / covariance.shape[0], index=covariance.index)
            )
        posterior = black_litterman(
            covariance,
            weights.reindex(covariance.index).fillna(0.0),
            views,
            tau=tau,
            market_excess_return=market_excess_return,
        )
        return max_sharpe(
            posterior.posterior_returns, posterior.posterior_covariance, bounds
        ).weights

    return build


def default_strategies(
    periods_per_year: float = 12.0, constraints: Constraints | None = None
) -> dict[str, PortfolioBuilder]:
    """The full comparison set used in the analysis script."""
    bounds = constraints or Constraints()
    return {
        "Equal weight (1/N)": make_equal_weight(),
        "Inverse volatility": make_inverse_volatility(periods_per_year),
        "Risk parity (ERC)": make_risk_parity(periods_per_year),
        "Hierarchical Risk Parity": make_hrp(periods_per_year),
        "Min variance (shrunk)": make_min_variance(periods_per_year, shrinkage_estimator, bounds),
        "Min variance (sample)": make_min_variance(periods_per_year, sample_estimator, bounds),
        "Markowitz max-Sharpe (sample)": make_max_sharpe(
            periods_per_year, sample_estimator, bounds
        ),
        "Markowitz max-Sharpe (shrunk)": make_max_sharpe(
            periods_per_year, shrinkage_estimator, bounds
        ),
        "Black-Litterman (no views)": make_black_litterman(
            periods_per_year, shrinkage_estimator, constraints=bounds
        ),
    }
