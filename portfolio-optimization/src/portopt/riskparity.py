r"""Risk parity: equalise each asset's contribution to portfolio risk.

Portfolio volatility decomposes exactly, by Euler's theorem on the homogeneous function
:math:`\sigma(w) = \sqrt{w'\Sigma w}`:

.. math::
    \sigma(w) = \sum_i w_i \frac{\partial \sigma}{\partial w_i}
              = \sum_i \underbrace{\frac{w_i(\Sigma w)_i}{\sqrt{w'\Sigma w}}}_{RC_i}.

The **equal risk contribution** portfolio sets every :math:`RC_i` equal. It uses the
covariance matrix but **not** expected returns at all, which is the whole point: expected
returns are the hardest thing in finance to estimate, and a portfolio that never touches
them inherits none of that error.

**How it is solved matters.** The obvious approach -- minimise
:math:`\sum_i (RC_i - \bar{RC})^2` -- is non-convex, has flat regions, and a
general-purpose solver lands in different places from different starts. There is a convex
alternative (Spinu 2013; Maillard, Roncalli and Teiletche 2010): minimise

.. math:: f(w) = \tfrac12 w'\Sigma w - \frac{1}{N}\sum_i \ln w_i, \qquad w > 0,

whose first-order condition is :math:`(\Sigma w)_i = \frac{1}{N w_i}`, i.e.
:math:`w_i(\Sigma w)_i` is the same for every :math:`i` -- exactly equal risk contribution.
The log barrier makes it strictly convex on the positive orthant, so the solution is unique
and any descent method finds it. Normalising the solution to sum to one preserves the
property, because risk contributions are scale-invariant in the right way.

A **naive risk parity** variant (inverse volatility, ignoring correlations) is also
provided. It is what most people mean by "risk parity" in practice, it is much simpler, and
the gap between the two is a useful measure of how much the correlation structure matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .optimizers import OptimizationResult, portfolio_return, portfolio_volatility
from .types import FloatArray, Weights

__all__ = [
    "equal_risk_contribution",
    "inverse_volatility",
    "risk_contributions",
]


def risk_contributions(weights: Weights, covariance: pd.DataFrame) -> pd.Series:
    r"""Each asset's contribution to portfolio volatility. These sum to :math:`\sigma_p`.

    Raises:
        ValueError: if the portfolio has zero volatility, where contributions are undefined.
    """
    common = weights.index.intersection(covariance.index)
    w = weights.loc[common].to_numpy(dtype=np.float64)
    sigma = covariance.loc[common, common].to_numpy(dtype=np.float64)
    volatility = float(np.sqrt(max(w @ sigma @ w, 0.0)))
    if volatility <= 0:
        raise ValueError("risk contributions are undefined for a zero-volatility portfolio")
    return pd.Series(w * (sigma @ w) / volatility, index=common, name="risk_contribution")


def inverse_volatility(
    covariance: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    r"""Naive risk parity: :math:`w_i \propto 1/\sigma_i`, correlations ignored.

    Equal risk contribution only if every pairwise correlation is identical. Otherwise it
    over-weights assets that are individually calm but move with everything else -- which
    is exactly what happened to "risk parity" funds holding credit in 2008.
    """
    volatilities = np.sqrt(np.diag(covariance.to_numpy(dtype=np.float64)))
    if np.any(volatilities <= 0):
        raise ValueError("every asset must have positive variance")
    raw = 1.0 / volatilities
    weights = pd.Series(raw / raw.sum(), index=covariance.index, name="weight")
    expected = portfolio_return(weights, expected_returns) if expected_returns is not None else 0.0
    volatility = portfolio_volatility(weights, covariance)
    return OptimizationResult(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe=(expected - risk_free) / volatility if volatility > 0 else float("nan"),
        method="inverse volatility",
    )


def equal_risk_contribution(
    covariance: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_free: float = 0.0,
    budget: pd.Series | None = None,
    tol: float = 1e-12,
    max_iter: int = 2_000,
) -> OptimizationResult:
    r"""Equal risk contribution via the convex log-barrier formulation.

    Args:
        covariance: Asset covariance matrix.
        expected_returns: Only used to report the portfolio's expected return.
        risk_free: Only used for the reported Sharpe ratio.
        budget: Optional *risk* budget -- target share of total risk per asset, summing to
            one. The default is equal shares. Passing an explicit budget turns this into
            general "risk budgeting", which is how the technique is actually used: a
            portfolio might deliberately allocate 60% of its risk to equities.
        tol: Solver tolerance.
        max_iter: Solver iteration cap.

    Returns:
        An :class:`~portopt.optimizers.OptimizationResult` whose weights are positive and
        sum to one.

    Raises:
        ValueError: if the risk budget is not positive and summing to one.
    """
    sigma = covariance.to_numpy(dtype=np.float64)
    n = sigma.shape[0]
    if budget is None:
        shares = np.full(n, 1.0 / n)
    else:
        shares = budget.reindex(covariance.index).to_numpy(dtype=np.float64)
        if np.any(~np.isfinite(shares)) or np.any(shares <= 0):
            raise ValueError("risk budget entries must all be positive")
        if not np.isclose(shares.sum(), 1.0):
            raise ValueError("risk budget must sum to 1")

    def objective(w: FloatArray) -> float:
        return float(0.5 * w @ sigma @ w - shares @ np.log(w))

    def gradient(w: FloatArray) -> FloatArray:
        return np.asarray(sigma @ w - shares / w, dtype=np.float64)

    # Start from inverse volatility: already a decent risk-parity approximation, so the
    # solver converges in a handful of iterations.
    start = 1.0 / np.sqrt(np.diag(sigma))
    start = start / start.sum()
    solution = minimize(
        objective,
        start,
        jac=gradient,
        bounds=[(1e-12, None)] * n,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol},
    )
    raw = np.maximum(solution.x, 0.0)
    weights = pd.Series(raw / raw.sum(), index=covariance.index, name="weight")

    expected = portfolio_return(weights, expected_returns) if expected_returns is not None else 0.0
    volatility = portfolio_volatility(weights, covariance)
    return OptimizationResult(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe=(expected - risk_free) / volatility if volatility > 0 else float("nan"),
        method="equal risk contribution",
        converged=bool(solution.success),
        message=str(solution.message),
    )
