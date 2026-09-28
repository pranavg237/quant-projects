r"""Mean-variance optimisation: the efficient frontier, minimum variance and maximum Sharpe.

**Unconstrained.** With :math:`A = \mathbf{1}'\Sigma^{-1}\mathbf{1}`,
:math:`B = \mathbf{1}'\Sigma^{-1}\mu`, :math:`C = \mu'\Sigma^{-1}\mu` and
:math:`D = AC - B^2`, the classical results are

.. math::
    w_{\text{GMV}} = \frac{\Sigma^{-1}\mathbf{1}}{A}, \qquad
    \sigma^2_{\text{GMV}} = \frac{1}{A},

.. math::
    \sigma_p^2(\mu_p) = \frac{A\mu_p^2 - 2B\mu_p + C}{D}, \qquad
    w_{\text{tan}} = \frac{\Sigma^{-1}(\mu - r_f\mathbf{1})}
                          {\mathbf{1}'\Sigma^{-1}(\mu - r_f\mathbf{1})}.

These are implemented exactly and are what the tests check the numerical solver against,
because "the optimiser agrees with itself" is not a test.

**Constrained.** Long-only or bounded problems have no closed form. Minimum variance and
target-return problems are quadratic programmes, solved here with SLSQP. Maximum Sharpe
looks harder because the objective is a ratio, but for a long-only portfolio there is an
exact trick (Cornuejols and Tutuncu): the change of variables :math:`y = w/\kappa` turns

.. math:: \max_w \frac{(\mu - r_f\mathbf{1})'w}{\sqrt{w'\Sigma w}}
    \quad\text{into}\quad
    \min_y y'\Sigma y \ \text{ s.t. } (\mu - r_f\mathbf{1})'y = 1,\ y \ge 0,

a convex QP whose solution recovers the tangency portfolio as :math:`w = y/\mathbf{1}'y`.
That is used whenever the constraints permit it, because it finds the *global* optimum,
where directly maximising a ratio with a general-purpose solver can and does land in a
local one. When the constraint set rules the transformation out (upper bounds, budgets
other than one), the code falls back to direct SLSQP on the Sharpe ratio and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .types import Constraints, FloatArray, Weights, align

__all__ = [
    "EfficientFrontier",
    "FrontierScalars",
    "OptimizationResult",
    "efficient_frontier",
    "frontier_scalars",
    "max_sharpe",
    "min_variance",
    "portfolio_return",
    "portfolio_volatility",
    "target_return_portfolio",
    "unconstrained_min_variance",
    "unconstrained_tangency",
]


def portfolio_return(weights: Weights, expected_returns: pd.Series) -> float:
    r""":math:`\mu'w`, on the assets the two have in common."""
    common = weights.index.intersection(expected_returns.index)
    w = weights.loc[common].to_numpy(dtype=np.float64)
    mu = expected_returns.loc[common].to_numpy(dtype=np.float64)
    return float(w @ mu)


def portfolio_volatility(weights: Weights, covariance: pd.DataFrame) -> float:
    r""":math:`\sqrt{w'\Sigma w}`."""
    common = weights.index.intersection(covariance.index)
    w = weights.loc[common].to_numpy(dtype=np.float64)
    sigma = covariance.loc[common, common].to_numpy(dtype=np.float64)
    return float(np.sqrt(max(w @ sigma @ w, 0.0)))


@dataclass(frozen=True)
class OptimizationResult:
    """A solved portfolio.

    Attributes:
        weights: Asset weights.
        expected_return: :math:`\\mu'w`.
        volatility: :math:`\\sqrt{w'\\Sigma w}`.
        sharpe: ``(expected_return - risk_free) / volatility``.
        method: Which solver produced it, for provenance.
        converged: Whether the solver reported success.
        message: Solver message.
    """

    weights: Weights
    expected_return: float
    volatility: float
    sharpe: float
    method: str
    converged: bool = True
    message: str = ""

    @property
    def leverage(self) -> float:
        """Sum of absolute weights. 1.0 for a fully invested long-only portfolio."""
        return float(self.weights.abs().sum())

    @property
    def effective_n(self) -> float:
        r"""Inverse Herfindahl :math:`1/\\sum w_i^2`: how many assets this *really* holds.

        A portfolio with a 90% position in one asset and 1% in ten others has an effective
        N near 1.2, however many line items it shows.

        Only interpretable for long-only portfolios. With offsetting long and short legs
        :math:`\sum w_i^2` can exceed 1 and the "effective number of assets" comes out
        below 1, which is a signal that the portfolio is levered, not that it holds less
        than one asset.
        """
        squared = float((self.weights**2).sum())
        return float(1.0 / squared) if squared > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"{self.method}: return {self.expected_return:.2%}, vol {self.volatility:.2%}, "
            f"Sharpe {self.sharpe:.3f}, leverage {self.leverage:.2f}, "
            f"effective N {self.effective_n:.1f}"
        )


@dataclass(frozen=True)
class FrontierScalars:
    r"""The four scalars that determine the whole unconstrained efficient frontier.

    Attributes:
        a: :math:`\mathbf{1}'\Sigma^{-1}\mathbf{1}`.
        b: :math:`\mathbf{1}'\Sigma^{-1}\mu`.
        c: :math:`\mu'\Sigma^{-1}\mu`.
        d: :math:`AC - B^2`, positive whenever returns are not all identical.
    """

    a: float
    b: float
    c: float
    d: float

    @property
    def gmv_return(self) -> float:
        """Expected return of the global minimum-variance portfolio, :math:`B/A`."""
        return self.b / self.a

    @property
    def gmv_variance(self) -> float:
        """Variance of the global minimum-variance portfolio, :math:`1/A`."""
        return 1.0 / self.a

    def variance_at(self, target_return: float | FloatArray) -> FloatArray:
        r""":math:`\sigma_p^2 = (A\mu_p^2 - 2B\mu_p + C)/D` -- the frontier parabola."""
        mu = np.asarray(target_return, dtype=np.float64)
        return np.asarray((self.a * mu**2 - 2 * self.b * mu + self.c) / self.d, dtype=np.float64)


def frontier_scalars(expected_returns: pd.Series, covariance: pd.DataFrame) -> FrontierScalars:
    """Compute ``(A, B, C, D)`` for the unconstrained frontier."""
    mu_series, sigma_frame = align(expected_returns, covariance)
    mu = mu_series.to_numpy(dtype=np.float64)
    precision = np.linalg.inv(sigma_frame.to_numpy(dtype=np.float64))
    ones = np.ones_like(mu)
    a = float(ones @ precision @ ones)
    b = float(ones @ precision @ mu)
    c = float(mu @ precision @ mu)
    return FrontierScalars(a=a, b=b, c=c, d=a * c - b * b)


def unconstrained_min_variance(covariance: pd.DataFrame) -> Weights:
    r"""Closed-form global minimum-variance weights :math:`\Sigma^{-1}\mathbf{1}/A`.

    No constraints at all, so this will happily short. It exists as the exact reference the
    constrained solver is tested against, and because the gap between it and the long-only
    solution is a good measure of how much the constraint is doing.
    """
    precision = np.linalg.inv(covariance.to_numpy(dtype=np.float64))
    ones = np.ones(covariance.shape[0])
    raw = precision @ ones
    return pd.Series(raw / raw.sum(), index=covariance.index, name="weight")


def unconstrained_tangency(
    expected_returns: pd.Series, covariance: pd.DataFrame, risk_free: float = 0.0
) -> Weights:
    r"""Closed-form tangency weights :math:`\Sigma^{-1}(\mu - r_f\mathbf{1})`, normalised.

    Raises:
        ValueError: if the excess returns are orthogonal to the precision-weighted ones, so
            the normalising constant vanishes and no tangency portfolio exists.
    """
    mu_series, sigma_frame = align(expected_returns, covariance)
    excess = mu_series.to_numpy(dtype=np.float64) - risk_free
    precision = np.linalg.inv(sigma_frame.to_numpy(dtype=np.float64))
    raw = precision @ excess
    total = float(raw.sum())
    if abs(total) < 1e-14:
        raise ValueError(
            "no tangency portfolio: 1'Sigma^-1(mu - rf) is zero, so the frontier's "
            "asymptote passes through the risk-free rate"
        )
    return pd.Series(raw / total, index=sigma_frame.index, name="weight")


def _solve_quadratic(
    covariance: pd.DataFrame,
    constraints: Constraints,
    extra_constraints: list[dict[str, object]] | None = None,
    initial: FloatArray | None = None,
) -> tuple[FloatArray, bool, str]:
    """Minimise ``w' Sigma w`` subject to the budget, bounds and any extra constraints."""
    sigma = covariance.to_numpy(dtype=np.float64)
    n = sigma.shape[0]
    if not constraints.is_feasible(n):
        raise ValueError(
            f"constraints are infeasible for {n} assets: budget {constraints.budget} "
            f"cannot be met within the per-asset bounds"
        )

    def objective(w: FloatArray) -> float:
        return float(w @ sigma @ w)

    def gradient(w: FloatArray) -> FloatArray:
        return np.asarray(2.0 * sigma @ w, dtype=np.float64)

    cons: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda w: float(w.sum() - constraints.budget)}
    ]
    if constraints.max_leverage is not None:
        limit = constraints.max_leverage
        cons.append({"type": "ineq", "fun": lambda w: float(limit - np.abs(w).sum())})
    cons.extend(extra_constraints or [])

    start = initial if initial is not None else np.full(n, constraints.budget / n)
    solution = minimize(
        objective,
        start,
        jac=gradient,
        bounds=constraints.bounds(n),
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    return np.asarray(solution.x, dtype=np.float64), bool(solution.success), str(solution.message)


def min_variance(
    covariance: pd.DataFrame,
    constraints: Constraints | None = None,
    expected_returns: pd.Series | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    r"""Minimum-variance portfolio, :math:`\min_w w'\Sigma w` subject to the constraints.

    Uses the closed form when the problem is genuinely unconstrained (no bounds, budget 1,
    shorting allowed) and SLSQP otherwise.

    Args:
        covariance: Asset covariance matrix.
        constraints: Portfolio constraints. Defaults to long-only, fully invested.
        expected_returns: Only used to report the portfolio's expected return; the
            minimum-variance portfolio does not depend on it. **This is the entire reason
            it out-performs out of sample**: expected returns are far harder to estimate
            than covariances, and a portfolio that ignores them inherits none of that error.
        risk_free: Used only for the reported Sharpe ratio.
    """
    constraints = constraints or Constraints()
    unconstrained = (
        not constraints.long_only
        and constraints.max_weight is None
        and constraints.min_weight <= -1e12
        and constraints.max_leverage is None
    )
    if unconstrained:
        weights = unconstrained_min_variance(covariance) * constraints.budget
        converged, message, method = True, "closed form", "min-variance (closed form)"
    else:
        raw, converged, message = _solve_quadratic(covariance, constraints)
        weights = pd.Series(raw, index=covariance.index, name="weight")
        method = "min-variance (SLSQP)"

    expected = portfolio_return(weights, expected_returns) if expected_returns is not None else 0.0
    volatility = portfolio_volatility(weights, covariance)
    return OptimizationResult(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe=(expected - risk_free) / volatility if volatility > 0 else float("nan"),
        method=method,
        converged=converged,
        message=message,
    )


def target_return_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    target: float,
    constraints: Constraints | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    r"""Minimum variance subject to :math:`\mu'w = \text{target}` -- one frontier point."""
    constraints = constraints or Constraints()
    mu_series, sigma_frame = align(expected_returns, covariance)
    mu = mu_series.to_numpy(dtype=np.float64)
    extra = [{"type": "eq", "fun": lambda w: float(w @ mu - target)}]
    raw, converged, message = _solve_quadratic(sigma_frame, constraints, extra)
    weights = pd.Series(raw, index=sigma_frame.index, name="weight")
    expected = portfolio_return(weights, mu_series)
    volatility = portfolio_volatility(weights, sigma_frame)
    return OptimizationResult(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe=(expected - risk_free) / volatility if volatility > 0 else float("nan"),
        method="target-return (SLSQP)",
        converged=converged,
        message=message,
    )


def max_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    constraints: Constraints | None = None,
    risk_free: float = 0.0,
) -> OptimizationResult:
    r"""Maximum-Sharpe (tangency) portfolio.

    Solved by the convex reformulation described in the module docstring when the
    constraints allow it -- long-only, budget 1, no upper bounds -- and by direct SLSQP on
    the Sharpe ratio otherwise. ``method`` records which was used, because the convex route
    is globally optimal and the direct route is not.
    """
    constraints = constraints or Constraints()
    mu_series, sigma_frame = align(expected_returns, covariance)
    mu = mu_series.to_numpy(dtype=np.float64)
    sigma = sigma_frame.to_numpy(dtype=np.float64)
    excess = mu - risk_free
    n = len(mu)

    fully_unconstrained = (
        not constraints.long_only
        and constraints.max_weight is None
        and constraints.min_weight <= -1e12
        and constraints.max_leverage is None
        and abs(constraints.budget - 1.0) < 1e-12
    )
    if fully_unconstrained:
        weights = unconstrained_tangency(mu_series, sigma_frame, risk_free)
        method, converged, message = "max-Sharpe (closed form)", True, "closed form"
    elif (
        constraints.long_only
        and constraints.max_weight is None
        and abs(constraints.budget - 1.0) < 1e-12
        and constraints.max_leverage is None
    ):
        if np.max(excess) <= 0:
            raise ValueError(
                "no asset has a positive excess return, so no long-only portfolio can "
                "have a positive Sharpe ratio"
            )

        # Convex reformulation: min y'Sigma y s.t. excess'y = 1, y >= 0.
        def objective(y: FloatArray) -> float:
            return float(y @ sigma @ y)

        def gradient(y: FloatArray) -> FloatArray:
            return np.asarray(2.0 * sigma @ y, dtype=np.float64)

        start = np.maximum(excess, 0.0)
        start = start / max(float(excess @ start), 1e-12)
        solution = minimize(
            objective,
            start,
            jac=gradient,
            bounds=[(0.0, None)] * n,
            constraints=[{"type": "eq", "fun": lambda y: float(excess @ y - 1.0)}],
            method="SLSQP",
            options={"maxiter": 500, "ftol": 1e-14},
        )
        total = float(solution.x.sum())
        raw = solution.x / total if total > 1e-14 else np.full(n, 1.0 / n)
        weights = pd.Series(raw, index=sigma_frame.index, name="weight")
        method = "max-Sharpe (convex QP)"
        converged, message = bool(solution.success), str(solution.message)
    else:

        def negative_sharpe(w: FloatArray) -> float:
            vol = float(np.sqrt(max(w @ sigma @ w, 1e-300)))
            return -float(excess @ w) / vol

        cons: list[dict[str, object]] = [
            {"type": "eq", "fun": lambda w: float(w.sum() - constraints.budget)}
        ]
        if constraints.max_leverage is not None:
            # sum|w| is non-smooth, so SLSQP is given a smooth surrogate: the constraint is
            # imposed on sqrt(sum(w^2 + eps)), which bounds sum|w| from above and has a
            # gradient everywhere. Forgetting this constraint entirely was a real bug --
            # the backtest ran at 785x average leverage while nominally capped at 1.5x.
            limit = constraints.max_leverage
            cons.append(
                {"type": "ineq", "fun": lambda w: float(limit - np.sum(np.sqrt(w * w + 1e-12)))}
            )
        solution = minimize(
            negative_sharpe,
            np.full(n, constraints.budget / n),
            bounds=constraints.bounds(n),
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 800, "ftol": 1e-12},
        )
        weights = pd.Series(solution.x, index=sigma_frame.index, name="weight")
        method = "max-Sharpe (SLSQP, local)"
        converged, message = bool(solution.success), str(solution.message)

    expected = portfolio_return(weights, mu_series)
    volatility = portfolio_volatility(weights, sigma_frame)
    return OptimizationResult(
        weights=weights,
        expected_return=expected,
        volatility=volatility,
        sharpe=(expected - risk_free) / volatility if volatility > 0 else float("nan"),
        method=method,
        converged=converged,
        message=message,
    )


@dataclass
class EfficientFrontier:
    """A sampled efficient frontier.

    Attributes:
        returns: Target returns.
        volatilities: Minimum volatility achievable at each target.
        weights: One row of weights per target.
        constrained: Whether the frontier was computed subject to constraints.
    """

    returns: FloatArray
    volatilities: FloatArray
    weights: pd.DataFrame
    constrained: bool

    def to_frame(self) -> pd.DataFrame:
        """Return / volatility / Sharpe as a tidy frame."""
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe = np.where(self.volatilities > 0, self.returns / self.volatilities, np.nan)
        return pd.DataFrame(
            {"target_return": self.returns, "volatility": self.volatilities, "sharpe": sharpe}
        )


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    n_points: int = 40,
    constraints: Constraints | None = None,
) -> EfficientFrontier:
    """Trace the efficient frontier between the minimum-variance and maximum-return points.

    The upper end of the range is the highest *achievable* return under the constraints --
    for a long-only portfolio that is the best single asset, not the unbounded maximum. The
    lower end is the minimum-variance portfolio's return, because everything below it is on
    the inefficient branch of the parabola and is never worth holding.
    """
    constraints = constraints or Constraints()
    mu_series, sigma_frame = align(expected_returns, covariance)
    gmv = min_variance(sigma_frame, constraints, mu_series)

    if constraints.long_only:
        highest = float(mu_series.max()) * constraints.budget
    else:  # pragma: no cover - unbounded above; pick a generous span
        highest = float(mu_series.max()) * 2.0 * constraints.budget
    lowest = gmv.expected_return
    if highest <= lowest:  # pragma: no cover - degenerate single-asset case
        highest = lowest * 1.5 + 1e-6

    targets = np.linspace(lowest, highest, n_points)
    volatilities = np.empty(n_points, dtype=np.float64)
    rows: list[pd.Series] = []
    for i, target in enumerate(targets):
        point = target_return_portfolio(mu_series, sigma_frame, float(target), constraints)
        volatilities[i] = point.volatility
        rows.append(point.weights)
    return EfficientFrontier(
        returns=targets,
        volatilities=volatilities,
        weights=pd.DataFrame(rows, index=targets),
        constrained=True,
    )
