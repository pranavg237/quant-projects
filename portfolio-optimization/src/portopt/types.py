"""Shared types and small helpers.

Two conventions run through the whole package and are worth stating once:

**Weights are a pandas Series indexed by ticker, never a bare array.** Every optimiser and
every covariance estimator returns something aligned to asset names. A silent misalignment
between an expected-return vector and a covariance matrix is the single easiest way to get
a plausible-looking portfolio that is complete nonsense, and using labelled containers
everywhere makes it a loud error instead.

**Returns are simple (arithmetic), not log.** Portfolio return is a *weighted average* of
simple asset returns, which is exactly what mean-variance assumes; log returns do not
aggregate across assets that way. Log returns are used only where compounding through time
matters, and the conversion is explicit at that point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""Canonical float array type."""

Weights: TypeAlias = "pd.Series[float]"
"""Portfolio weights, indexed by ticker."""

__all__ = [
    "TRADING_DAYS",
    "Constraints",
    "FloatArray",
    "Weights",
    "align",
    "annualise_return",
    "annualise_vol",
    "as_float",
    "as_timestamp",
    "is_positive_semidefinite",
    "nearest_positive_definite",
]

#: Trading days per year. Used for every annualisation in the package.
TRADING_DAYS = 252


@dataclass(frozen=True)
class Constraints:
    """Portfolio constraints shared by every optimiser.

    Attributes:
        long_only: Forbid short positions. Equivalent to ``min_weight = 0`` but stated
            separately because it is the switch that matters most: an unconstrained
            mean-variance optimiser will happily take 400% long / 300% short positions on
            estimation noise, and the long-only constraint is the cheapest and most
            effective regulariser in the whole toolkit.
        min_weight: Lower bound on each weight.
        max_weight: Upper bound on each weight. ``None`` means unbounded above.
        budget: Weights must sum to this. ``1.0`` is fully invested.
        max_leverage: Cap on the sum of absolute weights. ``None`` means uncapped. Only
            binds when shorting is allowed.
    """

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float | None = None
    budget: float = 1.0
    max_leverage: float | None = None

    def __post_init__(self) -> None:
        if self.max_weight is not None and self.max_weight <= self.min_weight:
            raise ValueError("max_weight must exceed min_weight")
        if self.max_leverage is not None and self.max_leverage < abs(self.budget):
            raise ValueError("max_leverage cannot be below |budget|")

    def bounds(self, n_assets: int) -> list[tuple[float, float | None]]:
        """Per-asset ``(low, high)`` bounds for ``scipy.optimize``."""
        low = max(self.min_weight, 0.0) if self.long_only else self.min_weight
        return [(low, self.max_weight) for _ in range(n_assets)]

    def is_feasible(self, n_assets: int) -> bool:
        """Whether the budget can be met at all given the bounds."""
        low, high = self.bounds(n_assets)[0]
        if n_assets * low > self.budget + 1e-12:
            return False
        return not (high is not None and n_assets * high < self.budget - 1e-12)


def align(expected_returns: pd.Series, covariance: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Restrict a return vector and a covariance matrix to their common assets, in order.

    Raises:
        ValueError: if fewer than two assets are common to both.
    """
    common = [a for a in expected_returns.index if a in covariance.index]
    if len(common) < 2:
        raise ValueError(
            f"expected returns and covariance share only {len(common)} asset(s); "
            "at least 2 are needed"
        )
    return expected_returns.loc[common], covariance.loc[common, common]


def as_float(value: Any) -> float:
    """Coerce to ``float``.

    Exists because pandas types reductions and index lookups as ``Hashable`` or a wide
    union, so a bare ``float(...)`` on an obviously numeric result fails type checking.
    """
    return float(cast(float, value))


def as_timestamp(value: Any) -> pd.Timestamp:
    """Coerce an index label to a ``Timestamp``, for the same reason as :func:`as_float`."""
    return pd.Timestamp(cast(Any, value))


def annualise_return(period_return: float, periods_per_year: float) -> float:
    """Geometrically annualise a per-period simple return."""
    return float((1.0 + period_return) ** periods_per_year - 1.0)


def annualise_vol(period_vol: float, periods_per_year: float) -> float:
    """Annualise a per-period volatility by the square-root-of-time rule."""
    return float(period_vol * np.sqrt(periods_per_year))


def is_positive_semidefinite(matrix: pd.DataFrame | FloatArray, tol: float = -1e-10) -> bool:
    """Whether a symmetric matrix has no meaningfully negative eigenvalue."""
    values = np.asarray(matrix, dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(0.5 * (values + values.T))
    return bool(eigenvalues.min() >= tol)


def nearest_positive_definite(matrix: pd.DataFrame, min_eigenvalue: float = 1e-10) -> pd.DataFrame:
    """Clip negative eigenvalues to ``min_eigenvalue`` and rebuild the matrix.

    A sample covariance matrix estimated from fewer observations than assets is singular by
    construction, and floating-point error can leave a nominally PSD matrix with tiny
    negative eigenvalues. Either makes ``Sigma^{-1}`` explode and produces optimiser output
    that looks precise and is meaningless. This is the minimal repair; it is **not** a
    substitute for shrinkage, which fixes the estimation problem rather than the symptom.
    """
    values = np.asarray(matrix, dtype=np.float64)
    values = 0.5 * (values + values.T)
    eigenvalues, eigenvectors = np.linalg.eigh(values)
    repaired = eigenvectors @ np.diag(np.maximum(eigenvalues, min_eigenvalue)) @ eigenvectors.T
    repaired = 0.5 * (repaired + repaired.T)
    return pd.DataFrame(repaired, index=matrix.index, columns=matrix.columns)
