r"""How long an estimation window mean-variance needs before it beats 1/N.

DeMiguel, Garlappi and Uppal (2009) asked how many months of data a sample-based
mean-variance investor needs before, *in expectation*, their portfolio beats 1/N out of
sample. This module runs the same kind of experiment on this project's own universe.

**Design.** Pretend the full-sample monthly excess-return mean :math:`\mu` and covariance
:math:`\Sigma` of the 15 ETFs are the truth. For an estimation window of :math:`M` months:

1. draw :math:`M` IID normal monthly excess returns from :math:`N(\mu, \Sigma)`;
2. estimate :math:`\hat\mu, \hat\Sigma` from them and build each portfolio;
3. score the portfolio by its **true** Sharpe ratio,
   :math:`w'\mu / \sqrt{w'\Sigma w}`, which is what it would earn out of sample on
   average. No out-of-sample returns are simulated, so there is no second layer of noise.

Repeat and average. 1/N does not estimate anything, so its true Sharpe is a constant. The
true tangency portfolio :math:`\Sigma^{-1}\mu` is the ceiling no estimated portfolio can
beat.

Because returns are IID normal and the moments are known, this is the *best case* for
mean-variance: no regime changes, no fat tails, no costs. And because the "true" moments
are themselves in-sample estimates from 261 months, the true tangency Sharpe is inflated
(it is an in-sample optimum), which makes the potential gain from optimising look larger
than it really is. Both effects favour mean-variance, so the crossover window found here is
if anything an understatement of what reality would need.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .optimizers import max_sharpe, min_variance
from .types import Constraints, FloatArray

__all__ = [
    "EstimatedPortfolio",
    "WindowExperiment",
    "default_estimated_portfolios",
    "run_window_experiment",
    "true_sharpe",
]

#: Maps an estimated monthly (mean, covariance) to portfolio weights.
EstimatedPortfolio = Callable[[pd.Series, pd.DataFrame], FloatArray]


def true_sharpe(weights: FloatArray, mu: FloatArray, sigma: FloatArray) -> float:
    """Per-period Sharpe ratio of ``weights`` under the true moments."""
    variance = float(weights @ sigma @ weights)
    if variance <= 0:
        return float("nan")
    return float(weights @ mu) / float(np.sqrt(variance))


def _sample_tangency(mu_hat: pd.Series, sigma_hat: pd.DataFrame) -> FloatArray:
    """Unconstrained sample tangency direction :math:`\\hat\\Sigma^{-1}\\hat\\mu`.

    With a risk-free asset available, the mean-variance investor holds this direction and
    lends or borrows the rest, so only the direction matters for the Sharpe ratio. It is
    scaled to unit gross exposure purely for readability.
    """
    raw = np.linalg.solve(sigma_hat.to_numpy(dtype=np.float64), mu_hat.to_numpy(np.float64))
    return np.asarray(raw / np.abs(raw).sum(), dtype=np.float64)


def _long_only_max_sharpe(mu_hat: pd.Series, sigma_hat: pd.DataFrame) -> FloatArray:
    """The walk-forward's long-only max-Sharpe, with the same min-variance fallback."""
    try:
        weights = max_sharpe(mu_hat, sigma_hat, Constraints()).weights
    except ValueError:  # no positive estimated excess return in this draw
        weights = min_variance(sigma_hat, Constraints()).weights
    return np.asarray(weights, dtype=np.float64)


def _long_only_min_variance(_mu_hat: pd.Series, sigma_hat: pd.DataFrame) -> FloatArray:
    """Long-only minimum variance: uses the covariance estimate, ignores the mean."""
    return np.asarray(min_variance(sigma_hat, Constraints()).weights, dtype=np.float64)


def default_estimated_portfolios() -> dict[str, EstimatedPortfolio]:
    """Three points on the "how much does it estimate" spectrum."""
    return {
        "MV tangency (unconstrained)": _sample_tangency,
        "MV max-Sharpe (long-only)": _long_only_max_sharpe,
        "Min variance (long-only)": _long_only_min_variance,
    }


@dataclass(frozen=True)
class WindowExperiment:
    """Output of :func:`run_window_experiment`.

    Attributes:
        table: One row per (window, portfolio): mean, 10th and 90th percentile of the true
            annualised Sharpe across draws, and the share of draws that beat 1/N.
        equal_weight_sharpe: True annualised Sharpe of 1/N (no estimation, so exact).
        tangency_sharpe: True annualised Sharpe of the true tangency portfolio -- the
            ceiling.
        n_reps: Draws per window.
    """

    table: pd.DataFrame
    equal_weight_sharpe: float
    tangency_sharpe: float
    n_reps: int

    def crossover(self, portfolio: str) -> int | None:
        """Shortest window at which ``portfolio``'s mean true Sharpe beats 1/N, if any."""
        rows = self.table[self.table["portfolio"] == portfolio]
        beating = rows[rows["mean_sharpe"] > self.equal_weight_sharpe]
        return int(beating["window"].min()) if len(beating) else None


def run_window_experiment(
    mu: pd.Series,
    sigma: pd.DataFrame,
    windows: list[int],
    n_reps: int = 200,
    seed: int = 0,
    portfolios: dict[str, EstimatedPortfolio] | None = None,
    periods_per_year: float = 12.0,
) -> WindowExperiment:
    """Expected out-of-sample Sharpe of estimated portfolios against the window length.

    Args:
        mu: True per-period **excess** mean returns.
        sigma: True per-period covariance.
        windows: Estimation window lengths, in periods. Each must exceed the asset count so
            the sample covariance is invertible.
        n_reps: Independent draws per window.
        seed: Seed for the random generator; results are reproducible.
        portfolios: Estimated portfolios to evaluate. Defaults to
            :func:`default_estimated_portfolios`.
        periods_per_year: For annualising the Sharpe ratios.

    Raises:
        ValueError: if a window is not longer than the number of assets.
    """
    n_assets = len(mu)
    if min(windows) <= n_assets:
        raise ValueError(f"every window must exceed the {n_assets} assets")
    builders = portfolios or default_estimated_portfolios()
    mu_true = mu.to_numpy(dtype=np.float64)
    sigma_true = sigma.loc[mu.index, mu.index].to_numpy(dtype=np.float64)
    scale = np.sqrt(periods_per_year)

    equal = np.full(n_assets, 1.0 / n_assets)
    equal_sharpe = true_sharpe(equal, mu_true, sigma_true) * scale
    tangency = np.linalg.solve(sigma_true, mu_true)
    tangency_sharpe = abs(true_sharpe(tangency, mu_true, sigma_true)) * scale

    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(sigma_true)
    rows = []
    for window in windows:
        sharpes: dict[str, list[float]] = {name: [] for name in builders}
        for _ in range(n_reps):
            draws = mu_true + rng.standard_normal((window, n_assets)) @ chol.T
            sample = pd.DataFrame(draws, columns=mu.index)
            mu_hat = sample.mean()
            sigma_hat = sample.cov(ddof=1)
            for name, build in builders.items():
                weights = build(mu_hat, sigma_hat)
                sharpes[name].append(true_sharpe(weights, mu_true, sigma_true) * scale)
        for name, values in sharpes.items():
            arr = np.asarray(values)
            rows.append(
                {
                    "window": window,
                    "portfolio": name,
                    "mean_sharpe": float(arr.mean()),
                    "p10_sharpe": float(np.quantile(arr, 0.10)),
                    "p90_sharpe": float(np.quantile(arr, 0.90)),
                    "share_beating_1N": float((arr > equal_sharpe).mean()),
                }
            )
    return WindowExperiment(
        table=pd.DataFrame(rows),
        equal_weight_sharpe=float(equal_sharpe),
        tangency_sharpe=float(tangency_sharpe),
        n_reps=n_reps,
    )
