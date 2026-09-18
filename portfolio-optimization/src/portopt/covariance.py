r"""Covariance estimation, including Ledoit-Wolf shrinkage.

The sample covariance matrix is unbiased, and that is almost the only good thing about it.
With :math:`N` assets and :math:`T` observations it has :math:`N(N+1)/2` parameters
estimated from :math:`NT` numbers, so when :math:`T` is not much larger than :math:`N` its
**extreme** eigenvalues are badly biased: the largest are too large, the smallest too small.
A mean-variance optimiser then does exactly the wrong thing with it. Minimising
:math:`w'\Sigma w` means loading up on whatever direction the estimator says has the least
variance -- which, when the small eigenvalues are biased downwards, is precisely the
direction where the estimate is least reliable. Michaud called optimisers
"estimation-error maximisers" for this reason.

**Shrinkage** (Ledoit and Wolf, 2004) pulls the sample matrix towards a heavily structured
target:

.. math:: \hat\Sigma = \delta F + (1-\delta) S,

and derives the :math:`\delta` that minimises expected squared Frobenius distance to the
true covariance. The estimator is biased and much lower variance, which is the right trade
for an input to an optimiser. It is also always well conditioned and invertible even when
:math:`T < N`, where :math:`S` is singular by construction.

Two targets are implemented:

* ``identity`` -- :math:`F = \bar\mu I` with :math:`\bar\mu = \mathrm{tr}(S)/N`. This is
  Ledoit-Wolf 2004, the version in ``sklearn``, and the tests check agreement with it.
* ``constant_correlation`` -- every pairwise correlation replaced by the average pairwise
  correlation, keeping the sample variances. Ledoit-Wolf 2003. This is usually the better
  target for equities, because a common correlation is a much more plausible description of
  a stock universe than "everything is uncorrelated".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import nearest_positive_definite

__all__ = [
    "CovarianceDiagnostics",
    "constant_correlation_target",
    "diagnose",
    "exponentially_weighted_covariance",
    "ledoit_wolf",
    "ledoit_wolf_shrinkage_intensity",
    "repair",
    "sample_covariance",
    "shrink_to_target",
]


def sample_covariance(returns: pd.DataFrame, annualise: float = 1.0) -> pd.DataFrame:
    r"""Plain sample covariance, :math:`S = \frac{1}{T-1}\sum_t (x_t-\bar x)(x_t-\bar x)'`.

    Args:
        returns: Asset returns, observations on the index.
        annualise: Multiply by this (e.g. 12 for monthly, 252 for daily).

    Raises:
        ValueError: if there are fewer than two observations.
    """
    if len(returns) < 2:
        raise ValueError("need at least 2 observations to estimate a covariance")
    return returns.cov(ddof=1) * annualise


def exponentially_weighted_covariance(
    returns: pd.DataFrame, half_life: float = 60.0, annualise: float = 1.0
) -> pd.DataFrame:
    r"""Exponentially weighted covariance with the given half-life, in observations.

    Weights decay as :math:`\lambda^{k}` with :math:`\lambda = 2^{-1/h}`, so recent
    observations dominate. This is the standard answer to non-stationarity -- correlations
    genuinely move, and 2008 tells you little about 2024 -- but it cuts the *effective*
    sample size to roughly :math:`h/\ln 2`, which makes the estimation-error problem worse
    at the same time as it makes the staleness problem better. Reported here so the
    trade-off can be measured rather than assumed away.
    """
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    if len(returns) < 2:
        raise ValueError("need at least 2 observations to estimate a covariance")
    decay = 0.5 ** (1.0 / half_life)
    n = len(returns)
    weights = decay ** np.arange(n - 1, -1, -1)
    weights = weights / weights.sum()

    values = returns.to_numpy(dtype=np.float64)
    mean = weights @ values
    centred = values - mean
    # Unbiased-style correction for weighted samples: divide by 1 - sum(w^2).
    covariance = (centred * weights[:, None]).T @ centred / (1.0 - np.sum(weights**2))
    return pd.DataFrame(covariance, index=returns.columns, columns=returns.columns) * annualise


def constant_correlation_target(sample: pd.DataFrame) -> pd.DataFrame:
    r"""Ledoit-Wolf 2003 target: sample variances, one common correlation.

    :math:`F_{ij} = \bar\rho\sqrt{s_{ii}s_{jj}}` off the diagonal and :math:`s_{ii}` on it,
    with :math:`\bar\rho` the average sample pairwise correlation.
    """
    std = np.sqrt(np.diag(sample.to_numpy(dtype=np.float64)))
    outer = np.outer(std, std)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = sample.to_numpy(dtype=np.float64) / outer
    n = correlation.shape[0]
    upper = np.triu_indices(n, k=1)
    mean_correlation = float(np.nanmean(correlation[upper]))
    target = mean_correlation * outer
    np.fill_diagonal(target, np.diag(sample.to_numpy(dtype=np.float64)))
    return pd.DataFrame(target, index=sample.index, columns=sample.columns)


def ledoit_wolf_shrinkage_intensity(
    returns: pd.DataFrame, target: str = "identity"
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    r"""The optimal shrinkage intensity :math:`\delta^*`, plus the sample matrix and target.

    For the identity target the Ledoit-Wolf (2004) estimator is

    .. math::
        m = \frac{\langle S, I\rangle}{N},\quad
        d^2 = \frac{\|S - mI\|_F^2}{N},\quad
        \bar b^2 = \frac{1}{T^2}\sum_t \frac{\|x_tx_t' - S\|_F^2}{N},

    .. math::
        b^2 = \min(\bar b^2, d^2),\qquad \delta^* = \frac{b^2}{d^2}.

    :math:`d^2` measures how far the sample matrix is from the target and :math:`\bar b^2`
    measures how noisy the sample matrix is. Shrink hard when the estimate is noisy relative
    to how much structure you would be throwing away. The :math:`\min` keeps
    :math:`\delta^* \le 1`.

    For the constant-correlation target the same decomposition is used with :math:`F` in
    place of :math:`mI`, which is the practical form of Ledoit-Wolf (2003).

    Returns:
        ``(delta, sample, target)``.

    Raises:
        ValueError: on an unknown target or too few observations.
    """
    if target not in {"identity", "constant_correlation"}:
        raise ValueError(f"unknown target {target!r}")
    if len(returns) < 2:
        raise ValueError("need at least 2 observations to estimate a covariance")

    values = returns.to_numpy(dtype=np.float64)
    t_obs, n_assets = values.shape
    centred = values - values.mean(axis=0)
    # Maximum-likelihood (1/T) scaling: the Ledoit-Wolf derivation is stated for it.
    sample_ml = centred.T @ centred / t_obs
    sample_df = pd.DataFrame(sample_ml, index=returns.columns, columns=returns.columns)

    if target == "identity":
        m = float(np.trace(sample_ml) / n_assets)
        target_matrix = m * np.eye(n_assets)
    else:
        target_matrix = constant_correlation_target(sample_df).to_numpy(dtype=np.float64)

    d_squared = float(np.sum((sample_ml - target_matrix) ** 2) / n_assets)
    # b_bar^2: the average squared deviation of the per-observation outer products from S.
    squared = (centred**2).T @ (centred**2) / t_obs
    b_bar_squared = float(np.sum(squared - sample_ml**2) / (n_assets * t_obs))
    b_squared = min(b_bar_squared, d_squared)
    delta = 0.0 if d_squared <= 0 else float(np.clip(b_squared / d_squared, 0.0, 1.0))

    return (
        delta,
        sample_df,
        pd.DataFrame(target_matrix, index=returns.columns, columns=returns.columns),
    )


def shrink_to_target(sample: pd.DataFrame, target: pd.DataFrame, delta: float) -> pd.DataFrame:
    r""":math:`\delta F + (1-\delta) S`, with ``delta`` clipped to :math:`[0,1]`."""
    weight = float(np.clip(delta, 0.0, 1.0))
    return weight * target + (1.0 - weight) * sample


def ledoit_wolf(
    returns: pd.DataFrame,
    target: str = "constant_correlation",
    annualise: float = 1.0,
    delta: float | None = None,
) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrunk covariance.

    Args:
        returns: Asset returns.
        target: ``"identity"`` or ``"constant_correlation"``.
        annualise: Scaling factor applied at the end.
        delta: Override the estimated shrinkage intensity. Used by the tests to check the
            endpoints, and occasionally useful for sensitivity analysis.

    Returns:
        ``(covariance, delta)``. The covariance is rescaled from the maximum-likelihood
        ``1/T`` convention back to the unbiased ``1/(T-1)`` one, so it is directly
        comparable with :func:`sample_covariance`.
    """
    estimated_delta, sample, target_matrix = ledoit_wolf_shrinkage_intensity(returns, target)
    used = estimated_delta if delta is None else float(np.clip(delta, 0.0, 1.0))
    shrunk = shrink_to_target(sample, target_matrix, used)
    t_obs = len(returns)
    return shrunk * (t_obs / (t_obs - 1)) * annualise, used


@dataclass(frozen=True)
class CovarianceDiagnostics:
    """How trustworthy a covariance estimate is.

    Attributes:
        n_assets: Matrix dimension.
        n_observations: Sample size used.
        aspect_ratio: ``n_assets / n_observations``. Above ~0.2 the sample eigenvalues are
            already badly biased; at 1.0 the sample matrix is singular.
        condition_number: Largest eigenvalue over smallest. This is the amplification factor
            from input noise to optimiser output: a condition number of 10,000 means a 1%
            error in a return estimate can move weights by 100%.
        min_eigenvalue: Smallest eigenvalue.
        effective_rank: ``exp`` of the entropy of the normalised eigenvalues. How many
            directions the matrix genuinely distinguishes, which is usually far fewer than
            ``n_assets`` for equities.
        is_psd: Whether no eigenvalue is meaningfully negative.
    """

    n_assets: int
    n_observations: int
    aspect_ratio: float
    condition_number: float
    min_eigenvalue: float
    effective_rank: float
    is_psd: bool

    def __str__(self) -> str:
        return (
            f"{self.n_assets} assets / {self.n_observations} obs "
            f"(ratio {self.aspect_ratio:.2f}) | condition {self.condition_number:,.0f} | "
            f"effective rank {self.effective_rank:.1f} | min eig {self.min_eigenvalue:.2e}"
        )


def diagnose(covariance: pd.DataFrame, n_observations: int) -> CovarianceDiagnostics:
    """Compute :class:`CovarianceDiagnostics` for an estimated covariance matrix."""
    values = np.asarray(covariance, dtype=np.float64)
    values = 0.5 * (values + values.T)
    eigenvalues = np.linalg.eigvalsh(values)
    positive = eigenvalues[eigenvalues > 0]
    if positive.size:
        share = positive / positive.sum()
        effective_rank = float(np.exp(-np.sum(share * np.log(share))))
    else:  # pragma: no cover - a covariance with no positive eigenvalue is degenerate
        effective_rank = 0.0
    smallest = float(eigenvalues.min())
    condition = float(eigenvalues.max() / smallest) if smallest > 0 else float("inf")
    n_assets = values.shape[0]
    return CovarianceDiagnostics(
        n_assets=n_assets,
        n_observations=n_observations,
        aspect_ratio=n_assets / max(n_observations, 1),
        condition_number=condition,
        min_eigenvalue=smallest,
        effective_rank=effective_rank,
        is_psd=smallest > -1e-10,
    )


def repair(covariance: pd.DataFrame) -> pd.DataFrame:
    """Clip negative eigenvalues so the matrix is usable. See the note in ``types``."""
    return nearest_positive_definite(covariance)
