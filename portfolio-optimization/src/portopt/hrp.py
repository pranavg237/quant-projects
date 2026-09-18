r"""Hierarchical Risk Parity (Lopez de Prado, 2016).

Markowitz inverts the covariance matrix. HRP never does. That is the whole idea.

Inverting :math:`\Sigma` forces every asset to be compared with every other asset
simultaneously, so a single unstable eigenvalue -- and with :math:`N` assets and
:math:`T \approx N` observations there are always several -- propagates into every weight.
HRP replaces the complete graph with a **tree**, so each comparison is local and
estimation error stays local with it.

Three steps.

**1. Tree clustering.** Turn the correlation matrix into a proper distance,

.. math:: d_{ij} = \sqrt{\tfrac12(1 - \rho_{ij})},

which is a genuine metric on :math:`[0,2]`: perfectly correlated assets are at distance 0,
uncorrelated at :math:`1/\sqrt2`, perfectly anti-correlated at 1. Then run hierarchical
clustering on it.

**2. Quasi-diagonalisation.** Reorder the assets by the dendrogram's leaf order. This puts
similar assets next to each other, so the reordered covariance matrix has its large entries
near the diagonal -- without any matrix algebra, and without discarding anything.

**3. Recursive bisection.** Split the ordered list in half, compute each half's variance
under inverse-variance weights, and split capital between the halves in inverse proportion
to those variances:

.. math:: \alpha = 1 - \frac{\tilde V_{\text{left}}}
                             {\tilde V_{\text{left}} + \tilde V_{\text{right}}}.

Recurse. Every allocation decision is between exactly two things, and uses only the
covariance *within* those two things.

The result is between naive risk parity (which ignores correlations entirely) and
mean-variance (which trusts them completely). Lopez de Prado's claim is that it beats both
out of sample. The walk-forward backtest in this repo tests that claim rather than
repeating it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .optimizers import OptimizationResult, portfolio_return, portfolio_volatility
from .types import FloatArray

__all__ = [
    "correlation_distance",
    "hierarchical_risk_parity",
    "quasi_diagonal_order",
]


def correlation_distance(correlation: pd.DataFrame) -> pd.DataFrame:
    r""":math:`d_{ij} = \sqrt{\tfrac12(1-\rho_{ij})}`, a true metric on correlations."""
    values = np.asarray(correlation, dtype=np.float64)
    values = np.clip(0.5 * (values + values.T), -1.0, 1.0)
    distance = np.sqrt(np.clip(0.5 * (1.0 - values), 0.0, None))
    np.fill_diagonal(distance, 0.0)
    return pd.DataFrame(distance, index=correlation.index, columns=correlation.columns)


def quasi_diagonal_order(link: FloatArray) -> list[int]:
    """Leaf order of a SciPy linkage matrix: the quasi-diagonalisation step.

    Implemented directly rather than via ``scipy.cluster.hierarchy.leaves_list`` so the
    algorithm in the paper is visible: repeatedly replace each cluster id with its two
    children until only original items remain.
    """
    link = link.astype(int)
    n_items = int(link[-1, 3])
    order = pd.Series([int(link[-1, 0]), int(link[-1, 1])])

    while order.max() >= n_items:
        # Double the index spacing so each cluster's right child has a slot after it.
        order = pd.Series(order.to_numpy(), index=pd.RangeIndex(0, order.shape[0] * 2, 2))
        clusters = order[order >= n_items]
        positions = clusters.index
        rows = clusters.to_numpy() - n_items
        order[positions] = link[rows, 0]  # left child
        order = pd.concat([order, pd.Series(link[rows, 1], index=positions + 1)])  # right child
        order = order.sort_index()
    return [int(x) for x in order.tolist()]


def _inverse_variance_weights(covariance: FloatArray) -> FloatArray:
    """Weights proportional to the reciprocal of each asset's variance."""
    inverse = 1.0 / np.diag(covariance)
    return np.asarray(inverse / inverse.sum(), dtype=np.float64)


def _cluster_variance(covariance: FloatArray, items: list[int]) -> float:
    """Variance of a sub-portfolio held at inverse-variance weights."""
    block = covariance[np.ix_(items, items)]
    weights = _inverse_variance_weights(block)
    return float(weights @ block @ weights)


def hierarchical_risk_parity(
    covariance: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    risk_free: float = 0.0,
    linkage_method: str = "single",
) -> OptimizationResult:
    """Allocate by Hierarchical Risk Parity.

    Args:
        covariance: Asset covariance matrix.
        expected_returns: Only used to report the portfolio's expected return. HRP, like
            risk parity, never uses expected returns to decide anything.
        risk_free: Only used for the reported Sharpe ratio.
        linkage_method: SciPy linkage method. The paper uses ``"single"``. Single linkage
            is the most sensitive to noise (one close pair can chain two clusters together),
            so ``"ward"`` and ``"average"`` are worth trying; the choice is exposed rather
            than hard-coded because it is a real modelling decision.

    Returns:
        An :class:`~portopt.optimizers.OptimizationResult` with positive weights summing
        to one.

    Raises:
        ValueError: if the covariance matrix has fewer than two assets or a non-positive
            variance.
    """
    if covariance.shape[0] < 2:
        raise ValueError("HRP needs at least 2 assets")
    values = np.asarray(covariance, dtype=np.float64)
    variances = np.diag(values)
    if np.any(variances <= 0):
        raise ValueError("every asset must have positive variance")

    std = np.sqrt(variances)
    correlation = pd.DataFrame(
        values / np.outer(std, std), index=covariance.index, columns=covariance.columns
    )
    distance = correlation_distance(correlation)
    link = linkage(squareform(distance.to_numpy(dtype=np.float64), checks=False), linkage_method)
    order = quasi_diagonal_order(np.asarray(link, dtype=np.float64))

    weights = np.ones(len(order), dtype=np.float64)
    position = {item: i for i, item in enumerate(order)}
    clusters: list[list[int]] = [order]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            midpoint = len(cluster) // 2
            left, right = cluster[:midpoint], cluster[midpoint:]
            variance_left = _cluster_variance(values, left)
            variance_right = _cluster_variance(values, right)
            total = variance_left + variance_right
            alpha = 1.0 - variance_left / total if total > 0 else 0.5
            for item in left:
                weights[position[item]] *= alpha
            for item in right:
                weights[position[item]] *= 1.0 - alpha
            next_clusters.extend([left, right])
        clusters = next_clusters

    ordered_names = [covariance.index[i] for i in order]
    series = pd.Series(weights, index=ordered_names, name="weight")
    series = series.reindex(covariance.index)
    series = series / series.sum()

    expected = portfolio_return(series, expected_returns) if expected_returns is not None else 0.0
    volatility = portfolio_volatility(series, covariance)
    return OptimizationResult(
        weights=series,
        expected_return=expected,
        volatility=volatility,
        sharpe=(expected - risk_free) / volatility if volatility > 0 else float("nan"),
        method=f"HRP ({linkage_method} linkage)",
    )
