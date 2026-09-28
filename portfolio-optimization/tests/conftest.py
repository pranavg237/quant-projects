"""Shared fixtures: synthetic return panels with known properties.

Every test runs offline. Where a result has a closed form, the test checks the code
against it rather than against a number copied from a previous run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_returns(
    n_periods: int = 240, n_assets: int = 6, seed: int = 0, start: str = "2000-01-31"
) -> pd.DataFrame:
    """Monthly returns drawn from a known multivariate normal with distinct means and vols."""
    rng = np.random.default_rng(seed)
    means = np.linspace(0.004, 0.010, n_assets)
    vols = np.linspace(0.03, 0.07, n_assets)
    correlation = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(correlation, 1.0)
    covariance = correlation * np.outer(vols, vols)
    draws = rng.multivariate_normal(means, covariance, size=n_periods)
    index = pd.date_range(start, periods=n_periods, freq="ME")
    columns = [f"A{i}" for i in range(n_assets)]
    return pd.DataFrame(draws, index=index, columns=columns)


@pytest.fixture
def returns() -> pd.DataFrame:
    """240 months of returns for six correlated assets."""
    return make_returns()


@pytest.fixture
def covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Annualised sample covariance of :func:`returns`."""
    return returns.cov() * 12.0


@pytest.fixture
def expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Annualised arithmetic mean returns of :func:`returns`."""
    return returns.mean() * 12.0
