"""Covariance estimation: Ledoit-Wolf against scikit-learn and the endpoints of shrinkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf

from portopt import covariance as cov


def test_sample_covariance_matches_pandas(returns: pd.DataFrame) -> None:
    np.testing.assert_allclose(cov.sample_covariance(returns, 12.0), returns.cov() * 12.0)


def test_ledoit_wolf_identity_target_matches_sklearn(returns: pd.DataFrame) -> None:
    # sklearn implements the identity-target estimator with the 1/T (maximum-likelihood)
    # convention; ours rescales to 1/(T-1), so undo that before comparing.
    reference = LedoitWolf().fit(returns.to_numpy())
    shrunk, delta = cov.ledoit_wolf(returns, "identity")
    t_obs = len(returns)
    assert delta == pytest.approx(reference.shrinkage_, abs=1e-12)
    np.testing.assert_allclose(
        shrunk.to_numpy() * (t_obs - 1) / t_obs, reference.covariance_, atol=1e-14
    )


@pytest.mark.parametrize("target", ["identity", "constant_correlation"])
def test_shrinkage_endpoints(returns: pd.DataFrame, target: str) -> None:
    t_obs = len(returns)
    none, _ = cov.ledoit_wolf(returns, target, delta=0.0)
    full, _ = cov.ledoit_wolf(returns, target, delta=1.0)
    _, sample_ml, target_matrix = cov.ledoit_wolf_shrinkage_intensity(returns, target)
    scale = t_obs / (t_obs - 1)
    np.testing.assert_allclose(none, sample_ml * scale, atol=1e-15)
    np.testing.assert_allclose(full, target_matrix * scale, atol=1e-15)


def test_estimated_intensity_is_a_fraction(returns: pd.DataFrame) -> None:
    delta, _, _ = cov.ledoit_wolf_shrinkage_intensity(returns, "constant_correlation")
    assert 0.0 <= delta <= 1.0


def test_shrinkage_increases_when_data_is_scarce() -> None:
    # Fewer observations per asset means a noisier sample matrix, so more shrinkage. The
    # identity target is used because the synthetic assets have different volatilities, so
    # it is misspecified and shrinkage only wins when the sample matrix is noisy. (Against
    # the constant-correlation target, which is exactly true for this data, delta is 1.)
    from conftest import make_returns

    sample = make_returns(n_periods=600, n_assets=10, seed=1)
    deltas = [cov.ledoit_wolf(sample.iloc[:n], "identity")[1] for n in (30, 60, 600)]
    assert deltas[0] > deltas[1] > deltas[2]


def test_constant_correlation_target_keeps_variances(returns: pd.DataFrame) -> None:
    sample = returns.cov()
    target = cov.constant_correlation_target(sample)
    np.testing.assert_allclose(np.diag(target), np.diag(sample))
    std = np.sqrt(np.diag(sample))
    implied = target.to_numpy() / np.outer(std, std)
    off_diagonal = implied[~np.eye(len(std), dtype=bool)]
    assert np.ptp(off_diagonal) < 1e-12  # one correlation everywhere


def test_unknown_target_raises(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown target"):
        cov.ledoit_wolf(returns, "diagonal")


def test_ewma_with_a_huge_half_life_is_the_sample_covariance(returns: pd.DataFrame) -> None:
    ewma = cov.exponentially_weighted_covariance(returns, half_life=1e9)
    np.testing.assert_allclose(ewma, returns.cov(), rtol=1e-6)


def test_ewma_rejects_bad_half_life(returns: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="half_life"):
        cov.exponentially_weighted_covariance(returns, half_life=0)


def test_diagnostics_of_the_identity() -> None:
    identity = pd.DataFrame(np.eye(4))
    report = cov.diagnose(identity, n_observations=40)
    assert report.condition_number == pytest.approx(1.0)
    assert report.effective_rank == pytest.approx(4.0)
    assert report.aspect_ratio == pytest.approx(0.1)
    assert report.is_psd


def test_singular_sample_when_assets_exceed_observations() -> None:
    from conftest import make_returns

    wide = make_returns(n_periods=5, n_assets=8, seed=2)
    report = cov.diagnose(cov.sample_covariance(wide), n_observations=5)
    assert report.condition_number > 1e10 or report.condition_number == float("inf")


def test_repair_makes_a_matrix_positive_definite() -> None:
    broken = pd.DataFrame([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    assert not cov.diagnose(broken, 10).is_psd
    fixed = cov.repair(broken)
    assert np.linalg.eigvalsh(fixed.to_numpy()).min() > 0


def _ledoit_wolf_2003_reference(values: np.ndarray) -> float:
    """Constant-correlation intensity written element by element from Ledoit-Wolf (2003).

    Deliberately loop-based and in the paper's notation (pi_ij, theta_ii,ij, the r_bar/2
    symmetric form of rho), so it shares no vectorised code with the implementation.
    """
    t_obs, n = values.shape
    x = values - values.mean(axis=0)
    s = x.T @ x / t_obs
    r = s / np.sqrt(np.outer(np.diag(s), np.diag(s)))
    r_bar = sum(r[i, j] for i in range(n) for j in range(n) if i < j) * 2 / (n * (n - 1))
    f = np.empty_like(s)
    for i in range(n):
        for j in range(n):
            f[i, j] = s[i, i] if i == j else r_bar * np.sqrt(s[i, i] * s[j, j])

    def pi_ij(i: int, j: int) -> float:
        return float(np.mean((x[:, i] * x[:, j] - s[i, j]) ** 2))

    def theta(k: int, i: int, j: int) -> float:  # theta_{kk,ij}
        return float(np.mean((x[:, k] ** 2 - s[k, k]) * (x[:, i] * x[:, j] - s[i, j])))

    pi_hat = sum(pi_ij(i, j) for i in range(n) for j in range(n))
    rho_hat = sum(pi_ij(i, i) for i in range(n))
    for i in range(n):
        for j in range(n):
            if i != j:
                rho_hat += (r_bar / 2) * (
                    np.sqrt(s[j, j] / s[i, i]) * theta(i, i, j)
                    + np.sqrt(s[i, i] / s[j, j]) * theta(j, i, j)
                )
    gamma_hat = float(np.sum((f - s) ** 2))
    kappa = (pi_hat - rho_hat) / gamma_hat
    return float(max(0.0, min(1.0, kappa / t_obs)))


@pytest.mark.parametrize(("n_periods", "n_assets", "seed"), [(60, 6, 3), (36, 10, 4), (240, 5, 5)])
def test_constant_correlation_intensity_matches_the_2003_formula(
    n_periods: int, n_assets: int, seed: int
) -> None:
    from conftest import make_returns

    # Add a factor with mixed-sign loadings so correlations differ, the target is
    # misspecified, and delta is interior rather than pinned at 1.
    base = make_returns(n_periods=n_periods, n_assets=n_assets, seed=seed)
    rng = np.random.default_rng(seed)
    factor = rng.standard_normal((n_periods, 1)) * rng.uniform(-1, 1, n_assets)
    panel = base + 0.04 * factor
    delta, _, _ = cov.ledoit_wolf_shrinkage_intensity(panel, "constant_correlation")
    assert 0.0 < delta < 1.0
    assert delta == pytest.approx(_ledoit_wolf_2003_reference(panel.to_numpy()), abs=1e-12)


def test_constant_correlation_intensity_is_close_to_the_oracle() -> None:
    # The oracle intensity minimises E||delta F + (1 - delta) S - Sigma||_F^2 with Sigma
    # known: delta* = E<S - F, S - Sigma> / E||S - F||^2, estimated by simulation. The
    # Ledoit-Wolf estimate should be close to it on average. Dropping the rho term (the
    # previous implementation) over-shrinks by ~0.04 here and fails this check.
    rng = np.random.default_rng(0)
    n_assets, t_obs, reps = 10, 60, 300
    loadings = rng.normal(size=(n_assets, 3))
    raw = loadings @ loadings.T + np.diag(rng.uniform(0.5, 2.0, n_assets))
    scale = np.sqrt(np.diag(raw))
    vols = np.linspace(0.03, 0.08, n_assets)
    sigma = raw / np.outer(scale, scale) * np.outer(vols, vols)
    numerator = denominator = 0.0
    estimates = []
    for _ in range(reps):
        panel = pd.DataFrame(rng.multivariate_normal(np.zeros(n_assets), sigma, size=t_obs))
        delta, sample, target = cov.ledoit_wolf_shrinkage_intensity(panel, "constant_correlation")
        gap = sample.to_numpy() - target.to_numpy()
        numerator += float(np.sum(gap * (sample.to_numpy() - sigma)))
        denominator += float(np.sum(gap**2))
        estimates.append(delta)
    oracle = numerator / denominator
    assert 0.05 < oracle < 0.5  # interior, so the check has teeth
    assert np.mean(estimates) == pytest.approx(oracle, abs=0.02)
