import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from ffmodel.asset_pricing import fama_macbeth, grs_test

FF3 = ["Mkt-RF", "SMB", "HML"]


def simulate_portfolios(factors, rng, n=10, alpha=0.0, noise=0.02):
    betas = rng.uniform([0.6, -0.5, -0.5], [1.5, 1.2, 1.0], size=(n, 3))
    f = factors[FF3].to_numpy()
    r = alpha + f @ betas.T + rng.normal(0, noise, (len(factors), n))
    return pd.DataFrame(r, index=factors.index, columns=[f"P{i}" for i in range(n)]), betas


def test_grs_matches_textbook_formula(factors, rng):
    R, _ = simulate_portfolios(factors.iloc[:240], rng, alpha=0.001)
    F = factors.loc[R.index, FF3]
    result = grs_test(R, F)

    # Independent computation: per-asset statsmodels OLS, unbiased residual
    # covariance, GRS (1989) scaling T/N * (T-N-L)/(T-L-1).
    T, N, L = len(R), R.shape[1], 3
    fits = [sm.OLS(R[c], sm.add_constant(F)).fit() for c in R]
    alpha = np.array([m.params["const"] for m in fits])
    E = np.column_stack([m.resid for m in fits])
    sigma = E.T @ E / (T - L - 1)
    mu = F.mean().to_numpy()
    omega = np.cov(F.to_numpy(), rowvar=False, ddof=0)
    expected = T / N * (T - N - L) / (T - L - 1) * alpha @ np.linalg.solve(sigma, alpha) / (1 + mu @ np.linalg.solve(omega, mu))

    assert result.statistic == pytest.approx(expected, rel=1e-8)
    assert result.pvalue == pytest.approx(stats.f.sf(expected, N, T - N - L))
    np.testing.assert_allclose(result.alpha_tstats, [m.tvalues["const"] for m in fits], rtol=1e-8)
    assert (result.df_num, result.df_den) == (N, T - N - L)


def test_grs_size_under_null_and_power_under_alternative():
    rng = np.random.default_rng(7)
    from conftest import make_factors

    pvalues = []
    for seed in range(300):
        f = make_factors(T=120, seed=seed)
        R, _ = simulate_portfolios(f, rng, alpha=0.0)
        pvalues.append(grs_test(R, f[FF3]).pvalue)
    rejection_rate = np.mean(np.array(pvalues) < 0.05)
    assert 0.02 < rejection_rate < 0.09

    f = make_factors(T=240, seed=1)
    R, _ = simulate_portfolios(f, rng, alpha=0.005)
    assert grs_test(R, f[FF3]).pvalue < 0.001


def test_grs_requires_enough_periods(factors, rng):
    R, _ = simulate_portfolios(factors.iloc[:12], rng, n=10)
    with pytest.raises(ValueError, match="T > N"):
        grs_test(R, factors[FF3])


def test_fama_macbeth_recovers_premia(factors, rng):
    R, true_betas = simulate_portfolios(factors, rng, n=25, noise=0.01)
    F = factors[FF3]
    fm = fama_macbeth(R, F, intercept=False)
    np.testing.assert_allclose(fm.lambdas.to_numpy(), F.mean().to_numpy(), atol=0.002)
    np.testing.assert_allclose(fm.betas.to_numpy(), true_betas, atol=0.1)
    assert (fm.se_shanken >= fm.se).all()

    with_const = fama_macbeth(R, F, intercept=True)
    assert abs(with_const.tstats_shanken["const"]) < 3
    assert list(with_const.table().columns) == ["lambda (ann.)", "t (FM)", "t (Shanken)", "factor mean (ann.)"]
    assert with_const.cs_rsquared > 0.5


def test_fama_macbeth_newey_west_and_rolling(factors, rng):
    R, _ = simulate_portfolios(factors, rng, n=25, noise=0.01)
    F = factors[FF3]
    classic = fama_macbeth(R, F)
    nw = fama_macbeth(R, F, lags=None)
    pd.testing.assert_series_equal(classic.lambdas, nw.lambdas)
    assert not np.allclose(classic.se, nw.se)

    rolling = fama_macbeth(R, F, beta_window=60)
    assert rolling.nobs == len(F) - 60
    assert rolling.se_shanken is None
    assert "t (Shanken)" not in rolling.table()
