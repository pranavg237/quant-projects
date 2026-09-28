"""Newey-West errors checked against a from-scratch formula and against statsmodels directly."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from ffmodel.regression import (
    compare_standard_errors,
    fit_factor_model,
    holm_adjust,
    newey_west_lags,
    rolling_regression,
)

FF3 = ["Mkt-RF", "SMB", "HML"]


def autocorrelated_asset(factors, seed=7, rho=0.5):
    """Excess returns whose errors are AR(1) and heteroskedastic, so OLS, White and NW all differ."""
    rng = np.random.default_rng(seed)
    n = len(factors)
    shocks = rng.normal(0, 0.01, n) * (1 + 20 * np.abs(factors["Mkt-RF"].to_numpy()))
    errors = np.zeros(n)
    for t in range(1, n):
        errors[t] = rho * errors[t - 1] + shocks[t]
    excess = 0.001 + factors[FF3].to_numpy() @ np.array([1.0, 0.3, -0.2]) + errors
    return pd.Series(excess, index=factors.index, name="asset")


def newey_west_by_hand(X, resid, lags):
    """(X'X)^-1 S (X'X)^-1 with S = sum_l w_l (Gamma_l + Gamma_l'), w_l = 1 - l/(L+1), no df correction."""
    scores = X * resid[:, None]
    S = scores.T @ scores
    for lag in range(1, lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        S += (1 - lag / (lags + 1)) * (gamma + gamma.T)
    bread = np.linalg.inv(X.T @ X)
    return np.sqrt(np.diag(bread @ S @ bread))


@pytest.mark.parametrize("lags", [0, 1, 4, 12])
def test_newey_west_matches_formula_and_statsmodels(factors, lags):
    y = autocorrelated_asset(factors)
    res = fit_factor_model(y, factors, model="ff3", excess=True, cov="hac", lags=lags)
    X = sm.add_constant(factors[FF3].to_numpy())
    np.testing.assert_allclose(res.bse.to_numpy(), newey_west_by_hand(X, res.resid.to_numpy(), lags), rtol=1e-10)

    direct = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    np.testing.assert_allclose(res.bse.to_numpy(), direct.bse, rtol=1e-12)
    np.testing.assert_allclose(res.pvalues.to_numpy(), direct.pvalues, rtol=1e-12)
    assert res.hac_lags == lags


def test_default_lags_follow_the_rule(factors):
    y = autocorrelated_asset(factors)
    res = fit_factor_model(y, factors, model="ff3", excess=True)
    assert res.hac_lags == newey_west_lags(len(factors)) == 5  # T = 600
    X = sm.add_constant(factors[FF3].to_numpy())
    direct = sm.OLS(y.to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    np.testing.assert_allclose(res.bse.to_numpy(), direct.bse, rtol=1e-12)
    assert fit_factor_model(y, factors, model="ff3", excess=True, cov="ols").hac_lags is None


def test_zero_lags_is_white_without_correction(factors):
    y = autocorrelated_asset(factors)
    nw0 = fit_factor_model(y, factors, model="ff3", excess=True, lags=0)
    X = sm.add_constant(factors[FF3].to_numpy())
    np.testing.assert_allclose(nw0.bse.to_numpy(), sm.OLS(y.to_numpy(), X).fit(cov_type="HC0").bse, rtol=1e-12)


def test_lag_rule_thresholds():
    assert [newey_west_lags(t) for t in (36, 60, 99, 100, 259, 272, 273, 620, 621)] == [3, 3, 3, 4, 4, 4, 5, 5, 6]
    with pytest.raises(ValueError):
        fit_factor_model(pd.Series([0.0] * 10), pd.DataFrame(), lags=-1)


def test_positive_autocorrelation_widens_errors(factors):
    """With positively autocorrelated errors, OLS understates uncertainty and NW corrects it."""
    y = autocorrelated_asset(factors, rho=0.6)
    table = compare_standard_errors(fit_factor_model(y, factors, model="ff3", excess=True))
    assert abs(table.loc["alpha", "t (NW, 5 lags)"]) < abs(table.loc["alpha", "t (OLS)"])


def test_compare_standard_errors_columns_match_individual_fits(factors):
    y = autocorrelated_asset(factors)
    res = fit_factor_model(y, factors, model="ff3", excess=True)
    table = compare_standard_errors(res, extra_lags=(12, 5))
    assert list(table.columns) == ["t (OLS)", "t (White HC1)", "t (NW, 5 lags)", "t (NW, 12 lags)",
                                   "p (OLS)", "p (NW, 5 lags)", "significance changes"]
    ols = fit_factor_model(y, factors, model="ff3", excess=True, cov="ols")
    pd.testing.assert_series_equal(table["t (OLS)"], ols.tvalues, check_names=False)
    pd.testing.assert_series_equal(table["t (NW, 5 lags)"], res.tvalues, check_names=False)
    flips = (table["p (OLS)"] < 0.05) != (table["p (NW, 5 lags)"] < 0.05)
    assert list(table["significance changes"] == "yes") == list(flips)


def test_holm_adjustment():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    adjusted = holm_adjust(p)
    np.testing.assert_allclose(adjusted[["a", "b", "c", "d"]].to_numpy(), [0.03, 0.06, 0.06, 0.02])
    assert holm_adjust(pd.Series({"x": 0.6, "y": 0.9})).max() == 1.0


def test_rolling_standard_errors_match_window_fits(factors):
    y = autocorrelated_asset(factors)
    window = 60
    rolled = rolling_regression(y, factors, model="ff3", window=window, excess=True, se="hac")
    assert list(rolled.columns) == ["alpha"] + FF3 + ["R2"] + [f"se({c})" for c in ["alpha"] + FF3]
    assert len(rolled) == len(factors) - window + 1
    for end in (window, len(factors)):
        chunk = slice(end - window, end)
        X = sm.add_constant(factors[FF3].to_numpy()[chunk])
        fit = sm.OLS(y.to_numpy()[chunk], X).fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(window)})
        row = rolled.loc[factors.index[end - 1]]
        np.testing.assert_allclose(row[["alpha"] + FF3].to_numpy(dtype=float), fit.params, rtol=1e-8)
        np.testing.assert_allclose(row[[f"se({c})" for c in ["alpha"] + FF3]].to_numpy(dtype=float), fit.bse,
                                   rtol=1e-10)
    no_se = rolling_regression(y, factors, model="ff3", window=window, excess=True)
    assert not any(c.startswith("se(") for c in no_se.columns)

    hc3 = rolling_regression(y, factors, model="ff3", window=window, excess=True, se="hc3")
    X = sm.add_constant(factors[FF3].to_numpy()[-window:])
    expected = sm.OLS(y.to_numpy()[-window:], X).fit(cov_type="HC3").bse
    np.testing.assert_allclose(hc3.iloc[-1][[f"se({c})" for c in ["alpha"] + FF3]].to_numpy(dtype=float), expected,
                               rtol=1e-10)
    with pytest.raises(ValueError):
        fit_factor_model(y, factors, model="ff3", excess=True, cov="hc3")  # windows only, not a --cov choice
