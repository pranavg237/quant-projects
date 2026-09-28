import numpy as np
import pandas as pd
import pytest

from ffmodel.regression import (
    compare_models,
    fit_factor_model,
    fit_many,
    infer_periods_per_year,
    newey_west_lags,
    rolling_regression,
    summarize,
)

FF5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
TRUE_BETAS = np.array([1.1, 0.4, -0.3, 0.2, 0.1])
TRUE_ALPHA = 0.002


def simulate_asset(factors, rng, noise=0.01):
    excess = TRUE_ALPHA + factors[FF5].to_numpy() @ TRUE_BETAS + rng.normal(0, noise, len(factors))
    return pd.Series(excess, index=factors.index, name="asset") + factors["RF"]


def test_recovers_alpha_and_betas(factors, rng):
    result = fit_factor_model(simulate_asset(factors, rng), factors, model="ff5")
    assert result.alpha == pytest.approx(TRUE_ALPHA, abs=0.0015)
    np.testing.assert_allclose(result.betas.to_numpy(), TRUE_BETAS, atol=0.1)
    assert result.rsquared > 0.9
    assert result.nobs == len(factors)
    assert result.periods_per_year == 12
    assert result.alpha_annual == pytest.approx(result.alpha * 12)


def test_excess_flag_is_equivalent(factors, rng):
    raw = simulate_asset(factors, rng)
    a = fit_factor_model(raw, factors, model="ff3")
    b = fit_factor_model(raw - factors["RF"], factors, model="ff3", excess=True)
    pd.testing.assert_series_equal(a.params, b.params)


def test_covariance_choice_changes_errors_not_estimates(factors, rng):
    asset = simulate_asset(factors, rng)
    ols = fit_factor_model(asset, factors, cov="ols")
    hac = fit_factor_model(asset, factors, cov="hac")
    pd.testing.assert_series_equal(ols.params, hac.params)
    assert not np.allclose(ols.bse, hac.bse)
    with pytest.raises(ValueError):
        fit_factor_model(asset, factors, cov="bogus")


def test_missing_rf_requires_excess(factors, rng):
    asset = simulate_asset(factors, rng)
    with pytest.raises(ValueError, match="RF"):
        fit_factor_model(asset, factors.drop(columns="RF"))


def test_frequency_inference_and_lag_rule():
    assert infer_periods_per_year(pd.bdate_range("2020-01-01", periods=50)) == 252
    assert infer_periods_per_year(pd.date_range("2020-01-31", periods=50, freq="ME")) == 12
    assert infer_periods_per_year(pd.date_range("2020-12-31", periods=10, freq="YE")) == 1
    assert newey_west_lags(600) == 5
    assert newey_west_lags(100) == 4


def test_rolling_regression_tracks_constant_betas(factors, rng):
    rolled = rolling_regression(simulate_asset(factors, rng), factors, model="ff5", window=120)
    assert len(rolled) == len(factors) - 120 + 1
    assert list(rolled.columns) == ["alpha"] + FF5 + ["R2"]
    np.testing.assert_allclose(rolled[FF5].mean().to_numpy(), TRUE_BETAS, atol=0.1)


def test_compare_models_nested_r2(factors, rng):
    asset = simulate_asset(factors, rng)
    sets = {m: factors for m in ("capm", "ff3", "ff5", "ff6")}
    table, results = compare_models(asset, sets)
    assert list(table.index) == ["capm", "ff3", "ff5", "ff6"]
    assert table["R2"].is_monotonic_increasing
    assert np.isnan(table.loc["capm", "SMB"])
    assert list(table.columns[:2]) == ["alpha (ann.)", "t(alpha)"]


def test_fit_many_and_summarize(factors, rng):
    returns = pd.DataFrame({"a": simulate_asset(factors, rng), "b": simulate_asset(factors, rng)})
    returns["short"] = np.nan
    returns.iloc[:3, 2] = 0.01
    with pytest.warns(UserWarning, match="short"):
        results = fit_many(returns, factors, model="ff3")
    assert set(results) == {"a", "b"}
    table = summarize(results)
    assert list(table.index) == ["a", "b"]
    assert {"Mkt-RF", "SMB", "HML", "IR"} <= set(table.columns)
