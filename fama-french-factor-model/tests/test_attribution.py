import numpy as np
import pandas as pd
import pytest

from ffmodel.attribution import attribute_returns
from ffmodel.regression import fit_factor_model


def test_components_add_up(factors, rng):
    excess = 0.001 + factors[["Mkt-RF", "HML"]].to_numpy() @ [0.9, 0.5] + rng.normal(0, 0.01, len(factors))
    asset = pd.Series(excess, index=factors.index)
    result = fit_factor_model(asset, factors, model="ff3", excess=True)
    attribution = attribute_returns(result, factors)

    pd.testing.assert_series_equal(attribution.contributions.sum(axis=1), asset, check_names=False)
    assert list(attribution.contributions.columns) == ["alpha", "Mkt-RF", "SMB", "HML", "residual"]

    summary = attribution.summary
    body = summary.drop(index="total")
    assert body["share of return"].sum() == pytest.approx(1.0)
    assert body["share of variance"].sum() == pytest.approx(1.0)
    assert summary.loc["total", "return (ann.)"] == pytest.approx(asset.mean() * 12)
    assert summary.loc["residual", "return (ann.)"] == pytest.approx(0.0, abs=1e-12)
    assert attribution.cumulative().iloc[-1].sum() == pytest.approx(asset.sum())
