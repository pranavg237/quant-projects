"""The estimation-window simulation: exact benchmarks, the ceiling, and convergence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portopt import estimation_error as ee


def _world() -> tuple[pd.Series, pd.DataFrame]:
    """Four assets with distinct Sharpe ratios, so 1/N is strictly below the ceiling."""
    names = ["A", "B", "C", "D"]
    vols = np.array([0.02, 0.03, 0.04, 0.05])
    correlation = np.full((4, 4), 0.3)
    np.fill_diagonal(correlation, 1.0)
    sigma = pd.DataFrame(correlation * np.outer(vols, vols), index=names, columns=names)
    mu = pd.Series([0.004, 0.002, 0.005, 0.003], index=names)
    return mu, sigma


def _tangency_only() -> dict[str, ee.EstimatedPortfolio]:
    return {"tangency": ee.default_estimated_portfolios()["MV tangency (unconstrained)"]}


def test_true_sharpe_is_scale_invariant() -> None:
    mu, sigma = _world()
    w = np.array([0.1, 0.2, 0.3, 0.4])
    base = ee.true_sharpe(w, mu.to_numpy(), sigma.to_numpy())
    assert ee.true_sharpe(3.0 * w, mu.to_numpy(), sigma.to_numpy()) == pytest.approx(base)
    expected = w @ mu.to_numpy() / np.sqrt(w @ sigma.to_numpy() @ w)
    assert base == pytest.approx(expected)


def test_benchmarks_are_exact() -> None:
    mu, sigma = _world()
    result = ee.run_window_experiment(mu, sigma, [20], n_reps=5, portfolios=_tangency_only())
    m, s = mu.to_numpy(), sigma.to_numpy()
    equal = np.full(4, 0.25)
    assert result.equal_weight_sharpe == pytest.approx(ee.true_sharpe(equal, m, s) * np.sqrt(12))
    ceiling = np.sqrt(m @ np.linalg.solve(s, m)) * np.sqrt(12)
    assert result.tangency_sharpe == pytest.approx(ceiling)
    assert result.tangency_sharpe > result.equal_weight_sharpe


def test_no_estimated_portfolio_beats_the_true_tangency() -> None:
    mu, sigma = _world()
    result = ee.run_window_experiment(mu, sigma, [10, 40], n_reps=30, seed=1)
    assert (result.table["p90_sharpe"] <= result.tangency_sharpe + 1e-12).all()


def test_estimated_tangency_converges_to_the_ceiling_as_the_window_grows() -> None:
    mu, sigma = _world()
    result = ee.run_window_experiment(
        mu, sigma, [12, 120, 20_000], n_reps=40, seed=2, portfolios=_tangency_only()
    )
    means = result.table.set_index("window")["mean_sharpe"]
    assert means[12] < means[120] < means[20_000]
    assert means[20_000] == pytest.approx(result.tangency_sharpe, abs=0.01)
    assert result.crossover("tangency") in (120, 20_000)
    assert result.crossover("no such portfolio") is None


def test_results_are_reproducible_from_the_seed() -> None:
    mu, sigma = _world()
    a = ee.run_window_experiment(mu, sigma, [30], n_reps=10, seed=3)
    b = ee.run_window_experiment(mu, sigma, [30], n_reps=10, seed=3)
    pd.testing.assert_frame_equal(a.table, b.table)


def test_window_must_exceed_the_asset_count() -> None:
    mu, sigma = _world()
    with pytest.raises(ValueError, match="exceed"):
        ee.run_window_experiment(mu, sigma, [4], n_reps=1)
