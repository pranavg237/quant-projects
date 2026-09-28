"""Multi-run comparisons, common random numbers and sensitivity sweeps."""

from __future__ import annotations

import numpy as np
import pytest

from mmsim.avellaneda_stoikov import AvellanedaStoikovParams
from mmsim.experiments import (
    build_policy_set,
    compare_policies_book,
    compare_policies_reference,
    paired_test,
    sensitivity_sweep,
)
from mmsim.flow import FlowConfig
from mmsim.strategies import AvellanedaStoikovPolicy, SymmetricPolicy, average_optimal_spread
from mmsim.types import MarketConfig


def test_policy_set_is_matched_on_spread(stationary_params: AvellanedaStoikovParams) -> None:
    policies = build_policy_set(stationary_params, inventory_limit=3.0)
    assert len(policies) == 3
    expected_half = average_optimal_spread(stationary_params) / 2.0
    assert policies[1].half_spread == pytest.approx(expected_half)  # type: ignore[union-attr]
    assert policies[2].half_spread == pytest.approx(expected_half)  # type: ignore[union-attr]
    assert policies[2].max_inventory == 3.0  # type: ignore[union-attr]


def test_common_random_numbers_pair_the_runs(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    """Run i must face the same price path for every policy -- that is the whole point."""
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)
    comparison = compare_policies_reference(
        build_policy_set(stationary_params, inventory_limit=3.0), world, n_runs=25
    )
    names = list(comparison.runs)
    first, second = comparison.runs[names[0]], comparison.runs[names[1]]
    # Same seed => same Brownian increments, even though the fills differ.
    for a, b in zip(first, second, strict=True):
        assert np.allclose(a.mid, b.mid)
    assert not np.allclose(first[0].inventory, second[0].inventory)


def test_pairing_reduces_the_variance_of_the_comparison(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    """The reason common random numbers are worth the trouble, measured.

    The paired standard error must be smaller than the unpaired one, because the shared
    price path is common to both and cancels in the difference.
    """
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)
    comparison = compare_policies_reference(
        build_policy_set(stationary_params, inventory_limit=3.0)[:2], world, n_runs=200
    )
    a = comparison.pnl.iloc[:, 0].to_numpy()
    b = comparison.pnl.iloc[:, 1].to_numpy()
    paired_se = float((a - b).std(ddof=1) / np.sqrt(a.size))
    unpaired_se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    assert paired_se < unpaired_se


def test_comparison_outputs(stationary_params: AvellanedaStoikovParams) -> None:
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)
    comparison = compare_policies_reference(
        build_policy_set(stationary_params, inventory_limit=3.0), world, n_runs=40
    )
    assert len(comparison.metrics) == 3
    assert comparison.pnl.shape == (40, 3)
    assert len(comparison.paired_tests) == 2
    assert set(comparison.paired_tests.columns) >= {
        "comparison",
        "mean_difference",
        "t_statistic",
        "p_value",
        "n_pairs",
    }
    assert (comparison.paired_tests["n_pairs"] == 40).all()


def test_avellaneda_stoikov_wins_on_sharpe_in_the_reference_world(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)
    comparison = compare_policies_reference(
        build_policy_set(stationary_params, inventory_limit=3.0), world, n_runs=250
    )
    metrics = comparison.metrics.set_index("policy")
    assert metrics.loc["Avellaneda-Stoikov", "sharpe"] > metrics.loc["Symmetric", "sharpe"]
    assert (
        metrics.loc["Avellaneda-Stoikov", "std_final_inventory"]
        < metrics.loc["Symmetric", "std_final_inventory"] / 3.0
    )
    # ...and it wins *despite* making less mean PnL, which is the honest framing.
    assert metrics.loc["Avellaneda-Stoikov", "mean_pnl"] < metrics.loc["Symmetric", "mean_pnl"]


def test_avellaneda_stoikov_also_beats_a_hard_position_limit(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    """The fair benchmark: a naive maker that at least caps its position."""
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)
    comparison = compare_policies_reference(
        build_policy_set(stationary_params, inventory_limit=3.0), world, n_runs=300
    )
    metrics = comparison.metrics.set_index("policy")
    assert (
        metrics.loc["Avellaneda-Stoikov", "sharpe"]
        > metrics.loc["Symmetric + position limit", "sharpe"]
    )
    test = comparison.paired_tests.set_index("comparison")
    row = test.loc["Avellaneda-Stoikov - Symmetric + position limit"]
    assert row["mean_difference"] > 0
    assert row["p_value"] < 0.01


def test_book_comparison_runs(informed_flow: FlowConfig, market: MarketConfig) -> None:
    from mmsim.calibration import fit_as_params_to_book

    params, _, _ = fit_as_params_to_book(
        informed_flow, market, gamma=0.5, risk_horizon=0.1, n_steps=1500, seed=1
    )
    comparison = compare_policies_book(
        build_policy_set(params, inventory_limit=6.0)[:2],
        informed_flow,
        market,
        n_runs=12,
        n_steps=800,
    )
    assert len(comparison.metrics) == 2
    assert comparison.pnl.shape == (12, 2)
    metrics = comparison.metrics.set_index("policy")
    assert (
        metrics.loc["Avellaneda-Stoikov", "std_final_inventory"]
        < metrics.loc["Symmetric", "std_final_inventory"]
    )


def test_paired_test_basics() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    result = paired_test(a, a - 1.0, "a", "b")
    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["n_pairs"] == 4
    assert result["comparison"] == "a - b"
    # Identical inputs produce a zero difference with no variance.
    degenerate = paired_test(a, a, "a", "a")
    assert degenerate["mean_difference"] == 0.0


def test_sensitivity_sweep_shape_and_monotonicity() -> None:
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)

    def build(value: float) -> tuple[AvellanedaStoikovPolicy, AvellanedaStoikovParams]:
        params = AvellanedaStoikovParams(
            gamma=value, kappa=1.5, arrival_rate=140.0, sigma=2.0, horizon=1.0
        )
        return AvellanedaStoikovPolicy(params), world

    frame = sensitivity_sweep([0.02, 0.1, 0.5, 2.0], build, "gamma", n_runs=80, n_steps=150)
    assert list(frame["gamma"]) == [0.02, 0.1, 0.5, 2.0]
    assert "sharpe" in frame.columns
    # More risk aversion => tighter inventory control and wider quotes, monotonically.
    assert frame["mean_abs_inventory"].is_monotonic_decreasing
    assert frame["mean_spread_captured"].is_monotonic_increasing
    # ...and fewer trades, because the quotes are further out.
    assert frame["mean_trades"].is_monotonic_decreasing


def test_sharpe_is_non_monotonic_in_risk_aversion() -> None:
    """There is an interior optimum: too little control is risky, too much stops trading."""
    world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)

    def build(value: float) -> tuple[AvellanedaStoikovPolicy, AvellanedaStoikovParams]:
        params = AvellanedaStoikovParams(
            gamma=value, kappa=1.5, arrival_rate=140.0, sigma=2.0, horizon=1.0
        )
        return AvellanedaStoikovPolicy(params), world

    frame = sensitivity_sweep([0.01, 0.05, 0.2, 1.0, 3.0], build, "gamma", n_runs=150, n_steps=200)
    sharpes = frame["sharpe"].to_numpy()
    best = int(np.argmax(sharpes))
    assert 0 < best < len(sharpes) - 1
    assert sharpes[best] > sharpes[0]
    assert sharpes[best] > sharpes[-1]


def test_symmetric_policy_appears_in_the_matched_set(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    policies = build_policy_set(stationary_params)
    assert isinstance(policies[1], SymmetricPolicy)
    assert policies[2].max_inventory == 5.0  # type: ignore[union-attr]
