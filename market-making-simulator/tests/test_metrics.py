"""Metrics: PnL decomposition, markout and run summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmsim.avellaneda_stoikov import AvellanedaStoikovParams
from mmsim.engine import SimulationResult, simulate_book, simulate_reference
from mmsim.flow import FlowConfig
from mmsim.metrics import markout_curve, pnl_decomposition, summarise_runs
from mmsim.strategies import AvellanedaStoikovPolicy, SymmetricPolicy
from mmsim.types import MarketConfig


def test_pnl_decomposition_is_an_exact_identity(
    quiet_flow: FlowConfig, market: MarketConfig
) -> None:
    """Spread capture plus inventory PnL must equal simulated wealth, to the last cent."""
    for seed in range(8):
        result = simulate_book(
            SymmetricPolicy(half_spread=0.05), quiet_flow, market, n_steps=500, seed=seed
        )
        parts = pnl_decomposition(result)
        assert parts["total"] == pytest.approx(parts["reported_total"], abs=1e-9)
        assert parts["reconciliation_error"] == pytest.approx(0.0, abs=1e-9)
        assert parts["spread_capture"] + parts["inventory_pnl"] == pytest.approx(
            parts["total"], abs=1e-9
        )


def test_spread_capture_is_positive_for_a_passive_maker(
    quiet_flow: FlowConfig, market: MarketConfig
) -> None:
    """Quoting outside the mid earns the spread on every fill, by construction."""
    result = simulate_book(
        SymmetricPolicy(half_spread=0.05), quiet_flow, market, n_steps=800, seed=1
    )
    assert pnl_decomposition(result)["spread_capture"] > 0


def test_informed_flow_shows_up_as_negative_inventory_pnl(market: MarketConfig) -> None:
    """Adverse selection has to land somewhere; it lands in the inventory leg."""
    quiet = FlowConfig(informed_fraction=0.0)
    toxic = FlowConfig(informed_fraction=0.35, info_impact_ticks=3.0, info_impact_speed=0.2)
    policy = SymmetricPolicy(half_spread=0.05)

    def mean_inventory_pnl(cfg: FlowConfig) -> float:
        runs = [simulate_book(policy, cfg, market, n_steps=1200, seed=s) for s in range(40)]
        return float(np.mean([pnl_decomposition(r)["inventory_pnl"] for r in runs]))

    assert mean_inventory_pnl(toxic) < mean_inventory_pnl(quiet)


def test_decomposition_of_an_empty_run() -> None:
    empty = SimulationResult(
        times=np.zeros(2),
        mid=np.ones(2) * 100.0,
        inventory=np.zeros(2),
        cash=np.zeros(2),
        bid_quotes=np.full(2, np.nan),
        ask_quotes=np.full(2, np.nan),
        buy_fills=np.zeros(2),
        sell_fills=np.zeros(2),
    )
    parts = pnl_decomposition(empty)
    assert parts["spread_capture"] == 0.0
    assert parts["total"] == 0.0


def test_markout_is_negative_under_informed_flow(
    quiet_flow: FlowConfig, informed_flow: FlowConfig, market: MarketConfig
) -> None:
    """The desk's measure of being picked off: the price moves against you after a fill."""
    policy = SymmetricPolicy(half_spread=0.04)

    def mean_markout(cfg: FlowConfig, horizon: int) -> float:
        total, weight = 0.0, 0.0
        for seed in range(25):
            result = simulate_book(policy, cfg, market, n_steps=900, seed=seed)
            curve = markout_curve(result, (horizon,))
            if curve.empty:
                continue
            total += float(curve["mean_markout"].iloc[0] * curve["n_fills"].iloc[0])
            weight += float(curve["n_fills"].iloc[0])
        return total / weight

    toxic = mean_markout(informed_flow, 50)
    benign = mean_markout(quiet_flow, 50)
    assert toxic < 0.0
    assert toxic < benign


def test_markout_builds_as_the_information_is_impounded(
    informed_flow: FlowConfig, market: MarketConfig
) -> None:
    """The canonical desk chart: markout deepens over the price-discovery horizon.

    ``info_impact_speed`` releases about 5% of an informed trade's impact per step, so the
    move takes roughly 20 steps to play out and the markout curve builds over that window
    rather than jumping. Past the discovery horizon the estimate is dominated by diffusive
    noise growing like sqrt(h), so the curve is only compared inside it.
    """
    horizons = (1, 5, 25)
    total = {h: [0.0, 0.0] for h in horizons}
    for seed in range(40):
        result = simulate_book(
            SymmetricPolicy(half_spread=0.04), informed_flow, market, n_steps=900, seed=seed
        )
        curve = markout_curve(result, horizons)
        for _, row in curve.iterrows():
            total[int(row["horizon"])][0] += row["mean_markout"] * row["n_fills"]
            total[int(row["horizon"])][1] += row["n_fills"]
    means = {h: v[0] / v[1] for h, v in total.items()}
    assert means[25] < means[5] < means[1] < 0.0


def test_instantaneous_impact_gives_a_flat_markout_curve(market: MarketConfig) -> None:
    """Why ``info_impact_speed`` exists: at speed 1 the whole move is already done at h=1.

    With instantaneous impact the markout curve carries no shape information at all, which
    is what the first version of this model produced and why it was changed.
    """
    config = FlowConfig(informed_fraction=0.2, info_impact_ticks=2.0, info_impact_speed=1.0)
    horizons = (1, 5, 25)
    total = {h: [0.0, 0.0] for h in horizons}
    for seed in range(40):
        result = simulate_book(
            SymmetricPolicy(half_spread=0.04), config, market, n_steps=900, seed=seed
        )
        for _, row in markout_curve(result, horizons).iterrows():
            total[int(row["horizon"])][0] += row["mean_markout"] * row["n_fills"]
            total[int(row["horizon"])][1] += row["n_fills"]
    means = {h: v[0] / v[1] for h, v in total.items()}
    assert means[1] < 0.0
    # Already saturated at the first step: h=25 is no deeper than h=1.
    assert means[25] > 1.5 * means[1]


def test_markout_requires_book_fills(paper_params: AvellanedaStoikovParams) -> None:
    result = simulate_reference(AvellanedaStoikovPolicy(paper_params), paper_params, seed=1)
    with pytest.raises(ValueError, match="simulate_book"):
        markout_curve(result)


def test_markout_of_an_empty_run() -> None:
    empty = SimulationResult(
        times=np.zeros(2),
        mid=np.ones(2),
        inventory=np.zeros(2),
        cash=np.zeros(2),
        bid_quotes=np.full(2, np.nan),
        ask_quotes=np.full(2, np.nan),
        buy_fills=np.zeros(2),
        sell_fills=np.zeros(2),
        fill_prices=pd.DataFrame(),
    )
    assert markout_curve(empty).empty


def test_summarise_runs(paper_params: AvellanedaStoikovParams) -> None:
    runs = [
        simulate_reference(AvellanedaStoikovPolicy(paper_params), paper_params, seed=s)
        for s in range(40)
    ]
    summary = summarise_runs(runs)
    assert summary.n_runs == 40
    assert summary.policy == "Avellaneda-Stoikov"
    assert summary.sharpe == pytest.approx(summary.mean_pnl / summary.std_pnl)
    assert summary.pnl_5th_percentile <= summary.median_pnl
    assert summary.max_abs_inventory >= summary.mean_abs_inventory
    assert summary.max_drawdown >= 0.0
    assert summary.mean_trades > 0
    assert summary.mean_spread_captured == pytest.approx(1.49, abs=0.02)
    assert set(summary.as_dict()) >= {"policy", "sharpe", "mean_pnl"}


def test_summarise_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError, match="empty list"):
        summarise_runs([])


def test_drawdown_is_zero_for_a_monotone_equity_curve() -> None:
    from mmsim.metrics import _max_drawdown

    assert _max_drawdown(np.array([0.0, 1.0, 2.0, 3.0])) == 0.0
    assert _max_drawdown(np.array([0.0, 5.0, 1.0, 4.0])) == 4.0
