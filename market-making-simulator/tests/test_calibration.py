"""Estimating model parameters from the simulated book."""

from __future__ import annotations

import numpy as np
import pytest

from mmsim.avellaneda_stoikov import HorizonMode
from mmsim.calibration import (
    estimate_fill_intensity,
    estimate_volatility,
    fit_as_params_to_book,
)
from mmsim.flow import FlowConfig
from mmsim.types import MarketConfig


def test_fill_intensity_decays_with_distance(quiet_flow: FlowConfig, market: MarketConfig) -> None:
    fit = estimate_fill_intensity(quiet_flow, market, n_steps=1500, seed=1)
    assert fit.kappa > 0
    assert fit.arrival_rate > 0
    # Monotone decay is the qualitative claim; the exponential form is the quantitative one.
    assert fit.intensities[0] > fit.intensities[-1]
    assert fit.r_squared > 0.7
    assert fit.half_life_ticks > 0


def test_fill_curve_is_only_approximately_exponential(
    quiet_flow: FlowConfig, market: MarketConfig
) -> None:
    """The model's core assumption holds to R^2 ~ 0.85 here, not to 1.

    Market-order sizes are Pareto, so the occasional large sweep reaches deep levels far
    more often than an exponential predicts and the tail of the fill curve flattens. Worth
    stating rather than hiding: it is a real limitation of applying this model to a book.
    """
    fit = estimate_fill_intensity(quiet_flow, market, n_steps=2500, seed=2)
    assert 0.75 < fit.r_squared < 0.99
    predicted = fit.predict(fit.distances)
    # The furthest point sits above its fitted value: the tail is fatter than exponential.
    assert fit.intensities[-1] > predicted[-1]


def test_humped_depth_steepens_the_fill_curve(market: MarketConfig) -> None:
    """The design decision in `flow.py`, measured.

    An exponential depth profile puts liquidity at the touch and leaves the deep book
    empty, so probes far from the mid still fill and the fitted kappa collapses.
    """
    exponential = FlowConfig(depth_shape=1.0, limit_depth_ticks=4.0)
    humped = FlowConfig(depth_shape=2.5, limit_depth_ticks=4.0)
    flat_fit = estimate_fill_intensity(exponential, market, n_steps=2000, seed=3)
    humped_fit = estimate_fill_intensity(humped, market, n_steps=2000, seed=3)
    # ~1.35x steeper decay and a visibly better log-linear fit at these settings.
    assert humped_fit.kappa > 1.25 * flat_fit.kappa
    assert humped_fit.r_squared > flat_fit.r_squared


def test_volatility_estimate_tracks_the_configured_diffusion(market: MarketConfig) -> None:
    """With no informed flow, realised vol must match `volatility_ticks` in price units."""
    for ticks in (4.0, 8.0, 16.0):
        config = FlowConfig(informed_fraction=0.0, volatility_ticks=ticks)
        sigma = estimate_volatility(config, market, n_steps=4000, seed=4)
        assert sigma == pytest.approx(ticks * market.tick_size, rel=0.08)


def test_informed_flow_raises_the_volatility_estimate(market: MarketConfig) -> None:
    """Price discovery is volatility too, and the model must be told about it."""
    quiet = estimate_volatility(FlowConfig(informed_fraction=0.0), market, n_steps=3000, seed=5)
    toxic = estimate_volatility(
        FlowConfig(informed_fraction=0.25, info_impact_ticks=2.0), market, n_steps=3000, seed=5
    )
    assert toxic > quiet


def test_fit_as_params_to_book_produces_sane_quotes(
    informed_flow: FlowConfig, market: MarketConfig
) -> None:
    params, fit, sigma = fit_as_params_to_book(
        informed_flow, market, gamma=0.5, risk_horizon=0.1, n_steps=2500, seed=6
    )
    assert params.horizon_mode is HorizonMode.STATIONARY
    assert params.kappa == fit.kappa
    assert params.arrival_rate == fit.arrival_rate
    assert params.sigma == sigma
    assert params.gamma == 0.5

    from mmsim.strategies import average_optimal_spread

    spread_ticks = average_optimal_spread(params) / market.tick_size
    # A quote that is neither inside the tick nor a hundred ticks wide.
    assert 2.0 < spread_ticks < 40.0


def test_estimation_needs_enough_fills(market: MarketConfig) -> None:
    """Probing at absurd distances produces no fills and must fail loudly, not silently."""
    config = FlowConfig(market_order_rate=1.0, volatility_ticks=0.1, seed_levels=2)
    with pytest.raises(ValueError, match="fewer than two probe distances"):
        estimate_fill_intensity(
            config, market, distances_ticks=(5_000, 6_000, 7_000), n_steps=200, seed=7
        )


def test_fit_predicts_and_reports(quiet_flow: FlowConfig, market: MarketConfig) -> None:
    fit = estimate_fill_intensity(quiet_flow, market, n_steps=1200, seed=8)
    assert fit.predict(0.0) == pytest.approx(fit.arrival_rate)
    assert float(fit.predict(fit.half_life_ticks)) == pytest.approx(fit.arrival_rate / 2.0)
    assert np.all(np.diff(np.asarray(fit.predict(np.linspace(0, 0.5, 20)))) < 0)
