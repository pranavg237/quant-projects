"""Order flow, informed traders and adverse selection."""

from __future__ import annotations

import numpy as np
import pytest

from mmsim.book import LimitOrderBook
from mmsim.flow import FlowConfig, OrderFlowGenerator
from mmsim.types import MarketConfig, Side


def _run(
    config: FlowConfig, market: MarketConfig, rng: np.random.Generator, n_steps: int = 600
) -> tuple[LimitOrderBook, float, list[float]]:
    book = LimitOrderBook()
    generator = OrderFlowGenerator(config, market, rng)
    efficient = float(market.initial_mid_ticks)
    generator.seed_book(book, efficient, 0.0)
    dt = 1.0 / n_steps
    path = [efficient]
    for i in range(n_steps):
        efficient, _ = generator.step(book, efficient, i * dt, dt)
        path.append(efficient)
    return book, efficient, path


def test_seed_book_populates_both_sides(
    quiet_flow: FlowConfig, market: MarketConfig, rng: np.random.Generator
) -> None:
    book = LimitOrderBook()
    generator = OrderFlowGenerator(quiet_flow, market, rng)
    generator.seed_book(book, float(market.initial_mid_ticks), 0.0)
    assert book.best_bid == market.initial_mid_ticks - 1
    assert book.best_ask == market.initial_mid_ticks + 1
    assert len(book.depth(Side.BID, 50)) == quiet_flow.seed_levels
    assert len(book.depth(Side.ASK, 50)) == quiet_flow.seed_levels


def test_flow_produces_a_two_sided_book(
    quiet_flow: FlowConfig, market: MarketConfig, rng: np.random.Generator
) -> None:
    book, _, _ = _run(quiet_flow, market, rng)
    assert book.best_bid is not None
    assert book.best_ask is not None
    assert book.spread_ticks is not None and book.spread_ticks > 0
    assert len(book.trades) > 50


def test_noise_limit_orders_never_cross(
    quiet_flow: FlowConfig, market: MarketConfig, rng: np.random.Generator
) -> None:
    """A noise *limit* order that crosses would really be a market order, double-counting."""
    book, _, _ = _run(quiet_flow, market, rng)
    assert book.best_bid is not None and book.best_ask is not None
    assert book.best_bid < book.best_ask
    noise_trades = [t for t in book.trades if t.taker_owner == "noise"]
    assert noise_trades == []


def test_uninformed_flow_leaves_the_price_a_martingale(
    quiet_flow: FlowConfig, market: MarketConfig
) -> None:
    """With no informed traders, the efficient price drifts nowhere."""
    finals = []
    for seed in range(60):
        _, efficient, _ = _run(quiet_flow, market, np.random.default_rng(seed), n_steps=400)
        finals.append(efficient - market.initial_mid_ticks)
    finals_arr = np.array(finals)
    std_error = finals_arr.std(ddof=1) / np.sqrt(finals_arr.size)
    assert abs(finals_arr.mean()) < 3 * std_error


def test_informed_flow_raises_realised_volatility(
    quiet_flow: FlowConfig, informed_flow: FlowConfig, market: MarketConfig
) -> None:
    """Permanent impact is a second source of variance on top of the diffusion.

    Measured over 40-step blocks, not single steps. Impact is released gradually
    (``info_impact_speed``), so at a one-step horizon most of it has not happened yet and
    the extra variance is nearly invisible; it only shows up once the window is long enough
    for price discovery to complete. The next test pins the other side of that.
    """

    def realised(cfg: FlowConfig, block: int) -> float:
        paths = [
            _run(cfg, market, np.random.default_rng(s), n_steps=400)[2][::block] for s in range(15)
        ]
        return float(np.concatenate([np.diff(p) for p in paths]).std(ddof=1))

    assert realised(informed_flow, 40) > 2.0 * realised(quiet_flow, 40)


def test_gradual_impact_is_nearly_invisible_at_a_one_step_horizon(
    quiet_flow: FlowConfig, informed_flow: FlowConfig, market: MarketConfig
) -> None:
    """Why the block size above is 40 and not 1."""

    def realised(cfg: FlowConfig) -> float:
        paths = [
            np.diff(_run(cfg, market, np.random.default_rng(s), n_steps=400)[2]) for s in range(15)
        ]
        return float(np.concatenate(paths).std(ddof=1))

    ratio = realised(informed_flow) / realised(quiet_flow)
    assert 1.0 < ratio < 2.0


def test_informed_trades_move_the_price_in_their_direction(market: MarketConfig) -> None:
    """The mechanism itself: an informed buy raises the efficient price, a sell lowers it."""
    config = FlowConfig(
        market_order_rate=500.0,
        limit_order_rate=8_000.0,
        informed_fraction=1.0,  # every market order is informed
        info_impact_ticks=3.0,
        info_impact_speed=1.0,  # released at once, so the move is attributable to one trade
        volatility_ticks=0.0,  # and no diffusion
    )
    rng = np.random.default_rng(5)
    book = LimitOrderBook()
    generator = OrderFlowGenerator(config, market, rng)
    efficient = float(market.initial_mid_ticks)
    generator.seed_book(book, efficient, 0.0)

    moves: list[tuple[int, float]] = []
    dt = 1.0 / 300
    for i in range(300):
        before = efficient
        efficient, trades = generator.step(book, efficient, i * dt, dt)
        informed = [t for t in trades if t.taker_owner == "informed"]
        if len(informed) == 1:
            moves.append((int(informed[0].aggressor_side), efficient - before))
    assert len(moves) > 20
    for side, move in moves:
        assert move == pytest.approx(side * 3.0)


def test_gradual_impact_releases_the_full_amount(market: MarketConfig) -> None:
    """Slower discovery changes *when* the price moves, not *how far* it ends up moving."""
    config = FlowConfig(
        market_order_rate=1.0,
        limit_order_rate=8_000.0,
        informed_fraction=1.0,
        info_impact_ticks=4.0,
        info_impact_speed=0.05,
        volatility_ticks=0.0,
    )
    rng = np.random.default_rng(11)
    book = LimitOrderBook()
    generator = OrderFlowGenerator(config, market, rng)
    efficient = float(market.initial_mid_ticks)
    generator.seed_book(book, efficient, 0.0)

    # One informed trade, then let the price discover with no further flow.
    generator.pending_impact = 4.0
    quiet = FlowConfig(
        market_order_rate=1e-12,
        limit_order_rate=1e-12,
        informed_fraction=0.0,
        info_impact_speed=0.05,
        volatility_ticks=0.0,
    )
    generator.config = quiet
    generator.pending_impact = 4.0
    start = efficient
    for i in range(400):
        efficient, _ = generator.step(book, efficient, i / 400, 1.0 / 400)
    assert efficient - start == pytest.approx(4.0, abs=1e-6)
    assert generator.pending_impact == pytest.approx(0.0, abs=1e-6)


def test_expected_adverse_selection_formula() -> None:
    config = FlowConfig(informed_fraction=0.2, info_impact_ticks=2.5)
    assert config.expected_adverse_selection_per_fill == pytest.approx(0.5)
    assert FlowConfig(informed_fraction=0.0).expected_adverse_selection_per_fill == 0.0


def test_order_sizes_have_the_configured_mean(market: MarketConfig) -> None:
    config = FlowConfig(mean_order_size=2.5, size_tail=1.8)
    generator = OrderFlowGenerator(config, market, np.random.default_rng(0))
    sizes = np.array([generator._draw_size() for _ in range(200_000)])
    assert float(sizes.mean()) == pytest.approx(2.5, rel=0.05)
    # Heavy tailed: the largest order dwarfs the mean, which is what sweeps the book.
    assert sizes.max() > 20 * sizes.mean()
    assert np.all(sizes > 0)


def test_depth_profile_is_humped_not_exponential(market: MarketConfig) -> None:
    """The shape that makes the fill curve decay properly; see the module docstring."""
    config = FlowConfig(cancel_rate=0.0, limit_depth_ticks=4.0, depth_shape=2.5, seed_levels=0)
    book = LimitOrderBook()
    generator = OrderFlowGenerator(config, market, np.random.default_rng(1))
    centre = float(market.initial_mid_ticks)
    # Add liquidity without any market orders, to see the raw placement profile.
    for _ in range(400):
        generator._add_noise_limits(book, centre, 0.0, 1.0 / 400)
    sizes = [size for _, size in book.depth(Side.BID, 30)]
    assert len(sizes) > 6
    # Thin at the touch, thicker a few ticks out: the peak is not at level 1.
    assert int(np.argmax(sizes)) > 0
    assert sizes[0] < max(sizes)


def test_cancellation_thins_the_book(market: MarketConfig, rng: np.random.Generator) -> None:
    dense = FlowConfig(cancel_rate=0.0, market_order_rate=1.0)
    sparse = FlowConfig(cancel_rate=20_000.0, market_order_rate=1.0)
    book_dense, _, _ = _run(dense, market, np.random.default_rng(2), n_steps=300)
    book_sparse, _, _ = _run(sparse, market, np.random.default_rng(2), n_steps=300)
    assert len(book_dense) > 3 * len(book_sparse)


def test_flow_config_validation() -> None:
    with pytest.raises(ValueError, match="informed_fraction"):
        FlowConfig(informed_fraction=1.5)
    with pytest.raises(ValueError, match="market_order_rate"):
        FlowConfig(market_order_rate=0.0)
    with pytest.raises(ValueError, match="cancel_rate"):
        FlowConfig(cancel_rate=-1.0)
    with pytest.raises(ValueError, match="size_tail"):
        FlowConfig(size_tail=0.9)  # infinite mean
    with pytest.raises(ValueError, match="seed_levels"):
        FlowConfig(seed_levels=-1)
    with pytest.raises(ValueError, match="depth_shape"):
        FlowConfig(depth_shape=0.0)
    with pytest.raises(ValueError, match="limit_depth_ticks"):
        FlowConfig(limit_depth_ticks=0.0)
    with pytest.raises(ValueError, match="info_impact_speed"):
        FlowConfig(info_impact_speed=0.0)
    with pytest.raises(ValueError, match="info_impact_speed"):
        FlowConfig(info_impact_speed=1.5)
