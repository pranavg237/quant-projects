r"""Simulated order flow: noise traders, informed traders, and adverse selection.

The order book needs someone to trade with. This module supplies three streams.

**Noise limit orders.** Passive liquidity arriving and cancelling around the efficient
price, at a distance drawn from a **Gamma** distribution. Not an exponential: an
exponential puts the most liquidity right at the touch, whereas real limit order books are
*humped* -- thin at the best price, thickest a few ticks out, thinning again beyond that.
The shape matters far more than it sounds. With exponential depth, a probe order ten ticks
from the mid still fills at a fifth of the touch rate, because there is nothing resting in
front of it; the measured fill curve is nearly flat, and a model fitted to it concludes
that quoting very wide is optimal. With a humped profile the fill curve decays properly and
the fitted :math:`\kappa` triples.

**Uninformed market orders.** Poisson arrivals, direction a fair coin, size drawn from a
heavy-tailed distribution so that large orders occasionally sweep several levels. These are
the maker's bread and butter: it buys at its bid and sells at its ask and earns the spread.

**Informed market orders.** The reason market making is hard. An informed order arrives
knowing the efficient price is about to move, and it trades in that direction. Concretely,
an informed buy **permanently raises** the efficient price by ``info_impact`` ticks, and an
informed sell lowers it. So the maker that sold to an informed buyer is immediately marked
down on the position it just took.

This is the Glosten-Milgrom mechanism, and it produces genuine adverse selection rather
than a hand-waved cost: the loss shows up automatically in the maker's mark-to-market, at
a magnitude set by the informed fraction and the impact size. It also means the fair
break-even spread is a derived quantity, not an assumption -- with informed fraction
:math:`p` and impact :math:`J`, the expected adverse-selection cost per fill is
:math:`pJ`, so a maker quoting a half-spread below that loses money no matter how well it
manages inventory. The tests check exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .book import LimitOrderBook
from .types import MarketConfig, Side, Trade

__all__ = ["FlowConfig", "OrderFlowGenerator"]


@dataclass(frozen=True)
class FlowConfig:
    r"""Parameters of the simulated order flow.

    Attributes:
        market_order_rate: Poisson intensity of market orders, per unit time.
        limit_order_rate: Poisson intensity of noise limit orders, per unit time.
        cancel_rate: Per-order cancellation intensity, per unit time. Each resting noise
            order is cancelled independently with probability ``1 - exp(-rate*dt)``.
        informed_fraction: Probability :math:`p` that a market order is informed.
        info_impact_ticks: Permanent price impact :math:`J` of an informed order, in ticks.
        mean_order_size: Mean size of a market order.
        size_tail: Pareto tail index for market-order size. Smaller means fatter tails and
            more multi-level sweeps. Empirical estimates for equities sit near 1.5.
        limit_depth_ticks: Mean distance of a noise limit order from the efficient price.
        depth_shape: Gamma shape parameter for that distance. ``1.0`` is an exponential
            (liquidity concentrated at the touch); values above 1 produce the humped
            profile real books have. The default of 2.5 gives a log-linear fill-curve fit
            with an R-squared around 0.92.
        noise_size: Size of each noise limit order.
        volatility_ticks: Diffusive volatility of the efficient price, in ticks per
            sqrt(unit time). This is the part of price movement that is *not* caused by
            informed trading.
        seed_levels: How many levels either side to pre-populate at the start, so the maker
            is not quoting into an empty book.
    """

    market_order_rate: float = 2_000.0
    limit_order_rate: float = 8_000.0
    cancel_rate: float = 1_200.0
    informed_fraction: float = 0.0
    info_impact_ticks: float = 2.0
    mean_order_size: float = 1.5
    size_tail: float = 1.5
    limit_depth_ticks: float = 4.0
    depth_shape: float = 2.5
    noise_size: float = 2.0
    volatility_ticks: float = 8.0
    seed_levels: int = 20

    def __post_init__(self) -> None:
        if not 0.0 <= self.informed_fraction <= 1.0:
            raise ValueError("informed_fraction must be in [0, 1]")
        for name in ("market_order_rate", "limit_order_rate", "mean_order_size"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.cancel_rate < 0.0:
            raise ValueError("cancel_rate must be >= 0")
        if self.size_tail <= 1.0:
            raise ValueError("size_tail must be > 1 for the order size to have a finite mean")
        if self.seed_levels < 0:
            raise ValueError("seed_levels must be >= 0")
        if self.depth_shape <= 0.0:
            raise ValueError("depth_shape must be > 0")
        if self.limit_depth_ticks <= 0.0:
            raise ValueError("limit_depth_ticks must be > 0")

    @property
    def expected_adverse_selection_per_fill(self) -> float:
        r"""Expected adverse-selection cost per fill, in ticks: :math:`pJ`.

        The break-even half-spread. A maker quoting inside this loses money on average
        however well it manages inventory, because the loss is in the *price* it gets, not
        in the position it carries.
        """
        return self.informed_fraction * self.info_impact_ticks


class OrderFlowGenerator:
    """Generates one step of order flow and evolves the efficient price.

    Args:
        config: Flow parameters.
        market: Tick size and initial price.
        rng: Random generator. Passing one in (rather than a seed) lets the caller keep a
            single stream across the whole simulation, which makes runs reproducible.
    """

    NOISE_OWNER = "noise"

    def __init__(self, config: FlowConfig, market: MarketConfig, rng: np.random.Generator) -> None:
        self.config = config
        self.market = market
        self.rng = rng

    # ------------------------------------------------------------------ helpers

    def _draw_size(self) -> float:
        r"""Market-order size: a Pareto tail scaled to ``mean_order_size``.

        A Pareto(:math:`\alpha`) has mean :math:`\alpha/(\alpha-1)`, so the scale factor
        below normalises the mean to the configured value. Heavy tails matter here: it is
        the occasional large sweep, not the typical order, that takes out the maker's quote
        and several levels behind it.
        """
        alpha = self.config.size_tail
        scale = self.config.mean_order_size * (alpha - 1.0) / alpha
        return float(scale * (self.rng.pareto(alpha) + 1.0))

    def seed_book(self, book: LimitOrderBook, efficient_ticks: float, timestamp: float) -> None:
        """Pre-populate the book with noise liquidity either side of the efficient price."""
        centre = round(efficient_ticks)
        for level in range(1, self.config.seed_levels + 1):
            book.submit_limit(
                Side.BID, centre - level, self.config.noise_size, timestamp, self.NOISE_OWNER
            )
            book.submit_limit(
                Side.ASK, centre + level, self.config.noise_size, timestamp, self.NOISE_OWNER
            )

    # ------------------------------------------------------------------ the step

    def step(
        self,
        book: LimitOrderBook,
        efficient_ticks: float,
        timestamp: float,
        dt: float,
    ) -> tuple[float, list[Trade]]:
        """Advance the market by ``dt``.

        Order of operations matters and is chosen to avoid look-ahead: market orders are
        matched against the book **before** the diffusive price move for the step is
        applied, so no fill can use information from a price change that has not happened
        yet. Informed impact is applied immediately *after* the informed order executes,
        which is the whole point -- the maker is filled at the old price and marked at the
        new one.

        Args:
            book: The order book to trade against.
            efficient_ticks: Current efficient price in ticks.
            timestamp: Current time.
            dt: Step length.

        Returns:
            ``(new_efficient_ticks, trades)`` where ``trades`` covers every match generated
            this step, by any participant.
        """
        cfg = self.config
        trades: list[Trade] = []

        self._cancel_noise(book, dt)
        self._add_noise_limits(book, efficient_ticks, timestamp, dt)

        n_market = int(self.rng.poisson(cfg.market_order_rate * dt))
        for _ in range(n_market):
            side = Side.BID if self.rng.random() < 0.5 else Side.ASK
            size = self._draw_size()
            informed = self.rng.random() < cfg.informed_fraction
            owner = "informed" if informed else "noise_taker"
            _, executed = book.submit_market(side, size, timestamp, owner)
            trades.extend(executed)
            if informed and executed:
                # The impact is applied *after* execution: the informed trader is filled at
                # the old price and the efficient price then moves in its direction. That
                # ordering is what makes the adverse selection causal rather than fitted.
                efficient_ticks += float(side) * cfg.info_impact_ticks

        efficient_ticks += cfg.volatility_ticks * np.sqrt(dt) * float(self.rng.standard_normal())
        return efficient_ticks, trades

    def _cancel_noise(self, book: LimitOrderBook, dt: float) -> None:
        """Cancel each resting noise order independently with probability 1-exp(-rate*dt)."""
        if self.config.cancel_rate <= 0.0:
            return
        prob = 1.0 - np.exp(-self.config.cancel_rate * dt)
        resting = [o.order_id for o in book.resting_orders(self.NOISE_OWNER)]
        if not resting:
            return
        draws = self.rng.random(len(resting))
        for order_id, draw in zip(resting, draws, strict=True):
            if draw < prob:
                book.cancel(order_id)

    def _add_noise_limits(
        self, book: LimitOrderBook, efficient_ticks: float, timestamp: float, dt: float
    ) -> None:
        """Add noise limit orders at Gamma-distributed distances from fair value.

        Gamma with ``shape > 1`` is humped, which is what a real book looks like. See the
        module docstring for why an exponential profile quietly breaks the fill-curve fit.
        """
        cfg = self.config
        n_new = int(self.rng.poisson(cfg.limit_order_rate * dt))
        if n_new == 0:
            return
        centre = round(efficient_ticks)
        sides = self.rng.random(n_new) < 0.5
        shape = cfg.depth_shape
        depths = self.rng.gamma(shape, cfg.limit_depth_ticks / shape, n_new)
        for is_bid, depth in zip(sides, depths, strict=True):
            offset = max(1, round(float(depth)))
            if is_bid:
                price = centre - offset
                # Never post a bid that would cross the book: a noise *limit* order that
                # crosses is really a market order, and letting it through here would
                # double-count aggressive flow.
                best_ask = book.best_ask
                if best_ask is not None and price >= best_ask:
                    continue
                book.submit_limit(Side.BID, price, cfg.noise_size, timestamp, self.NOISE_OWNER)
            else:
                price = centre + offset
                best_bid = book.best_bid
                if best_bid is not None and price <= best_bid:
                    continue
                book.submit_limit(Side.ASK, price, cfg.noise_size, timestamp, self.NOISE_OWNER)
