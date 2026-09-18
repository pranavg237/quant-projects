r"""Two simulation engines, deliberately at different levels of fidelity.

**`simulate_reference`** is the Avellaneda-Stoikov world exactly as published: the mid is
an arithmetic Brownian motion, there is no order book, and a quote at distance
:math:`\delta` fills as a Poisson process with intensity :math:`A e^{-\kappa\delta}`. This
is the only setting in which the model's closed-form solution is actually optimal, so it is
the setting in which the implementation can be *validated* -- against the published Table 1
of the paper.

**`simulate_book`** replaces all of that with the real matching engine from
:mod:`mmsim.book`: the maker posts limit orders that join a queue, noise traders add and
cancel liquidity, market orders arrive and walk the book, and a fraction of those market
orders are **informed** -- they move the efficient price permanently in their own
direction. The maker only gets filled when a market order actually reaches its order, which
depends on queue position.

Running the same policy in both is the point. The reference engine says what the theory
promises; the book engine says what survives once fills are contingent on queue position
and a fraction of the flow knows something you do not. The gap between them is the honest
answer to "does Avellaneda-Stoikov work".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import pandas as pd

from .avellaneda_stoikov import AvellanedaStoikovParams
from .book import LimitOrderBook
from .flow import FlowConfig, OrderFlowGenerator
from .strategies import MakerState, QuotePolicy
from .types import FloatArray, MarketConfig, Side

__all__ = ["FillModel", "SimulationResult", "simulate_book", "simulate_reference"]


class FillModel(StrEnum):
    r"""How a Poisson fill intensity is turned into a per-step fill probability.

    ``EXACT``
        :math:`1 - e^{-\lambda\Delta t}`, the true probability of at least one arrival.
        The default, and correct for any step size.

    ``LINEAR``
        :math:`\lambda\Delta t`, the first-order approximation used in the original
        Avellaneda-Stoikov discretisation. It agrees with ``EXACT`` to
        :math:`O(\Delta t^2)` but **overstates** the fill rate -- by about 10% at the
        paper's own parameters -- and exceeds 1 for tight quotes on a coarse grid.
        Provided so the published Table 1 can be reproduced exactly; not recommended.
    """

    EXACT = "exact"
    LINEAR = "linear"


def _fill_probability(intensity: float, dt: float, model: FillModel) -> float:
    """Per-step fill probability for a Poisson process of the given intensity."""
    if model is FillModel.LINEAR:
        return min(intensity * dt, 1.0)
    return float(1.0 - np.exp(-intensity * dt))


_MAKER = "mm"


@dataclass
class SimulationResult:
    """One simulation path.

    Attributes:
        times: Time grid.
        mid: Efficient mid price on that grid.
        inventory: Signed position after each step.
        cash: Cash balance after each step.
        bid_quotes / ask_quotes: Quoted prices, ``nan`` where the side was withdrawn.
        buy_fills / sell_fills: Cumulative filled size on each side.
        fill_prices: Per-fill records used for markout / adverse-selection analysis.
        engine: Which engine produced this.
        policy_name: Which policy produced this.
    """

    times: FloatArray
    mid: FloatArray
    inventory: FloatArray
    cash: FloatArray
    bid_quotes: FloatArray
    ask_quotes: FloatArray
    buy_fills: FloatArray
    sell_fills: FloatArray
    fill_prices: pd.DataFrame = field(default_factory=pd.DataFrame)
    engine: str = "reference"
    policy_name: str = ""

    @property
    def mark_to_market(self) -> FloatArray:
        """Cash plus inventory marked at the current mid, at every step."""
        return np.asarray(self.cash + self.inventory * self.mid, dtype=np.float64)

    @property
    def final_pnl(self) -> float:
        """Terminal mark-to-market wealth."""
        return float(self.mark_to_market[-1])

    @property
    def final_inventory(self) -> float:
        """Terminal signed position."""
        return float(self.inventory[-1])

    @property
    def n_trades(self) -> int:
        """Total fills on both sides."""
        return int(self.buy_fills[-1] + self.sell_fills[-1])

    def to_frame(self) -> pd.DataFrame:
        """Tidy per-step frame, handy for plotting."""
        return pd.DataFrame(
            {
                "time": self.times,
                "mid": self.mid,
                "inventory": self.inventory,
                "cash": self.cash,
                "bid": self.bid_quotes,
                "ask": self.ask_quotes,
                "pnl": self.mark_to_market,
                "buy_fills": self.buy_fills,
                "sell_fills": self.sell_fills,
            }
        )


def simulate_reference(
    policy: QuotePolicy,
    params: AvellanedaStoikovParams,
    n_steps: int = 200,
    initial_mid: float = 100.0,
    order_size: float = 1.0,
    fill_model: FillModel | str = FillModel.EXACT,
    seed: int | np.random.Generator | None = None,
) -> SimulationResult:
    r"""Simulate the Avellaneda-Stoikov model world.

        Each step of length :math:`\Delta t`:

        1. the maker quotes;
        2. each side fills with probability :math:`1 - e^{-\lambda(\delta)\Delta t}`, where
           :math:`\lambda(\delta) = Ae^{-\kappa\delta}` and :math:`\delta` is the distance
           from the **current** mid;
        3. the mid moves by :math:`\sigma\sqrt{\Delta t}\,Z`.

    The default :class:`FillModel` is the exact Poisson probability rather than the
        first-order :math:`\lambda\Delta t` the paper's discretisation implies. They agree to
        :math:`O(\Delta t^2)`, but at the paper's own parameters
        (:math:`\lambda \approx 45`, :math:`\Delta t = 0.005`) the linear form overstates the
        fill rate by 10%, and it exceeds 1 outright for tight quotes on a coarse grid. Pass
        ``fill_model="linear"`` to reproduce the published numbers.

        Quotes are evaluated against the mid *before* the move, so a fill never uses
        information from the price change that follows it. There is no look-ahead.

        Args:
            policy: Quoting rule.
            params: Model parameters; supplies ``sigma``, ``arrival_rate``, ``kappa`` and the
                horizon.
            n_steps: Steps in the session.
            initial_mid: Starting price.
            order_size: Size quoted and filled on each side.
            fill_model: ``"exact"`` or ``"linear"``; see :class:`FillModel`.
            seed: Seed or ``Generator``.

        Returns:
            A :class:`SimulationResult`.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    model = (
        FillModel(str(fill_model).lower()) if not isinstance(fill_model, FillModel) else fill_model
    )

    dt = params.horizon / n_steps
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0.0, params.horizon, n_steps + 1)
    mid = np.empty(n_steps + 1, dtype=np.float64)
    inventory = np.zeros(n_steps + 1, dtype=np.float64)
    cash = np.zeros(n_steps + 1, dtype=np.float64)
    bid_quotes = np.full(n_steps + 1, np.nan, dtype=np.float64)
    ask_quotes = np.full(n_steps + 1, np.nan, dtype=np.float64)
    buy_fills = np.zeros(n_steps + 1, dtype=np.float64)
    sell_fills = np.zeros(n_steps + 1, dtype=np.float64)
    mid[0] = initial_mid

    # Pre-draw every random number: one call into the RNG instead of 4n.
    shocks = rng.standard_normal(n_steps)
    fill_draws = rng.random((n_steps, 2))
    fills: list[dict[str, float | str]] = []

    for i in range(n_steps):
        state = MakerState(
            mid=float(mid[i]),
            inventory=float(inventory[i]),
            cash=float(cash[i]),
            time=float(times[i]),
            horizon=params.horizon,
        )
        bid, ask = policy.quote(state)
        q, c = state.inventory, state.cash

        if bid is not None:
            bid_quotes[i] = bid
            delta_b = state.mid - bid
            intensity = params.arrival_rate * np.exp(-params.kappa * delta_b)
            if fill_draws[i, 0] < _fill_probability(float(intensity), dt, model):
                q += order_size
                c -= bid * order_size
                buy_fills[i + 1 :] += order_size
                fills.append(
                    {"time": float(times[i]), "side": "buy", "price": bid, "mid": state.mid}
                )
        if ask is not None:
            ask_quotes[i] = ask
            delta_a = ask - state.mid
            intensity = params.arrival_rate * np.exp(-params.kappa * delta_a)
            if fill_draws[i, 1] < _fill_probability(float(intensity), dt, model):
                q -= order_size
                c += ask * order_size
                sell_fills[i + 1 :] += order_size
                fills.append(
                    {"time": float(times[i]), "side": "sell", "price": ask, "mid": state.mid}
                )

        inventory[i + 1] = q
        cash[i + 1] = c
        mid[i + 1] = mid[i] + params.sigma * sqrt_dt * shocks[i]

    return SimulationResult(
        times=times,
        mid=mid,
        inventory=inventory,
        cash=cash,
        bid_quotes=bid_quotes,
        ask_quotes=ask_quotes,
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        fill_prices=pd.DataFrame(fills),
        engine="reference",
        policy_name=getattr(policy, "name", type(policy).__name__),
    )


def _refresh_side(
    book: LimitOrderBook,
    side: Side,
    target_tick: int | None,
    resting: tuple[int, int] | None,
    size: float,
    now: float,
) -> tuple[int, int] | None:
    """Move the maker's quote on one side to ``target_tick``, keeping it if unchanged.

    **An order is only cancelled and re-posted when its price actually changes.** Blindly
    cancelling and replacing every step -- which is what the first version did -- sends the
    maker to the back of the queue on every single step, so it never accumulates any time
    priority at all. That silently understates fill rates for every strategy, and it
    understates them *most* for the strategy that changes its quotes least, which is
    exactly the comparison this repo is trying to make.

    Args:
        book: The order book.
        side: Which side to refresh.
        target_tick: Desired price, or ``None`` to withdraw the side.
        resting: The maker's current ``(order_id, price_ticks)`` on this side, if any.
        size: Size to post.
        now: Current time.

    Returns:
        The new ``(order_id, price_ticks)``, or ``None`` if nothing is resting.
    """
    if resting is not None and resting[0] not in book:
        resting = None  # it was filled or otherwise removed
    if target_tick is None:
        if resting is not None:
            book.cancel(resting[0])
        return None
    if resting is not None and resting[1] == target_tick:
        return resting  # unchanged: leave it alone and keep its queue position

    # Never post through the opposite side: a quote that crosses is a market order.
    opposite_best = book.best(side.opposite)
    if opposite_best is not None:
        crosses = target_tick >= opposite_best if side is Side.BID else target_tick <= opposite_best
        if crosses:
            if resting is not None:
                book.cancel(resting[0])
            return None

    if resting is not None:
        book.cancel(resting[0])
    order_id, _ = book.submit_limit(side, target_tick, size, now, _MAKER)
    return (order_id, target_tick)


def simulate_book(  # noqa: PLR0915  (one loop; splitting it would hide the order of operations)
    policy: QuotePolicy,
    flow_config: FlowConfig,
    market: MarketConfig | None = None,
    n_steps: int = 2_000,
    horizon: float = 1.0,
    order_size: float = 1.0,
    requote_every: int = 1,
    seed: int | np.random.Generator | None = None,
) -> SimulationResult:
    """Simulate a full limit order book with noise and informed traders.

    Each step:

    1. the maker cancels its resting quotes and re-posts (every ``requote_every`` steps);
    2. noise traders add and cancel limit orders around the efficient price;
    3. market orders arrive; informed ones move the efficient price permanently, which is
       where adverse selection comes from;
    4. the efficient price also diffuses.

    The maker's fills come out of the matching engine, so they depend on queue position:
    posting at the same price as everyone else means waiting behind them.

    Args:
        policy: Quoting rule.
        flow_config: Order-flow parameters.
        market: Tick size and fees.
        n_steps: Steps in the session.
        horizon: Session length in time units.
        order_size: Size posted on each side.
        requote_every: Re-quote every ``n`` steps. Larger values let an order age and
            earn queue priority, but also leave the maker out of the market after each fill
            until its next refresh, and let the quote go stale against a moving mid. In
            this market the second pair of effects wins: refreshing every 20 steps instead
            of every step costs about 30% of fills.
        seed: Seed or ``Generator``.

    Returns:
        A :class:`SimulationResult` whose ``fill_prices`` frame carries the efficient mid
        at fill time and at several markout horizons.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if requote_every < 1:
        raise ValueError("requote_every must be >= 1")
    market = market or MarketConfig()
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    book = LimitOrderBook()
    generator = OrderFlowGenerator(flow_config, market, rng)
    dt = horizon / n_steps

    times = np.linspace(0.0, horizon, n_steps + 1)
    mid = np.empty(n_steps + 1, dtype=np.float64)
    inventory = np.zeros(n_steps + 1, dtype=np.float64)
    cash = np.zeros(n_steps + 1, dtype=np.float64)
    bid_quotes = np.full(n_steps + 1, np.nan, dtype=np.float64)
    ask_quotes = np.full(n_steps + 1, np.nan, dtype=np.float64)
    buy_fills = np.zeros(n_steps + 1, dtype=np.float64)
    sell_fills = np.zeros(n_steps + 1, dtype=np.float64)

    efficient_ticks = float(market.initial_mid_ticks)
    mid[0] = market.to_currency(efficient_ticks)
    generator.seed_book(book, efficient_ticks, timestamp=0.0)
    fills: list[dict[str, float | str]] = []
    # (order_id, price_ticks) of the maker's live quote on each side, or None.
    resting_bid: tuple[int, int] | None = None
    resting_ask: tuple[int, int] | None = None

    for i in range(n_steps):
        now = float(times[i])
        state = MakerState(
            mid=market.to_currency(efficient_ticks),
            inventory=float(inventory[i]),
            cash=float(cash[i]),
            time=now,
            horizon=horizon,
        )

        if i % requote_every == 0:
            bid_price, ask_price = policy.quote(state)
            bid_tick = (
                int(np.floor(bid_price / market.tick_size)) if bid_price is not None else None
            )
            ask_tick = int(np.ceil(ask_price / market.tick_size)) if ask_price is not None else None
            resting_bid = _refresh_side(book, Side.BID, bid_tick, resting_bid, order_size, now)
            resting_ask = _refresh_side(book, Side.ASK, ask_tick, resting_ask, order_size, now)
        bid_quotes[i] = market.to_currency(resting_bid[1]) if resting_bid is not None else np.nan
        ask_quotes[i] = market.to_currency(resting_ask[1]) if resting_ask is not None else np.nan

        q, c = state.inventory, state.cash
        mid_before = state.mid  # the mid the maker quoted against, before this step moves it
        efficient_ticks, trades = generator.step(book, efficient_ticks, now, dt)

        for trade in trades:
            if trade.maker_owner != _MAKER:
                continue
            # The maker was resting, so it traded against the aggressor: a buyer lifting
            # our offer means we sold.
            signed = -float(trade.aggressor_side) * trade.size
            price = market.to_currency(trade.price_ticks)
            q += signed
            c -= signed * price
            c -= market.maker_fee * trade.size
            if signed > 0:
                buy_fills[i + 1 :] += trade.size
                if resting_bid is not None and trade.maker_order_id == resting_bid[0]:
                    resting_bid = None
            else:
                sell_fills[i + 1 :] += trade.size
                if resting_ask is not None and trade.maker_order_id == resting_ask[0]:
                    resting_ask = None
            fills.append(
                {
                    "time": now,
                    "step": i,
                    "side": "buy" if signed > 0 else "sell",
                    "price": price,
                    # The mid *before* this step's price move, which is what the maker
                    # quoted against. Recording the post-move mid instead would silently
                    # fold the adverse-selection loss into "spread captured" and make the
                    # first markout horizon blind to informed impact.
                    "mid": mid_before,
                    "size": trade.size,
                }
            )

        inventory[i + 1] = q
        cash[i + 1] = c
        mid[i + 1] = market.to_currency(efficient_ticks)

    book.cancel_all(_MAKER)
    fill_frame = pd.DataFrame(fills)
    return SimulationResult(
        times=times,
        mid=mid,
        inventory=inventory,
        cash=cash,
        bid_quotes=bid_quotes,
        ask_quotes=ask_quotes,
        buy_fills=buy_fills,
        sell_fills=sell_fills,
        fill_prices=fill_frame,
        engine="book",
        policy_name=getattr(policy, "name", type(policy).__name__),
    )
