"""Core value types for the order book and the simulation.

Prices are integers throughout: **ticks**, not currency. A matching engine compares and
sorts prices constantly, and doing that in floating point invites the classic bug where
two orders that should sit at the same level land in different levels because
``0.1 + 0.2 != 0.3``. Conversion to currency happens once, at the boundary, via
:attr:`MarketConfig.tick_size`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
"""Canonical float array type."""

__all__ = [
    "Fill",
    "FloatArray",
    "MarketConfig",
    "Order",
    "OrderType",
    "Quote",
    "Side",
    "Trade",
]


class Side(IntEnum):
    """Which side of the book an order rests on or an aggressor trades against.

    An :class:`~enum.IntEnum` with values ``+1`` / ``-1`` so that ``side * price`` and
    ``side * size`` express signed quantities directly: a buy adds ``+size`` to inventory,
    a sell adds ``-size``.
    """

    BID = 1
    ASK = -1

    @property
    def opposite(self) -> Side:
        """The other side."""
        return Side.ASK if self is Side.BID else Side.BID

    @property
    def label(self) -> str:
        """``"bid"`` or ``"ask"``."""
        return "bid" if self is Side.BID else "ask"


class OrderType(StrEnum):
    """Order types the engine accepts."""

    LIMIT = "limit"
    MARKET = "market"


@dataclass(slots=True)
class Order:
    """A resting or incoming order.

    Attributes:
        order_id: Unique, monotonically increasing.
        side: Which side of the book.
        price_ticks: Limit price in ticks. Ignored for market orders.
        size: Remaining size. Mutated as the order fills.
        timestamp: Arrival time in seconds. Sets time priority within a price level.
        owner: Tag identifying who submitted it, e.g. ``"mm"`` or ``"noise"``. The
            simulation uses this to attribute fills without the book needing to know
            anything about strategies.
        order_type: Limit or market.
        original_size: Size at submission, kept so partial fills are measurable.
    """

    order_id: int
    side: Side
    price_ticks: int
    size: float
    timestamp: float
    owner: str = "anon"
    order_type: OrderType = OrderType.LIMIT
    original_size: float = field(default=0.0)

    def __post_init__(self) -> None:
        if self.size <= 0.0:
            raise ValueError("order size must be > 0")
        if self.original_size == 0.0:
            self.original_size = self.size

    @property
    def filled(self) -> float:
        """Size filled so far."""
        return self.original_size - self.size

    @property
    def is_active(self) -> bool:
        """Whether any size remains."""
        return self.size > 1e-12


@dataclass(frozen=True, slots=True)
class Trade:
    """A single match between an aggressor and one resting order.

    A market order that walks several levels produces several trades; that is deliberate,
    because the whole point of modelling the book is that a large order does not execute
    at one price.

    Attributes:
        price_ticks: Execution price (always the *resting* order's price).
        size: Size traded.
        timestamp: Execution time.
        aggressor_side: Side of the incoming order. ``Side.BID`` means a buyer lifted the
            offer, so the trade sign is positive for the taker.
        maker_order_id: The resting order that was hit.
        maker_owner: Owner tag of the resting order.
        taker_owner: Owner tag of the aggressor.
    """

    price_ticks: int
    size: float
    timestamp: float
    aggressor_side: Side
    maker_order_id: int
    maker_owner: str
    taker_owner: str

    @property
    def maker_side(self) -> Side:
        """The resting side. A buyer lifting the offer hits a resting ask."""
        return self.aggressor_side.opposite


@dataclass(frozen=True, slots=True)
class Fill:
    """One participant's view of a trade: signed size and price.

    Attributes:
        signed_size: ``+size`` if the participant bought, ``-size`` if it sold.
        price_ticks: Execution price.
        timestamp: Execution time.
        was_maker: Whether the participant was the resting side (and so earned the spread
            rather than paying it).
    """

    signed_size: float
    price_ticks: int
    timestamp: float
    was_maker: bool


@dataclass(frozen=True, slots=True)
class Quote:
    """A two-sided quote from a market maker, in ticks relative to nothing in particular.

    ``None`` on a side means "do not quote that side", which is how inventory limits are
    expressed: a maker at its long limit simply stops bidding.
    """

    bid_ticks: int | None
    ask_ticks: int | None
    bid_size: float = 1.0
    ask_size: float = 1.0

    def __post_init__(self) -> None:
        if (
            self.bid_ticks is not None
            and self.ask_ticks is not None
            and self.bid_ticks >= self.ask_ticks
        ):
            raise ValueError(
                f"quote is crossed: bid {self.bid_ticks} >= ask {self.ask_ticks}. "
                "A maker quoting through itself would trade with itself."
            )


@dataclass(frozen=True)
class MarketConfig:
    """Fixed properties of the simulated market.

    Attributes:
        tick_size: Currency value of one tick.
        initial_mid_ticks: Starting mid price in ticks.
        maker_fee: Fee per unit traded as a maker. Negative means a rebate, which is how
            most equity venues actually pay market makers.
        taker_fee: Fee per unit traded as a taker.
    """

    tick_size: float = 0.01
    initial_mid_ticks: int = 10_000
    maker_fee: float = 0.0
    taker_fee: float = 0.0

    def to_currency(self, ticks: float) -> float:
        """Convert a tick quantity to currency."""
        return float(ticks) * self.tick_size
