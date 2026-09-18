"""Orders, fills, commission and slippage models, and the execution simulator.

The simulator only ever fills an order on a bar *after* the one on which it was
submitted. That single rule is the framework's main defence against lookahead: a
strategy sees the close of bar ``t`` and gets the open (default) or close of ``t+1``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol

import pandas as pd


class OrderKind(StrEnum):
    QUANTITY = "quantity"
    TARGET_WEIGHT = "target_weight"


@dataclass(frozen=True)
class Order:
    """An instruction submitted by a strategy at the close of ``submitted``.

    ``amount`` is a share count for ``QUANTITY`` orders (negative = sell) or a fraction of
    equity for ``TARGET_WEIGHT`` orders (negative = short).
    """

    symbol: str
    kind: OrderKind
    amount: float
    submitted: pd.Timestamp
    id: int = field(default_factory=itertools.count().__next__)

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount):
            raise ValueError(f"order amount must be finite, got {self.amount}")


@dataclass(frozen=True)
class Fill:
    order_id: int
    symbol: str
    timestamp: pd.Timestamp
    quantity: float  # signed
    price: float  # after slippage
    commission: float
    reference_price: float  # the bar price before slippage

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price

    @property
    def slippage_cost(self) -> float:
        return abs(self.quantity) * abs(self.price - self.reference_price)


@dataclass(frozen=True)
class Rejection:
    order: Order
    timestamp: pd.Timestamp
    reason: str


# -- commission ---------------------------------------------------------------------


class CommissionModel(Protocol):
    def calculate(self, quantity: float, price: float) -> float:
        """Commission for filling ``|quantity|`` shares at ``price``."""
        ...

    def shares_for_notional(self, notional: float, price: float) -> float:
        """Largest share count whose cost *including commission* is ``notional``."""
        ...


@dataclass(frozen=True)
class NoCommission:
    def calculate(self, quantity: float, price: float) -> float:
        return 0.0

    def shares_for_notional(self, notional: float, price: float) -> float:
        return notional / price


@dataclass(frozen=True)
class PercentageCommission:
    """A fraction of the traded notional (``rate=0.0001`` is 1 bp)."""

    rate: float

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("rate must be >= 0")

    def calculate(self, quantity: float, price: float) -> float:
        return abs(quantity) * price * self.rate

    def shares_for_notional(self, notional: float, price: float) -> float:
        return notional / (price * (1.0 + self.rate))


@dataclass(frozen=True)
class PerShareCommission:
    """A fixed amount per share with a per-order minimum (Interactive Brokers style)."""

    per_share: float = 0.005
    minimum: float = 1.0

    def calculate(self, quantity: float, price: float) -> float:
        if quantity == 0:
            return 0.0
        return max(abs(quantity) * self.per_share, self.minimum)

    def shares_for_notional(self, notional: float, price: float) -> float:
        shares = notional / (price + self.per_share)
        if shares * self.per_share < self.minimum:
            shares = (notional - self.minimum) / price
        return max(shares, 0.0)


# -- slippage -----------------------------------------------------------------------


class SlippageModel(Protocol):
    def fill_price(self, price: float, quantity: float, volume: float) -> float:
        """Execution price for a signed ``quantity`` against a bar with ``volume`` shares."""
        ...


@dataclass(frozen=True)
class NoSlippage:
    def fill_price(self, price: float, quantity: float, volume: float) -> float:
        return price


@dataclass(frozen=True)
class FixedBpsSlippage:
    """Pay ``bps`` of the price against you on every fill (half-spread plus impact)."""

    bps: float = 5.0

    def __post_init__(self) -> None:
        if self.bps < 0:
            raise ValueError("bps must be >= 0")

    def fill_price(self, price: float, quantity: float, volume: float) -> float:
        adj = self.bps / 1e4
        return price * (1.0 + adj) if quantity > 0 else price * (1.0 - adj)


@dataclass(frozen=True)
class VolumeShareSlippage:
    """Quadratic impact in the order's share of bar volume (zipline's model).

    ``price_impact`` of 0.1 means an order that is 10% of the bar's volume moves the
    price by ``0.1 * 0.1**2 = 0.1%``.
    """

    price_impact: float = 0.1
    fixed_bps: float = 0.0

    def fill_price(self, price: float, quantity: float, volume: float) -> float:
        share = abs(quantity) / volume if volume and volume > 0 else 0.0
        impact = self.price_impact * share**2 + self.fixed_bps / 1e4
        return price * (1.0 + impact) if quantity > 0 else price * (1.0 - impact)


# -- the simulator ------------------------------------------------------------------

FillAt = Literal["open", "close"]


@dataclass(frozen=True)
class ExecutionSimulator:
    """Turns pending orders into fills against the *next* bar.

    ``fill_at`` selects the open (default) or close of that bar. ``max_volume_share``
    truncates any fill to that fraction of the bar's volume; the remainder is cancelled,
    not carried. ``lot_size`` rounds quantities toward zero to a multiple (``None`` keeps
    fractional shares, which is what makes the engine agree with a vectorised model).
    """

    commission: CommissionModel = field(default_factory=lambda: PercentageCommission(1e-4))
    slippage: SlippageModel = field(default_factory=lambda: FixedBpsSlippage(5.0))
    fill_at: FillAt = "open"
    max_volume_share: float | None = None
    lot_size: float | None = None

    def __post_init__(self) -> None:
        if self.fill_at not in ("open", "close"):
            raise ValueError("fill_at must be 'open' or 'close'")
        if self.max_volume_share is not None and not 0 < self.max_volume_share <= 1:
            raise ValueError("max_volume_share must be in (0, 1]")

    def size_target(
        self,
        order: Order,
        price: float,
        equity: float,
        current_quantity: float,
    ) -> float:
        """Signed share quantity that moves ``current_quantity`` to ``order``'s target."""
        if order.kind is OrderKind.QUANTITY:
            return order.amount
        target_value = order.amount * equity
        delta_value = target_value - current_quantity * price
        if delta_value > 0:
            return self.commission.shares_for_notional(delta_value, price)
        return delta_value / price

    def fill(
        self,
        order: Order,
        timestamp: pd.Timestamp,
        price: float,
        volume: float,
        equity: float,
        current_quantity: float,
    ) -> Fill | Rejection:
        if not (math.isfinite(price) and price > 0):
            return Rejection(order, timestamp, "no price on fill bar")
        # Size at the price the order will actually pay: slippage direction follows the
        # sign of the trade, and impact models depend on its size, so size twice.
        estimate = self.size_target(order, price, equity, current_quantity)
        if estimate == 0:
            return Rejection(order, timestamp, "zero quantity after sizing")
        fill_price = self.slippage.fill_price(price, estimate, volume)
        quantity = self.size_target(order, fill_price, equity, current_quantity)
        if self.lot_size:
            quantity = math.trunc(quantity / self.lot_size) * self.lot_size
        if self.max_volume_share is not None and math.isfinite(volume) and volume > 0:
            cap = self.max_volume_share * volume
            if abs(quantity) > cap:
                quantity = math.copysign(cap, quantity)
        if quantity == 0:
            return Rejection(order, timestamp, "zero quantity after sizing")
        fill_price = self.slippage.fill_price(price, quantity, volume)
        commission = self.commission.calculate(quantity, fill_price)
        return Fill(
            order_id=order.id,
            symbol=order.symbol,
            timestamp=timestamp,
            quantity=quantity,
            price=fill_price,
            commission=commission,
            reference_price=price,
        )
