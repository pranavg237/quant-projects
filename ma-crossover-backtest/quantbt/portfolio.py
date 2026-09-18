"""Cash, positions, mark-to-market and round-trip bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from quantbt.execution import Fill


@dataclass
class Position:
    symbol: str
    quantity: float = 0.0
    cost_basis: float = 0.0  # average price paid for the open quantity
    last_price: float = float("nan")

    @property
    def market_value(self) -> float:
        return self.quantity * self.last_price if self.quantity else 0.0


@dataclass(frozen=True)
class RoundTrip:
    """A position opened and fully closed (or flipped), with realised P&L net of costs."""

    symbol: str
    entry: pd.Timestamp
    exit: pd.Timestamp
    direction: int  # +1 long, -1 short
    quantity: float
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    bars_held: int


@dataclass
class _OpenTrip:
    entry: pd.Timestamp
    direction: int
    quantity: float
    entry_notional: float
    costs: float
    bars: int = 0


@dataclass
class Portfolio:
    """Holds cash and positions; applies fills; marks to market.

    Cash may go negative (a margin loan); short positions are allowed. Whether a
    strategy *should* do either is a strategy-level constraint, not a bookkeeping one.
    """

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    fills: list[Fill] = field(default_factory=list)
    round_trips: list[RoundTrip] = field(default_factory=list)
    _open: dict[str, _OpenTrip] = field(default_factory=dict)
    total_commission: float = 0.0
    total_slippage: float = 0.0

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions.values())

    def quantity(self, symbol: str) -> float:
        pos = self.positions.get(symbol)
        return pos.quantity if pos else 0.0

    def weights(self) -> dict[str, float]:
        eq = self.equity
        if eq == 0:
            return {s: 0.0 for s in self.positions}
        return {s: p.market_value / eq for s, p in self.positions.items()}

    def accrue_interest(self, rate: float) -> None:
        """Apply one period of ``rate`` to the cash balance (negative cash pays it)."""
        self.cash *= 1.0 + rate

    def mark(self, prices: dict[str, float]) -> None:
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is not None and price == price:  # not NaN
                pos.last_price = price
        for trip in self._open.values():
            trip.bars += 1

    def apply_fill(self, fill: Fill) -> None:
        pos = self.positions.setdefault(symbol := fill.symbol, Position(symbol))
        old_qty = pos.quantity
        new_qty = old_qty + fill.quantity
        self.cash -= fill.quantity * fill.price + fill.commission
        self.total_commission += fill.commission
        self.total_slippage += fill.slippage_cost
        self.fills.append(fill)

        # cost basis: average in when adding, unchanged when reducing
        if old_qty == 0 or (old_qty > 0) == (fill.quantity > 0):
            total_cost = pos.cost_basis * abs(old_qty) + fill.price * abs(fill.quantity)
            pos.cost_basis = total_cost / abs(new_qty) if new_qty else 0.0
        elif abs(new_qty) < 1e-12 or (old_qty > 0) != (new_qty > 0):
            # closed or flipped: the remainder (if any) starts a new basis at the fill price
            pos.cost_basis = fill.price if abs(new_qty) >= 1e-12 else 0.0

        pos.quantity = 0.0 if abs(new_qty) < 1e-12 else new_qty
        pos.last_price = fill.price
        self._track_round_trip(fill, old_qty, pos.quantity)

    def _track_round_trip(self, fill: Fill, old_qty: float, new_qty: float) -> None:
        symbol = fill.symbol
        direction = 1 if fill.quantity > 0 else -1
        trip = self._open.get(symbol)
        if old_qty == 0:
            self._open[symbol] = _OpenTrip(
                fill.timestamp, direction, abs(fill.quantity), fill.notional, fill.commission
            )
            return
        assert trip is not None
        same_side = (old_qty > 0) == (fill.quantity > 0)
        if same_side:
            trip.quantity += abs(fill.quantity)
            trip.entry_notional += fill.notional
            trip.costs += fill.commission
            return
        # reducing, closing or flipping
        closed_qty = min(abs(fill.quantity), abs(old_qty))
        avg_entry = trip.entry_notional / trip.quantity
        exit_notional = closed_qty * fill.price
        cost_share = fill.commission * (closed_qty / abs(fill.quantity))
        entry_cost_share = trip.costs * (closed_qty / trip.quantity)
        pnl = (
            trip.direction * (exit_notional - closed_qty * avg_entry)
            - cost_share
            - entry_cost_share
        )
        if new_qty == 0 or (new_qty > 0) != (old_qty > 0):
            self.round_trips.append(
                RoundTrip(
                    symbol=symbol,
                    entry=trip.entry,
                    exit=fill.timestamp,
                    direction=trip.direction,
                    quantity=trip.quantity,
                    entry_price=avg_entry,
                    exit_price=fill.price,
                    pnl=pnl,
                    return_pct=pnl / (trip.quantity * avg_entry),
                    bars_held=trip.bars,
                )
            )
            del self._open[symbol]
            if new_qty != 0:  # flipped: remainder opens a new trip
                remainder = abs(new_qty)
                self._open[symbol] = _OpenTrip(
                    fill.timestamp,
                    direction,
                    remainder,
                    remainder * fill.price,
                    fill.commission - cost_share,
                )
        else:
            # partial reduction: realise proportionally, keep the trip open
            trip.quantity -= closed_qty
            trip.entry_notional -= closed_qty * avg_entry
            trip.costs -= entry_cost_share
            self.round_trips.append(
                RoundTrip(
                    symbol=symbol,
                    entry=trip.entry,
                    exit=fill.timestamp,
                    direction=trip.direction,
                    quantity=closed_qty,
                    entry_price=avg_entry,
                    exit_price=fill.price,
                    pnl=pnl,
                    return_pct=pnl / (closed_qty * avg_entry),
                    bars_held=trip.bars,
                )
            )
