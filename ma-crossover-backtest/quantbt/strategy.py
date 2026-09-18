"""Strategy interface and the per-bar :class:`Context` a strategy sees.

A strategy is called once per bar at the close and can only observe bars up to and
including that close. It expresses intent as orders; the engine fills them on the next
bar. Strategies should be *functions of history*: hold parameters, not market state, so
they can be re-run on any window without leaking information across windows.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from quantbt.data import PriceData
from quantbt.execution import Order, OrderKind
from quantbt.portfolio import Portfolio


class LookaheadError(RuntimeError):
    """Raised when a strategy asks for data past the current bar."""


@dataclass
class Context:
    """Everything a strategy may touch on one bar. Created by the engine."""

    data: PriceData
    portfolio: Portfolio
    now: pd.Timestamp
    position_index: int  # integer location of ``now`` in data.index
    symbols: list[str]  # universe members on this bar
    params: Mapping[str, Any] = field(default_factory=dict)
    _orders: list[Order] = field(default_factory=list, repr=False)
    _records: dict[str, float] = field(default_factory=dict, repr=False)

    # -- data access (never past ``now``) ---------------------------------------------
    def history(
        self,
        field_name: str = "close",
        bars: int | None = None,
        symbols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """The last ``bars`` rows of ``field_name`` ending at ``now`` (inclusive)."""
        frame = self.data.field(field_name)
        stop = self.position_index + 1
        start = 0 if bars is None else max(0, stop - bars)
        cols = list(symbols) if symbols is not None else self.symbols
        return frame.iloc[start:stop][cols]

    def price(self, symbol: str, field_name: str = "close") -> float:
        """Latest value of ``field_name`` for ``symbol`` (NaN if it has no bar today)."""
        value = self.data.field(field_name).iat[self.position_index, self._col(symbol)]
        return float(cast("float", value))

    def _col(self, symbol: str) -> int:
        return int(self.data.close.columns.get_loc(symbol))  # type: ignore[arg-type]

    def has_price(self, symbol: str) -> bool:
        p = self.price(symbol)
        return p == p and p > 0

    # -- portfolio views ---------------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.portfolio.equity

    @property
    def cash(self) -> float:
        return self.portfolio.cash

    def quantity(self, symbol: str) -> float:
        return self.portfolio.quantity(symbol)

    def weight(self, symbol: str) -> float:
        return self.portfolio.weights().get(symbol, 0.0)

    @property
    def weights(self) -> dict[str, float]:
        return self.portfolio.weights()

    # -- orders --------------------------------------------------------------------------
    def order(self, symbol: str, quantity: float) -> Order:
        """Buy (``quantity > 0``) or sell a share count at the next bar."""
        return self._submit(symbol, OrderKind.QUANTITY, quantity)

    def order_target_weight(self, symbol: str, weight: float) -> Order:
        """Move ``symbol`` to ``weight`` of equity (sized at the fill)."""
        return self._submit(symbol, OrderKind.TARGET_WEIGHT, weight)

    def order_target_weights(
        self, weights: Mapping[str, float], close_others: bool = True
    ) -> list[Order]:
        """Rebalance to ``weights``; with ``close_others`` any held symbol not listed goes to 0."""
        orders = [self.order_target_weight(s, w) for s, w in weights.items()]
        if close_others:
            for symbol, pos in self.portfolio.positions.items():
                if symbol not in weights and pos.quantity != 0:
                    orders.append(self.order_target_weight(symbol, 0.0))
        return orders

    def _submit(self, symbol: str, kind: OrderKind, amount: float) -> Order:
        if symbol not in self.symbols:
            raise KeyError(
                f"{symbol!r} is not in the universe on {self.now.date()}; "
                "trading it would be survivorship bias (or a typo)"
            )
        order = Order(symbol=symbol, kind=kind, amount=amount, submitted=self.now)
        self._orders.append(order)
        return order

    def record(self, **values: float) -> None:
        """Store custom per-bar diagnostics (z-scores, signals...) for the results layer."""
        self._records.update({k: float(v) for k, v in values.items()})


class Strategy(ABC):
    """Base class. Override :meth:`on_bar`; optionally :meth:`on_start` and ``warmup``.

    ``warmup`` is the number of bars of history the strategy needs before its first
    decision; the engine warns if fewer are available before the evaluation start.
    """

    warmup: int = 0
    name: str = "strategy"

    def on_start(self, ctx: Context) -> None:  # noqa: B027 (intentional no-op hook)
        """Called once before the first evaluated bar."""

    @abstractmethod
    def on_bar(self, ctx: Context) -> None:
        """Called at the close of every evaluated bar."""

    def params(self) -> dict[str, Any]:
        """Parameters for reporting; defaults to public instance attributes."""
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}
