"""Point-in-time universes.

A universe answers "which symbols could I have traded on date ``t``?". Building it from
*today's* index membership is the survivorship bias: every name that was delisted,
acquired or dropped is missing, and the survivors are, by construction, the winners.

Two safeguards are built in:

* :class:`PointInTimeUniverse` takes explicit membership intervals, so a symbol can enter
  and leave. The engine refuses orders for symbols outside the universe on that date
  and force-liquidates positions in symbols that leave.
* :func:`from_data` derives membership from where a symbol actually has prices, so a
  name that stops trading drops out instead of being carried as a stale position.

Neither replaces a real point-in-time constituent file, which is what a production
system needs; see ``README.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd

from quantbt.data import PriceData


@runtime_checkable
class Universe(Protocol):
    """Anything that can list the tradable symbols on a date."""

    @property
    def all_symbols(self) -> list[str]: ...

    def members(self, date: pd.Timestamp) -> list[str]: ...


@dataclass(frozen=True)
class StaticUniverse:
    """The same symbols on every date. Only appropriate for instruments that cannot die
    (broad index ETFs, futures continuations) and even then carries selection bias."""

    symbols: Sequence[str]

    @property
    def all_symbols(self) -> list[str]:
        return list(self.symbols)

    def members(self, date: pd.Timestamp) -> list[str]:
        return list(self.symbols)


@dataclass(frozen=True)
class PointInTimeUniverse:
    """Symbols with explicit membership intervals ``[start, end]`` (inclusive, ``None`` = open)."""

    memberships: Mapping[str, Sequence[tuple[pd.Timestamp | None, pd.Timestamp | None]]]
    _order: list[str] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_order", list(self.memberships))
        for symbol, intervals in self.memberships.items():
            for start, end in intervals:
                if start is not None and end is not None and start > end:
                    raise ValueError(f"{symbol}: interval start {start} after end {end}")

    @property
    def all_symbols(self) -> list[str]:
        return list(self._order)

    def members(self, date: pd.Timestamp) -> list[str]:
        out = []
        for symbol in self._order:
            for start, end in self.memberships[symbol]:
                if (start is None or start <= date) and (end is None or date <= end):
                    out.append(symbol)
                    break
        return out

    @classmethod
    def from_table(cls, rows: Iterable[tuple[str, str | None, str | None]]) -> PointInTimeUniverse:
        """Build from ``(symbol, start, end)`` rows, e.g. read from a constituent CSV."""
        memberships: dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None]]] = {}
        for symbol, start, end in rows:
            s = pd.Timestamp(start) if start else None
            e = pd.Timestamp(end) if end else None
            memberships.setdefault(symbol, []).append((s, e))
        return cls(memberships)


def from_data(data: PriceData, min_history: int = 0) -> PointInTimeUniverse:
    """Membership = the span over which a symbol has a close, optionally after ``min_history`` bars.

    A symbol whose prices stop (delisting) leaves the universe on its last bar.
    """
    memberships: dict[str, list[tuple[pd.Timestamp | None, pd.Timestamp | None]]] = {}
    for symbol in data.symbols:
        valid = data.close[symbol].dropna()
        if len(valid) <= min_history:
            memberships[symbol] = []
            continue
        first = pd.Timestamp(valid.index[min_history])
        last = pd.Timestamp(valid.index[-1])
        memberships[symbol] = [(first, last)]
    return PointInTimeUniverse(memberships)
