"""Quoting policies.

A policy is a pure function from the maker's state to a pair of prices. Keeping them
stateless-by-default and free of any simulator knowledge means the same policy object runs
unchanged in the idealised Avellaneda-Stoikov world and in the full order-book simulator,
which is what makes the comparison between those two worlds meaningful.

Three policies are implemented, and the comparison between them is the point of the repo:

``AvellanedaStoikovPolicy``
    Quotes symmetrically around the *reservation* price, so inventory skews both sides.

``SymmetricPolicy``
    Quotes symmetrically around the *mid*, ignoring inventory. The naive benchmark.

``InventoryLimitPolicy``
    Symmetric around the mid, but stops quoting a side at a hard position limit. This is
    what most simple market makers actually do, and it is a much fairer benchmark than
    pure symmetric quoting -- it isolates how much of Avellaneda-Stoikov's advantage comes
    from *continuous* skewing rather than from merely having some inventory control.

All three are compared at a **matched average spread**, so they trade at comparable rates
and the comparison measures inventory management rather than how wide you are willing to
quote. That is the same control Avellaneda and Stoikov use in their own paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .avellaneda_stoikov import (
    AvellanedaStoikovParams,
    HorizonMode,
    optimal_quotes,
    optimal_spread,
)

__all__ = [
    "AvellanedaStoikovPolicy",
    "InventoryLimitPolicy",
    "MakerState",
    "QuotePolicy",
    "SymmetricPolicy",
    "average_optimal_spread",
]


@dataclass(frozen=True, slots=True)
class MakerState:
    """Everything a policy is allowed to see.

    Attributes:
        mid: Current efficient mid price.
        inventory: Signed position.
        cash: Cash balance.
        time: Elapsed time.
        horizon: Total session length.
    """

    mid: float
    inventory: float
    cash: float
    time: float
    horizon: float

    @property
    def time_remaining(self) -> float:
        """``horizon - time``, floored at zero."""
        return max(self.horizon - self.time, 0.0)

    @property
    def mark_to_market(self) -> float:
        """Cash plus inventory marked at the mid."""
        return self.cash + self.inventory * self.mid


@runtime_checkable
class QuotePolicy(Protocol):
    """A two-sided quoting rule."""

    name: str

    def quote(self, state: MakerState) -> tuple[float | None, float | None]:
        """Return ``(bid_price, ask_price)``; ``None`` on a side means do not quote it."""
        ...


def average_optimal_spread(params: AvellanedaStoikovParams) -> float:
    r"""Time-average of the Avellaneda-Stoikov spread over a whole session.

    In finite-horizon mode the spread is :math:`\gamma\sigma^2(T-t) + \text{markup}`, whose
    average over :math:`t \in [0, T]` is :math:`\tfrac12\gamma\sigma^2 T + \text{markup}`.
    In stationary mode the spread is constant.

    This is the number the benchmark policies are matched to, so that every strategy quotes
    the same average width and the comparison isolates inventory management.
    """
    markup = float(optimal_spread(0.0, params))
    if params.horizon_mode is HorizonMode.STATIONARY:
        return float(optimal_spread(params.horizon, params))
    return markup + 0.5 * params.gamma * params.sigma**2 * params.horizon


@dataclass(frozen=True)
class AvellanedaStoikovPolicy:
    """Quotes around the reservation price with the model-optimal spread."""

    params: AvellanedaStoikovParams
    name: str = "Avellaneda-Stoikov"

    def quote(self, state: MakerState) -> tuple[float | None, float | None]:
        """Delegate to :func:`~mmsim.avellaneda_stoikov.optimal_quotes`."""
        return optimal_quotes(state.mid, state.inventory, state.time, self.params)


@dataclass(frozen=True)
class SymmetricPolicy:
    """Quotes a fixed half-spread either side of the mid, ignoring inventory entirely.

    The naive benchmark. It has no mechanism at all for controlling inventory: its position
    is a random walk driven by the imbalance of buy and sell fills, and its variance grows
    linearly in time without bound.
    """

    half_spread: float
    name: str = "Symmetric"

    def __post_init__(self) -> None:
        if self.half_spread <= 0.0:
            raise ValueError("half_spread must be > 0")

    def quote(self, state: MakerState) -> tuple[float | None, float | None]:
        """Mid plus and minus the fixed half-spread."""
        return state.mid - self.half_spread, state.mid + self.half_spread


@dataclass(frozen=True)
class InventoryLimitPolicy:
    """Symmetric quoting with a hard position limit -- the realistic naive strategy.

    Included because comparing Avellaneda-Stoikov against a policy with *no* inventory
    control at all is too easy a win. A hard limit is what a simple desk actually runs, and
    the interesting question is how much the continuous skew adds on top of it.
    """

    half_spread: float
    max_inventory: float
    name: str = "Symmetric + position limit"

    def __post_init__(self) -> None:
        if self.half_spread <= 0.0:
            raise ValueError("half_spread must be > 0")
        if self.max_inventory <= 0.0:
            raise ValueError("max_inventory must be > 0")

    def quote(self, state: MakerState) -> tuple[float | None, float | None]:
        """Symmetric quotes, with a side withdrawn once the limit is reached."""
        bid: float | None = state.mid - self.half_spread
        ask: float | None = state.mid + self.half_spread
        if state.inventory >= self.max_inventory:
            bid = None
        if state.inventory <= -self.max_inventory:
            ask = None
        return bid, ask
