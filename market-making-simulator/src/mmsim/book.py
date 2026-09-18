"""A price-time-priority limit order book.

The matching rules are the ones every lit venue uses:

1. **Price priority.** A buyer willing to pay more trades first.
2. **Time priority within a price level.** Among orders at the same price, the one that
   arrived first trades first. This is the reason queue position matters, and it is the
   single most important thing a market-making simulator has to get right -- a model that
   fills your order the instant a trade prints at your price will overstate your fill rate
   and understate adverse selection, which flatters every strategy you test.
3. **Resting price wins.** A marketable order executes at the price of the order it hits,
   not at its own limit, so the aggressor can receive price improvement.

Data structures: one ``dict`` from price level to a FIFO ``deque`` per side, plus a heap of
active price levels with lazy deletion for O(log n) best-price lookup. An ``O(1)`` index
from order id to its order object makes cancellation ``O(1)`` amortised -- cancels vastly
outnumber fills in real markets and in this simulator, so that is the operation to make
cheap.

The book knows nothing about strategies. Every order carries an ``owner`` tag, and
attribution happens in the simulation layer.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from collections.abc import Iterator

from .types import Order, OrderType, Side, Trade

__all__ = ["BookLevel", "LimitOrderBook"]

#: A price level as reported by :meth:`LimitOrderBook.depth`: ``(price_ticks, total_size)``.
BookLevel = tuple[int, float]


class LimitOrderBook:
    """A single-instrument limit order book with price-time priority.

    Args:
        allow_self_trade: If ``False`` (the default) an incoming order will not match
            against a resting order with the same ``owner`` tag; it skips past it in the
            queue instead. Real venues enforce self-trade prevention, and without it a
            market maker whose quotes cross would print fictitious trades with itself.
    """

    def __init__(self, allow_self_trade: bool = False) -> None:
        self._bids: dict[int, deque[Order]] = {}
        self._asks: dict[int, deque[Order]] = {}
        # Max-heap for bids (negated), min-heap for asks. Entries are lazily deleted:
        # a price stays on the heap after its level empties and is skipped on read.
        self._bid_heap: list[int] = []
        self._ask_heap: list[int] = []
        self._orders: dict[int, Order] = {}
        self._next_id = itertools.count(1)
        self.allow_self_trade = allow_self_trade
        self.trades: list[Trade] = []

    # ---------------------------------------------------------------- introspection

    def _side_maps(self, side: Side) -> tuple[dict[int, deque[Order]], list[int]]:
        return (self._bids, self._bid_heap) if side is Side.BID else (self._asks, self._ask_heap)

    @property
    def best_bid(self) -> int | None:
        """Highest bid price in ticks, or ``None`` if the bid side is empty."""
        while self._bid_heap:
            price = -self._bid_heap[0]
            if self._bids.get(price):
                return price
            heapq.heappop(self._bid_heap)  # lazily drop an emptied level
        return None

    @property
    def best_ask(self) -> int | None:
        """Lowest ask price in ticks, or ``None`` if the ask side is empty."""
        while self._ask_heap:
            price = self._ask_heap[0]
            if self._asks.get(price):
                return price
            heapq.heappop(self._ask_heap)
        return None

    def best(self, side: Side) -> int | None:
        """Best price on the given side."""
        return self.best_bid if side is Side.BID else self.best_ask

    @property
    def mid_ticks(self) -> float | None:
        """Midpoint of the best bid and offer, or ``None`` if either side is empty."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return 0.5 * (bid + ask)

    @property
    def spread_ticks(self) -> int | None:
        """Best ask minus best bid, or ``None`` if either side is empty."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return ask - bid

    def size_at(self, side: Side, price_ticks: int) -> float:
        """Total resting size at one price level."""
        levels, _ = self._side_maps(side)
        return sum(o.size for o in levels.get(price_ticks, ()))

    def depth(self, side: Side, n_levels: int = 5) -> list[BookLevel]:
        """The top ``n_levels`` price levels on one side, best first."""
        levels, _ = self._side_maps(side)
        prices = sorted((p for p, q in levels.items() if q), reverse=(side is Side.BID))[:n_levels]
        return [(p, self.size_at(side, p)) for p in prices]

    def queue_ahead(self, order_id: int) -> float:
        """Total size resting ahead of an order at its own price level.

        This is the quantity that decides whether a quote actually gets filled. A market
        maker joining the back of a 500-lot queue has a very different experience from one
        that just improved the price by a tick, and no model without a book can tell them
        apart.

        Raises:
            KeyError: if the order is not resting in the book.
        """
        order = self._orders[order_id]
        levels, _ = self._side_maps(order.side)
        ahead = 0.0
        for resting in levels.get(order.price_ticks, ()):
            if resting.order_id == order_id:
                return ahead
            ahead += resting.size
        raise KeyError(f"order {order_id} is not resting in the book")

    def __contains__(self, order_id: object) -> bool:
        return order_id in self._orders

    def __len__(self) -> int:
        """Number of resting orders."""
        return len(self._orders)

    def resting_orders(self, owner: str | None = None) -> Iterator[Order]:
        """Iterate resting orders, optionally filtered to one owner."""
        for order in self._orders.values():
            if owner is None or order.owner == owner:
                yield order

    # ---------------------------------------------------------------- mutation

    def submit_limit(
        self,
        side: Side,
        price_ticks: int,
        size: float,
        timestamp: float,
        owner: str = "anon",
    ) -> tuple[int, list[Trade]]:
        """Submit a limit order, matching any marketable portion first.

        Args:
            side: Buy or sell.
            price_ticks: Limit price.
            size: Quantity.
            timestamp: Arrival time; sets time priority.
            owner: Attribution tag.

        Returns:
            ``(order_id, trades)``. ``trades`` is empty for a passive order. The order is
            only added to the book if size remains after matching.
        """
        order = Order(
            order_id=next(self._next_id),
            side=side,
            price_ticks=price_ticks,
            size=size,
            timestamp=timestamp,
            owner=owner,
            order_type=OrderType.LIMIT,
        )
        trades = self._match(order, limit_price=price_ticks)
        if order.is_active:
            self._rest(order)
        return order.order_id, trades

    def submit_market(
        self, side: Side, size: float, timestamp: float, owner: str = "anon"
    ) -> tuple[int, list[Trade]]:
        """Submit a market order.

        Any unfilled remainder is **discarded**, not rested: a market order that exhausts
        the book has no limit price to rest at. The caller can detect this by comparing
        the traded size against ``size``.

        Returns:
            ``(order_id, trades)``.
        """
        order = Order(
            order_id=next(self._next_id),
            side=side,
            price_ticks=0,
            size=size,
            timestamp=timestamp,
            owner=owner,
            order_type=OrderType.MARKET,
        )
        trades = self._match(order, limit_price=None)
        return order.order_id, trades

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order.

        Returns:
            ``True`` if the order was resting and is now removed, ``False`` if it was
            already fully filled or never rested. Returning a bool rather than raising is
            deliberate: in a simulation a cancel racing a fill is normal, not exceptional.
        """
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        levels, _ = self._side_maps(order.side)
        queue = levels.get(order.price_ticks)
        if queue is None:  # pragma: no cover - index and levels cannot diverge
            return False
        try:
            queue.remove(order)
        except ValueError:  # pragma: no cover - same
            return False
        if not queue:
            del levels[order.price_ticks]
        return True

    def cancel_all(self, owner: str) -> int:
        """Cancel every resting order belonging to ``owner``. Returns how many."""
        ids = [o.order_id for o in self._orders.values() if o.owner == owner]
        return sum(self.cancel(i) for i in ids)

    # ---------------------------------------------------------------- internals

    def _rest(self, order: Order) -> None:
        levels, heap = self._side_maps(order.side)
        if order.price_ticks not in levels:
            levels[order.price_ticks] = deque()
            heapq.heappush(
                heap, -order.price_ticks if order.side is Side.BID else order.price_ticks
            )
        levels[order.price_ticks].append(order)  # FIFO: time priority
        self._orders[order.order_id] = order

    def _crosses(self, side: Side, limit_price: int | None, resting_price: int) -> bool:
        """Whether an incoming order at ``limit_price`` can trade at ``resting_price``."""
        if limit_price is None:  # market order: any price is acceptable
            return True
        return limit_price >= resting_price if side is Side.BID else limit_price <= resting_price

    def _match(self, incoming: Order, limit_price: int | None) -> list[Trade]:
        """Walk the opposite side of the book, filling ``incoming`` in price-time order.

        Self-trade prevention steps *past* the participant's own resting orders and keeps
        going to the next price level, rather than stopping. Stopping would silently
        under-fill an aggressor whose own quote happens to sit at the touch -- which is
        precisely the situation a market maker is in most of the time, so getting it wrong
        would bias every result in the repo.
        """
        opposite = incoming.side.opposite
        levels, _ = self._side_maps(opposite)
        trades: list[Trade] = []
        # Price levels consisting entirely of the aggressor's own orders. Almost always
        # empty, so the common path stays on the O(log n) heap lookup.
        blocked: set[int] = set()

        while incoming.is_active:
            best_price = self._best_excluding(opposite, blocked)
            if best_price is None or not self._crosses(incoming.side, limit_price, best_price):
                break
            queue = levels[best_price]

            skipped: list[Order] = []
            while queue and incoming.is_active:
                resting = queue[0]
                if not self.allow_self_trade and resting.owner == incoming.owner:
                    skipped.append(queue.popleft())
                    continue
                traded = min(incoming.size, resting.size)
                incoming.size -= traded
                resting.size -= traded
                trade = Trade(
                    price_ticks=best_price,
                    size=traded,
                    timestamp=incoming.timestamp,
                    aggressor_side=incoming.side,
                    maker_order_id=resting.order_id,
                    maker_owner=resting.owner,
                    taker_owner=incoming.owner,
                )
                trades.append(trade)
                self.trades.append(trade)
                if not resting.is_active:
                    queue.popleft()
                    self._orders.pop(resting.order_id, None)

            # Restore skipped same-owner orders at the front, preserving their queue order.
            for order in reversed(skipped):
                queue.appendleft(order)
            if not queue:
                del levels[best_price]
            elif skipped and incoming.is_active:
                # Only our own orders are left here; move on to the next price level.
                blocked.add(best_price)
        return trades

    def _best_excluding(self, side: Side, blocked: set[int]) -> int | None:
        """Best price on ``side``, ignoring levels in ``blocked``."""
        if not blocked:
            return self.best(side)
        levels, _ = self._side_maps(side)
        candidates = [p for p, q in levels.items() if q and p not in blocked]
        if not candidates:
            return None
        return max(candidates) if side is Side.BID else min(candidates)
