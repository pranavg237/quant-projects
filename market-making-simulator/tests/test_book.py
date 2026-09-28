"""The matching engine.

A market-making study is only as trustworthy as its matching engine, so these tests are
deliberately picky about the invariants that flatter a strategy when they are broken:
time priority (get it wrong and your fill rate is overstated), resting-price execution
(get it wrong and you invent price improvement), and self-trade prevention.
"""

from __future__ import annotations

import pytest

from mmsim.book import LimitOrderBook
from mmsim.types import Side


def test_empty_book_reports_nothing(book: LimitOrderBook) -> None:
    assert book.best_bid is None
    assert book.best_ask is None
    assert book.mid_ticks is None
    assert book.spread_ticks is None
    assert len(book) == 0
    assert book.depth(Side.BID) == []
    assert list(book.resting_orders()) == []


def test_touch_and_depth(populated_book: LimitOrderBook) -> None:
    assert populated_book.best_bid == 99
    assert populated_book.best_ask == 101
    assert populated_book.mid_ticks == 100.0
    assert populated_book.spread_ticks == 2
    assert populated_book.depth(Side.BID) == [(99, 8.0), (98, 6.0)]
    assert populated_book.depth(Side.ASK) == [(101, 7.0), (102, 4.0)]
    assert populated_book.depth(Side.ASK, n_levels=1) == [(101, 7.0)]
    assert populated_book.size_at(Side.BID, 99) == 8.0
    assert populated_book.size_at(Side.BID, 50) == 0.0
    assert len(populated_book) == 6


def test_price_priority(populated_book: LimitOrderBook) -> None:
    """A market sell takes the best bid first, then walks down."""
    _, trades = populated_book.submit_market(Side.ASK, 10.0, 1.0, "taker")
    assert [t.price_ticks for t in trades] == [99, 99, 98]
    assert [t.size for t in trades] == [5.0, 3.0, 2.0]
    assert populated_book.best_bid == 98
    assert populated_book.size_at(Side.BID, 98) == 4.0


def test_time_priority_within_a_level(populated_book: LimitOrderBook) -> None:
    """Alice was first at 99, so Alice fills first -- this is the whole point of a queue."""
    _, trades = populated_book.submit_market(Side.ASK, 6.0, 1.0, "taker")
    assert [t.maker_owner for t in trades] == ["alice", "bob"]
    assert [t.size for t in trades] == [5.0, 1.0]


def test_new_order_joins_the_back_of_its_level(populated_book: LimitOrderBook) -> None:
    order_id, trades = populated_book.submit_limit(Side.BID, 99, 2.0, 1.0, "dave")
    assert trades == []
    assert populated_book.queue_ahead(order_id) == 8.0
    assert populated_book.size_at(Side.BID, 99) == 10.0


def test_queue_ahead_for_a_missing_order(populated_book: LimitOrderBook) -> None:
    with pytest.raises(KeyError):
        populated_book.queue_ahead(9999)


def test_execution_happens_at_the_resting_price(populated_book: LimitOrderBook) -> None:
    """A buyer willing to pay 105 still pays 101 -- price improvement accrues to the taker."""
    _, trades = populated_book.submit_limit(Side.BID, 105, 5.0, 1.0, "taker")
    assert [t.price_ticks for t in trades] == [101, 101]
    assert sum(t.size for t in trades) == 5.0


def test_marketable_limit_rests_its_remainder(populated_book: LimitOrderBook) -> None:
    order_id, trades = populated_book.submit_limit(Side.BID, 101, 10.0, 1.0, "taker")
    assert sum(t.size for t in trades) == 7.0  # all of the 101 level
    assert populated_book.best_bid == 101  # the unfilled 3 rests and becomes the new touch
    assert populated_book.size_at(Side.BID, 101) == 3.0
    assert order_id in populated_book


def test_passive_limit_does_not_cross(populated_book: LimitOrderBook) -> None:
    order_id, trades = populated_book.submit_limit(Side.BID, 100, 5.0, 1.0, "dave")
    assert trades == []
    assert populated_book.best_bid == 100
    assert populated_book.spread_ticks == 1
    assert order_id in populated_book


def test_market_order_remainder_is_discarded(book: LimitOrderBook) -> None:
    """There is no limit price to rest a market order at, so the remainder disappears."""
    book.submit_limit(Side.ASK, 101, 2.0, 0.0, "maker")
    _, trades = book.submit_market(Side.BID, 10.0, 1.0, "taker")
    assert sum(t.size for t in trades) == 2.0
    assert len(book) == 0
    assert book.best_ask is None


def test_market_order_into_an_empty_book(book: LimitOrderBook) -> None:
    _, trades = book.submit_market(Side.BID, 5.0, 0.0, "taker")
    assert trades == []


def test_cancel(populated_book: LimitOrderBook) -> None:
    order_id, _ = populated_book.submit_limit(Side.BID, 97, 4.0, 1.0, "dave")
    assert populated_book.cancel(order_id) is True
    assert populated_book.size_at(Side.BID, 97) == 0.0
    # Cancelling twice is not an error: in a simulation a cancel racing a fill is normal.
    assert populated_book.cancel(order_id) is False
    assert populated_book.cancel(123456) is False


def test_cancelling_the_whole_level_removes_it(book: LimitOrderBook) -> None:
    order_id, _ = book.submit_limit(Side.BID, 99, 1.0, 0.0, "a")
    book.cancel(order_id)
    assert book.best_bid is None
    # And the price can be re-used afterwards without leaking the stale heap entry.
    book.submit_limit(Side.BID, 99, 2.0, 1.0, "b")
    assert book.best_bid == 99
    assert book.size_at(Side.BID, 99) == 2.0


def test_cancel_all_by_owner(populated_book: LimitOrderBook) -> None:
    assert populated_book.cancel_all("alice") == 2
    assert populated_book.size_at(Side.BID, 99) == 3.0
    assert populated_book.size_at(Side.ASK, 101) == 4.0
    assert populated_book.cancel_all("nobody") == 0


def test_filled_orders_leave_the_index(populated_book: LimitOrderBook) -> None:
    populated_book.submit_market(Side.ASK, 8.0, 1.0, "taker")
    assert populated_book.size_at(Side.BID, 99) == 0.0
    assert populated_book.best_bid == 98
    assert len(list(populated_book.resting_orders())) == len(populated_book)


def test_partial_fill_keeps_queue_position(book: LimitOrderBook) -> None:
    first, _ = book.submit_limit(Side.BID, 99, 10.0, 0.0, "a")
    book.submit_limit(Side.BID, 99, 5.0, 0.1, "b")
    book.submit_market(Side.ASK, 4.0, 1.0, "taker")
    assert book.size_at(Side.BID, 99) == 11.0
    assert book.queue_ahead(first) == 0.0  # still at the front, just smaller
    order = next(o for o in book.resting_orders("a"))
    assert order.size == 6.0
    assert order.filled == 4.0
    assert order.original_size == 10.0


def test_self_trade_prevention_skips_own_order(book: LimitOrderBook) -> None:
    """The aggressor steps past its own resting order and trades with the next one."""
    book.submit_limit(Side.ASK, 101, 3.0, 0.0, "mm")
    book.submit_limit(Side.ASK, 101, 4.0, 0.1, "other")
    _, trades = book.submit_market(Side.BID, 2.0, 1.0, "mm")
    assert [t.maker_owner for t in trades] == ["other"]
    assert book.size_at(Side.ASK, 101) == 5.0
    # ...and our own order keeps its place at the front of the queue.
    own = next(o for o in book.resting_orders("mm"))
    assert book.queue_ahead(own.order_id) == 0.0


def test_self_trade_prevention_continues_to_the_next_level(book: LimitOrderBook) -> None:
    """A level that is entirely our own must not stop the walk -- it must be stepped over.

    If it stopped, an aggressor whose own quote sits at the touch would be silently
    under-filled, which is the situation a market maker is in essentially always.
    """
    book.submit_limit(Side.ASK, 101, 5.0, 0.0, "mm")
    book.submit_limit(Side.ASK, 102, 4.0, 0.1, "other")
    _, trades = book.submit_market(Side.BID, 3.0, 1.0, "mm")
    assert [(t.price_ticks, t.maker_owner) for t in trades] == [(102, "other")]
    assert book.size_at(Side.ASK, 101) == 5.0
    assert book.size_at(Side.ASK, 102) == 1.0


def test_self_trade_allowed_when_configured() -> None:
    b = LimitOrderBook(allow_self_trade=True)
    b.submit_limit(Side.ASK, 101, 5.0, 0.0, "mm")
    _, trades = b.submit_market(Side.BID, 2.0, 1.0, "mm")
    assert [t.maker_owner for t in trades] == ["mm"]


def test_trade_records_are_complete(populated_book: LimitOrderBook) -> None:
    _, trades = populated_book.submit_market(Side.ASK, 2.0, 7.5, "taker")
    trade = trades[0]
    assert trade.aggressor_side is Side.ASK
    assert trade.maker_side is Side.BID
    assert trade.taker_owner == "taker"
    assert trade.maker_owner == "alice"
    assert trade.timestamp == 7.5
    assert trade.price_ticks == 99
    assert populated_book.trades == trades  # the book keeps a full tape


def test_conservation_of_size(populated_book: LimitOrderBook) -> None:
    """Nothing is created or destroyed: traded + resting == originally submitted."""
    before = sum(o.size for o in populated_book.resting_orders())
    _, trades = populated_book.submit_market(Side.ASK, 9.0, 1.0, "taker")
    after = sum(o.size for o in populated_book.resting_orders())
    assert before - after == pytest.approx(sum(t.size for t in trades))


def test_order_validation() -> None:
    from mmsim.types import Order

    with pytest.raises(ValueError, match="size must be"):
        Order(1, Side.BID, 100, 0.0, 0.0)


def test_side_helpers() -> None:
    assert Side.BID.opposite is Side.ASK
    assert Side.ASK.opposite is Side.BID
    assert Side.BID.label == "bid"
    assert int(Side.BID) == 1
    assert int(Side.ASK) == -1
