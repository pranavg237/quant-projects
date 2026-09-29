"""
Run with: python3 -m unittest discover -s tests

These tests never touch the network - compute_signal/position_size/decide_order
are pure functions, tested with synthetic price series.
"""

import datetime as dt
import os
import sys
import unittest
from typing import Any, ClassVar

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy import (
    Signal,
    completed_bars,
    compute_signal,
    decide_order,
    position_size,
    validate_bars,
)


class TestComputeSignal(unittest.TestCase):
    def test_uptrend_gives_long_signal(self):
        # steadily rising prices -> short MA ends up above long MA
        closes = pd.Series(range(1, 251))
        signal = compute_signal(closes, short_window=10, long_window=50)
        self.assertEqual(signal, Signal.LONG)

    def test_downtrend_gives_flat_signal(self):
        closes = pd.Series(range(250, 0, -1))
        signal = compute_signal(closes, short_window=10, long_window=50)
        self.assertEqual(signal, Signal.FLAT)

    def test_raises_when_not_enough_bars(self):
        closes = pd.Series(range(1, 10))
        with self.assertRaises(ValueError):
            compute_signal(closes, short_window=5, long_window=50)


class TestPositionSize(unittest.TestCase):
    def test_basic_sizing(self):
        # $100,000 equity, 20% risk fraction, $50/share -> 400 shares
        qty = position_size(equity=100_000, price=50, risk_fraction=0.20)
        self.assertEqual(qty, 400)

    def test_capped_by_max_position_fraction(self):
        # risk_fraction asks for 80%, but max_position_fraction caps it at 25%
        qty_uncapped = position_size(
            equity=100_000, price=50, risk_fraction=0.80, max_position_fraction=1.0
        )
        qty_capped = position_size(
            equity=100_000, price=50, risk_fraction=0.80, max_position_fraction=0.25
        )
        self.assertEqual(qty_uncapped, 1600)
        self.assertEqual(qty_capped, 500)

    def test_capped_by_buying_power(self):
        # 20% of $100k is $20k, but only $5k of buying power is left -> 100 shares at $50
        qty = position_size(equity=100_000, price=50, risk_fraction=0.20, buying_power=5_000)
        self.assertEqual(qty, 100)

    def test_negative_buying_power_buys_nothing(self):
        qty = position_size(equity=100_000, price=50, risk_fraction=0.20, buying_power=-10)
        self.assertEqual(qty, 0)

    def test_rejects_invalid_fractions(self):
        with self.assertRaises(ValueError):
            position_size(equity=100_000, price=50, risk_fraction=0)
        with self.assertRaises(ValueError):
            position_size(equity=100_000, price=50, risk_fraction=1.5)


class TestDecideOrder(unittest.TestCase):
    def test_flat_to_long_buys_target_qty(self):
        order = decide_order(current_qty=0, signal=Signal.LONG, target_qty=100)
        self.assertEqual(order, ("buy", 100))

    def test_long_to_flat_sells_full_position(self):
        order = decide_order(current_qty=37, signal=Signal.FLAT, target_qty=100)
        self.assertEqual(order, ("sell", 37))

    def test_already_long_does_nothing(self):
        order = decide_order(current_qty=100, signal=Signal.LONG, target_qty=100)
        self.assertIsNone(order)

    def test_already_flat_does_nothing(self):
        order = decide_order(current_qty=0, signal=Signal.FLAT, target_qty=100)
        self.assertIsNone(order)

    def test_partial_fill_below_half_target_tops_up(self):
        # A 143-share buy filled only 50 before being cancelled: buy the other 93.
        self.assertEqual(
            decide_order(current_qty=50, signal=Signal.LONG, target_qty=143), ("buy", 93)
        )

    def test_normal_drift_above_half_target_does_not_trade(self):
        # Price moved so the target is now 160 shares; holding 143 is close enough.
        self.assertIsNone(decide_order(current_qty=143, signal=Signal.LONG, target_qty=160))

    def test_zero_target_qty_blocks_entry(self):
        # not enough buying power for even 1 share - must not submit a 0-qty order
        order = decide_order(current_qty=0, signal=Signal.LONG, target_qty=0)
        self.assertIsNone(order)


class TestCompletedBars(unittest.TestCase):
    # Alpaca stamps a daily bar at midnight New York time, i.e. 04:00 or 05:00 UTC.
    BARS: ClassVar[list[dict[str, Any]]] = [
        {"t": "2026-09-24T04:00:00Z", "c": 100.0},
        {"t": "2026-09-25T04:00:00Z", "c": 101.0},
    ]

    def _at(self, hour, minute=0):
        # 2026-09-25 is a Friday; EDT is UTC-4.
        return dt.datetime(2026, 9, 25, hour + 4, minute, tzinfo=dt.UTC)

    def test_drops_todays_bar_during_the_session(self):
        self.assertEqual(completed_bars(self.BARS, self._at(11, 30)), self.BARS[:1])

    def test_keeps_todays_bar_after_the_close(self):
        self.assertEqual(completed_bars(self.BARS, self._at(16, 5)), self.BARS)

    def test_keeps_yesterdays_bar_before_the_open(self):
        next_morning = dt.datetime(2026, 9, 28, 13, 0, tzinfo=dt.UTC)  # Mon 9am ET
        self.assertEqual(completed_bars(self.BARS, next_morning), self.BARS)

    def test_requires_a_timezone(self):
        with self.assertRaises(ValueError):
            completed_bars(self.BARS, dt.datetime(2026, 9, 25, 12, 0))  # noqa: DTZ001 (the point)


class TestValidateBars(unittest.TestCase):
    FRI_AFTER_CLOSE = dt.datetime(2026, 9, 25, 20, 15, tzinfo=dt.UTC)

    def bars(self, n=25, last="2026-09-25"):
        end = pd.Timestamp(last)
        return [
            {"t": f"{(end - pd.offsets.BDay(n - 1 - i)).date()}T04:00:00Z", "c": 100.0 + i}
            for i in range(n)
        ]

    def test_good_bars_pass(self):
        self.assertIsNone(validate_bars(self.bars(), self.FRI_AFTER_CLOSE, long_window=20))

    def test_too_few_bars(self):
        self.assertIn("only 10", validate_bars(self.bars(10), self.FRI_AFTER_CLOSE, 20))

    def test_invalid_closes(self):
        for bad in (None, "101.5", float("nan"), 0.0, -3.0):
            bars = self.bars()
            bars[5]["c"] = bad
            self.assertIn("invalid close", validate_bars(bars, self.FRI_AFTER_CLOSE, 20))

    def test_missing_timestamp(self):
        bars = self.bars()
        del bars[3]["t"]
        self.assertIn("no timestamp", validate_bars(bars, self.FRI_AFTER_CLOSE, 20))

    def test_thursday_bar_on_monday_after_a_holiday_is_not_stale(self):
        monday = dt.datetime(2026, 9, 28, 20, 15, tzinfo=dt.UTC)
        self.assertIsNone(validate_bars(self.bars(last="2026-09-24"), monday, 20))

    def test_week_old_bar_is_stale(self):
        later = self.FRI_AFTER_CLOSE + dt.timedelta(days=6)
        self.assertIn("stale", validate_bars(self.bars(), later, 20))


if __name__ == "__main__":
    unittest.main()
