"""
Run with: python3 -m unittest discover -s tests

These tests never touch the network - compute_signal/position_size/decide_order
are pure functions, tested with synthetic price series.
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy import compute_signal, position_size, decide_order, Signal  # noqa: E402


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
        qty_uncapped = position_size(equity=100_000, price=50, risk_fraction=0.80,
                                       max_position_fraction=1.0)
        qty_capped = position_size(equity=100_000, price=50, risk_fraction=0.80,
                                     max_position_fraction=0.25)
        self.assertEqual(qty_uncapped, 1600)
        self.assertEqual(qty_capped, 500)

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

    def test_zero_target_qty_blocks_entry(self):
        # not enough buying power for even 1 share - must not submit a 0-qty order
        order = decide_order(current_qty=0, signal=Signal.LONG, target_qty=0)
        self.assertIsNone(order)


if __name__ == "__main__":
    unittest.main()
