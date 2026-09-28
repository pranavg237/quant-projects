"""
End-to-end test of one decision cycle against a fake Alpaca client: no network,
no credentials, but the same code path bot.py runs for real.
"""
import datetime as dt
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from alpaca_client import Position  # noqa: E402
from bot import run_once  # noqa: E402


class FakeClient:
    def __init__(self, closes, qty=0.0, equity=100_000.0, buying_power=100_000.0,
                 status="ACTIVE"):
        start = dt.date(2025, 1, 1)
        self.bars = [
            {"t": f"{start + dt.timedelta(days=i)}T05:00:00Z", "c": float(c)}
            for i, c in enumerate(closes)
        ]
        self.qty = qty
        self.account = {"status": status, "equity": str(equity),
                        "buying_power": str(buying_power)}
        self.orders = []

    def get_account(self):
        return self.account

    def get_daily_bars(self, symbol, start):
        return self.bars

    def get_position(self, symbol):
        return Position(symbol, self.qty, self.qty * 100, 100) if self.qty else None

    def submit_market_order(self, symbol, qty, side):
        self.orders.append((side, qty, symbol))
        return {"id": "fake", "status": "accepted"}


def run(client, dry_run=False):
    run_once(client, "SPY", short_window=5, long_window=20, risk_fraction=0.2,
             max_position_fraction=0.25, dry_run=dry_run)


class TestRunOnce(unittest.TestCase):
    def test_uptrend_from_flat_buys(self):
        client = FakeClient(range(100, 140))
        run(client)
        # 20% of $100k at the last close of 139 -> 143 shares
        self.assertEqual(client.orders, [("buy", 143, "SPY")])

    def test_downtrend_while_long_sells_everything(self):
        client = FakeClient(range(140, 100, -1), qty=37)
        run(client)
        self.assertEqual(client.orders, [("sell", 37, "SPY")])

    def test_rerun_in_the_right_state_is_a_no_op(self):
        client = FakeClient(range(100, 140), qty=143)
        run(client)
        self.assertEqual(client.orders, [])

    def test_dry_run_never_submits(self):
        client = FakeClient(range(100, 140))
        run(client, dry_run=True)
        self.assertEqual(client.orders, [])

    def test_buying_power_limits_the_order(self):
        client = FakeClient(range(100, 140), buying_power=1_390)
        run(client)
        self.assertEqual(client.orders, [("buy", 10, "SPY")])

    def test_refuses_a_blocked_account(self):
        client = FakeClient(range(100, 140), status="ACCOUNT_CLOSED")
        with self.assertRaises(RuntimeError):
            run(client)
        self.assertEqual(client.orders, [])

    def test_refuses_too_little_history(self):
        client = FakeClient(range(100, 110))
        with self.assertRaises(RuntimeError):
            run(client)


if __name__ == "__main__":
    unittest.main()
