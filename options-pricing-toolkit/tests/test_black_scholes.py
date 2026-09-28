"""
Run with: python3 -m unittest discover -s tests

No pytest dependency on purpose - just stdlib unittest, since nothing else in
this project needs installing.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from black_scholes import price, greeks, implied_vol  # noqa: E402


class TestBlackScholesPrice(unittest.TestCase):
    def test_textbook_call_hull(self):
        # Hull, "Options, Futures, and Other Derivatives" - classic worked example:
        # S=42, K=40, T=0.5y, r=0.10, sigma=0.20 -> call ~= 4.76, put ~= 0.81
        c = price(42, 40, 0.5, 0.10, 0.20, "call")
        p = price(42, 40, 0.5, 0.10, 0.20, "put")
        self.assertAlmostEqual(c, 4.759, places=2)
        self.assertAlmostEqual(p, 0.809, places=2)

    def test_put_call_parity(self):
        # C - P = S - K*exp(-rT), must hold exactly regardless of sigma
        S, K, T, r, sigma = 137.0, 140.0, 0.25, 0.03, 0.45
        c = price(S, K, T, r, sigma, "call")
        p = price(S, K, T, r, sigma, "put")
        lhs = c - p
        rhs = S - K * math.exp(-r * T)
        self.assertAlmostEqual(lhs, rhs, places=8)

    def test_deep_itm_call_converges_to_intrinsic(self):
        # Deep ITM, tiny time/vol -> price should approach intrinsic value
        c = price(200, 100, 1 / 365, 0.05, 0.05, "call")
        self.assertAlmostEqual(c, 200 - 100 * math.exp(-0.05 / 365), places=2)

    def test_atm_call_more_expensive_with_higher_vol(self):
        low = price(100, 100, 30 / 365, 0.05, 0.15, "call")
        high = price(100, 100, 30 / 365, 0.05, 0.60, "call")
        self.assertGreater(high, low)


class TestGreeks(unittest.TestCase):
    def test_call_delta_between_0_and_1(self):
        g = greeks(100, 100, 30 / 365, 0.05, 0.30, "call")
        self.assertTrue(0 < g.delta < 1)

    def test_put_delta_between_minus1_and_0(self):
        g = greeks(100, 100, 30 / 365, 0.05, 0.30, "put")
        self.assertTrue(-1 < g.delta < 0)

    def test_gamma_matches_finite_difference(self):
        S, K, T, r, sigma = 100, 100, 30 / 365, 0.05, 0.30
        h = 0.01
        c_up = price(S + h, K, T, r, sigma, "call")
        c_mid = price(S, K, T, r, sigma, "call")
        c_down = price(S - h, K, T, r, sigma, "call")
        numeric_gamma = (c_up - 2 * c_mid + c_down) / (h ** 2)
        g = greeks(S, K, T, r, sigma, "call")
        self.assertAlmostEqual(g.gamma, numeric_gamma, places=3)

    def test_theta_matches_finite_difference_in_time(self):
        # theta should roughly match (price tomorrow - price today), both
        # measured "per day" and with the standard sign convention (time decay
        # is usually negative for long options)
        S, K, T, r, sigma = 100, 100, 30 / 365, 0.05, 0.30
        dt = 1 / 365
        c_today = price(S, K, T, r, sigma, "call")
        c_tomorrow = price(S, K, T - dt, r, sigma, "call")
        numeric_theta = c_tomorrow - c_today  # T already stepped by exactly one day
        g = greeks(S, K, T, r, sigma, "call")
        self.assertAlmostEqual(g.theta, numeric_theta, delta=0.02)


class TestImpliedVol(unittest.TestCase):
    def test_round_trips_for_call_and_put(self):
        S, K, T, r, sigma = 100, 105, 45 / 365, 0.04, 0.35
        for opt in ("call", "put"):
            p = price(S, K, T, r, sigma, opt)
            iv = implied_vol(p, S, K, T, r, opt)
            self.assertAlmostEqual(iv, sigma, places=4)

    def test_rejects_price_below_intrinsic(self):
        with self.assertRaises(ValueError):
            implied_vol(0.0, 150, 100, 30 / 365, 0.05, "call")  # 50 intrinsic, priced at 0


if __name__ == "__main__":
    unittest.main()
