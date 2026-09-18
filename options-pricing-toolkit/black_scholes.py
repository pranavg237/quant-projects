"""
Black-Scholes European option pricer, full Greeks, and implied volatility solver.

All formulas assume:
  S      - current underlying price
  K      - strike price
  T      - time to expiry, in YEARS (e.g. 3 days = 3/365)
  r      - risk-free rate, annualized, continuously compounded (e.g. 0.05)
  sigma  - annualized volatility (e.g. 0.30 for 30%)
  q      - continuous dividend yield (default 0)

No external dependencies beyond numpy/scipy (both already in the stdlib-adjacent
scientific stack). No lookahead, no data fetching - pure math, so it's a good
reference / interview-prep piece as well as a working calculator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm
from scipy.optimize import brentq

N = norm.cdf          # standard normal CDF
n = norm.pdf          # standard normal PDF


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton price of a European call or put."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return S * math.exp(-q * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    elif option_type == "put":
        return K * math.exp(-r * T) * N(-d2) - S * math.exp(-q * T) * N(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


@dataclass
class Greeks:
    delta: float
    gamma: float
    vega: float     # price change per 1 vol point (0.01)
    theta: float    # price change per calendar day
    rho: float       # price change per 1% rate move (0.01)

    def __str__(self) -> str:
        return (f"delta={self.delta:+.4f}  gamma={self.gamma:.4f}  "
                f"vega={self.vega:+.4f}  theta={self.theta:+.4f}/day  rho={self.rho:+.4f}")


def greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call", q: float = 0.0) -> Greeks:
    """Full Greeks for a European call or put. Theta and vega are scaled to
    'per day' and 'per 1 vol point' respectively, matching how brokers display them."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    sqrtT = math.sqrt(T)

    gamma = math.exp(-q * T) * n(d1) / (S * sigma * sqrtT)
    vega = S * math.exp(-q * T) * n(d1) * sqrtT / 100  # per 1 vol point

    if option_type == "call":
        delta = math.exp(-q * T) * N(d1)
        theta_annual = (
            -S * math.exp(-q * T) * n(d1) * sigma / (2 * sqrtT)
            - r * K * math.exp(-r * T) * N(d2)
            + q * S * math.exp(-q * T) * N(d1)
        )
        rho = K * T * math.exp(-r * T) * N(d2) / 100
    elif option_type == "put":
        delta = -math.exp(-q * T) * N(-d1)
        theta_annual = (
            -S * math.exp(-q * T) * n(d1) * sigma / (2 * sqrtT)
            + r * K * math.exp(-r * T) * N(-d2)
            - q * S * math.exp(-q * T) * N(-d1)
        )
        rho = -K * T * math.exp(-r * T) * N(-d2) / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    theta_per_day = theta_annual / 365.0
    return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta_per_day, rho=rho)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                 option_type: str = "call", q: float = 0.0,
                 lo: float = 1e-4, hi: float = 5.0) -> float:
    """Solve for the volatility that reproduces market_price, via Brent's method
    (robust bisection-style root finder - no starting guess needed, unlike Newton)."""

    def f(sigma):
        return price(S, K, T, r, sigma, option_type, q) - market_price

    # Sanity check: market price must be within no-arbitrage bounds, otherwise
    # there's no sigma that reproduces it.
    intrinsic = max(0.0, (S - K) if option_type == "call" else (K - S))
    if market_price < intrinsic * math.exp(-q * T) - 1e-9:
        raise ValueError("market_price is below intrinsic value - not arbitrage-free")

    try:
        return brentq(f, lo, hi, xtol=1e-8, maxiter=200)
    except ValueError as e:
        raise ValueError(
            f"Could not solve for implied vol in range [{lo}, {hi}]. "
            f"Check inputs; f(lo)={f(lo):.4f}, f(hi)={f(hi):.4f}"
        ) from e


if __name__ == "__main__":
    # Quick sanity demo
    S, K, T, r, sigma = 100, 100, 30 / 365, 0.05, 0.30
    for opt in ("call", "put"):
        p = price(S, K, T, r, sigma, opt)
        g = greeks(S, K, T, r, sigma, opt)
        iv = implied_vol(p, S, K, T, r, opt)
        print(f"{opt.upper():5s} price={p:.4f}  {g}  implied_vol_check={iv:.4f}")
