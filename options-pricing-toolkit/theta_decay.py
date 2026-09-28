"""
Visualizes how theta and price behave as an option runs down to 0-2 DTE.

Directly aimed at the low-DTE premium-capture approach: this plots (a) option
price vs. days-to-expiry at a few fixed spot scenarios, and (b) theta itself
accelerating as expiry nears, so you can see - not just intuit - how sharply
time decay steepens and how much of the remaining price is "time value" that
evaporates independent of the underlying moving your way.

Usage:
  python3 theta_decay.py --S 230 --K 235 --sigma 0.55 --r 0.045 --type call
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from black_scholes import price, greeks


def decay_curve(S: float, K: float, r: float, sigma: float, option_type: str, q: float = 0.0,
                max_dte: float = 10, points: int = 200
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Price, theta (per day) and gamma on a grid from ``max_dte`` days down to 0.05 days.

    Spot and volatility are held fixed, so this isolates the effect of time alone.
    Returns ``(dtes, prices, thetas, gammas)``.
    """
    dtes = np.linspace(max_dte, 0.05, points)  # avoid T=0 (undefined)
    prices, thetas, gammas = [], [], []
    for dte in dtes:
        T = dte / 365.0
        prices.append(price(S, K, T, r, sigma, option_type, q))
        g = greeks(S, K, T, r, sigma, option_type, q)
        thetas.append(g.theta)
        gammas.append(g.gamma)
    return dtes, np.array(prices), np.array(thetas), np.array(gammas)


def plot_decay(S: float, K: float, r: float, sigma: float, option_type: str, q: float = 0.0,
               max_dte: float = 10, out: str = "theta_decay.png") -> None:
    """Save a three-panel chart of price, theta and gamma against days to expiry, and print
    the values at 2, 1 and 0.25 days."""
    dtes, prices, thetas, gammas = decay_curve(S, K, r, sigma, option_type, q, max_dte)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    ax1.plot(dtes, prices, color="tab:blue")
    ax1.axvspan(0, 2, color="tab:red", alpha=0.1, label="0-2 DTE zone")
    ax1.set_ylabel("Option price")
    ax1.set_title(f"{option_type.upper()} S={S} K={K} sigma={sigma:.0%} r={r:.2%} - price/theta/gamma vs DTE")
    ax1.legend()
    ax1.invert_xaxis()

    ax2.plot(dtes, thetas, color="tab:orange")
    ax2.axvspan(0, 2, color="tab:red", alpha=0.1)
    ax2.axhline(0, color="grey", linewidth=0.8)
    ax2.set_ylabel("Theta ($/day)")

    ax3.plot(dtes, gammas, color="tab:green")
    ax3.axvspan(0, 2, color="tab:red", alpha=0.1)
    ax3.set_ylabel("Gamma")
    ax3.set_xlabel("Days to expiry")

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")

    # Also print the numbers for the specific 0/1/2 DTE points, since that's
    # the zone actually being traded.
    print("\nDTE   price     theta/day    gamma")
    for target in (2.0, 1.0, 0.25):
        idx = (np.abs(dtes - target)).argmin()
        print(f"{dtes[idx]:.2f}  {prices[idx]:8.4f}  {thetas[idx]:9.4f}  {gammas[idx]:.4f}")


def main() -> None:
    """Command-line entry point."""
    p = argparse.ArgumentParser(description="Plot theta/gamma decay into 0-2 DTE")
    p.add_argument("--S", type=float, required=True)
    p.add_argument("--K", type=float, required=True)
    p.add_argument("--sigma", type=float, required=True)
    p.add_argument("--r", type=float, default=0.045)
    p.add_argument("--q", type=float, default=0.0)
    p.add_argument("--type", dest="option_type", choices=["call", "put"], required=True)
    p.add_argument("--max-dte", type=float, default=10)
    p.add_argument("--out", default="theta_decay.png")
    args = p.parse_args()

    plot_decay(args.S, args.K, args.r, args.sigma, args.option_type, args.q, args.max_dte, args.out)


if __name__ == "__main__":
    main()
