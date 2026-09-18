r"""Implied volatility: invert Black-Scholes for :math:`\sigma`.

The Black-Scholes price is strictly increasing in :math:`\sigma` between the no-arbitrage
bounds, so the inverse exists and is unique whenever the quote is arbitrage-free:

.. math::
    \max(\phi(Se^{-q\tau} - Ke^{-r\tau}), 0) < V < \begin{cases}
        Se^{-q\tau} & \text{call} \\ Ke^{-r\tau} & \text{put.}\end{cases}

Newton-Raphson on :math:`f(\sigma) = V_{BS}(\sigma) - V_{mkt}` converges quadratically
using the analytic vega, but it is **not** globally safe: vega collapses to zero for deep
in- or out-of-the-money options and for very short expiries, which sends a raw Newton
step off to infinity. That is precisely where real option chains live.

This module therefore implements the Numerical-Recipes ``rtsafe`` hybrid, fully
vectorised: a bracket :math:`[\sigma_{lo}, \sigma_{hi}]` is maintained at all times, a
Newton step is taken when it lands inside the bracket and is shrinking the interval fast
enough, and a bisection step is taken otherwise. It cannot diverge and it keeps Newton's
speed on the well-conditioned majority of the surface.

Quotes outside the no-arbitrage bounds return ``nan`` rather than raising -- on real
chains a few percent of strikes are stale or crossed, and the caller should filter them,
not crash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs
from .types import FloatArray, Numeric, OptionType, as_array, to_option_type

__all__ = ["ImpliedVolResult", "implied_vol", "implied_vol_scalar", "no_arbitrage_bounds"]

_MIN_VOL = 1e-9
_MAX_VOL = 10.0


@dataclass(frozen=True)
class ImpliedVolResult:
    """Implied vols plus solver diagnostics, all broadcast to a common shape.

    Attributes:
        vol: Implied volatility, ``nan`` where the quote is not invertible.
        iterations: Iterations actually consumed per element.
        converged: Whether the price residual met the tolerance.
        newton_fraction: Share of *steps taken across all elements* that were Newton
            rather than bisection. A low number means the chain is dominated by
            low-vega quotes -- a useful health check on the data.
    """

    vol: FloatArray
    iterations: FloatArray
    converged: FloatArray
    newton_fraction: float


def no_arbitrage_bounds(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> tuple[FloatArray, FloatArray]:
    r"""Model-free lower and upper bounds on a European option price.

    A quote must lie strictly inside these for an implied volatility to exist.
    """
    opt = to_option_type(option_type)
    s, k, t, r, q = (as_array(x) for x in (spot, strike, tau, rate, dividend_yield))
    disc_s = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)
    if opt is OptionType.CALL:
        lower = np.maximum(disc_s - disc_k, 0.0)
        upper = disc_s
    else:
        lower = np.maximum(disc_k - disc_s, 0.0)
        upper = disc_k
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def _initial_guess(
    spot: FloatArray, strike: FloatArray, tau: FloatArray, rate: FloatArray, q: FloatArray
) -> FloatArray:
    r"""Manaster-Koehler (1982) seed :math:`\sigma_0 = \sqrt{2|\ln(F/K)|/\tau}`.

    This is the volatility at which vega is maximised for the given moneyness, which is
    the region where Newton behaves best. Floored away from zero for the ATM case where
    the formula degenerates.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = spot * np.exp((rate - q) * tau)
        guess = np.sqrt(np.abs(np.log(fwd / strike)) * 2.0 / np.maximum(tau, 1e-12))
    guess = np.where(np.isfinite(guess), guess, 0.5)
    return np.asarray(np.clip(guess, 0.05, 3.0), dtype=np.float64)


def implied_vol(
    price: Numeric,
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
    return_diagnostics: bool = False,
) -> FloatArray | ImpliedVolResult:
    """Invert Black-Scholes for implied volatility (vectorised, safeguarded Newton).

    Args:
        price: Observed option price(s).
        spot, strike, tau, rate, dividend_yield: Market inputs, broadcastable.
        option_type: ``"call"`` or ``"put"``.
        tol: Absolute tolerance on the **price** residual.
        max_iter: Iteration cap. The bisection fallback halves the bracket every
            iteration, so 100 iterations bounds the vol error by ``10 * 2**-100``.
        return_diagnostics: If ``True`` return an :class:`ImpliedVolResult` instead of a
            bare array.

    Returns:
        Implied volatilities, ``nan`` where the quote is outside the no-arbitrage bounds.
    """
    opt = to_option_type(option_type)
    v_mkt, s, k, t, r, q = np.broadcast_arrays(
        *(as_array(x) for x in (price, spot, strike, tau, rate, dividend_yield))
    )
    v_mkt, s, k, t, r, q = (np.array(x, dtype=np.float64) for x in (v_mkt, s, k, t, r, q))

    lower, upper = no_arbitrage_bounds(s, k, t, r, opt, q)
    # Strict inequality: a quote sitting exactly on a bound implies sigma = 0 or infinity.
    invertible = (
        (v_mkt > lower + 1e-12)
        & (v_mkt < upper - 1e-12)
        & (t > 0.0)
        & (s > 0.0)
        & (k > 0.0)
    )

    lo = np.full(v_mkt.shape, _MIN_VOL)
    hi = np.full(v_mkt.shape, _MAX_VOL)
    vol = _initial_guess(s, k, t, r, q)
    iterations = np.zeros(v_mkt.shape)
    converged = np.zeros(v_mkt.shape, dtype=bool)
    active = invertible.copy()

    step_prev = hi - lo
    step = step_prev.copy()
    newton_steps = 0
    total_steps = 0

    for _ in range(max_iter):
        if not active.any():
            break
        resid = bs.price(s, k, t, r, vol, opt, q) - v_mkt
        vega = bs.vega(s, k, t, r, vol, q)

        # Keep the bracket valid: price is increasing in sigma, so the sign of the
        # residual tells us which side of the root we are on.
        lo = np.where(active & (resid < 0.0), vol, lo)
        hi = np.where(active & (resid >= 0.0), vol, hi)

        newly_done = active & (np.abs(resid) < tol)
        converged |= newly_done
        active &= ~newly_done
        if not active.any():
            break

        # rtsafe test: bisect when the Newton step would leave the bracket, or when it is
        # not at least halving the interval.
        newton_out_of_range = ((vol - hi) * vega - resid) * ((vol - lo) * vega - resid) > 0.0
        too_slow = np.abs(2.0 * resid) > np.abs(step_prev * vega)
        use_bisection = newton_out_of_range | too_slow | ~(np.abs(vega) > 0.0)

        step_prev = np.where(active, step, step_prev)
        bisect_step = 0.5 * (hi - lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton_step = np.where(np.abs(vega) > 0.0, resid / vega, 0.0)
        step = np.where(use_bisection, bisect_step, newton_step)
        new_vol = np.where(use_bisection, lo + bisect_step, vol - newton_step)

        n_active = int(active.sum())
        total_steps += n_active
        newton_steps += int((active & ~use_bisection).sum())

        vol = np.where(active, new_vol, vol)
        iterations = iterations + active.astype(np.float64)

    vol = np.where(invertible, vol, np.nan)
    result_vol = np.asarray(vol, dtype=np.float64)
    if not return_diagnostics:
        return result_vol
    return ImpliedVolResult(
        vol=result_vol,
        iterations=iterations,
        converged=converged,
        newton_fraction=(newton_steps / total_steps) if total_steps else 1.0,
    )


def implied_vol_scalar(
    price: float,
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Scalar convenience wrapper around :func:`implied_vol`."""
    out = implied_vol(price, spot, strike, tau, rate, option_type, dividend_yield, tol, max_iter)
    assert isinstance(out, np.ndarray)
    return float(out)
