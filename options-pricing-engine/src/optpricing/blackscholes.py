r"""Analytic Black-Scholes-Merton pricing and Greeks.

The model assumes the spot follows a geometric Brownian motion under the risk-neutral
measure with constant rate :math:`r`, continuous dividend yield :math:`q` and constant
volatility :math:`\sigma`:

.. math::
    dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t.

With :math:`\tau = T - t`,

.. math::
    d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)\tau}{\sigma\sqrt{\tau}},
    \qquad d_2 = d_1 - \sigma\sqrt{\tau},

.. math::
    C = S e^{-q\tau} N(d_1) - K e^{-r\tau} N(d_2), \qquad
    P = K e^{-r\tau} N(-d_2) - S e^{-q\tau} N(-d_1).

Every function here is fully vectorised: pass scalars or broadcastable arrays for any
argument. Degenerate inputs (``tau <= 0`` or ``sigma <= 0``) fall back to the discounted
forward intrinsic value :math:`\max(\phi(Se^{-q\tau} - Ke^{-r\tau}), 0)`, which is the
correct no-uncertainty limit and keeps the functions total rather than raising.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from scipy.stats import norm

from .types import FloatArray, Numeric, OptionType, as_array, to_option_type

__all__ = [
    "Greeks",
    "charm",
    "d1_d2",
    "delta",
    "dual_delta",
    "forward_price",
    "gamma",
    "greeks",
    "price",
    "put_call_parity_gap",
    "rho",
    "theta",
    "vanna",
    "vega",
    "volga",
]

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(x: FloatArray) -> FloatArray:
    """Standard normal density, written out to avoid a scipy call in hot loops."""
    return np.asarray(np.exp(-0.5 * x * x) / _SQRT_2PI, dtype=np.float64)


def _norm_cdf(x: FloatArray) -> FloatArray:
    return np.asarray(norm.cdf(x), dtype=np.float64)


def d1_d2(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    dividend_yield: Numeric = 0.0,
) -> tuple[FloatArray, FloatArray]:
    r"""Return :math:`(d_1, d_2)`.

    Degenerate cells (``tau <= 0``, ``sigma <= 0``, non-positive spot or strike) are
    returned as ``+/-inf`` according to moneyness so that ``N(d)`` collapses to the
    correct 0/1 indicator; callers overwrite those cells anyway.
    """
    s, k, t, r, vol, q = (
        as_array(spot),
        as_array(strike),
        as_array(tau),
        as_array(rate),
        as_array(sigma),
        as_array(dividend_yield),
    )
    total_vol = vol * np.sqrt(np.maximum(t, 0.0))
    degenerate = ~(total_vol > 0.0) | ~(s > 0.0) | ~(k > 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_moneyness = np.log(np.where(s > 0.0, s, 1.0) / np.where(k > 0.0, k, 1.0))
        drift = (r - q + 0.5 * vol * vol) * t
        d1 = (log_moneyness + drift) / np.where(degenerate, 1.0, total_vol)
        d2 = d1 - np.where(degenerate, 0.0, total_vol)

    # In the zero-volatility / zero-time limit the option is worth its forward intrinsic,
    # which is what +/-inf in d1 and d2 reproduces.
    fwd = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)
    limit = np.where(fwd > disc_k, np.inf, np.where(fwd < disc_k, -np.inf, 0.0))
    d1 = np.where(degenerate, limit, d1)
    d2 = np.where(degenerate, limit, d2)
    return np.asarray(d1, dtype=np.float64), np.asarray(d2, dtype=np.float64)


def forward_price(
    spot: Numeric, tau: Numeric, rate: Numeric, dividend_yield: Numeric = 0.0
) -> FloatArray:
    r"""Forward price :math:`F = S e^{(r - q)\tau}`."""
    s, t, r, q = as_array(spot), as_array(tau), as_array(rate), as_array(dividend_yield)
    return np.asarray(s * np.exp((r - q) * t), dtype=np.float64)


def price(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r"""Black-Scholes-Merton price of a European vanilla.

    Args:
        spot: Spot price :math:`S`.
        strike: Strike :math:`K`.
        tau: Time to expiry in years. Non-positive values give the intrinsic value.
        rate: Continuously compounded risk-free rate :math:`r`.
        sigma: Volatility :math:`\sigma` (annualised, in decimals, e.g. ``0.2``).
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous dividend yield :math:`q`.

    Returns:
        Option price, broadcast over the inputs.
    """
    opt = to_option_type(option_type)
    phi = opt.sign
    s, k, t, r, q = (
        as_array(spot),
        as_array(strike),
        as_array(tau),
        as_array(rate),
        as_array(dividend_yield),
    )
    d1, d2 = d1_d2(s, k, t, r, sigma, q)
    disc_s = s * np.exp(-q * t)
    disc_k = k * np.exp(-r * t)
    value = phi * (disc_s * _norm_cdf(phi * d1) - disc_k * _norm_cdf(phi * d2))
    # Guard against -0.0 and tiny negative round-off in the deep-OTM tail.
    return np.asarray(np.maximum(value, 0.0), dtype=np.float64)


def delta(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial V/\partial S`. Call: :math:`e^{-q\tau}N(d_1)`."""
    opt = to_option_type(option_type)
    phi = opt.sign
    t, q = as_array(tau), as_array(dividend_yield)
    d1, _ = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    return np.asarray(phi * np.exp(-q * t) * _norm_cdf(phi * d1), dtype=np.float64)


def gamma(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial^2 V/\partial S^2 = e^{-q\tau} n(d_1) / (S\sigma\sqrt{\tau})`.

    Identical for calls and puts (put-call parity is linear in :math:`S`).
    """
    s, t, vol, q = (
        as_array(spot),
        as_array(tau),
        as_array(sigma),
        as_array(dividend_yield),
    )
    d1, _ = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    denom = s * vol * np.sqrt(np.maximum(t, 0.0))
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    out = np.where(denom > 0.0, np.exp(-q * t) * _norm_pdf(d1) / safe_denom, 0.0)
    return np.asarray(out, dtype=np.float64)


def vega(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial V/\partial\sigma = S e^{-q\tau} n(d_1)\sqrt{\tau}`.

    Quoted per **one unit** of volatility (i.e. per 100 vol points), not per vol point.
    """
    s, t, q = as_array(spot), as_array(tau), as_array(dividend_yield)
    d1, _ = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    out = np.where(t > 0.0, s * np.exp(-q * t) * _norm_pdf(d1) * np.sqrt(np.maximum(t, 0.0)), 0.0)
    return np.asarray(out, dtype=np.float64)


def theta(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial V/\partial t`, **per year** (divide by 365 for a daily decay)."""
    opt = to_option_type(option_type)
    phi = opt.sign
    s, k, t, r, vol, q = (
        as_array(spot),
        as_array(strike),
        as_array(tau),
        as_array(rate),
        as_array(sigma),
        as_array(dividend_yield),
    )
    d1, d2 = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    sqrt_t = np.sqrt(np.maximum(t, 0.0))
    safe_sqrt_t = np.where(sqrt_t > 0.0, sqrt_t, 1.0)
    decay = -np.exp(-q * t) * s * _norm_pdf(d1) * vol / (2.0 * safe_sqrt_t)
    carry = phi * (
        q * s * np.exp(-q * t) * _norm_cdf(phi * d1)
        - r * k * np.exp(-r * t) * _norm_cdf(phi * d2)
    )
    out = np.where(t > 0.0, decay + carry, 0.0)
    return np.asarray(out, dtype=np.float64)


def rho(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial V/\partial r`. Call: :math:`K\tau e^{-r\tau}N(d_2)`."""
    opt = to_option_type(option_type)
    phi = opt.sign
    k, t, r = as_array(strike), as_array(tau), as_array(rate)
    _, d2 = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    return np.asarray(phi * k * t * np.exp(-r * t) * _norm_cdf(phi * d2), dtype=np.float64)


def vanna(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial^2 V/\partial S\partial\sigma = -e^{-q\tau} n(d_1) d_2/\sigma`."""
    t, vol, q = as_array(tau), as_array(sigma), as_array(dividend_yield)
    d1, d2 = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    safe_vol = np.where(vol > 0.0, vol, 1.0)
    out = np.where((vol > 0.0) & (t > 0.0), -np.exp(-q * t) * _norm_pdf(d1) * d2 / safe_vol, 0.0)
    return np.asarray(out, dtype=np.float64)


def volga(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial^2 V/\partial\sigma^2 = \mathcal{V}\, d_1 d_2/\sigma` (a.k.a. vomma)."""
    vol, t = as_array(sigma), as_array(tau)
    d1, d2 = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    v = vega(spot, strike, tau, rate, sigma, dividend_yield)
    safe_vol = np.where(vol > 0.0, vol, 1.0)
    out = np.where((vol > 0.0) & (t > 0.0), v * d1 * d2 / safe_vol, 0.0)
    return np.asarray(out, dtype=np.float64)


def charm(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial\Delta/\partial t` (delta decay), per year."""
    opt = to_option_type(option_type)
    phi = opt.sign
    t, r, vol, q = (
        as_array(tau),
        as_array(rate),
        as_array(sigma),
        as_array(dividend_yield),
    )
    d1, d2 = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    sqrt_t = np.sqrt(np.maximum(t, 0.0))
    safe = np.where(sqrt_t > 0.0, sqrt_t, 1.0)
    safe_t = np.maximum(t, 1e-300)
    term = _norm_pdf(d1) * (2.0 * (r - q) * t - d2 * vol * safe) / (2.0 * safe_t * vol * safe)
    out = np.exp(-q * t) * (phi * q * _norm_cdf(phi * d1) - term)
    return np.asarray(np.where(t > 0.0, out, 0.0), dtype=np.float64)


def dual_delta(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r""":math:`\partial V/\partial K`. Call: :math:`-e^{-r\tau}N(d_2)`.

    Minus this, discounted back, is the risk-neutral CDF of :math:`S_T` -- the quantity
    a surface has to keep monotone for the implied density to stay non-negative.
    """
    opt = to_option_type(option_type)
    phi = opt.sign
    t, r = as_array(tau), as_array(rate)
    _, d2 = d1_d2(spot, strike, tau, rate, sigma, dividend_yield)
    return np.asarray(-phi * np.exp(-r * t) * _norm_cdf(phi * d2), dtype=np.float64)


@dataclass(frozen=True)
class Greeks:
    """Container for every Greek this module computes, all broadcast to a common shape."""

    price: FloatArray
    delta: FloatArray
    gamma: FloatArray
    vega: FloatArray
    theta: FloatArray
    rho: FloatArray
    vanna: FloatArray
    volga: FloatArray
    charm: FloatArray
    dual_delta: FloatArray

    def as_dict(self) -> dict[str, FloatArray]:
        """Return the Greeks keyed by name, handy for building a DataFrame."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def greeks(
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    sigma: Numeric,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: Numeric = 0.0,
) -> Greeks:
    """Compute price and all Greeks in one call."""
    args = (spot, strike, tau, rate, sigma)
    return Greeks(
        price=price(*args, option_type, dividend_yield),
        delta=delta(*args, option_type, dividend_yield),
        gamma=gamma(*args, dividend_yield),
        vega=vega(*args, dividend_yield),
        theta=theta(*args, option_type, dividend_yield),
        rho=rho(*args, option_type, dividend_yield),
        vanna=vanna(*args, dividend_yield),
        volga=volga(*args, dividend_yield),
        charm=charm(*args, option_type, dividend_yield),
        dual_delta=dual_delta(*args, option_type, dividend_yield),
    )


def put_call_parity_gap(
    call_price: Numeric,
    put_price: Numeric,
    spot: Numeric,
    strike: Numeric,
    tau: Numeric,
    rate: Numeric,
    dividend_yield: Numeric = 0.0,
) -> FloatArray:
    r"""Residual of put-call parity :math:`C - P - (Se^{-q\tau} - Ke^{-r\tau})`.

    Model-free: any non-zero value on real quotes is either a wrong rate/dividend
    assumption, stale quotes, or genuine arbitrage (almost always the first two).
    """
    c, p, s, k, t, r, q = (
        as_array(x)
        for x in (call_price, put_price, spot, strike, tau, rate, dividend_yield)
    )
    return np.asarray(c - p - (s * np.exp(-q * t) - k * np.exp(-r * t)), dtype=np.float64)
