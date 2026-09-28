r"""Heston (1993) stochastic volatility: characteristic-function pricing.

Under the risk-neutral measure the spot and its instantaneous variance follow

.. math::
    dS_t &= (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW^S_t, \\
    dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t, \qquad
    d\langle W^S, W^v\rangle_t = \rho\,dt.

Five parameters: long-run variance :math:`\theta`, mean-reversion speed :math:`\kappa`,
vol-of-vol :math:`\xi`, correlation :math:`\rho` and initial variance :math:`v_0`. In
loose terms :math:`\rho` tilts the smile (equity indices need :math:`\rho \approx -0.7`),
:math:`\xi` sets its curvature, and :math:`\kappa, \theta, v_0` shape the term structure.

The model has no closed-form price, but the characteristic function of :math:`x_T =
\ln S_T` is known in closed form, so prices follow from a single numerical integral. With
:math:`F = S_0 e^{(r-q)\tau}`, :math:`k = \ln(K/F)` and :math:`\psi` the characteristic
function of :math:`\ln(S_T/F)`, the Gil-Pelaez inversion gives

.. math::
    P_j = \frac12 + \frac1\pi\int_0^\infty
          \Re\!\left[\frac{e^{-iuk}\psi(u - i\,\mathbb{1}_{j=1})}{iu}\right]du,
    \qquad C = S_0e^{-q\tau}P_1 - Ke^{-r\tau}P_2.

and :math:`\psi(u) = \exp(C(u,\tau) + D(u,\tau)v_0)` with

.. math::
    d &= \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)},\quad
    g = \frac{\kappa - \rho\xi iu - d}{\kappa - \rho\xi iu + d}, \\
    C &= \frac{\kappa\theta}{\xi^2}\Big[(\kappa-\rho\xi iu - d)\tau
         - 2\ln\frac{1-ge^{-d\tau}}{1-g}\Big],\qquad
    D = \frac{\kappa-\rho\xi iu-d}{\xi^2}\,\frac{1-e^{-d\tau}}{1-ge^{-d\tau}}.

This is the **"little trap"** formulation of Albrecher et al. (2007), which picks the
branch of ``g`` whose complex logarithm stays on the principal sheet. Heston's original
grouping is mathematically identical but crosses a branch cut for long maturities and
silently returns garbage prices -- a classic and hard-to-spot bug.

Two independent implementations are provided (``"gil-pelaez"`` and Lewis' single-integral
``"lewis"``) plus a Monte Carlo scheme, so the quadrature can be validated against a
completely different formula rather than against itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .types import FloatArray, Numeric, OptionType, as_array, to_option_type

__all__ = ["HestonParams", "char_func", "implied_vol_surface", "mc_price", "price"]


@dataclass(frozen=True)
class HestonParams:
    r"""The five Heston parameters.

    Attributes:
        v0: Initial instantaneous variance :math:`v_0` (not volatility).
        kappa: Mean-reversion speed :math:`\kappa`.
        theta: Long-run variance :math:`\theta`.
        xi: Volatility of variance :math:`\xi`.
        rho: Spot/variance correlation :math:`\rho \in (-1, 1)`.
    """

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if self.v0 < 0.0:
            raise ValueError("v0 must be >= 0")
        if self.theta < 0.0:
            raise ValueError("theta must be >= 0")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if self.xi < 0.0:
            raise ValueError("xi must be >= 0")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must be in (-1, 1)")

    @property
    def feller_ratio(self) -> float:
        r"""The Feller ratio :math:`2\kappa\theta/\xi^2`.

        Variance stays strictly positive iff this exceeds 1. Equity calibrations very
        often violate it; the characteristic function is still valid (the variance
        process just touches zero), but simulation schemes need care.
        """
        if self.xi == 0.0:
            return float("inf")
        return 2.0 * self.kappa * self.theta / (self.xi * self.xi)

    @property
    def satisfies_feller(self) -> bool:
        """Whether ``2*kappa*theta > xi**2``."""
        return self.feller_ratio > 1.0

    def to_array(self) -> FloatArray:
        """Pack as ``[v0, kappa, theta, xi, rho]`` for an optimiser."""
        return np.array([self.v0, self.kappa, self.theta, self.xi, self.rho], dtype=np.float64)

    @classmethod
    def from_array(cls, x: Numeric) -> HestonParams:
        """Unpack from ``[v0, kappa, theta, xi, rho]``."""
        a = as_array(x).ravel()
        if a.size != 5:
            raise ValueError("expected 5 parameters [v0, kappa, theta, xi, rho]")
        return cls(float(a[0]), float(a[1]), float(a[2]), float(a[3]), float(a[4]))

    def replace(self, **changes: float) -> HestonParams:
        """Return a copy with the given fields overridden."""
        return replace(self, **changes)


def char_func(
    u: Numeric,
    tau: float,
    params: HestonParams,
) -> np.ndarray:
    r"""Characteristic function of :math:`\ln(S_\tau/F)` under Heston.

    Drift-free by construction, so it depends only on ``tau`` and ``params`` -- the spot,
    rate and dividend enter only through the forward :math:`F`. Accepts complex ``u``,
    which is what the :math:`P_1` integrand needs (it evaluates at :math:`u - i`).

    Returns:
        Complex array :math:`\psi(u) = \mathbb{E}[e^{iu\ln(S_\tau/F)}]`.
    """
    uu = np.asarray(u, dtype=np.complex128)
    kappa, theta, xi, rho, v0 = params.kappa, params.theta, params.xi, params.rho, params.v0

    if xi == 0.0:
        # Deterministic variance: v_t = theta + (v0 - theta)e^{-kappa t}; the integrated
        # variance is available in closed form and the CF collapses to Black-Scholes.
        integrated = theta * tau + (v0 - theta) * (1.0 - np.exp(-kappa * tau)) / kappa
        return np.asarray(np.exp(-0.5 * integrated * (uu * uu + 1j * uu)), dtype=np.complex128)

    iu = 1j * uu
    beta = kappa - rho * xi * iu
    d = np.sqrt(beta * beta + xi * xi * (iu + uu * uu))
    # "Little trap": use g = (beta - d)/(beta + d) so |g| <= 1 and log stays principal.
    g = (beta - d) / (beta + d)
    exp_dt = np.exp(-d * tau)
    one_minus_g_exp = 1.0 - g * exp_dt

    c_term = (kappa * theta / (xi * xi)) * (
        (beta - d) * tau - 2.0 * np.log(one_minus_g_exp / (1.0 - g))
    )
    d_term = ((beta - d) / (xi * xi)) * (1.0 - exp_dt) / one_minus_g_exp
    return np.asarray(np.exp(c_term + d_term * v0), dtype=np.complex128)


def _gauss_legendre(n: int, lo: float, hi: float) -> tuple[FloatArray, FloatArray]:
    """Gauss-Legendre nodes and weights rescaled from ``[-1, 1]`` to ``[lo, hi]``."""
    x, w = np.polynomial.legendre.leggauss(n)
    mid, half = 0.5 * (hi + lo), 0.5 * (hi - lo)
    return mid + half * x, half * w


def _integration_limit(tau: float, params: HestonParams) -> float:
    r"""Pick an upper limit for the Fourier integral.

    :math:`|\psi(u)|` decays roughly like :math:`e^{-\frac12 \bar{v}\tau u^2}` for small
    vol-of-vol, so the required limit scales like :math:`1/\sqrt{\bar{v}\tau}`. Short
    expiries and low variance therefore need a much wider domain -- using a fixed limit
    is the second-most-common Heston bug after the branch cut.
    """
    typical_var = max(0.5 * (params.v0 + params.theta), 1e-4)
    scale = np.sqrt(typical_var * max(tau, 1e-4))
    return float(np.clip(40.0 / scale, 100.0, 5000.0))


def price(
    spot: float,
    strike: Numeric,
    tau: float,
    rate: float,
    params: HestonParams,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    n_quad: int = 512,
    u_max: float | None = None,
    method: str = "gil-pelaez",
) -> FloatArray:
    """Price European vanillas under Heston by Fourier inversion.

    Args:
        spot: Spot price.
        strike: Strike or array of strikes (vectorised: all strikes share one quadrature).
        tau: Time to expiry in years.
        rate: Risk-free rate.
        params: Heston parameters.
        option_type: ``"call"`` or ``"put"`` (puts come from put-call parity, which is
            model-free and therefore exact).
        dividend_yield: Continuous dividend yield.
        n_quad: Gauss-Legendre nodes. 256 is ample for typical equity parameters.
        u_max: Upper integration limit; auto-selected from ``tau`` and ``params`` if
            ``None``.
        method: ``"gil-pelaez"`` (two-integral Heston form) or ``"lewis"`` (single
            integral). They are algebraically equivalent; disagreement between them
            signals a quadrature problem.

    Returns:
        Prices, broadcast over ``strike``.
    """
    opt = to_option_type(option_type)
    k_arr = as_array(strike)
    if tau <= 0.0:
        payoff = np.maximum(opt.sign * (spot - k_arr), 0.0)
        return np.asarray(payoff, dtype=np.float64)

    forward = spot * np.exp((rate - dividend_yield) * tau)
    disc = float(np.exp(-rate * tau))
    log_moneyness = np.log(np.maximum(k_arr, 1e-300) / forward)  # k = ln(K/F)

    limit = _integration_limit(tau, params) if u_max is None else u_max
    u, w = _gauss_legendre(n_quad, 1e-10, limit)

    if method == "gil-pelaez":
        psi2 = char_func(u, tau, params)
        psi1 = char_func(u - 1j, tau, params)
        phase = np.exp(-1j * np.outer(u, log_moneyness))  # (n_quad, n_strikes)
        denom = (1j * u)[:, None]
        integrand1 = np.real(phase * psi1[:, None] / denom)
        integrand2 = np.real(phase * psi2[:, None] / denom)
        p1 = 0.5 + (w @ integrand1) / np.pi
        p2 = 0.5 + (w @ integrand2) / np.pi
        call = spot * np.exp(-dividend_yield * tau) * p1 - k_arr * disc * p2
    elif method == "lewis":
        # C = e^{-r tau} F [ 1 - (e^{k/2}/pi) \int Re( e^{-iuk} psi(u - i/2) ) / (u^2+1/4) du ]
        psi = char_func(u - 0.5j, tau, params)
        phase = np.exp(-1j * np.outer(u, log_moneyness))
        kernel = (psi / (u * u + 0.25))[:, None]
        integral = w @ np.real(phase * kernel)
        call = disc * forward * (1.0 - np.exp(0.5 * log_moneyness) * integral / np.pi)
    else:
        raise ValueError(f"unknown method {method!r}; expected 'gil-pelaez' or 'lewis'")

    call = np.maximum(call, 0.0)
    if opt is OptionType.CALL:
        return np.asarray(call, dtype=np.float64)
    put = call - spot * np.exp(-dividend_yield * tau) + k_arr * disc
    return np.asarray(np.maximum(put, 0.0), dtype=np.float64)


def mc_price(
    spot: float,
    strike: Numeric,
    tau: float,
    rate: float,
    params: HestonParams,
    option_type: OptionType | str = OptionType.CALL,
    dividend_yield: float = 0.0,
    n_paths: int = 200_000,
    n_steps: int = 500,
    seed: int | np.random.Generator | None = None,
    antithetic: bool = True,
) -> tuple[FloatArray, FloatArray]:
    r"""Monte Carlo price under Heston using the full-truncation Euler scheme.

    Full truncation (Lord et al., 2010) replaces :math:`v_t` with :math:`v_t^+` everywhere
    it appears under a square root **and** in the drift, which is the least-biased of the
    simple fixes for the fact that an Euler step can push the variance negative. It is
    slower and less accurate than the characteristic-function price, so it exists purely
    as an independent check on the Fourier implementation -- two different derivations of
    the same number is the only real defence against a sign error.

    Returns:
        ``(price, std_error)`` arrays broadcast over ``strike``.
    """
    opt = to_option_type(option_type)
    phi = opt.sign
    k_arr = as_array(strike)
    gen = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

    dt = tau / n_steps
    sqrt_dt = np.sqrt(dt)
    n_base = (n_paths + 1) // 2 if antithetic else n_paths

    log_spot = np.full(n_base * (2 if antithetic else 1), np.log(spot))
    variance = np.full_like(log_spot, params.v0)

    for _ in range(n_steps):
        z1 = gen.standard_normal(n_base)
        z2 = gen.standard_normal(n_base)
        if antithetic:
            z1 = np.concatenate([z1, -z1])
            z2 = np.concatenate([z2, -z2])
        w_v = z1
        w_s = params.rho * z1 + np.sqrt(1.0 - params.rho**2) * z2

        v_pos = np.maximum(variance, 0.0)
        sqrt_v = np.sqrt(v_pos)
        log_spot += (rate - dividend_yield - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * w_s
        variance = (
            variance
            + params.kappa * (params.theta - v_pos) * dt
            + params.xi * sqrt_v * sqrt_dt * w_v
        )

    terminal = np.exp(log_spot)
    payoff = np.maximum(phi * (terminal[:, None] - np.atleast_1d(k_arr)[None, :]), 0.0)

    if antithetic:
        half = payoff.shape[0] // 2
        payoff = 0.5 * (payoff[:half] + payoff[half : 2 * half])

    disc = float(np.exp(-rate * tau))
    n = payoff.shape[0]
    mean = disc * payoff.mean(axis=0)
    se = disc * payoff.std(axis=0, ddof=1) / np.sqrt(n)
    return (
        np.asarray(mean.reshape(np.shape(k_arr)), dtype=np.float64),
        np.asarray(se.reshape(np.shape(k_arr)), dtype=np.float64),
    )


def implied_vol_surface(
    spot: float,
    strikes: Numeric,
    taus: Numeric,
    rate: float,
    params: HestonParams,
    dividend_yield: float = 0.0,
) -> FloatArray:
    """Heston implied-vol surface on a ``(len(taus), len(strikes))`` grid.

    Prices each expiry with one quadrature, then inverts Black-Scholes. Out-of-the-money
    options are used on each side of the forward, because they carry all the time value
    and are what a real desk quotes from.
    """
    from . import implied_vol as iv_mod  # noqa: PLC0415  (local: avoids an import cycle)

    k = np.atleast_1d(as_array(strikes))
    t = np.atleast_1d(as_array(taus))
    out = np.empty((t.size, k.size), dtype=np.float64)
    for i, tau in enumerate(t):
        forward = spot * np.exp((rate - dividend_yield) * float(tau))
        is_call = k >= forward
        call_px = price(spot, k, float(tau), rate, params, OptionType.CALL, dividend_yield)
        put_px = price(spot, k, float(tau), rate, params, OptionType.PUT, dividend_yield)
        px = np.where(is_call, call_px, put_px)
        vol_c = iv_mod.implied_vol(px, spot, k, float(tau), rate, OptionType.CALL, dividend_yield)
        vol_p = iv_mod.implied_vol(px, spot, k, float(tau), rate, OptionType.PUT, dividend_yield)
        out[i] = np.where(is_call, np.asarray(vol_c), np.asarray(vol_p))
    return out
