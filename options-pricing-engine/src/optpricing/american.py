r"""De-Americanising option quotes: European-equivalent prices from American ones.

SPY options are American. Black-Scholes, Heston and the no-arbitrage checks are all
European, so before a quote is inverted or calibrated to, its **early-exercise premium**
has to come off:

.. math:: V^{E}_{\text{mkt}} = V^{A}_{\text{mkt}} - e(\sigma),\qquad
          e(\sigma) = V^{A}_{\text{tree}}(\sigma) - V^{E}_{\text{tree}}(\sigma).

The premium is American minus European on the *same* lattice (Leisen-Reimer by default,
which keeps it smooth across strikes; see :func:`early_exercise_premium`), so the
lattice's own discretisation error -- larger than the premium itself for short expiries --
cancels in the difference. Nothing is fitted to the quotes being corrected.

**The circularity.** The premium depends on the volatility, which is the thing being
solved for. :func:`deamericanise` treats this as a fixed point,

.. math:: \sigma_{n+1} = \mathrm{BS}^{-1}\big(V^{A}_{\text{mkt}} - e(\sigma_n)\big),
          \qquad \sigma_0 = \mathrm{BS}^{-1}(V^{A}_{\text{mkt}}),

whose fixed point is exactly the vol at which "European price plus tree premium"
reproduces the market quote. Differentiating, the map's slope is
:math:`-\,(\partial e/\partial\sigma)/\mathcal{V}_{BS}`: the premium's vega over the
option's vega. For an out-of-the-money option the premium is a small, smooth part of the
price, so the ratio is far below one and each iteration shrinks the error by that factor
(``contraction`` in :class:`Deamericanised` measures it). Starting from the uncorrected
vol, which is an upper bound because the premium is non-negative, the iterates alternate
around the answer. The map need not contract for deep in-the-money options, whose value
is almost all exercise; the surface never inverts those.

The second circularity is the **forward**. Parity pins the forward only once the premia of
the in-the-money legs in the fitting window are removed, those premia need a vol, and the
vol comes from the surface, which is built at the forward.
:func:`american_forward` solves the inner problem for one expiry;
:func:`optpricing.surface.build_surface` iterates the outer one (forward -> surface ->
forward) to convergence.

**What the tree cannot represent.** It uses a continuous dividend yield. SPY pays
discrete quarterly dividends, which give deep in-the-money *calls* an exercise premium
just before each ex-date; a continuous yield smears that out, so call premia are
understated around ex-dates. For the out-of-the-money calls the surface uses, this is
second order; it is not second order for a single stock around a large dividend.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import implied_vol as iv
from .types import FloatArray, OptionType, to_option_type

__all__ = [
    "Deamericanised",
    "ForwardSolution",
    "american_forward",
    "deamericanise",
    "early_exercise_premium",
]


def _peizer_pratt(z: FloatArray, n: int) -> FloatArray:
    """Peizer-Pratt inversion (method 2), vectorised; see :mod:`optpricing.binomial`."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    inner = 1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0))
    return np.asarray(0.5 + np.sign(z) * 0.5 * np.sqrt(inner), dtype=np.float64)


def _american_european(
    spot: float,
    strikes: FloatArray,
    tau: float,
    rate: float,
    dividend_yield: float,
    vols: FloatArray,
    phi: float,
    steps: int,
    method: str,
) -> tuple[FloatArray, FloatArray]:
    """American and European lattice prices for many (strike, vol) pairs in one sweep.

    Row ``i`` is an independent tree with volatility ``vols[i]`` (and, for Leisen-Reimer,
    centred on ``strikes[i]``); the backward induction runs on all rows at once. It is the
    algorithm of :func:`optpricing.binomial.price` vectorised over strikes (a test pins
    the two together), which is what makes the fixed point below affordable on ~2,000
    quotes. Rows whose risk-neutral probability falls outside ``[0, 1]`` come back
    ``nan``.
    """
    dt = tau / steps
    growth = np.exp((rate - dividend_yield) * dt)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if method == "crr":
            log_u = vols * np.sqrt(dt)
            log_d = -log_u
            u, d = np.exp(log_u), np.exp(log_d)
            p = (growth - d) / (u - d)
        else:  # Leisen-Reimer: each row's tree is centred on its own strike
            vst = vols * np.sqrt(tau)
            d2 = (np.log(spot / strikes) + (rate - dividend_yield - 0.5 * vols**2) * tau) / vst
            p = _peizer_pratt(d2, steps)
            u = growth * _peizer_pratt(d2 + vst, steps) / p
            d = (growth - p * u) / (1.0 - p)
            log_u, log_d = np.log(u), np.log(d)
    disc = float(np.exp(-rate * dt))
    ok = np.isfinite(p) & (p >= 0.0) & (p <= 1.0) & np.isfinite(log_u) & np.isfinite(log_d)
    p = np.where(ok, p, 0.5)[:, None]
    log_u = np.where(ok, log_u, 0.0)[:, None]
    log_d = np.where(ok, log_d, 0.0)[:, None]
    k = strikes[:, None]
    log_s = float(np.log(spot))

    j = np.arange(steps + 1, dtype=np.float64)
    terminal = np.exp(log_s + j * log_u + (steps - j) * log_d)
    european = np.maximum(phi * (terminal - k), 0.0)
    american = european.copy()
    for step in range(steps - 1, -1, -1):
        european = disc * (p * european[:, 1:] + (1.0 - p) * european[:, :-1])
        american = disc * (p * american[:, 1:] + (1.0 - p) * american[:, :-1])
        jj = np.arange(step + 1, dtype=np.float64)
        node = np.exp(log_s + jj * log_u + (step - jj) * log_d)
        american = np.maximum(american, phi * (node - k))
    am = np.where(ok, american[:, 0], np.nan)
    eu = np.where(ok, european[:, 0], np.nan)
    return np.asarray(am, dtype=np.float64), np.asarray(eu, dtype=np.float64)


def early_exercise_premium(
    spot: float,
    strikes: FloatArray,
    tau: float,
    rate: float,
    dividend_yield: float,
    vols: FloatArray,
    option_type: OptionType | str,
    steps: int = 201,
    method: str = "lr",
) -> FloatArray:
    """American-minus-European value on one lattice, per (strike, vol).

    Args:
        spot: Underlying price.
        strikes: Strikes, one per quote.
        tau: Time to expiry in years (one expiry per call).
        rate: Continuously compounded rate for ``tau``.
        dividend_yield: Continuous yield for ``tau`` (the carry that reproduces the
            forward: ``rate - ln(F/S)/tau``).
        vols: Volatility per strike. ``nan`` gives a ``nan`` premium.
        option_type: ``"call"`` or ``"put"``.
        steps: Lattice steps (forced odd for Leisen-Reimer, as in
            :func:`optpricing.binomial.price`).
        method: ``"lr"`` (Leisen-Reimer, default) or ``"crr"``. CRR's premium is
            accurate on average but jagged *in strike*: which side of a node each strike
            falls on changes from one listed strike to the next, and on $1 strikes that
            sawtooth is as large as a short-dated butterfly's value. Leisen-Reimer centres
            each tree on its own strike, so the premium is smooth across strikes and
            closer to the converged value at the same step count.

    Returns:
        The premium per strike, floored at zero (on one lattice it is non-negative up to
        round-off).
    """
    if method not in {"lr", "crr"}:
        raise ValueError(f"unknown lattice method {method!r}; expected 'lr' or 'crr'")
    if method == "lr" and steps % 2 == 0:
        steps += 1
    phi = to_option_type(option_type).sign
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    sig = np.broadcast_to(np.asarray(vols, dtype=np.float64), k.shape).copy()
    out = np.full(k.shape, np.nan)
    good = np.isfinite(sig) & (sig > 0.0)
    if tau <= 0.0:
        return np.where(good, 0.0, np.nan)
    if good.any():
        am, eu = _american_european(
            spot, k[good], tau, rate, dividend_yield, sig[good], phi, steps, method
        )
        out[good] = np.maximum(am - eu, 0.0)
    return out


@dataclass(frozen=True)
class Deamericanised:
    """The fixed point of :func:`deamericanise`, with its convergence record.

    Attributes:
        european_price: Market price minus the premium at the converged vol.
        vol: Black-Scholes implied vol of ``european_price`` (``nan`` if not invertible).
        premium: Early-exercise premium at ``vol``.
        iterations: Iterations each quote needed to meet the tolerance.
        converged: Whether each quote met the tolerance within ``max_iter``.
        step_history: Largest ``|sigma_{n+1} - sigma_n|`` over all quotes, per iteration.
        contraction: Largest observed ratio of successive per-quote steps
            ``|d_{n+1}| / |d_n|`` (only where ``|d_n|`` is above round-off): the empirical
            slope of the fixed-point map. Below one means a contraction.
    """

    european_price: FloatArray
    vol: FloatArray
    premium: FloatArray
    iterations: FloatArray
    converged: np.ndarray
    step_history: list[float]
    contraction: float


def deamericanise(
    prices: FloatArray,
    spot: float,
    strikes: FloatArray,
    tau: float,
    rate: float,
    dividend_yield: float,
    option_type: OptionType | str,
    steps: int = 201,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> Deamericanised:
    r"""Solve :math:`\sigma = \mathrm{BS}^{-1}(V^A - e(\sigma))` by fixed-point iteration.

    Args:
        prices: American market prices (mids, typically), one per strike.
        spot, strikes, tau, rate, dividend_yield: As in :func:`early_exercise_premium`.
        option_type: ``"call"`` or ``"put"`` (one side per call).
        steps: Lattice steps for the premium.
        tol: Stop when every quote's vol moves by less than this between iterations.
        max_iter: Iteration cap. Quotes still moving after it are flagged unconverged.

    Returns:
        :class:`Deamericanised`. Quotes whose American price is not invertible, or whose
        de-Americanised price falls outside the European no-arbitrage bounds, come back
        with ``nan`` vol.
    """
    opt = to_option_type(option_type)
    px = np.atleast_1d(np.asarray(prices, dtype=np.float64))
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))

    def invert(target: FloatArray) -> FloatArray:
        return np.asarray(
            iv.implied_vol(target, spot, k, tau, rate, opt, dividend_yield), dtype=np.float64
        )

    vol = invert(px)
    premium = np.zeros_like(px)
    iterations = np.zeros_like(px)
    converged = np.zeros(px.shape, dtype=bool)
    history: list[float] = []
    ratio = 0.0
    prev_step = np.full(px.shape, np.nan)
    for n in range(1, max_iter + 1):
        premium = early_exercise_premium(spot, k, tau, rate, dividend_yield, vol, opt, steps)
        new_vol = invert(px - premium)
        with np.errstate(invalid="ignore"):
            step = np.abs(new_vol - vol)
        vol = new_vol
        live = np.isfinite(step)
        newly = live & ~converged & (step < tol)
        iterations = np.where(newly, float(n), iterations)
        converged |= newly
        # The slope of the map, from successive steps well above round-off.
        measurable = live & np.isfinite(prev_step) & (prev_step > 1e3 * tol)
        if measurable.any():
            ratio = max(ratio, float(np.max(step[measurable] / prev_step[measurable])))
        prev_step = step
        history.append(float(np.max(step[live])) if live.any() else 0.0)
        if (converged | ~live).all():
            break
    iterations = np.where(converged, iterations, float(max_iter))
    # The premium that matches the returned vol (one more lattice pass would change it by
    # less than tol x premium-vega, i.e. nothing).
    return Deamericanised(
        european_price=px - premium,
        vol=vol,
        premium=premium,
        iterations=iterations,
        converged=converged & np.isfinite(vol),
        step_history=history,
        contraction=ratio,
    )


@dataclass(frozen=True)
class ForwardSolution:
    """One expiry's parity forward with the early-exercise premia removed."""

    forward: float
    per_strike: FloatArray
    call_premium: FloatArray
    put_premium: FloatArray
    iterations: int


def american_forward(
    strikes: FloatArray,
    call_mid: FloatArray,
    put_mid: FloatArray,
    spot: float,
    tau: float,
    rate: float,
    discount: float,
    vol_at: Callable[[FloatArray], FloatArray],
    forward0: float,
    steps: int = 201,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> ForwardSolution:
    r"""Re-solve parity for the forward with each leg's early-exercise premium removed.

    .. math:: \hat F = \operatorname{median}_i\,
              \big[K_i + (C_i - P_i - e^C_i + e^P_i)/D\big].

    The premia depend on the carry :math:`q = r - \ln(F/S)/\tau`, and the vol at each
    strike is read at :math:`k = \ln(K/F)`, so both move with :math:`F`; the two are
    iterated from ``forward0`` until :math:`F` moves by less than ``tol`` (relative).

    Args:
        strikes, call_mid, put_mid: The matched pairs used for the forward fit.
        spot, tau, rate, discount: Market inputs for this expiry.
        vol_at: Vol as a function of log-moneyness :math:`\ln(K/F)` (from the surface).
        forward0: Starting forward (the European-parity one).
        steps: Lattice steps.
        tol: Relative tolerance on the forward.
        max_iter: Iteration cap.
    """
    k = np.asarray(strikes, dtype=np.float64)
    y = np.asarray(call_mid, dtype=np.float64) - np.asarray(put_mid, dtype=np.float64)
    forward = float(forward0)
    ee_c = ee_p = np.zeros_like(k)
    per_strike = k + y / discount
    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        q = rate - float(np.log(forward / spot)) / tau
        vols = np.asarray(vol_at(np.log(k / forward)), dtype=np.float64)
        ee_c = early_exercise_premium(spot, k, tau, rate, q, vols, OptionType.CALL, steps)
        ee_p = early_exercise_premium(spot, k, tau, rate, q, vols, OptionType.PUT, steps)
        per_strike = k + (y - (ee_c - ee_p)) / discount
        new_forward = float(np.median(per_strike))
        moved = abs(new_forward / forward - 1.0)
        forward = new_forward
        if moved < tol:
            break
    return ForwardSolution(
        forward=forward,
        per_strike=per_strike,
        call_premium=ee_c,
        put_premium=ee_p,
        iterations=iterations,
    )
