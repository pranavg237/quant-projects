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

**Dividends.** By default the tree uses a continuous dividend yield. SPY pays discrete
quarterly dividends, which give deep in-the-money *calls* an exercise value just before
each ex-date that a continuous yield smears out (and, when the implied yield is not
positive, removes entirely). Every function here therefore also takes a
:class:`~optpricing.dividends.DividendSchedule` of cash dividends, priced with the
escrowed-dividend model (:func:`lattice_prices`); with ``None`` the arithmetic is the
continuous-yield tree's, bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import blackscholes as bs
from . import implied_vol as iv
from .dividends import DividendSchedule
from .types import FloatArray, OptionType, to_option_type

__all__ = [
    "Deamericanised",
    "ForwardSolution",
    "LatticePrices",
    "american_forward",
    "deamericanise",
    "early_exercise_premium",
    "escrowed_spot",
    "lattice_prices",
]


def _peizer_pratt(z: FloatArray, n: int) -> FloatArray:
    """Peizer-Pratt inversion (method 2), vectorised; see :mod:`optpricing.binomial`."""
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    inner = 1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0))
    return np.asarray(0.5 + np.sign(z) * 0.5 * np.sqrt(inner), dtype=np.float64)


def escrowed_spot(
    spot: float, tau: float, rate: float, dividends: DividendSchedule | None
) -> float:
    r"""Spot minus the present value of the cash dividends going ex before ``tau``.

    :math:`S^* = S - \sum_{0 < t_i < \tau} D_i e^{-r t_i}` -- the part of the spot that
    the escrowed-dividend model treats as lognormal (see :func:`lattice_prices`).

    Raises:
        ValueError: if the dividends are worth at least the spot.
    """
    if dividends is None:
        return float(spot)
    s_star = float(spot) - dividends.present_value(tau, rate)
    if s_star <= 0.0:
        raise ValueError("the dividends before expiry are worth more than the spot")
    return s_star


@dataclass(frozen=True)
class LatticePrices:
    """American and European prices from one lattice, plus where it exercised.

    Attributes:
        american: American value per row (``nan`` where the tree is arbitrageable).
        european: European value on the same tree.
        exercise_steps: Boolean ``(rows, steps + 1)``: whether any node of row ``i``
            exercises early at step ``m`` (time ``step_times[m]``; step 0 is today). The
            last column is the lattice's final node: expiry (never an early exercise) or,
            when the last ex-date falls inside the final step, that ex-date (exercise
            just before it). Only filled when requested, otherwise shape ``(rows, 0)``.
        step_times: The time of each lattice node, years (length ``steps + 1``).
    """

    american: FloatArray
    european: FloatArray
    exercise_steps: np.ndarray
    step_times: FloatArray


def _tree_parameters(
    spot: float,
    strikes: FloatArray,
    tau: float,
    rate: float,
    dividend_yield: float,
    vols: FloatArray,
    steps: int,
    method: str,
) -> tuple[FloatArray, FloatArray, FloatArray, np.ndarray]:
    """Per-row ``(p, log u, log d)`` as ``(rows, 1)`` columns, and which rows are valid."""
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
    ok = np.isfinite(p) & (p >= 0.0) & (p <= 1.0) & np.isfinite(log_u) & np.isfinite(log_d)
    return (
        np.asarray(np.where(ok, p, 0.5)[:, None], dtype=np.float64),
        np.asarray(np.where(ok, log_u, 0.0)[:, None], dtype=np.float64),
        np.asarray(np.where(ok, log_d, 0.0)[:, None], dtype=np.float64),
        ok,
    )


def _pv_ahead(
    cash: DividendSchedule | None, step_times: FloatArray, rate: float, last_is_cum: bool
) -> FloatArray:
    """PV at each node time of the dividends still to go ex after it (0 without any).

    ``last_is_cum``: the final node sits *at* the last ex-date (the stub case) and is
    still cum-dividend.
    """
    if cash is None or not len(cash):
        return np.zeros(step_times.size)
    ahead = cash.times[None, :] > step_times[:, None]
    ahead[-1, -1] = last_is_cum  # only the final dividend can be at the final node
    growth = np.exp(-rate * (cash.times[None, :] - step_times[:, None]))
    return np.asarray(np.sum(np.where(ahead, cash.amounts * growth, 0.0), axis=1), dtype=np.float64)


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
    dividends: DividendSchedule | None = None,
    record_exercise: bool = False,
) -> LatticePrices:
    """American and European lattice prices for many (strike, vol) pairs in one sweep.

    Row ``i`` is an independent tree with volatility ``vols[i]`` (and, for Leisen-Reimer,
    centred on ``strikes[i]``); the backward induction runs on all rows at once. It is the
    algorithm of :func:`optpricing.binomial.price` vectorised over strikes (a test pins
    the two together), which is what makes the fixed point below affordable on ~2,000
    quotes. Rows whose risk-neutral probability falls outside ``[0, 1]`` come back
    ``nan``.

    With ``dividends`` the tree is built on the escrowed spot and the dividends still to
    go ex are added back for the exercise decision (:func:`lattice_prices`). With no
    dividends before expiry the arithmetic is exactly the continuous-yield tree's.
    """
    cash = dividends.within(tau) if dividends is not None else None
    s_tree = escrowed_spot(spot, tau, rate, cash)
    # An ex-date inside the final step (SPY's quarterly expiries go ex on expiry day) is
    # where a call's exercise decision matters most and where a step-early decision
    # costs most: stop the lattice at that ex-date and price the stub to expiry in
    # closed form (the binomial Black-Scholes idea of Broadie & Detemple, 1996).
    stub = 0.0
    if cash is not None and len(cash) and tau - float(cash.times[-1]) < tau / steps:
        stub = tau - float(cash.times[-1])
        tau = float(cash.times[-1])
    dt = tau / steps
    p, log_u, log_d, ok = _tree_parameters(
        s_tree, strikes, tau, rate, dividend_yield, vols, steps, method
    )
    disc = float(np.exp(-rate * dt))
    k = strikes[:, None]
    log_s = float(np.log(s_tree))
    step_times = np.arange(steps + 1, dtype=np.float64) * dt
    pv_ahead = _pv_ahead(cash, step_times, rate, stub > 0.0)
    exercised = np.zeros((strikes.size, steps + 1 if record_exercise else 0), dtype=bool)

    j = np.arange(steps + 1, dtype=np.float64)
    terminal = np.exp(log_s + j * log_u + (steps - j) * log_d)
    if stub > 0.0:
        # Holding past the ex-date is worth the European option over the stub (exercise
        # inside a stub shorter than one step is ignored, as between any two nodes).
        side = OptionType.CALL if phi > 0 else OptionType.PUT
        european = np.asarray(
            bs.price(terminal, k, stub, rate, vols[:, None], side, dividend_yield),
            dtype=np.float64,
        )
        cum = phi * (terminal + pv_ahead[-1] - k)  # exercise just before the ex-date
        american = np.maximum(np.maximum(european, cum), phi * (terminal - k))
        if record_exercise:
            exercised[:, steps] = np.any(cum > european, axis=1)
    else:
        european = np.maximum(phi * (terminal - k), 0.0)
        american = european.copy()
    for step in range(steps - 1, -1, -1):
        european = disc * (p * european[:, 1:] + (1.0 - p) * european[:, :-1])
        american = disc * (p * american[:, 1:] + (1.0 - p) * american[:, :-1])
        jj = np.arange(step + 1, dtype=np.float64)
        node = np.exp(log_s + jj * log_u + (step - jj) * log_d)
        intrinsic = phi * (node + pv_ahead[step] - k)  # + 0.0 is exact without dividends
        if record_exercise:
            exercised[:, step] = np.any(intrinsic > american, axis=1)
        american = np.maximum(american, intrinsic)
    am = np.where(ok, american[:, 0], np.nan)
    eu = np.where(ok, european[:, 0], np.nan)
    return LatticePrices(
        american=np.asarray(am, dtype=np.float64),
        european=np.asarray(eu, dtype=np.float64),
        exercise_steps=exercised & ok[:, None],
        step_times=step_times,
    )


def lattice_prices(
    spot: float,
    strikes: FloatArray,
    tau: float,
    rate: float,
    vols: FloatArray,
    option_type: OptionType | str,
    dividend_yield: float = 0.0,
    dividends: DividendSchedule | None = None,
    steps: int = 201,
    method: str = "lr",
    record_exercise: bool = False,
) -> LatticePrices:
    r"""American and European prices on one lattice, with optional discrete cash dividends.

    **The escrowed-dividend model** (Hull, *Options, Futures, and Other Derivatives*,
    "known dollar dividends" in the binomial-tree chapter; the model behind the
    Roll-Geske-Whaley formula). The stock is split into the present value of the cash
    dividends that go ex before expiry, which is riskless, and the rest,

    .. math:: S^*_t = S_t - \sum_{t < t_i < \tau} D_i e^{-r(t_i - t)},

    which is lognormal with volatility :math:`\sigma` and carry :math:`r - q`. The tree
    is built on :math:`S^*` (so it recombines) and the dividends still to come are added
    back only where the stock price itself matters: early exercise at a node at time
    :math:`t` pays :math:`\phi\,(S^* + \mathrm{PV}_t - K)`. At expiry every dividend has
    gone ex, so the European price is Black-Scholes on the escrowed spot exactly (up to
    the lattice's discretisation error).

    Why it is the right model *here*: the surface quotes Black-76 vols on the forward,
    and the escrowed model's European price is Black-76 on the same forward, so the
    European half of the surface is unchanged; only the early-exercise premium sees the
    dividends. Its known biases (Beneder & Vorst 2001; Bos & Vandermark 2002;
    Haug, Haug & Lewis 2003, "Back to basics: a new approach to the discrete dividend
    problem"): the volatility belongs to :math:`S^*`, not the stock, so for a given
    :math:`\sigma` the stock is less volatile than a lognormal stock and the model
    underprices options relative to one -- most for long-dated options with several
    dividends -- and the vol of one underlying depends on which dividends fall before
    each expiry. For a fitted implied vol the European bias is absorbed into the vol; what
    is left is second order in the premium.

    **Timing.** Exercise is checked on the lattice's time grid, so the "just before the
    ex-date" exercise happens at the last node before each ex-time -- up to one step
    early, which forgoes interest on the strike and the option's remaining time value
    and so biases the call premium *down*. The case where that bias is largest, and the
    one that matters for SPY, is an ex-date inside the final step (SPY goes ex on its
    quarterly expiry days, hours before the close, while a 201-step lattice over 21
    months has 3-day steps). There the lattice stops *at* the ex-date, where the holder
    chooses between exercising cum-dividend, :math:`\phi(S^* + D - K)`, and holding the
    European option over the stub to expiry, priced in closed form (the "binomial
    Black-Scholes" device of Broadie & Detemple, 1996). Earlier ex-dates keep the
    grid-timing bias; for SPY they are rarely worth exercising for at all (the quarterly
    dividend is below the interest on the strike to the next ex-date,
    :math:`K(1 - e^{-r/4})`, for any strike above about 200).

    Args:
        spot: Cum-dividend spot.
        strikes: Strikes (one tree per strike).
        tau: Years to expiry.
        rate: Continuously compounded rate (discounting and dividend PVs).
        vols: Volatility per strike (of :math:`S^*` when there are dividends).
        option_type: ``"call"`` or ``"put"``.
        dividend_yield: Continuous yield *in addition to* the cash dividends (use it for a
            residual carry such as a funding basis; 0 for the textbook model).
        dividends: Cash dividends; only those with :math:`0 < t_i < \tau` matter. ``None``
            or an empty schedule is the continuous-yield tree, bit for bit.
        steps: Lattice steps (forced odd for Leisen-Reimer).
        method: ``"lr"`` or ``"crr"``.
        record_exercise: Fill :attr:`LatticePrices.exercise_steps`.
    """
    if method not in {"lr", "crr"}:
        raise ValueError(f"unknown lattice method {method!r}; expected 'lr' or 'crr'")
    if method == "lr" and steps % 2 == 0:
        steps += 1
    if tau <= 0.0:
        raise ValueError("tau must be > 0 to build a tree")
    phi = to_option_type(option_type).sign
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    sig = np.broadcast_to(np.asarray(vols, dtype=np.float64), k.shape).copy()
    return _american_european(
        spot, k, tau, rate, dividend_yield, sig, phi, steps, method, dividends, record_exercise
    )


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
    dividends: DividendSchedule | None = None,
) -> FloatArray:
    """American-minus-European value on one lattice, per (strike, vol).

    Args:
        spot: Underlying price.
        strikes: Strikes, one per quote.
        tau: Time to expiry in years (one expiry per call).
        rate: Continuously compounded rate for ``tau``.
        dividend_yield: Continuous yield for ``tau``. Without ``dividends`` it is the
            carry that reproduces the forward, ``rate - ln(F/S)/tau``; with them it is the
            residual on top of the cash dividends, ``rate - ln(F/S*)/tau`` with ``S*``
            from :func:`escrowed_spot`.
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
        dividends: Discrete cash dividends (escrowed-dividend model, see
            :func:`lattice_prices`). ``None`` is the continuous-yield tree.

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
        lat = _american_european(
            spot, k[good], tau, rate, dividend_yield, sig[good], phi, steps, method, dividends
        )
        out[good] = np.maximum(lat.american - lat.european, 0.0)
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
    dividends: DividendSchedule | None = None,
) -> Deamericanised:
    r"""Solve :math:`\sigma = \mathrm{BS}^{-1}(V^A - e(\sigma))` by fixed-point iteration.

    Args:
        prices: American market prices (mids, typically), one per strike.
        spot, strikes, tau, rate, dividend_yield: As in :func:`early_exercise_premium`.
        option_type: ``"call"`` or ``"put"`` (one side per call).
        steps: Lattice steps for the premium.
        tol: Stop when every quote's vol moves by less than this between iterations.
        max_iter: Iteration cap. Quotes still moving after it are flagged unconverged.
        dividends: Discrete cash dividends. The European price is then inverted with
            Black-Scholes on the escrowed spot and the residual yield, which is Black-76
            on the same forward: the vol means the same thing with or without them.

    Returns:
        :class:`Deamericanised`. Quotes whose American price is not invertible, or whose
        de-Americanised price falls outside the European no-arbitrage bounds, come back
        with ``nan`` vol.
    """
    opt = to_option_type(option_type)
    px = np.atleast_1d(np.asarray(prices, dtype=np.float64))
    k = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    s_bs = escrowed_spot(spot, tau, rate, dividends.within(tau) if dividends else None)

    def invert(target: FloatArray) -> FloatArray:
        return np.asarray(
            iv.implied_vol(target, s_bs, k, tau, rate, opt, dividend_yield), dtype=np.float64
        )

    vol = invert(px)
    premium = np.zeros_like(px)
    iterations = np.zeros_like(px)
    converged = np.zeros(px.shape, dtype=bool)
    history: list[float] = []
    ratio = 0.0
    prev_step = np.full(px.shape, np.nan)
    for n in range(1, max_iter + 1):
        premium = early_exercise_premium(
            spot, k, tau, rate, dividend_yield, vol, opt, steps, dividends=dividends
        )
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
    dividends: DividendSchedule | None = None,
) -> ForwardSolution:
    r"""Re-solve parity for the forward with each leg's early-exercise premium removed.

    .. math:: \hat F = \operatorname{median}_i\,
              \big[K_i + (C_i - P_i - e^C_i + e^P_i)/D\big].

    The premia depend on the carry :math:`q = r - \ln(F/S)/\tau`, and the vol at each
    strike is read at :math:`k = \ln(K/F)`, so both move with :math:`F`; the two are
    iterated from ``forward0`` until :math:`F` moves by less than ``tol`` (relative).
    With ``dividends`` the premia come from the escrowed-dividend tree and the carry is
    the residual :math:`q = r - \ln(F/S^*)/\tau` on top of the cash dividends: the
    forward is still whatever parity says, and the dividends only decide how the carry
    is split between cash and a continuous remainder (a funding basis).

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
    s_star = escrowed_spot(spot, tau, rate, dividends.within(tau) if dividends else None)
    for _ in range(max_iter):
        iterations += 1
        q = rate - float(np.log(forward / s_star)) / tau
        vols = np.asarray(vol_at(np.log(k / forward)), dtype=np.float64)
        args = (spot, k, tau, rate, q, vols)
        ee_c = early_exercise_premium(*args, OptionType.CALL, steps, dividends=dividends)
        ee_p = early_exercise_premium(*args, OptionType.PUT, steps, dividends=dividends)
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
