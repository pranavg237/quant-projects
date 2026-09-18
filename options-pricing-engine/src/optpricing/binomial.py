r"""Binomial lattice pricing for European and American vanillas.

A binomial tree discretises the log-spot into ``n`` steps of length
:math:`\Delta t = \tau/n`. Over one step the spot moves to :math:`Su` with risk-neutral
probability :math:`p` and to :math:`Sd` with probability :math:`1-p`, where :math:`p` is
chosen so the discounted spot is a martingale:

.. math::
    p = \frac{e^{(r-q)\Delta t} - d}{u - d}.

Three parameterisations are supported:

``CRR`` (Cox-Ross-Rubinstein, 1979)
    :math:`u = e^{\sigma\sqrt{\Delta t}},\; d = 1/u`. Recombining and symmetric in
    log-space; converges as :math:`O(1/n)` with a sawtooth caused by the strike sitting
    between two terminal nodes.

``JR`` (Jarrow-Rudd)
    :math:`u = e^{(r-q-\sigma^2/2)\Delta t + \sigma\sqrt{\Delta t}}`, :math:`p = 1/2`.
    Equal-probability ("risk-neutral drift in the nodes") variant.

``LR`` (Leisen-Reimer, 1996)
    Inverts the Peizer-Pratt normal approximation so the tree is centred on the strike.
    Converges as :math:`O(1/n^2)` **without** the sawtooth, so it reaches 1e-6 accuracy
    in a few hundred steps where CRR needs tens of thousands. Requires an odd ``n``.

American exercise is handled by taking :math:`\max(\text{continuation}, \text{intrinsic})`
at every node during the backward induction.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from .types import ExerciseStyle, FloatArray, OptionType, to_exercise_style, to_option_type

__all__ = [
    "TreeMethod",
    "american_exercise_boundary",
    "price",
    "tree_parameters",
]


class TreeMethod(StrEnum):
    """Lattice parameterisation."""

    CRR = "crr"
    JR = "jr"
    LR = "lr"


def _peizer_pratt(z: float, n: int) -> float:
    """Peizer-Pratt inversion method 2: a normal approximation to a binomial CDF.

    Maps a standard normal deviate ``z`` to the binomial probability that makes an
    ``n``-step tree reproduce :math:`N(z)`. This is what removes the CRR sawtooth.
    """
    if n % 2 == 0:  # pragma: no cover - guarded by the caller
        raise ValueError("Leisen-Reimer requires an odd number of steps")
    denom = n + 1.0 / 3.0 + 0.1 / (n + 1.0)
    inner = 1.0 - np.exp(-((z / denom) ** 2) * (n + 1.0 / 6.0))
    return float(0.5 + np.sign(z) * 0.5 * np.sqrt(inner))


def tree_parameters(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    steps: int,
    method: TreeMethod | str = TreeMethod.CRR,
    dividend_yield: float = 0.0,
) -> tuple[float, float, float, float]:
    r"""Return ``(u, d, p, discount_per_step)`` for the requested parameterisation.

    Exposed separately so the tests can assert the martingale property
    :math:`p u + (1-p) d = e^{(r-q)\Delta t}` directly.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if tau <= 0.0:
        raise ValueError("tau must be > 0 to build a tree")
    meth = TreeMethod(str(method).lower()) if not isinstance(method, TreeMethod) else method
    dt = tau / steps
    growth = np.exp((rate - dividend_yield) * dt)
    discount = float(np.exp(-rate * dt))

    if meth is TreeMethod.CRR:
        u = float(np.exp(sigma * np.sqrt(dt)))
        d = 1.0 / u
        p = float((growth - d) / (u - d))
    elif meth is TreeMethod.JR:
        nu = rate - dividend_yield - 0.5 * sigma * sigma
        u = float(np.exp(nu * dt + sigma * np.sqrt(dt)))
        d = float(np.exp(nu * dt - sigma * np.sqrt(dt)))
        p = 0.5
    else:  # Leisen-Reimer
        if steps % 2 == 0:
            raise ValueError("Leisen-Reimer requires an odd number of steps")
        if spot <= 0.0 or strike <= 0.0:
            raise ValueError("Leisen-Reimer needs positive spot and strike")
        vol_sqrt_t = sigma * np.sqrt(tau)
        d2 = (np.log(spot / strike) + (rate - dividend_yield - 0.5 * sigma**2) * tau) / vol_sqrt_t
        d1 = d2 + vol_sqrt_t
        p = _peizer_pratt(float(d2), steps)
        p_dash = _peizer_pratt(float(d1), steps)
        u = float(growth * p_dash / p)
        d = float((growth - p * u) / (1.0 - p))

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"risk-neutral probability {p:.4f} outside [0, 1]: the tree is arbitrageable. "
            "Increase `steps` or check that sigma is large enough for the carry (r - q)."
        )
    return u, d, p, discount


def price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType | str = OptionType.CALL,
    exercise: ExerciseStyle | str = ExerciseStyle.EUROPEAN,
    steps: int = 500,
    method: TreeMethod | str = TreeMethod.CRR,
    dividend_yield: float = 0.0,
) -> float:
    """Price a vanilla option on a binomial lattice.

    Args:
        spot: Spot price.
        strike: Strike.
        tau: Time to expiry in years.
        rate: Continuously compounded risk-free rate.
        sigma: Volatility.
        option_type: ``"call"`` or ``"put"``.
        exercise: ``"european"`` or ``"american"``.
        steps: Number of time steps. Leisen-Reimer forces this odd.
        method: ``"crr"``, ``"jr"`` or ``"lr"``.
        dividend_yield: Continuous dividend yield.

    Returns:
        The option value at time zero.
    """
    opt = to_option_type(option_type)
    style = to_exercise_style(exercise)
    phi = opt.sign

    if tau <= 0.0 or sigma <= 0.0:
        # No time or no uncertainty: the tree degenerates to the deterministic payoff.
        fwd = spot * np.exp(-dividend_yield * tau) if tau > 0 else spot
        disc_k = strike * np.exp(-rate * tau) if tau > 0 else strike
        intrinsic_now = max(phi * (spot - strike), 0.0)
        deterministic = max(phi * (fwd - disc_k), 0.0)
        if style is ExerciseStyle.AMERICAN:
            return float(max(intrinsic_now, deterministic))
        return float(deterministic)

    meth = TreeMethod(str(method).lower()) if not isinstance(method, TreeMethod) else method
    if meth is TreeMethod.LR and steps % 2 == 0:
        steps += 1

    u, d, p, discount = tree_parameters(spot, strike, tau, rate, sigma, steps, meth, dividend_yield)

    # Terminal spot lattice: node j has had j up-moves and (steps - j) down-moves.
    j = np.arange(steps + 1, dtype=np.float64)
    spot_terminal = spot * (u**j) * (d ** (steps - j))
    values = np.maximum(phi * (spot_terminal - strike), 0.0)

    american = style is ExerciseStyle.AMERICAN
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            j = np.arange(step + 1, dtype=np.float64)
            node_spot = spot * (u**j) * (d ** (step - j))
            values = np.maximum(values, phi * (node_spot - strike))
    return float(values[0])


def american_exercise_boundary(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    option_type: OptionType | str = OptionType.PUT,
    steps: int = 400,
    dividend_yield: float = 0.0,
) -> tuple[FloatArray, FloatArray]:
    r"""Extract the early-exercise boundary of an American option from a CRR tree.

    At each time step the boundary is the most extreme node spot at which immediate
    exercise beats continuation: the highest such spot for a put, the lowest for a call.

    Returns:
        ``(times, boundary)`` where ``times`` are years from today and ``boundary`` is the
        critical spot (``nan`` where no node exercises at that step).
    """
    opt = to_option_type(option_type)
    phi = opt.sign
    u, d, p, discount = tree_parameters(
        spot, strike, tau, rate, sigma, steps, TreeMethod.CRR, dividend_yield
    )
    dt = tau / steps

    j = np.arange(steps + 1, dtype=np.float64)
    spot_terminal = spot * (u**j) * (d ** (steps - j))
    values = np.maximum(phi * (spot_terminal - strike), 0.0)

    boundary = np.full(steps + 1, np.nan, dtype=np.float64)
    boundary[steps] = strike
    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        j = np.arange(step + 1, dtype=np.float64)
        node_spot = spot * (u**j) * (d ** (step - j))
        intrinsic = phi * (node_spot - strike)
        exercise_now = intrinsic > values
        values = np.where(exercise_now, intrinsic, values)
        if exercise_now.any():
            exercised_spots = node_spot[exercise_now]
            boundary[step] = float(exercised_spots.max() if phi < 0 else exercised_spots.min())

    times = np.arange(steps + 1, dtype=np.float64) * dt
    return times, boundary
