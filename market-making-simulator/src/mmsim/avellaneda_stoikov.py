r"""The Avellaneda-Stoikov (2008) optimal market-making model.

**Setup.** The efficient mid price is an arithmetic Brownian motion,
:math:`dS_t = \sigma\,dW_t`. The maker quotes a bid :math:`\delta^b` below and an ask
:math:`\delta^a` above the mid. Fills arrive as Poisson processes whose intensity falls
exponentially with distance from the mid,

.. math:: \lambda(\delta) = A e^{-\kappa\delta},

which is what you get if market-order sizes are power-law distributed and a market order
of size :math:`Q` walks the book by :math:`\Delta p \propto \ln Q`. The maker holds
inventory :math:`q` and cash :math:`X`, and maximises exponential utility
:math:`\mathbb{E}[-e^{-\gamma(X_T + q_T S_T)}]` over a horizon :math:`T`.

**Result.** Solving the Hamilton-Jacobi-Bellman equation and expanding to first order in
inventory gives two quantities. The **reservation price** is the price at which the maker
is indifferent to its current position:

.. math:: r(s, q, t) = s - q\gamma\sigma^2(T-t),

and the **optimal total spread** is

.. math::
    \delta^a + \delta^b = \gamma\sigma^2(T-t)
        + \frac{2}{\gamma}\ln\!\left(1 + \frac{\gamma}{\kappa}\right).

The quotes sit symmetrically around the reservation price, *not* around the mid:

.. math::
    \delta^a = \tfrac12(\delta^a + \delta^b) - q\gamma\sigma^2(T-t), \qquad
    \delta^b = \tfrac12(\delta^a + \delta^b) + q\gamma\sigma^2(T-t).

**The one idea.** Everything the model does is in that skew term :math:`q\gamma\sigma^2(T-t)`.
A maker who is long shifts *both* quotes down: it becomes keener to sell and more reluctant
to buy, so inventory mean-reverts to zero without the maker ever crossing the spread. The
size of the shift scales with risk aversion :math:`\gamma`, with variance :math:`\sigma^2`
(inventory is more dangerous when the asset moves more), and with time remaining (a
position held to a distant horizon has more time to hurt you).

**The horizon problem.** As :math:`t \to T` the skew term vanishes and the spread collapses
to :math:`(2/\gamma)\ln(1+\gamma/\kappa)`, because the model knows the game is about to end
and inventory risk with it. A real desk has no terminal time. The standard fix, and the one
implemented here as ``HorizonMode.STATIONARY``, is to freeze :math:`T - t` at a constant
"risk horizon": the quotes stop depending on the clock and the strategy becomes
time-homogeneous. Both modes are implemented so the difference can be measured rather than
argued about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from .types import FloatArray

__all__ = [
    "AvellanedaStoikovParams",
    "HorizonMode",
    "inventory_skew",
    "optimal_quotes",
    "optimal_spread",
    "reservation_price",
]


class HorizonMode(StrEnum):
    """How the model treats the terminal time.

    ``FINITE``
        The model as published: the effective horizon is :math:`T - t`, which shrinks to
        zero at the end of the session. Quotes tighten and inventory control weakens as
        the clock runs out.

    ``STATIONARY``
        The effective horizon is held at a constant. The strategy is time-homogeneous,
        which is what a desk that trades every day actually wants.
    """

    FINITE = "finite"
    STATIONARY = "stationary"


@dataclass(frozen=True)
class AvellanedaStoikovParams:
    r"""Parameters of the model.

    Attributes:
        gamma: Risk aversion :math:`\gamma`. Zero means risk neutral, in which case the
            model degenerates -- see :func:`optimal_spread`.
        kappa: Order-arrival decay :math:`\kappa`. Large ``kappa`` means fill probability
            drops off fast with distance, forcing tight quotes.
        arrival_rate: Intensity scale :math:`A`, the fill rate at zero distance. It does
            not enter the optimal quotes at all (it scales both sides equally), only the
            simulated fill rate.
        sigma: Volatility of the mid, in **price units per sqrt(time)** -- arithmetic, not
            lognormal. Mixing this up with a percentage volatility is the most common way
            to get nonsense out of this model.
        horizon: Terminal time :math:`T`, or the frozen risk horizon in stationary mode.
        horizon_mode: See :class:`HorizonMode`.
        max_inventory: Optional hard position limit. Beyond it the maker stops quoting the
            side that would make the position worse. The model itself has no such limit --
            it relies entirely on the skew -- but every real desk does.
    """

    gamma: float = 0.1
    kappa: float = 1.5
    arrival_rate: float = 140.0
    sigma: float = 2.0
    horizon: float = 1.0
    horizon_mode: HorizonMode = HorizonMode.FINITE
    max_inventory: float | None = None

    def __post_init__(self) -> None:
        if self.gamma < 0.0:
            raise ValueError("gamma must be >= 0")
        if self.kappa <= 0.0:
            raise ValueError("kappa must be > 0")
        if self.arrival_rate <= 0.0:
            raise ValueError("arrival_rate must be > 0")
        if self.sigma < 0.0:
            raise ValueError("sigma must be >= 0")
        if self.horizon <= 0.0:
            raise ValueError("horizon must be > 0")
        if self.max_inventory is not None and self.max_inventory <= 0:
            raise ValueError("max_inventory must be > 0 when set")

    def time_remaining(self, t: float) -> float:
        r"""The effective :math:`T - t` used by the quoting formulas.

        In stationary mode this is the constant ``horizon``; in finite mode it is
        ``max(horizon - t, 0)``.
        """
        if self.horizon_mode is HorizonMode.STATIONARY:
            return self.horizon
        return max(self.horizon - t, 0.0)


def reservation_price(
    mid: float | FloatArray,
    inventory: float | FloatArray,
    time_remaining: float | FloatArray,
    params: AvellanedaStoikovParams,
) -> FloatArray:
    r"""Indifference price :math:`r = s - q\gamma\sigma^2(T-t)`.

    The price at which the maker would be indifferent between holding its current
    inventory and holding none. A long maker's reservation price is *below* the mid: it
    values the asset less than the market does, because it already owns too much of it.
    """
    s = np.asarray(mid, dtype=np.float64)
    q = np.asarray(inventory, dtype=np.float64)
    tau = np.asarray(time_remaining, dtype=np.float64)
    return np.asarray(s - q * params.gamma * params.sigma**2 * tau, dtype=np.float64)


def inventory_skew(
    inventory: float | FloatArray,
    time_remaining: float | FloatArray,
    params: AvellanedaStoikovParams,
) -> FloatArray:
    r"""How far the quotes shift from the mid: :math:`q\gamma\sigma^2(T-t)`.

    Exposed on its own because it is the entire mechanism of the model, and because
    plotting it against inventory is the clearest single picture of what the strategy does.
    """
    q = np.asarray(inventory, dtype=np.float64)
    tau = np.asarray(time_remaining, dtype=np.float64)
    return np.asarray(q * params.gamma * params.sigma**2 * tau, dtype=np.float64)


def optimal_spread(
    time_remaining: float | FloatArray, params: AvellanedaStoikovParams
) -> FloatArray:
    r"""Total optimal spread, inventory-risk term plus monopolistic markup.

        .. math:: \gamma\sigma^2(T-t) + \frac{2}{\gamma}\ln(1 + \gamma/\kappa).

        The first term is compensation for inventory risk over the remaining horizon; the
        second is the monopolistic markup a maker can charge given how fast fill probability
        decays, and is independent of volatility.

    As :math:`\gamma \to 0` the second term has a removable singularity: expanding
        :math:`\tfrac{2}{\gamma}\ln(1+\gamma/\kappa) = \tfrac{2}{\kappa} - \tfrac{\gamma}{\kappa^2}
        + O(\gamma^2)`, the risk-neutral limit is the finite value :math:`2/\kappa`. Evaluating
        the formula literally at small ``gamma`` loses precision to cancellation, so the
        series is used below a cutoff.

        Note the markup is **decreasing** in :math:`\gamma`, approaching :math:`2/\kappa` from
        below -- a more risk-averse maker charges a *smaller* monopolistic markup, preferring a
        likelier small gain to a rarer large one. The total spread nevertheless widens with
        :math:`\gamma`, because the inventory term grows linearly and dominates.
    """
    tau = np.asarray(time_remaining, dtype=np.float64)
    gamma, kappa = params.gamma, params.kappa
    inventory_term = gamma * params.sigma**2 * tau
    if gamma < 1e-8:
        markup = 2.0 / kappa - gamma / kappa**2
    else:
        markup = (2.0 / gamma) * math.log1p(gamma / kappa)
    return np.asarray(inventory_term + markup, dtype=np.float64)


def optimal_quotes(
    mid: float,
    inventory: float,
    t: float,
    params: AvellanedaStoikovParams,
) -> tuple[float | None, float | None]:
    r"""Optimal bid and ask **prices** (not distances).

    Args:
        mid: Current efficient mid price.
        inventory: Current signed inventory ``q``.
        t: Current time, used only in ``FINITE`` horizon mode.
        params: Model parameters.

    Returns:
        ``(bid_price, ask_price)``. A side is ``None`` when ``max_inventory`` blocks it:
        a maker already at its long limit stops bidding entirely rather than quoting a
        price it does not want filled.
    """
    tau = params.time_remaining(t)
    reservation = float(reservation_price(mid, inventory, tau, params))
    half_spread = 0.5 * float(optimal_spread(tau, params))

    bid: float | None = reservation - half_spread
    ask: float | None = reservation + half_spread

    limit = params.max_inventory
    if limit is not None:
        if inventory >= limit:
            bid = None
        if inventory <= -limit:
            ask = None
    return bid, ask
