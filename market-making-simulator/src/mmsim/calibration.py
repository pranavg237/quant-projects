r"""Estimating Avellaneda-Stoikov parameters from the simulated order book.

The model has four inputs the user does not get to invent: :math:`\sigma`, :math:`\kappa`,
:math:`A` and the tick/price scale. Picking them by hand is where most implementations of
this model go wrong -- :math:`\kappa` in particular is measured in *inverse price units*,
so a value copied from the paper (where the asset trades at 100 with a spread near 1.5)
produces quotes 100 ticks wide on a book whose spread is 2 ticks, and the maker simply
never trades.

So they are estimated, from the same simulated market the strategies will run in, exactly
as a desk estimates them from its own fill logs:

**Fill intensity.** Post a passive probe order at a fixed distance :math:`\delta` from the
mid, measure fills per unit time, repeat across a grid of distances, and fit

.. math:: \ln \lambda(\delta) = \ln A - \kappa\delta

by ordinary least squares. The exponential form is an assumption of the model, and the
:math:`R^2` of that regression says how well the simulated book actually obeys it -- which
is worth knowing, and is reported.

**Volatility.** The realised standard deviation of efficient-price increments, scaled to
the time unit. Arithmetic, not log: the model is built on arithmetic Brownian motion. It is
measured over **blocks of steps, not single steps**, and that is not a detail. Informed
impact is released gradually, so the efficient price has positive autocorrelation at short
horizons: a one-step variance scaled by :math:`1/\Delta t` understates the variance a maker
actually faces over the horizon it holds inventory, by a factor of four on the headline
configuration. :func:`volatility_signature` exposes the whole variance-versus-horizon curve,
which is the standard microstructure diagnostic for exactly this.

Estimating rather than assuming also means the comparison between the idealised engine and
the book engine is a fair one: both are told the same thing about the world.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .avellaneda_stoikov import AvellanedaStoikovParams, HorizonMode
from .book import LimitOrderBook
from .flow import FlowConfig, OrderFlowGenerator
from .types import FloatArray, MarketConfig, Side

__all__ = [
    "FillIntensityFit",
    "estimate_fill_intensity",
    "estimate_volatility",
    "fit_as_params_to_book",
    "volatility_signature",
]

_PROBE = "probe"


@dataclass(frozen=True)
class FillIntensityFit:
    r"""Result of fitting :math:`\lambda(\delta) = Ae^{-\kappa\delta}`.

    Attributes:
        arrival_rate: :math:`A`, the fitted intensity at zero distance, per unit time.
        kappa: :math:`\kappa`, the decay rate, in inverse price units.
        distances: The probe distances used, in price units.
        intensities: Measured fill intensity at each distance.
        r_squared: Fit quality of the log-linear regression. A low value means the book's
            fill curve is not exponential and the model's core assumption is shaky here.
    """

    arrival_rate: float
    kappa: float
    distances: FloatArray
    intensities: FloatArray
    r_squared: float

    @property
    def half_life_ticks(self) -> float:
        """How far the probe must move, in price units, to halve its fill rate."""
        return float(np.log(2.0) / self.kappa)

    def predict(self, distance: FloatArray | float) -> FloatArray:
        """Fitted intensity at arbitrary distances."""
        d = np.asarray(distance, dtype=np.float64)
        return np.asarray(self.arrival_rate * np.exp(-self.kappa * d), dtype=np.float64)


def _efficient_price_path(
    flow_config: FlowConfig,
    market: MarketConfig,
    n_steps: int,
    horizon: float,
    rng: np.random.Generator,
) -> FloatArray:
    """Simulate the efficient price with no strategic maker present."""
    book = LimitOrderBook()
    generator = OrderFlowGenerator(flow_config, market, rng)
    dt = horizon / n_steps
    efficient = float(market.initial_mid_ticks)
    generator.seed_book(book, efficient, 0.0)
    path = np.empty(n_steps + 1, dtype=np.float64)
    path[0] = efficient
    for i in range(n_steps):
        efficient, _ = generator.step(book, efficient, i * dt, dt)
        path[i + 1] = efficient
    return path


def volatility_signature(
    flow_config: FlowConfig,
    market: MarketConfig,
    block_steps: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100, 200),
    n_steps: int = 8_000,
    horizon: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> pd.DataFrame:
    r"""Annualised volatility estimated at a range of sampling horizons.

    The **volatility signature plot** is the standard microstructure diagnostic. For a pure
    martingale it is flat: variance scales linearly with horizon, so the annualised number
    is the same however you sample. A rising signature means positive autocorrelation --
    here, from informed impact being released over many steps rather than at once. A falling
    signature would mean bid-ask bounce or other mean reversion.

    This is what says which horizon's volatility to feed the model: the one matching the
    time the maker actually holds inventory, not the finest grid available.

    Returns:
        Frame with ``block_steps``, ``block_time`` and ``sigma``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    path = _efficient_price_path(flow_config, market, n_steps, horizon, rng)
    dt = horizon / n_steps
    rows: list[dict[str, float]] = []
    for block in block_steps:
        sampled = path[::block] * market.tick_size
        increments = np.diff(sampled)
        if increments.size < 2:
            continue
        rows.append(
            {
                "block_steps": float(block),
                "block_time": float(block * dt),
                "sigma": float(np.std(increments, ddof=1) / np.sqrt(block * dt)),
            }
        )
    return pd.DataFrame(rows)


def estimate_volatility(
    flow_config: FlowConfig,
    market: MarketConfig,
    n_steps: int = 4_000,
    horizon: float = 1.0,
    block_steps: int = 25,
    seed: int | np.random.Generator | None = None,
) -> float:
    r"""Realised volatility of the efficient price, in price units per sqrt(time).

    Includes **both** sources of price movement: the diffusive noise term and the permanent
    impact of informed order flow. That matters -- a maker facing informed flow sees a more
    volatile price than ``volatility_ticks`` alone implies, and telling the model otherwise
    would make it under-price inventory risk exactly when inventory is most dangerous.

    Args:
        flow_config: Flow parameters.
        market: Tick size and initial price.
        n_steps: Simulation length.
        horizon: Simulated time span.
        block_steps: Sample the price every ``block_steps`` steps before differencing.
            **Not 1.** Informed impact is released gradually, so the price is positively
            autocorrelated at short horizons and a one-step estimate understates the
            variance a maker faces over the horizon it holds inventory -- by about 4x on the
            headline configuration. See :func:`volatility_signature`.
        seed: Seed or ``Generator``.
    """
    if block_steps < 1:
        raise ValueError("block_steps must be >= 1")
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    path = _efficient_price_path(flow_config, market, n_steps, horizon, rng)
    dt = horizon / n_steps
    increments = np.diff(path[::block_steps]) * market.tick_size
    return float(np.std(increments, ddof=1) / np.sqrt(block_steps * dt))


def estimate_fill_intensity(
    flow_config: FlowConfig,
    market: MarketConfig,
    distances_ticks: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12),
    n_steps: int = 4_000,
    horizon: float = 1.0,
    probe_size: float = 1.0,
    seed: int | np.random.Generator | None = None,
) -> FillIntensityFit:
    r"""Measure the book's fill curve by probing it, then fit :math:`Ae^{-\kappa\delta}`.

    For each distance the probe re-posts a one-sided order every step at exactly that many
    ticks from the current efficient price, and fills are counted. Both sides are probed and
    averaged, because a one-sided probe would confound the fill curve with any transient
    imbalance in the flow.

    The probe re-posts every step, so it never accumulates queue priority. That makes this a
    *conservative* estimate of :math:`A`: a maker that leaves an order to age will do better.
    The alternative -- letting the probe rest -- would measure a fill curve that depends on
    how long it happened to sit there, which is not a property of the market.

    Args:
        flow_config: Flow parameters.
        market: Tick size and initial price.
        distances_ticks: Probe distances, in ticks.
        n_steps: Steps per probe run.
        horizon: Length of each probe run.
        probe_size: Size posted by the probe.
        seed: Seed or ``Generator``.

    Returns:
        A :class:`FillIntensityFit`.

    Raises:
        ValueError: if fewer than two distances produced any fills, so no line can be fit.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    dt = horizon / n_steps
    measured: list[float] = []

    for offset in distances_ticks:
        book = LimitOrderBook()
        generator = OrderFlowGenerator(flow_config, market, rng)
        efficient = float(market.initial_mid_ticks)
        generator.seed_book(book, efficient, 0.0)
        filled = 0.0

        for i in range(n_steps):
            now = i * dt
            book.cancel_all(_PROBE)
            centre = round(efficient)
            # Probe both sides so a transient flow imbalance cannot bias the estimate.
            book.submit_limit(Side.BID, centre - offset, probe_size, now, _PROBE)
            book.submit_limit(Side.ASK, centre + offset, probe_size, now, _PROBE)
            efficient, trades = generator.step(book, efficient, now, dt)
            filled += sum(t.size for t in trades if t.maker_owner == _PROBE)

        # Two sides quoted for `horizon` time each, so divide by 2 for a per-side rate.
        measured.append(filled / (2.0 * horizon * probe_size))

    distances = np.array(distances_ticks, dtype=np.float64) * market.tick_size
    intensities = np.array(measured, dtype=np.float64)

    usable = intensities > 0
    if usable.sum() < 2:
        raise ValueError(
            "fewer than two probe distances produced fills; the flow is too thin or the "
            "distances are too wide to estimate a fill curve"
        )
    x = distances[usable]
    y = np.log(intensities[usable])
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else float("nan")

    kappa = -float(coef[1])
    if kappa <= 0:
        raise ValueError(
            f"fitted kappa is {kappa:.3f}, i.e. fills increase with distance. "
            "That is not a market; check the flow configuration."
        )
    return FillIntensityFit(
        arrival_rate=float(np.exp(coef[0])),
        kappa=kappa,
        distances=distances,
        intensities=intensities,
        r_squared=r_squared,
    )


def fit_as_params_to_book(
    flow_config: FlowConfig,
    market: MarketConfig,
    gamma: float = 0.5,
    risk_horizon: float = 0.1,
    max_inventory: float | None = None,
    n_steps: int = 4_000,
    seed: int | np.random.Generator | None = None,
) -> tuple[AvellanedaStoikovParams, FillIntensityFit, float]:
    r"""Estimate :math:`\sigma, \kappa, A` from the book and package them for the model.

    ``gamma`` and ``risk_horizon`` remain free -- they are preferences, not properties of
    the market, and the sensitivity analysis sweeps them deliberately. Everything else is
    measured.

    Stationary horizon mode is used, because a book simulation has no meaningful terminal
    time and the finite-horizon model throws away its inventory control as the clock runs
    out (see the README).

    The volatility is measured at a sampling horizon matched to ``risk_horizon`` rather
    than at one step, because informed impact is released gradually and a one-step estimate
    understates what the maker faces by about 4x. See :func:`volatility_signature`.

    Returns:
        ``(params, fill_fit, sigma)``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    # Measure volatility at the horizon the maker actually holds inventory for, capped so
    # enough blocks remain for the estimate to be stable.
    block = int(np.clip(round(risk_horizon * n_steps), 1, max(n_steps // 40, 1)))
    sigma = estimate_volatility(flow_config, market, n_steps=n_steps, block_steps=block, seed=rng)
    fit = estimate_fill_intensity(flow_config, market, n_steps=n_steps, seed=rng)
    params = AvellanedaStoikovParams(
        gamma=gamma,
        kappa=fit.kappa,
        arrival_rate=fit.arrival_rate,
        sigma=sigma,
        horizon=risk_horizon,
        horizon_mode=HorizonMode.STATIONARY,
        max_inventory=max_inventory,
    )
    return params, fit, sigma
