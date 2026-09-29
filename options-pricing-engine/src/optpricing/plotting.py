"""Charts for the pricing engine.

Every figure follows the house style in :mod:`optpricing.style`: one validated palette,
colour assigned by job (categorical for identity, a single blue ramp for the *ordered*
set of expiries, diverging blue/red for signed errors), recessive grids, a legend whenever
there is more than one series, and never two y-axes on one plot.

Functions return the created :class:`~matplotlib.figure.Figure` so a caller can save it or
embed it; none of them call ``plt.show``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import binomial, blackscholes, montecarlo
from .heston import HestonParams
from .style import (
    CATEGORICAL,
    DIVERGING,
    INK_MUTED,
    INK_SECONDARY,
    SEQUENTIAL_BLUE,
    SURFACE,
    apply_house_style,
    ordinal_colors,
)
from .surface import atm_term_structure, surface_grid
from .types import ExerciseStyle, FloatArray, OptionType

__all__ = [
    "plot_binomial_convergence",
    "plot_exercise_boundary",
    "plot_greeks_fd",
    "plot_greeks_panel",
    "plot_heston_fit",
    "plot_heston_fit_errors",
    "plot_mc_convergence",
    "plot_parity_residuals",
    "plot_smile_grid",
    "plot_smiles",
    "plot_surface_3d",
    "plot_surface_heatmap",
    "plot_term_structure",
    "save_all",
]


def _finish(fig: Figure, subtitle: str | None = None) -> Figure:
    if subtitle:
        fig.text(0.0, 1.0, subtitle, ha="left", va="bottom", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_smiles(surface: pd.DataFrame, max_expiries: int = 6) -> Figure:
    """Implied-volatility smiles, one line per expiry, in log-moneyness.

    Expiries are an *ordered* set, so they are coloured with the sequential blue ramp
    (light = near, dark = far) rather than with arbitrary categorical hues. Each line is
    also directly labelled at its right end, so identity never depends on colour alone.
    """
    apply_house_style()
    taus = sorted(surface["tau"].unique())
    if len(taus) > max_expiries:
        idx = np.linspace(0, len(taus) - 1, max_expiries).round().astype(int)
        taus = [taus[i] for i in idx]
    colors = ordinal_colors(len(taus))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for i, (tau, color) in enumerate(zip(taus, colors, strict=True)):
        sl = surface.loc[surface["tau"] == tau].sort_values("log_moneyness")
        ax.plot(
            sl["log_moneyness"], 100.0 * sl["implied_vol"], color=color, label=f"{tau * 365:.0f}d"
        )
        # Direct-label every line when there are few, otherwise only the two ends -- enough
        # to anchor which way the colour ramp runs without a pile-up of overlapping text.
        if len(taus) <= 4 or i in (0, len(taus) - 1):
            ax.annotate(
                f"{tau * 365:.0f}d",
                xy=(sl["log_moneyness"].iloc[-1], 100.0 * sl["implied_vol"].iloc[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
                fontweight="semibold",
            )
    ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
    ax.annotate(
        "forward",
        xy=(0.0, ax.get_ylim()[1]),
        xytext=(3, -10),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.set_xlabel("log-moneyness  $k=\\ln(K/F)$")
    ax.set_ylabel("implied volatility (%)")
    ax.set_title("The smile steepens as expiry shortens")
    ax.legend(title="expiry", ncol=2, loc="upper right")
    return _finish(fig, "Out-of-the-money quotes only; forward from put-call parity")


def plot_term_structure(surface: pd.DataFrame) -> Figure:
    """At-the-money volatility and total variance against expiry.

    Two panels rather than two y-axes: the level and the accumulated variance are on
    different scales and a dual-axis chart would invent a crossing point that means
    nothing.
    """
    apply_house_style()
    ts = atm_term_structure(surface)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    axes[0].plot(ts["tau"] * 365, 100.0 * ts["atm_vol"], color=CATEGORICAL[0], marker="o")
    axes[0].set_xlabel("days to expiry")
    axes[0].set_ylabel("ATM implied volatility (%)")
    axes[0].set_title("ATM term structure")
    axes[0].set_xscale("log")

    axes[1].plot(ts["tau"] * 365, ts["atm_total_variance"], color=CATEGORICAL[0], marker="o")
    axes[1].set_xlabel("days to expiry")
    axes[1].set_ylabel(r"total variance  $\sigma^2\tau$")
    axes[1].set_title("Total variance must be monotone")
    axes[1].set_xscale("log")
    return _finish(fig, "Vega-weighted average of quotes within +/-5% log-moneyness of the forward")


def plot_surface_heatmap(surface: pd.DataFrame) -> Figure:
    """Implied-volatility surface as a heatmap -- the readable view.

    A heatmap beats a 3-D wireframe for actually reading values off a surface: nothing is
    hidden behind a ridge, and the eye compares cells directly instead of estimating
    heights through a perspective projection. The 3-D version exists because a vol surface
    is conventionally shown that way, but this is the one to read.

    Magnitude, so a *sequential* single-hue ramp (light = low vol, dark = high). Cells
    outside an expiry's quoted strike range are left blank rather than extrapolated.
    """
    apply_house_style()
    taus, ks, grid = surface_grid(surface, n_moneyness=45)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    mesh = ax.pcolormesh(
        ks,
        np.arange(taus.size),
        100.0 * grid,
        cmap=SEQUENTIAL_BLUE,
        shading="nearest",
        edgecolors=SURFACE,
        linewidth=0.4,
    )
    ax.set_yticks(np.arange(taus.size))
    ax.set_yticklabels([f"{t * 365:.0f}d" for t in taus])
    ax.axvline(0.0, color="white", linewidth=1.2, linestyle=(0, (4, 4)))
    ax.set_xlabel("log-moneyness $k=\\ln(K/F)$")
    ax.set_ylabel("expiry")
    ax.set_title("Implied volatility surface")
    ax.grid(visible=False)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("implied vol (%)", color=INK_SECONDARY, fontsize=9)
    return _finish(fig, "Blank cells are strikes that expiry does not quote -- not extrapolated")


def plot_surface_3d(surface: pd.DataFrame) -> Figure:
    """Three-dimensional implied-volatility surface over log-moneyness and expiry.

    The expiry axis is spaced by ``sqrt(days)``, not linearly. Listed expiries cluster
    heavily at the front -- half of them inside three months on SPY -- and a linear axis
    squashes exactly the region where the surface has all of its structure into the first
    few percent of the plot.

    Gaps are left as gaps: grid points outside an expiry's quoted strike range are not
    extrapolated, so the plot shows the shape of the *data*, not an invented wing.
    """
    apply_house_style()
    taus, ks, grid = surface_grid(surface, n_moneyness=45)
    days = taus * 365.0
    mesh_k, mesh_t = np.meshgrid(ks, np.sqrt(days))

    fig = plt.figure(figsize=(8.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        mesh_k,
        mesh_t,
        100.0 * grid,
        cmap=SEQUENTIAL_BLUE,
        linewidth=0.25,
        edgecolor="white",
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    tick_days = np.array([7, 30, 90, 180, 365, 730])
    tick_days = tick_days[(tick_days >= days.min() * 0.9) & (tick_days <= days.max() * 1.1)]
    ax.set_yticks(np.sqrt(tick_days))
    ax.set_yticklabels([str(int(d)) for d in tick_days])
    ax.set_xlabel("log-moneyness $k$", labelpad=8)
    ax.set_ylabel("days to expiry (sqrt scale)", labelpad=8)
    ax.set_zlabel("implied vol (%)", labelpad=6)
    ax.set_title("Implied volatility surface")
    ax.view_init(elev=26, azim=-128)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label("implied vol (%)", color=INK_SECONDARY, fontsize=9)
    return fig


def plot_heston_fit(errors: pd.DataFrame, max_expiries: int = 6) -> Figure:
    """Small multiples of market smile vs calibrated Heston smile, one panel per expiry.

    Small multiples rather than one crowded axes: with six expiries and two series each,
    a single plot would need twelve lines and the comparison that matters -- model against
    market *within* an expiry -- would be the hardest one to make.
    """
    apply_house_style()
    taus = sorted(errors["tau"].unique())
    if len(taus) > max_expiries:
        idx = np.linspace(0, len(taus) - 1, max_expiries).round().astype(int)
        taus = [taus[i] for i in idx]

    ncols = 3
    nrows = int(np.ceil(len(taus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 3.2 * nrows), sharey=False)
    flat = np.atleast_1d(axes).ravel()

    for ax, tau in zip(flat, taus, strict=False):
        sl = errors.loc[errors["tau"] == tau].sort_values("log_moneyness")
        ax.plot(
            sl["log_moneyness"],
            100.0 * sl["market_vol"],
            color=CATEGORICAL[0],
            marker="o",
            markersize=3.5,
            linewidth=0,
            label="market",
        )
        ax.plot(sl["log_moneyness"], 100.0 * sl["model_vol"], color=CATEGORICAL[1], label="Heston")
        ax.set_title(f"{tau * 365:.0f} days")
        ax.set_xlabel("$k$")
        ax.set_ylabel("IV (%)")
    for ax in flat[len(taus) :]:
        ax.set_visible(False)
    flat[0].legend(loc="upper right")
    fig.suptitle("Calibrated Heston against the quoted smile", x=0.01, ha="left")
    return _finish(fig)


def plot_heston_fit_errors(errors: pd.DataFrame, params: HestonParams) -> Figure:
    """Signed Heston fit error in vol points, across moneyness and expiry.

    A *diverging* map: the sign is the message (model above or below market), so the
    palette has two poles around a neutral grey zero, and the colour limits are symmetric.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    valid = errors.dropna(subset=["vol_error"])
    limit = float(np.nanpercentile(np.abs(valid["vol_error"]), 98)) * 100.0
    scatter = ax.scatter(
        valid["log_moneyness"],
        valid["tau"] * 365.0,
        c=100.0 * valid["vol_error"],
        cmap=DIVERGING,
        vmin=-limit,
        vmax=limit,
        s=26,
        linewidths=0.4,
        edgecolors="white",
    )
    ax.set_yscale("log")
    ax.set_xlabel("log-moneyness $k$")
    ax.set_ylabel("days to expiry")
    ax.set_title("Where Heston misses: the short-dated put wing")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("model - market (vol points)", color=INK_SECONDARY, fontsize=9)
    subtitle = (
        f"v0={params.v0:.4f}  kappa={params.kappa:.2f}  theta={params.theta:.4f}  "
        f"xi={params.xi:.2f}  rho={params.rho:+.2f}"
    )
    return _finish(fig, subtitle)


def plot_binomial_convergence(
    spot: float = 100.0,
    strike: float = 100.0,
    tau: float = 1.0,
    rate: float = 0.05,
    sigma: float = 0.2,
    max_steps: int = 400,
) -> Figure:
    """Absolute pricing error of CRR and Leisen-Reimer trees against Black-Scholes.

    Log-log, so the convergence *order* is the slope: CRR sits on a ``-1`` slope with the
    familiar sawtooth (the strike drifts between terminal nodes as ``n`` changes);
    Leisen-Reimer sits on a ``-2`` slope and is smooth.
    """
    apply_house_style()
    exact = float(blackscholes.price(spot, strike, tau, rate, sigma, OptionType.CALL))
    # Every n, not every other n: the CRR sawtooth is driven by the parity of n, which
    # decides whether a terminal node lands exactly on the strike. Sampling even n only
    # would hide the very effect this chart exists to show.
    crr_steps = np.arange(10, max_steps + 1, 1)
    lr_steps = np.arange(11, max_steps + 1, 10)

    crr_err = [
        abs(
            binomial.price(
                spot,
                strike,
                tau,
                rate,
                sigma,
                OptionType.CALL,
                ExerciseStyle.EUROPEAN,
                int(n),
                binomial.TreeMethod.CRR,
            )
            - exact
        )
        for n in crr_steps
    ]
    lr_err = [
        abs(
            binomial.price(
                spot,
                strike,
                tau,
                rate,
                sigma,
                OptionType.CALL,
                ExerciseStyle.EUROPEAN,
                int(n),
                binomial.TreeMethod.LR,
            )
            - exact
        )
        for n in lr_steps
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.loglog(crr_steps, crr_err, color=CATEGORICAL[0], linewidth=1.0, label="Cox-Ross-Rubinstein")
    ax.loglog(
        lr_steps, lr_err, color=CATEGORICAL[1], marker="o", markersize=4, label="Leisen-Reimer"
    )
    # Reference slopes anchored to each series' own first point, so the eye compares
    # slope rather than level.
    ref_n = np.array([10, max_steps], dtype=float)
    ax.loglog(
        ref_n,
        crr_err[0] * crr_steps[0] / ref_n,
        color=INK_MUTED,
        linewidth=0.9,
        linestyle=(0, (4, 4)),
        zorder=0,
    )
    ax.loglog(
        ref_n,
        lr_err[0] * lr_steps[0] ** 2 / ref_n**2,
        color=INK_MUTED,
        linewidth=0.9,
        linestyle=(0, (1, 3)),
        zorder=0,
    )
    ax.annotate(
        "$O(1/n)$",
        xy=(max_steps, crr_err[0] * crr_steps[0] / max_steps),
        xytext=(-46, 8),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.annotate(
        "$O(1/n^2)$",
        xy=(max_steps, lr_err[0] * lr_steps[0] ** 2 / max_steps**2),
        xytext=(-52, -14),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    ax.set_xlabel("tree steps $n$")
    ax.set_ylabel("|tree price - Black-Scholes|")
    ax.set_title("Leisen-Reimer converges two orders faster, without the sawtooth")
    ax.legend(loc="lower left")
    return _finish(
        fig, f"European call, S=K={spot:.0f}, tau={tau:.0f}y, r={rate:.0%}, sigma={sigma:.0%}"
    )


def plot_mc_convergence(
    spot: float = 100.0,
    strike: float = 100.0,
    tau: float = 1.0,
    rate: float = 0.05,
    sigma: float = 0.2,
    seed: int = 20260918,
    n_replications: int = 24,
) -> Figure:
    """Monte Carlo standard error by variance-reduction mix, plus a calibration check.

    **Left:** reported standard error against effective sample size. All four series use
    the same *effective* sample count, so the comparison is like for like -- an antithetic
    run at ``n`` effective samples draws ``n`` normals and evaluates ``2n`` payoffs, the
    same work as a plain run at ``2n``.

    **Right:** the ratio of the realised root-mean-square error over
    ``n_replications`` independent runs to the standard error those runs reported. It
    should sit at 1. This is the chart that catches the classic antithetic bug (counting
    the ``Z`` and ``-Z`` legs as independent draws), which shows up here as a ratio near
    ``sqrt(2)`` rather than as anything wrong with the price.
    """
    apply_house_style()
    exact = float(blackscholes.price(spot, strike, tau, rate, sigma, OptionType.CALL))
    sizes = np.array([2_000, 5_000, 12_000, 30_000, 75_000, 200_000, 500_000])
    ratio_sizes = np.array([2_000, 8_000, 30_000, 120_000])
    configs = [
        ("plain", False, False),
        ("antithetic", True, False),
        ("control variate", False, True),
        ("both", True, True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for (label, anti, cv), color in zip(configs, CATEGORICAL, strict=True):
        ses = []
        for n in sizes:
            draws = int(2 * n) if anti else int(n)
            res = montecarlo.european_price(
                spot,
                strike,
                tau,
                rate,
                sigma,
                OptionType.CALL,
                draws,
                antithetic=anti,
                control_variate=cv,
                seed=seed + int(n),
            )
            ses.append(res.std_error)
        axes[0].loglog(sizes, ses, color=color, marker="o", markersize=4, label=label)
        axes[0].annotate(
            label,
            xy=(sizes[-1], ses[-1]),
            xytext=(5, -3),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )

        ratios = []
        for n in ratio_sizes:
            draws = int(2 * n) if anti else int(n)
            prices, reported = [], []
            for rep in range(n_replications):
                res = montecarlo.european_price(
                    spot,
                    strike,
                    tau,
                    rate,
                    sigma,
                    OptionType.CALL,
                    draws,
                    antithetic=anti,
                    control_variate=cv,
                    seed=seed + 977 * rep + int(n),
                )
                prices.append(res.price)
                reported.append(res.std_error)
            rmse = float(np.sqrt(np.mean((np.array(prices) - exact) ** 2)))
            ratios.append(rmse / float(np.mean(reported)))
        axes[1].semilogx(ratio_sizes, ratios, color=color, marker="o", markersize=4, label=label)

    ref = np.array([sizes[0], sizes[-1]], dtype=float)
    axes[0].loglog(ref, 15.0 / np.sqrt(ref), color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 4)))
    axes[0].annotate(
        "$O(n^{-1/2})$",
        xy=(ref[0], 15.0 / np.sqrt(ref[0])),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    axes[0].set_xlabel("effective samples")
    axes[0].set_ylabel("standard error")
    axes[0].set_title("Both together: 58x variance reduction")
    axes[0].set_xlim(right=sizes[-1] * 3)

    axes[1].axhline(1.0, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 4)), zorder=0)
    axes[1].axhline(np.sqrt(2.0), color=INK_MUTED, linewidth=0.7, linestyle=(0, (1, 3)), zorder=0)
    axes[1].annotate(
        r"$\sqrt{2}$ = the classic antithetic bug",
        xy=(ratio_sizes[0], np.sqrt(2.0)),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    axes[1].set_ylim(0.0, 1.8)
    axes[1].set_xlabel("effective samples")
    axes[1].set_ylabel("realised RMSE / reported SE")
    axes[1].set_title(f"Error bars are honest ({n_replications} replications)")
    axes[1].legend(loc="lower right", ncol=2)
    return _finish(fig, "European call, S=K=100, 1y, r=5%, sigma=20%")


def plot_exercise_boundary(
    spot: float = 100.0,
    strike: float = 100.0,
    tau: float = 1.0,
    rate: float = 0.05,
    sigma: float = 0.2,
    dividend_yield: float = 0.0,
    steps: int = 600,
) -> Figure:
    """American put early-exercise boundary extracted from a CRR tree."""
    apply_house_style()
    times, boundary = binomial.american_exercise_boundary(
        spot, strike, tau, rate, sigma, OptionType.PUT, steps, dividend_yield
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(times, boundary, color=CATEGORICAL[0], label="exercise boundary")
    ax.axhline(strike, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 4)))
    ax.annotate(
        "strike",
        xy=(0.0, strike),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    finite = np.isfinite(boundary)
    ax.fill_between(
        times[finite], 0.0, boundary[finite], color=CATEGORICAL[0], alpha=0.10, linewidth=0
    )
    ax.annotate(
        "exercise immediately", xy=(tau * 0.45, strike * 0.80), fontsize=9, color=CATEGORICAL[0]
    )
    ax.annotate("hold", xy=(tau * 0.45, strike * 1.02), fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(strike * 0.7, strike * 1.15)
    ax.set_xlabel("years from today")
    ax.set_ylabel("critical spot")
    ax.set_title("American put: exercise when spot falls through the boundary")
    ax.legend(loc="lower left")
    return _finish(
        fig,
        "Left end is blank where the CRR lattice has no node that low -- not resolved, not zero",
    )


def plot_greeks_panel(
    spot_range: tuple[float, float] = (60.0, 140.0),
    strike: float = 100.0,
    rate: float = 0.05,
    sigma: float = 0.2,
    taus: tuple[float, ...] = (1.0 / 12, 0.25, 1.0),
) -> Figure:
    """Price, delta, gamma, vega, theta and vanna against spot, for three expiries."""
    apply_house_style()
    spots = np.linspace(*spot_range, 300)
    colors = ordinal_colors(len(taus))
    panels: list[tuple[str, Callable[[float], FloatArray]]] = [
        ("price", lambda t: blackscholes.price(spots, strike, t, rate, sigma, OptionType.CALL)),
        ("delta", lambda t: blackscholes.delta(spots, strike, t, rate, sigma, OptionType.CALL)),
        ("gamma", lambda t: blackscholes.gamma(spots, strike, t, rate, sigma)),
        ("vega", lambda t: blackscholes.vega(spots, strike, t, rate, sigma)),
        (
            "theta (per year)",
            lambda t: blackscholes.theta(spots, strike, t, rate, sigma, OptionType.CALL),
        ),
        ("vanna", lambda t: blackscholes.vanna(spots, strike, t, rate, sigma)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.4))
    for ax, (name, fn) in zip(axes.ravel(), panels, strict=True):
        for tau, color in zip(taus, colors, strict=True):
            ax.plot(spots, np.asarray(fn(tau)), color=color, label=f"{tau * 12:.0f}m")
        ax.axvline(strike, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
        ax.set_title(name)
        ax.set_xlabel("spot")
    axes[0, 0].legend(title="expiry", loc="upper left")
    fig.suptitle("Black-Scholes call Greeks", x=0.01, ha="left")
    return _finish(fig, f"K={strike:.0f}, r={rate:.0%}, sigma={sigma:.0%}")


def plot_smile_grid(surface: pd.DataFrame, ncols: int = 4) -> Figure:
    """Every expiry's smile on its own axes, with the bid-ask implied-vol band.

    The overlay in :func:`plot_smiles` shows how the smile changes with expiry; this one
    shows how well each smile is actually *known*. The shaded band runs from the vol
    implied by the bid to the vol implied by the ask, so a wide band means the mid-price
    vol could sit anywhere inside it. Puts (left of the forward) and calls (right) are
    marked separately, so the switch at :math:`k = 0` -- where a wrong forward would show
    up as a step -- is visible.
    """
    apply_house_style()
    taus = sorted(surface["tau"].unique())
    nrows = int(np.ceil(len(taus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows))
    flat = np.atleast_1d(axes).ravel()
    has_band = {"iv_bid", "iv_ask"}.issubset(surface.columns)
    for ax, tau in zip(flat, taus, strict=False):
        sl = surface.loc[surface["tau"] == tau].sort_values("log_moneyness")
        k = sl["log_moneyness"].to_numpy(dtype=np.float64)
        if has_band:
            lo = 100.0 * sl["iv_bid"].to_numpy(dtype=np.float64)
            hi = 100.0 * sl["iv_ask"].to_numpy(dtype=np.float64)
            ax.fill_between(
                k, lo, hi, color=CATEGORICAL[0], alpha=0.18, linewidth=0, label="bid-ask IV"
            )
        for side, color in (("put", CATEGORICAL[0]), ("call", CATEGORICAL[1])):
            leg = sl.loc[sl["option_type"] == side]
            ax.plot(
                leg["log_moneyness"],
                100.0 * leg["implied_vol"],
                color=color,
                marker="o",
                markersize=2.2,
                linewidth=0.9,
                label=f"{side} mid",
            )
        ax.axvline(0.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
        ax.set_title(f"{tau * 365:.0f} days", fontsize=10)
    for ax in flat[len(taus) :]:
        ax.set_visible(False)
    for ax in flat[::ncols]:
        ax.set_ylabel("IV (%)")
    for ax in flat[max(len(taus) - ncols, 0) : len(taus)]:
        ax.set_xlabel("$k=\\ln(K/F)$")
    flat[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("SPY smiles by expiry, with the bid-ask implied-vol band", x=0.01, ha="left")
    return _finish(
        fig,
        "Shaded: bid-to-ask implied vol. Thinner than the line almost everywhere -- widest in"
        " the short-dated deep put wing",
    )


def plot_parity_residuals(
    residuals: pd.DataFrame, ncols: int = 4, k_range: tuple[float, float] = (-0.2, 0.2)
) -> Figure:
    r"""Put-call parity residuals per expiry, against the European and American theory.

    Each vertical bar is one strike: the tradeable range of the synthetic forward
    :math:`[C_{bid} - P_{ask},\ C_{ask} - P_{bid}]`, measured relative to European parity
    with the pipeline's forward, :math:`D(F - K)`. A bar that misses zero is a European
    parity violation beyond the spread. The line is what American exercise predicts for
    the same quantity (early-exercise premia plus the shift to the American-adjusted
    forward); a bar that misses the line is a violation even after allowing for it.
    """
    apply_house_style()
    taus = sorted(residuals["tau"].unique())
    nrows = int(np.ceil(len(taus) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.7 * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, tau in zip(flat, taus, strict=False):
        g = residuals.loc[residuals["tau"] == tau].sort_values("log_moneyness")
        g = g.loc[g["log_moneyness"].between(*k_range)]
        k = g["log_moneyness"].to_numpy(dtype=np.float64)
        theo = g["theo_european"].to_numpy(dtype=np.float64)
        lo = g["lower"].to_numpy(dtype=np.float64) - theo
        hi = g["upper"].to_numpy(dtype=np.float64) - theo
        ax.vlines(k, lo, hi, color=CATEGORICAL[0], linewidth=1.1, alpha=0.8, label="bid-ask range")
        american = (g["theo_american"] - g["theo_european"]).to_numpy(dtype=np.float64)
        ax.plot(k, american, color=CATEGORICAL[1], linewidth=1.4, label="American prediction")
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
        # Scale to the well-quoted strikes; stale deep-ITM quotes can be $100 off and
        # would flatten everything else. They are counted in results/parity.md.
        inside = g.loc[g["in_window"]]
        span = np.concatenate(
            [
                (inside["lower"] - inside["theo_european"]).to_numpy(dtype=np.float64),
                (inside["upper"] - inside["theo_european"]).to_numpy(dtype=np.float64),
                american,
            ]
        )
        if span.size:
            lo_lim, hi_lim = np.nanpercentile(span, [2, 98])
            pad = 0.15 * max(hi_lim - lo_lim, 0.2)
            ax.set_ylim(lo_lim - pad, hi_lim + pad)
        ax.set_title(f"{tau * 365:.0f} days", fontsize=10)
    for ax in flat[len(taus) :]:
        ax.set_visible(False)
    for ax in flat[::ncols]:
        ax.set_ylabel(r"$C-P-D(F-K)$  (\$)")
    for ax in flat[max(len(taus) - ncols, 0) : len(taus)]:
        ax.set_xlabel("$k=\\ln(K/F)$")
    flat[0].legend(loc="lower left", fontsize=7)
    fig.suptitle(
        "Put-call parity: European theory misses on the high-strike side; American fits",
        x=0.01,
        ha="left",
    )
    return _finish(
        fig, "Zero = European parity, pipeline forward. y-axis scaled to the fitted strikes"
    )


def plot_greeks_fd(
    errors: pd.DataFrame,
    sweeps: dict[str, pd.DataFrame],
    default_step: dict[str, float],
    tolerances: tuple[float, float] | None = None,
) -> Figure:
    """Finite-difference check of the Greeks: the step-size trade-off and the worst errors.

    Left: relative error of a finite-difference Greek against its analytic value as the
    step shrinks, at a one-day option -- round-off on the left, truncation on the right,
    with the step actually used marked. Right: the worst relative error over the whole
    grid at each maturity, first- and second-order Greeks separately.
    """
    apply_house_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for (name, sweep), color in zip(sweeps.items(), CATEGORICAL, strict=False):
        left.loglog(sweep["step"], sweep["rel_error"], color=color, label=name)
        chosen = default_step[name]
        err = float(np.interp(np.log(chosen), np.log(sweep["step"]), sweep["rel_error"]))
        left.plot([chosen], [err], marker="o", color=color, markersize=6)
    left.set_xlabel("relative step (x natural scale)")
    left.set_ylabel("relative error vs analytic")
    left.set_title("Step size: round-off left, truncation right")
    left.legend(loc="upper center", ncol=2)

    worst = errors.groupby(["tau_days", "order"])["rel_error"].max().unstack("order")
    for order, color, label in ((1, CATEGORICAL[0], "first-order"), (2, CATEGORICAL[1], "second")):
        right.loglog(worst.index, worst[order], color=color, marker="o", label=label)
    if tolerances is not None:
        for tol, color in zip(tolerances, CATEGORICAL[:2], strict=True):
            right.axhline(tol, color=color, linewidth=0.9, linestyle=(0, (4, 4)))
    right.set_xlabel("days to expiry")
    right.set_ylabel("worst relative error on the grid")
    right.set_title("Worst case across moneyness, vol, calls and puts")
    right.legend(loc="center right")
    return _finish(fig, "Dots on the left: the step used. Dashed on the right: the test tolerances")


def save_all(figures: dict[str, Figure], out_dir: Path) -> list[Path]:
    """Write every figure to ``out_dir`` as a PNG and return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written
