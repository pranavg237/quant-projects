"""Charts for the simulator.

Colour is assigned by job, as in :mod:`mmsim.style`: the three policies are *identities*
so they take the fixed categorical slots in a fixed order (a filter that drops one must not
repaint the others); inventory is an ordered magnitude so it uses the sequential ramp;
signed markout uses the diverging pair. No chart uses two y-scales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .avellaneda_stoikov import AvellanedaStoikovParams, inventory_skew, optimal_spread
from .calibration import FillIntensityFit
from .engine import SimulationResult
from .experiments import ComparisonResult
from .metrics import markout_curve
from .style import CATEGORICAL, GRID, INK_MUTED, INK_SECONDARY, apply_house_style
from .types import MarketConfig

__all__ = [
    "plot_book_snapshot",
    "plot_fill_intensity_fit",
    "plot_inventory_paths",
    "plot_markout",
    "plot_pnl_distribution",
    "plot_quotes_and_inventory",
    "plot_reservation_price",
    "plot_risk_return",
    "plot_sensitivity",
    "plot_volatility_signature",
    "save_all",
]


def _finish(fig: Figure, subtitle: str | None = None) -> Figure:
    if subtitle:
        fig.text(0.0, 1.0, subtitle, ha="left", va="bottom", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_reservation_price(params: AvellanedaStoikovParams) -> Figure:
    """The whole model in one picture: how quotes skew with inventory and with time.

    **Left:** quoted bid and ask against inventory. Both move together -- the width is
    constant, only the centre shifts -- which is the mechanism by which inventory
    mean-reverts without the maker ever crossing the spread.

    **Right:** the optimal spread against time remaining, in both horizon modes. The
    finite-horizon spread collapses to the pure markup as the session ends, taking the
    model's inventory control with it.
    """
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    inventories = np.linspace(-6.0, 6.0, 121)
    tau = params.time_remaining(0.0)
    skew = np.asarray(inventory_skew(inventories, tau, params))
    half = 0.5 * float(optimal_spread(tau, params))
    mid = 100.0
    axes[0].plot(inventories, mid - skew + half, color=CATEGORICAL[1], label="ask")
    axes[0].plot(inventories, mid - skew - half, color=CATEGORICAL[0], label="bid")
    axes[0].plot(
        inventories,
        mid - skew,
        color=INK_MUTED,
        linewidth=1.0,
        linestyle=(0, (4, 4)),
        label="reservation price",
    )
    axes[0].axhline(mid, color=GRID, linewidth=1.2, zorder=0)
    axes[0].annotate(
        "mid",
        xy=(inventories[0], mid),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    axes[0].set_xlabel("inventory $q$")
    axes[0].set_ylabel("quoted price")
    axes[0].set_title("Long inventory pushes both quotes down")
    axes[0].legend(loc="upper right")

    taus = np.linspace(0.0, params.horizon, 200)
    finite = np.asarray(optimal_spread(taus[::-1], params))
    stationary = np.full_like(taus, float(optimal_spread(params.horizon, params)))
    axes[1].plot(taus, finite, color=CATEGORICAL[0], label="finite horizon")
    axes[1].plot(taus, stationary, color=CATEGORICAL[1], label="stationary")
    markup = float(optimal_spread(0.0, params))
    axes[1].axhline(markup, color=INK_MUTED, linewidth=0.9, linestyle=(0, (1, 3)))
    axes[1].annotate(
        r"pure markup $\frac{2}{\gamma}\ln(1+\gamma/\kappa)$",
        xy=(0.02 * params.horizon, markup),
        xytext=(0, 6),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    axes[1].set_xlabel("time elapsed")
    axes[1].set_ylabel("optimal total spread")
    axes[1].set_title("Finite horizon gives up its inventory control at the bell")
    axes[1].legend(loc="upper right")
    return _finish(
        fig,
        f"gamma={params.gamma:g}, kappa={params.kappa:g}, sigma={params.sigma:g}, "
        f"T={params.horizon:g}",
    )


def plot_quotes_and_inventory(result: SimulationResult, max_points: int = 1200) -> Figure:
    """One session: mid, quotes, inventory and PnL, on a shared time axis.

    Stacked panels rather than twin axes. Price, inventory and PnL live on three different
    scales, and overlaying them on shared axes would invent visual crossings that carry no
    information.
    """
    apply_house_style()
    frame = result.to_frame()
    if len(frame) > max_points:
        frame = frame.iloc[:: len(frame) // max_points]

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 8.0), sharex=True, height_ratios=[2, 1, 1])
    axes[0].plot(frame["time"], frame["mid"], color=INK_SECONDARY, linewidth=1.2, label="mid")
    axes[0].plot(frame["time"], frame["bid"], color=CATEGORICAL[0], linewidth=0.9, label="bid")
    axes[0].plot(frame["time"], frame["ask"], color=CATEGORICAL[1], linewidth=0.9, label="ask")
    axes[0].set_ylabel("price")
    axes[0].set_title(f"{result.policy_name}: one session")
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].plot(frame["time"], frame["inventory"], color=CATEGORICAL[3])
    axes[1].axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
    axes[1].fill_between(frame["time"], 0.0, frame["inventory"], color=CATEGORICAL[3], alpha=0.12)
    axes[1].set_ylabel("inventory")

    axes[2].plot(frame["time"], frame["pnl"], color=CATEGORICAL[2])
    axes[2].axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
    axes[2].set_ylabel("mark-to-market")
    axes[2].set_xlabel("time")
    return _finish(fig)


def plot_inventory_paths(
    comparison: ComparisonResult, n_paths: int = 40, max_points: int = 600
) -> Figure:
    """Inventory paths per policy, small multiples.

    The clearest single picture of what the model buys you: the symmetric maker's position
    is an unbounded random walk, the Avellaneda-Stoikov maker's mean-reverts.
    """
    apply_house_style()
    names = list(comparison.runs)
    fig, axes = plt.subplots(1, len(names), figsize=(4.0 * len(names), 4.2), sharey=True)
    flat = np.atleast_1d(axes).ravel()

    limit = 0.0
    for name, ax, color in zip(names, flat, CATEGORICAL, strict=False):
        runs = comparison.runs[name][:n_paths]
        for run in runs:
            step = max(1, run.times.size // max_points)
            ax.plot(
                run.times[::step], run.inventory[::step], color=color, linewidth=0.7, alpha=0.35
            )
        final = np.array([r.final_inventory for r in comparison.runs[name]])
        limit = max(limit, float(np.percentile(np.abs(final), 99)) * 1.3)
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
        ax.set_title(f"{name}\nstd(final q) = {final.std(ddof=1):.2f}")
        ax.set_xlabel("time")
    flat[0].set_ylabel("inventory")
    flat[0].set_ylim(-limit, limit)
    fig.suptitle("Inventory: mean-reverting versus a random walk", x=0.01, ha="left")
    return _finish(fig)


def plot_pnl_distribution(comparison: ComparisonResult) -> Figure:
    """Terminal PnL distributions and the paired per-run differences.

    **Right panel** is the one that matters: because every policy ran against the same
    seeds, the per-run difference is paired, and its distribution answers "did this
    strategy beat that one on the same day" rather than "were the two averages different".
    """
    apply_house_style()
    names = list(comparison.pnl.columns)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))

    bins = np.histogram_bin_edges(comparison.pnl.to_numpy().ravel(), bins=40)
    for name, color in zip(names, CATEGORICAL, strict=False):
        axes[0].hist(
            comparison.pnl[name],
            bins=bins,
            histtype="step",
            linewidth=1.6,
            color=color,
            label=name,
        )
    axes[0].set_xlabel("terminal PnL")
    axes[0].set_ylabel("runs")
    axes[0].set_title("PnL distribution")
    axes[0].legend(loc="upper left")

    base = names[0]
    for name, color in zip(names[1:], CATEGORICAL[1:], strict=False):
        diff = comparison.pnl[base] - comparison.pnl[name]
        axes[1].hist(diff, bins=35, histtype="step", linewidth=1.6, color=color, label=f"vs {name}")
        axes[1].axvline(diff.mean(), color=color, linewidth=1.0, linestyle=(0, (4, 4)))
    axes[1].axvline(0.0, color=INK_MUTED, linewidth=1.0)
    axes[1].set_xlabel(f"paired PnL difference ({base} minus)")
    axes[1].set_ylabel("runs")
    axes[1].set_title("Paired differences, same seeds")
    axes[1].legend(loc="upper left")
    return _finish(fig, "Common random numbers: run i faces the same market for every policy")


def plot_risk_return(comparison: ComparisonResult) -> Figure:
    """Mean PnL against PnL volatility, with Sharpe isoclines.

    A scatter, not a bar chart of Sharpe: the point is that the strategies sit at different
    places on a risk-return trade-off, and a single ratio hides which way each one moved.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    metrics = comparison.metrics

    x_max = float(metrics["std_pnl"].max()) * 1.25
    y_max = float(metrics["mean_pnl"].max()) * 1.2
    grid = np.linspace(1e-9, x_max, 100)
    for sharpe in (2, 4, 6, 8, 10, 14):
        line = sharpe * grid
        if line[-1] < y_max * 0.15:
            continue
        ax.plot(grid, line, color=GRID, linewidth=0.9, zorder=0)
        inside = line < y_max
        if inside.any():
            idx = int(np.argmax(~inside)) - 1 if (~inside).any() else len(grid) - 1
            ax.annotate(
                f"Sharpe {sharpe}",
                xy=(grid[idx], line[idx]),
                fontsize=8,
                color=INK_MUTED,
                xytext=(-46, 2),
                textcoords="offset points",
            )

    for (_, row), color in zip(metrics.iterrows(), CATEGORICAL, strict=False):
        ax.scatter(
            row["std_pnl"],
            row["mean_pnl"],
            s=110,
            color=color,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
            label=str(row["policy"]),
        )
        ax.annotate(
            str(row["policy"]),
            xy=(row["std_pnl"], row["mean_pnl"]),
            xytext=(9, -4),
            textcoords="offset points",
            fontsize=9,
            color=color,
        )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("PnL standard deviation across runs")
    ax.set_ylabel("mean PnL")
    ax.set_title("Avellaneda-Stoikov trades mean PnL for far less risk")
    ax.legend(loc="lower right")
    return _finish(fig)


def _looks_geometric(x: np.ndarray) -> bool:
    """Whether a value grid is geometrically rather than linearly spaced.

    A geometric sweep plotted on a linear axis crowds every point except the last into the
    left edge, which hides exactly the region a risk-aversion or intensity sweep is about.
    """
    if x.size < 3 or np.any(x <= 0):
        return False
    ratios = x[1:] / x[:-1]
    return bool(np.std(ratios) / np.mean(ratios) < 0.05 and np.mean(ratios) > 1.2)


def plot_sensitivity(sweeps: dict[str, pd.DataFrame]) -> Figure:
    """Sharpe, mean PnL and inventory against each swept parameter, small multiples."""
    apply_house_style()
    n = len(sweeps)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 7.0), squeeze=False)

    for col, (parameter, frame) in enumerate(sweeps.items()):
        x = frame[parameter].to_numpy(dtype=np.float64)
        if _looks_geometric(x):
            axes[0][col].set_xscale("log")
            axes[1][col].set_xscale("log")
        axes[0][col].plot(x, frame["sharpe"], color=CATEGORICAL[0], marker="o", markersize=4)
        axes[0][col].set_ylabel("Sharpe" if col == 0 else "")
        axes[0][col].set_title(parameter)
        axes[1][col].plot(
            x,
            frame["std_final_inventory"],
            color=CATEGORICAL[3],
            marker="o",
            markersize=4,
            label="std(final q)",
        )
        axes[1][col].plot(
            x,
            frame["mean_abs_inventory"],
            color=CATEGORICAL[2],
            marker="s",
            markersize=4,
            label="mean |q|",
        )
        axes[1][col].set_xlabel(parameter)
        axes[1][col].set_ylabel("inventory" if col == 0 else "")
    axes[1][0].legend(loc="upper right")
    fig.suptitle("Parameter sensitivity of the Avellaneda-Stoikov maker", x=0.01, ha="left")
    return _finish(fig)


def plot_fill_intensity_fit(fit: FillIntensityFit, market: MarketConfig) -> Figure:
    r"""Measured fill intensity against distance, with the fitted :math:`Ae^{-\kappa\delta}`.

    Log-linear axes, because that is where the model's assumption is a straight line and a
    departure from it is visible. It is not quite straight: the tail flattens, because
    market-order sizes are power-law and the occasional large sweep reaches deep levels far
    more often than an exponential predicts.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ticks = fit.distances / market.tick_size
    ax.semilogy(
        ticks,
        fit.intensities,
        color=CATEGORICAL[0],
        marker="o",
        linewidth=0,
        markersize=6,
        label="measured",
    )
    dense = np.linspace(fit.distances.min(), fit.distances.max(), 200)
    ax.semilogy(
        dense / market.tick_size,
        fit.predict(dense),
        color=CATEGORICAL[1],
        label=r"fit $Ae^{-\kappa\delta}$",
    )
    ax.set_xlabel("distance from mid (ticks)")
    ax.set_ylabel("fill intensity (per unit time)")
    ax.set_title("The book's fill curve is exponential-ish, not exponential")
    ax.legend(loc="upper right")
    return _finish(
        fig,
        f"A={fit.arrival_rate:.0f}, kappa={fit.kappa:.1f} per price unit "
        f"(1/kappa = {fit.half_life_ticks / np.log(2) / market.tick_size:.1f} ticks), "
        f"R^2={fit.r_squared:.3f}",
    )


def plot_volatility_signature(signatures: dict[str, pd.DataFrame]) -> Figure:
    """Annualised volatility against sampling horizon -- the microstructure diagnostic.

    Flat means the efficient price is a martingale: variance scales with time, so the
    annualised number does not depend on how you sample. A **rising** signature means
    positive autocorrelation, which here comes from informed impact being released over
    many steps rather than at once. It is also the reason the model is fed a
    block-sampled volatility rather than a one-step one: at one step the informed
    contribution has barely happened yet.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (label, frame), color in zip(signatures.items(), CATEGORICAL, strict=False):
        ax.semilogx(
            frame["block_steps"],
            frame["sigma"],
            color=color,
            marker="o",
            markersize=5,
            label=label,
        )
        ax.annotate(
            label,
            xy=(frame["block_steps"].iloc[-1], frame["sigma"].iloc[-1]),
            xytext=(6, -3),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    ax.set_xlabel("sampling horizon (steps)")
    ax.set_ylabel(r"estimated $\sigma$ (price units per $\sqrt{\mathrm{time}}$)")
    ax.set_title("A flat signature means a martingale; a rising one means slow price discovery")
    ax.set_xlim(right=float(next(iter(signatures.values()))["block_steps"].max()) * 3)
    ax.legend(loc="upper left")
    return _finish(fig, "Feed the model the volatility at the horizon it holds inventory for")


def plot_markout(
    runs: dict[str, list[SimulationResult]],
    horizons: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100, 250),
) -> Figure:
    """Adverse selection: signed mid move after a fill, averaged across runs and fills.

    Negative and falling means the flow that hits you knows something. The level the curve
    settles at is the per-fill cost the spread has to cover.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for (name, results), color in zip(runs.items(), CATEGORICAL, strict=False):
        curves = [markout_curve(r, horizons) for r in results if not r.fill_prices.empty]
        if not curves:
            continue
        stacked = pd.concat(curves)
        # Weight each run's markout by its fill count, so a quiet run does not count as
        # much as a busy one.
        stacked["weighted"] = stacked["mean_markout"] * stacked["n_fills"]
        grouped = stacked.groupby("horizon")[["weighted", "n_fills"]].sum()
        agg = grouped["weighted"] / grouped["n_fills"]
        ax.plot(
            agg.index.to_numpy(), agg.to_numpy(), color=color, marker="o", markersize=4, label=name
        )
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.9, linestyle=(0, (4, 4)), zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("steps after fill")
    ax.set_ylabel("signed mid move (price units)")
    ax.set_title("Markout: how much of the captured spread the informed flow takes back")
    ax.legend(loc="lower left")
    return _finish(fig, "Negative means the price moved against the maker after it traded")


def plot_book_snapshot(
    depth_bids: list[tuple[int, float]], depth_asks: list[tuple[int, float]], market: MarketConfig
) -> Figure:
    """A single order-book snapshot as a depth chart."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    if depth_bids:
        prices, sizes = zip(*depth_bids, strict=True)
        ax.bar(
            [p * market.tick_size for p in prices],
            sizes,
            width=market.tick_size * 0.8,
            color=CATEGORICAL[0],
            label="bids",
        )
    if depth_asks:
        prices, sizes = zip(*depth_asks, strict=True)
        ax.bar(
            [p * market.tick_size for p in prices],
            sizes,
            width=market.tick_size * 0.8,
            color=CATEGORICAL[1],
            label="asks",
        )
    ax.set_xlabel("price")
    ax.set_ylabel("resting size")
    ax.set_title("Order book depth")
    if depth_bids or depth_asks:
        ax.legend(loc="upper right")
    return _finish(fig)


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
