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
from matplotlib.ticker import FuncFormatter, NullFormatter

from .avellaneda_stoikov import AvellanedaStoikovParams, inventory_skew, optimal_spread
from .calibration import FillIntensityFit
from .engine import SimulationResult
from .experiments import ComparisonResult
from .metrics import informed_markout_theory
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
    return _finish(
        fig,
        "Grey isoclines: Sharpe per simulated session (mean / std across sessions), "
        "not annualised.",
    )


def _looks_geometric(x: np.ndarray) -> bool:
    """Whether a value grid is geometrically rather than linearly spaced.

    A geometric sweep plotted on a linear axis crowds every point except the last into the
    left edge, which hides exactly the region a risk-aversion or intensity sweep is about.
    """
    if x.size < 3 or np.any(x <= 0):
        return False
    ratios = x[1:] / x[:-1]
    return bool(np.std(ratios) / np.mean(ratios) < 0.05 and np.mean(ratios) > 1.2)


#: Rows of the sensitivity grid: (column, its standard-error column, axis label).
_SENSITIVITY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("mean_pnl", "mean_pnl_se", "mean PnL per session"),
    ("std_pnl", "std_pnl_se", "std of session PnL"),
    ("sharpe", "sharpe_se", "Sharpe per session\n(not annualised)"),
    ("std_final_inventory", "std_final_inventory_se", "std of final inventory"),
)


def plot_sensitivity(sweep: pd.DataFrame, labels: dict[str, str] | None = None) -> Figure:
    """Four statistics against each swept parameter, per policy, with 95% bands.

    Small multiples: one column per swept parameter, one row per statistic, one line per
    policy. The shaded band is +-1.96 standard errors across sessions (bootstrap for the
    std, Sharpe and inventory rows). Colours follow the policy, in the fixed slot order of
    first appearance, so every panel paints the same policy the same way.

    Args:
        sweep: Long-format output of :func:`~mmsim.experiments.policy_sensitivity_sweep`,
            possibly several sweeps concatenated (distinguished by ``parameter``).
        labels: Optional x-axis label per parameter name.
    """
    apply_house_style()
    labels = labels or {}
    parameters = list(dict.fromkeys(sweep["parameter"]))
    policies = list(dict.fromkeys(sweep["policy"]))
    colors = dict(zip(policies, CATEGORICAL, strict=False))
    n_rows, n_cols = len(_SENSITIVITY_ROWS), len(parameters)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.6 * n_cols, 2.7 * n_rows), squeeze=False, sharex="col"
    )

    for col, parameter in enumerate(parameters):
        block = sweep[sweep["parameter"] == parameter]
        grid = np.sort(block["value"].unique().astype(np.float64))
        geometric = _looks_geometric(grid)
        for row, (stat, se_col, ylabel) in enumerate(_SENSITIVITY_ROWS):
            ax = axes[row][col]
            if geometric:
                ax.set_xscale("log")
                # Label every other swept value in plain numbers, not 2x10^0 notation.
                ax.set_xticks(grid[::2])
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 10 else f"{v:.2g}")
                )
                ax.xaxis.set_minor_formatter(NullFormatter())
            for policy in policies:
                sub = block[block["policy"] == policy].sort_values("value")
                if sub.empty:
                    continue
                x = sub["value"].to_numpy(dtype=np.float64)
                y = sub[stat].to_numpy(dtype=np.float64)
                se = sub[se_col].to_numpy(dtype=np.float64)
                ax.fill_between(
                    x, y - 1.96 * se, y + 1.96 * se, color=colors[policy], alpha=0.18, lw=0
                )
                ax.plot(x, y, color=colors[policy], marker="o", markersize=3.5, label=policy)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(labels.get(parameter, parameter))
            if row == n_rows - 1:
                ax.set_xlabel(labels.get(parameter, parameter))
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.suptitle("Parameter sensitivity, three policies", x=0.01, y=0.995, ha="left")
    fig.text(
        0.01,
        0.962,
        "Bands: +-1.96 standard errors across sessions. Sharpe is per simulated session, "
        "not annualised, and not comparable to a trading strategy's Sharpe.",
        ha="left",
        fontsize=9,
        color=INK_SECONDARY,
    )
    fig.legend(
        handles, names, loc="upper right", bbox_to_anchor=(0.995, 0.995), ncol=3, frameon=False
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return fig


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
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="lower right")
    return _finish(fig, "Feed the model the volatility at the horizon it holds inventory for")


def plot_markout(
    decomposition: pd.DataFrame,
    impact_ticks: float,
    impact_speed: float,
    bar_horizon: int = 100,
) -> Figure:
    """Adverse selection, split by who the maker traded with.

    **Left:** per-unit markout (signed mid move after the fill, in ticks) against horizon,
    for each policy, separately for fills against informed (solid) and uninformed
    (dashed) takers, with 95% bands clustered by session. The grey dotted line is the
    model's own prediction for an informed fill, :math:`-J(1-(1-v)^h)`.

    **Right:** the same thing as PnL per session at one horizon: the edge earned at the
    fill from each counterparty type, the adverse-selection cost each one then imposed,
    and what is left -- the realised spread.

    Args:
        decomposition: Output of :func:`~mmsim.metrics.markout_decomposition`.
        impact_ticks: Informed impact :math:`J`, for the theory line.
        impact_speed: Informed impact release speed :math:`v`, for the theory line.
        bar_horizon: Horizon, in steps, for the right-hand panel. Must be in the frame.
    """
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), gridspec_kw={"width_ratios": [1.1, 1]})
    policies = list(dict.fromkeys(decomposition["policy"]))
    colors = dict(zip(policies, CATEGORICAL, strict=False))

    ax = axes[0]
    for policy in policies:
        for counterparty, style_ in (("informed", "-"), ("uninformed", (0, (4, 3)))):
            sub = decomposition[
                (decomposition["policy"] == policy)
                & (decomposition["counterparty"] == counterparty)
            ].sort_values("horizon")
            h = sub["horizon"].to_numpy(dtype=np.float64)
            y = sub["markout_ticks"].to_numpy(dtype=np.float64)
            se = sub["markout_ticks_se"].to_numpy(dtype=np.float64)
            ax.fill_between(h, y - 1.96 * se, y + 1.96 * se, color=colors[policy], alpha=0.15)
            ax.plot(
                h,
                y,
                color=colors[policy],
                linestyle=style_,
                marker="o",
                markersize=3.5,
                label=f"{policy}, {counterparty}",
            )
    dense = np.geomspace(1.0, float(decomposition["horizon"].max()), 200)
    ax.plot(
        dense,
        informed_markout_theory(dense, impact_ticks, impact_speed),
        color=INK_MUTED,
        linestyle=(0, (1, 2)),
        linewidth=1.6,
        label=r"theory, informed: $-J(1-(1-v)^h)$",
    )
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.9, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("steps after fill")
    ax.set_ylabel("markout per unit filled (ticks)")
    ax.set_title("Markout by counterparty")
    ax.legend(loc="lower left", fontsize=7.5)

    ax = axes[1]
    at = decomposition[decomposition["horizon"] == float(bar_horizon)]
    components = (
        ("edge_pnl_per_session", "uninformed", "edge earned\nfrom uninformed"),
        ("edge_pnl_per_session", "informed", "edge earned\nfrom informed"),
        ("adverse_selection_pnl_per_session", "uninformed", "adverse sel.\nuninformed"),
        ("adverse_selection_pnl_per_session", "informed", "adverse sel.\ninformed"),
        ("realised_pnl_per_session", "all", "realised\nspread, total"),
    )
    width = 0.8 / max(len(policies), 1)
    x = np.arange(len(components), dtype=np.float64)
    for k, policy in enumerate(policies):
        values = []
        for column, counterparty, _ in components:
            cell = at[(at["policy"] == policy) & (at["counterparty"] == counterparty)][column]
            value = float(cell.iloc[0]) if not cell.empty else float("nan")
            # Costs are drawn below zero so the bars read as a ledger.
            values.append(-value if column.startswith("adverse") else value)
        ax.bar(
            x + (k - (len(policies) - 1) / 2) * width,
            values,
            width=width * 0.92,
            color=colors[policy],
            label=policy,
        )
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.9)
    ax.set_xticks(x, [c[2] for c in components], fontsize=8)
    ax.set_ylabel("PnL per session (price units)")
    ax.set_title(f"Where the spread goes, marked {bar_horizon} steps after each fill")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=False)
    return _finish(
        fig,
        "Negative markout means the mid moved against the maker after it traded. "
        "Bands: +-1.96 session-clustered standard errors.",
    )


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
