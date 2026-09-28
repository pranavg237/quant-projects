"""Charts for the portfolio study.

Colour by job, as in :mod:`portopt.style`: strategies are *identities* so they take the
fixed categorical slots in fixed order; ordered quantities (leverage caps, lookbacks) use
the sequential blue ramp; signed quantities use the diverging pair. Never two y-scales.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .backtest import BacktestResult
from .covariance import CovarianceDiagnostics
from .metrics import drawdown_series
from .optimizers import EfficientFrontier, OptimizationResult
from .style import (
    CATEGORICAL,
    DIVERGING,
    GRID,
    INK_MUTED,
    INK_SECONDARY,
    apply_house_style,
    ordinal_colors,
)

__all__ = [
    "plot_correlation_heatmap",
    "plot_covariance_diagnostics",
    "plot_drawdowns",
    "plot_efficient_frontier",
    "plot_equity_curves",
    "plot_leverage_sweep",
    "plot_risk_contributions",
    "plot_risk_return_scatter",
    "plot_shrinkage_intensity",
    "plot_turnover_vs_sharpe",
    "plot_weights_over_time",
    "save_all",
]

_MAX_SERIES = len(CATEGORICAL)


def _finish(fig: Figure, subtitle: str | None = None) -> Figure:
    if subtitle:
        fig.text(0.0, 1.0, subtitle, ha="left", va="bottom", fontsize=9, color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def _series_colors(n: int) -> list[str]:
    """Categorical slots in fixed order; beyond the validated four, fall back to the ramp.

    The palette validates four hues against the all-pairs colour-vision gates. A fifth
    would have to be a generated hue, which the method forbids, so past four the series are
    treated as an ordered set and take the sequential ramp instead -- and the charts that
    need more than four always carry direct labels as well.
    """
    if n <= _MAX_SERIES:
        return list(CATEGORICAL[:n])
    return ordinal_colors(n)


def plot_equity_curves(results: dict[str, BacktestResult], log_scale: bool = True) -> Figure:
    """Cumulative growth of one unit, net of costs, per strategy.

    Log scale by default: on a 17-year run a linear axis makes the last five years look
    like the whole story and compresses the 2008 drawdown into invisibility. On a log axis
    equal vertical distances are equal *percentage* moves, which is what a portfolio
    manager actually cares about.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    colors = _series_colors(len(results))
    for (name, result), color in zip(results.items(), colors, strict=True):
        curve = result.equity_curve
        if result.is_ruined:
            continue  # a ruined curve goes non-positive and cannot be drawn on a log axis
        ax.plot(curve.index, curve.to_numpy(), color=color, label=name, linewidth=1.6)
        ax.annotate(
            name,
            xy=(curve.index[-1], curve.iloc[-1]),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color=color,
            va="center",
        )
    if log_scale:
        ax.set_yscale("log")
    ax.axhline(1.0, color=INK_MUTED, linewidth=0.8, linestyle=(0, (4, 4)), zorder=0)
    ax.set_xlabel("date")
    ax.set_ylabel("growth of 1 unit (log scale)" if log_scale else "growth of 1 unit")
    ax.set_title("Out-of-sample equity curves, net of 10bp one-way costs")
    ax.legend(loc="upper left", ncol=2)
    ruined = [n for n, r in results.items() if r.is_ruined]
    note = (
        f"Not shown: {', '.join(ruined)} (ruined -- lost 100% in a single period)"
        if ruined
        else None
    )
    return _finish(fig, note)


def plot_drawdowns(results: dict[str, BacktestResult]) -> Figure:
    """Drawdown from the running peak, per strategy."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    colors = _series_colors(len(results))
    for (name, result), color in zip(results.items(), colors, strict=True):
        if result.is_ruined:
            continue
        series = drawdown_series(result.returns)
        ax.plot(series.index, 100.0 * series.to_numpy(), color=color, label=name, linewidth=1.4)
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=0)
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown (%)")
    ax.set_title("Drawdowns: every strategy shares the same 2008")
    ax.legend(loc="lower left", ncol=2)
    return _finish(fig)


def plot_risk_return_scatter(summary: pd.DataFrame) -> Figure:
    """Annualised return against volatility, with Sharpe isoclines.

    A scatter rather than a bar chart of Sharpe: the ratio hides whether a strategy got
    there by earning more or by risking less, and those are very different products.
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    usable = summary.loc[~summary["is_ruined"]] if "is_ruined" in summary else summary
    x_max = float(usable["annual_volatility"].max()) * 1.35
    y_max = float(usable["annual_return"].max()) * 1.35

    grid = np.linspace(1e-9, x_max, 100)
    for sharpe in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5):
        line = sharpe * grid
        ax.plot(grid, line, color=GRID, linewidth=0.9, zorder=0)
        inside = np.where(line < y_max)[0]
        if inside.size:
            idx = int(inside[-1])
            ax.annotate(
                f"Sharpe {sharpe:g}",
                xy=(grid[idx], line[idx]),
                xytext=(-52, 2),
                textcoords="offset points",
                fontsize=8,
                color=INK_MUTED,
            )

    colors = _series_colors(len(usable))
    for (_, row), color in zip(usable.iterrows(), colors, strict=True):
        ax.scatter(
            row["annual_volatility"],
            row["annual_return"],
            s=110,
            color=color,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            str(row["strategy"]),
            xy=(row["annual_volatility"], row["annual_return"]),
            xytext=(9, -4),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("annualised volatility")
    ax.set_ylabel("annualised return")
    ax.set_title("Out-of-sample risk and return")
    return _finish(fig, "Net of costs; ruined strategies omitted")


def plot_efficient_frontier(
    frontier: EfficientFrontier,
    highlights: dict[str, OptimizationResult] | None = None,
    assets: pd.DataFrame | None = None,
) -> Figure:
    """The in-sample efficient frontier, with individual assets and named portfolios."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.plot(
        100.0 * frontier.volatilities,
        100.0 * frontier.returns,
        color=CATEGORICAL[0],
        linewidth=2.0,
        label="efficient frontier",
    )
    if assets is not None:
        ax.scatter(
            100.0 * assets["volatility"],
            100.0 * assets["return"],
            s=28,
            color=INK_MUTED,
            zorder=2,
            label="individual assets",
        )
        for name, row in assets.iterrows():
            ax.annotate(
                str(name),
                xy=(100.0 * row["volatility"], 100.0 * row["return"]),
                xytext=(4, -3),
                textcoords="offset points",
                fontsize=7.5,
                color=INK_MUTED,
            )
    if highlights:
        for (label, result), color in zip(highlights.items(), CATEGORICAL[1:], strict=False):
            ax.scatter(
                100.0 * result.volatility,
                100.0 * result.expected_return,
                s=120,
                color=color,
                zorder=4,
                edgecolors="white",
                linewidths=1.5,
                label=label,
            )
    ax.set_xlabel("annualised volatility (%)")
    ax.set_ylabel("annualised expected return (%)")
    ax.set_title("The efficient frontier is an in-sample object")
    ax.legend(loc="lower right")
    return _finish(fig, "Every point here is fitted to the same data it is evaluated on")


def plot_leverage_sweep(sweep: pd.DataFrame) -> Figure:
    """Out-of-sample Sharpe and turnover against the gross-leverage cap.

    Two panels, not two y-axes. Sharpe and turnover differ by three orders of magnitude and
    overlaying them would invent a crossing point that means nothing.
    """
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    finite = sweep.loc[np.isfinite(sweep["leverage_cap"])]
    uncapped = sweep.loc[~np.isfinite(sweep["leverage_cap"])]

    axes[0].plot(
        finite["leverage_cap"], finite["sharpe"], color=CATEGORICAL[0], marker="o", markersize=5
    )
    if not uncapped.empty:
        axes[0].axhline(
            float(uncapped["sharpe"].iloc[0]),
            color=CATEGORICAL[1],
            linewidth=1.2,
            linestyle=(0, (4, 4)),
        )
        axes[0].annotate(
            "uncapped",
            xy=(finite["leverage_cap"].min(), float(uncapped["sharpe"].iloc[0])),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
            color=CATEGORICAL[1],
        )
    best = finite.loc[finite["sharpe"].idxmax()]
    axes[0].scatter(
        [best["leverage_cap"]],
        [best["sharpe"]],
        s=110,
        color=CATEGORICAL[0],
        zorder=4,
        edgecolors="white",
        linewidths=1.5,
    )
    axes[0].annotate(
        f"best at {best['leverage_cap']:.1f}x",
        xy=(best["leverage_cap"], best["sharpe"]),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=9,
        color=CATEGORICAL[0],
    )
    axes[0].set_xlabel("gross leverage cap")
    axes[0].set_ylabel("out-of-sample Sharpe")
    axes[0].set_title("Sharpe peaks at a modest cap")

    axes[1].semilogy(
        finite["leverage_cap"],
        100.0 * finite["annual_turnover"],
        color=CATEGORICAL[2],
        marker="o",
        markersize=5,
    )
    if not uncapped.empty:
        axes[1].axhline(
            100.0 * float(uncapped["annual_turnover"].iloc[0]),
            color=CATEGORICAL[1],
            linewidth=1.2,
            linestyle=(0, (4, 4)),
        )
        axes[1].annotate(
            "uncapped",
            xy=(finite["leverage_cap"].min(), 100.0 * float(uncapped["annual_turnover"].iloc[0])),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
            color=CATEGORICAL[1],
        )
    axes[1].set_xlabel("gross leverage cap")
    axes[1].set_ylabel("annual turnover (%, log scale)")
    axes[1].set_title("Turnover explodes without one")
    return _finish(fig, "Markowitz max-Sharpe on sample moments, 60-month window, 10bp costs")


def plot_shrinkage_intensity(frame: pd.DataFrame) -> Figure:
    """Ledoit-Wolf shrinkage intensity and covariance condition number against sample size."""
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(
        frame["n_observations"],
        frame["delta_identity"],
        color=CATEGORICAL[0],
        marker="o",
        markersize=4,
        label="identity target",
    )
    axes[0].plot(
        frame["n_observations"],
        frame["delta_constant_correlation"],
        color=CATEGORICAL[1],
        marker="s",
        markersize=4,
        label="constant correlation",
    )
    axes[0].set_xlabel("estimation window (months)")
    axes[0].set_ylabel(r"shrinkage intensity $\delta^*$")
    axes[0].set_title("Shrink harder when you have less data")
    axes[0].legend(loc="upper right")

    axes[1].semilogy(
        frame["n_observations"],
        frame["condition_sample"],
        color=CATEGORICAL[2],
        marker="o",
        markersize=4,
        label="sample",
    )
    axes[1].semilogy(
        frame["n_observations"],
        frame["condition_shrunk"],
        color=CATEGORICAL[0],
        marker="s",
        markersize=4,
        label="shrunk",
    )
    axes[1].set_xlabel("estimation window (months)")
    axes[1].set_ylabel("condition number (log scale)")
    axes[1].set_title("...which is what keeps the matrix invertible")
    axes[1].legend(loc="upper left")
    return _finish(fig)


def plot_correlation_heatmap(correlation: pd.DataFrame, order: list[str] | None = None) -> Figure:
    """Asset correlation matrix, optionally in the HRP quasi-diagonal order.

    Diverging palette because correlation is *signed* around zero, and symmetric limits so
    the neutral grey lands exactly on zero.
    """
    apply_house_style()
    frame = correlation.loc[order, order] if order else correlation
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    mesh = ax.pcolormesh(
        frame.to_numpy(), cmap=DIVERGING, vmin=-1.0, vmax=1.0, edgecolors="white", linewidth=0.5
    )
    ax.set_xticks(np.arange(len(frame)) + 0.5)
    ax.set_yticks(np.arange(len(frame)) + 0.5)
    ax.set_xticklabels(frame.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(frame.index, fontsize=8)
    ax.invert_yaxis()
    ax.grid(visible=False)
    ax.set_title("Correlations, clustered" if order else "Correlations, listed order")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("correlation", color=INK_SECONDARY, fontsize=9)
    return _finish(fig)


def plot_risk_contributions(contributions: dict[str, pd.Series]) -> Figure:
    """Share of portfolio risk per asset, one panel per strategy."""
    apply_house_style()
    n = len(contributions)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.6), sharey=True, squeeze=False)
    for ax, (name, series), color in zip(axes[0], contributions.items(), CATEGORICAL, strict=False):
        share = series / series.sum()
        ax.barh(np.arange(len(share)), 100.0 * share.to_numpy(), color=color, height=0.72)
        ax.axvline(100.0 / len(share), color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
        ax.set_yticks(np.arange(len(share)))
        ax.set_yticklabels(share.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("share of portfolio risk (%)")
        ax.set_title(name)
    axes[0][0].annotate(
        "equal share",
        xy=(100.0 / len(next(iter(contributions.values()))), 0),
        xytext=(4, -10),
        textcoords="offset points",
        fontsize=8,
        color=INK_MUTED,
    )
    fig.suptitle("Risk contributions: naive risk parity is not risk parity", x=0.01, ha="left")
    return _finish(fig)


def plot_weights_over_time(result: BacktestResult, max_assets: int = 15) -> Figure:
    """Stacked weights through the backtest, showing how much the portfolio actually moves."""
    apply_house_style()
    weights = result.weights.iloc[:, :max_assets]
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    colors = ordinal_colors(weights.shape[1])
    ax.stackplot(
        weights.index,
        *[weights[c].to_numpy() for c in weights.columns],
        labels=list(weights.columns),
        colors=colors,
        edgecolor="white",
        linewidth=0.2,
    )
    ax.set_xlabel("date")
    ax.set_ylabel("weight")
    ax.set_ylim(0, float(weights.sum(axis=1).max()) * 1.02)
    ax.set_title(f"{result.name}: weights through time")
    ax.legend(loc="upper center", ncol=8, fontsize=7.5)
    return _finish(fig)


def plot_turnover_vs_sharpe(summary: pd.DataFrame) -> Figure:
    """Out-of-sample Sharpe against annual turnover, with cost sensitivity.

    The chart that says whether a strategy's edge survives implementation. Each point also
    carries an arrow showing where it moves if costs are 5x higher.
    """
    apply_house_style()
    usable = summary.loc[~summary["is_ruined"]] if "is_ruined" in summary else summary
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    colors = _series_colors(len(usable))
    for (_, row), color in zip(usable.iterrows(), colors, strict=True):
        ax.scatter(
            100.0 * row["annual_turnover"],
            row["sharpe"],
            s=110,
            color=color,
            zorder=3,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            str(row["strategy"]),
            xy=(100.0 * row["annual_turnover"], row["sharpe"]),
            xytext=(9, -4),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_xlabel("annual one-way turnover (%, log scale)")
    ax.set_ylabel("out-of-sample Sharpe, net of costs")
    ax.set_title("Turnover is what you pay for; Sharpe is what you get")
    return _finish(fig)


def plot_covariance_diagnostics(diagnostics: dict[str, CovarianceDiagnostics]) -> Figure:
    """Condition number and effective rank per estimator."""
    apply_house_style()
    names = list(diagnostics)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    positions = np.arange(len(names))
    axes[0].barh(
        positions,
        [diagnostics[n].condition_number for n in names],
        color=CATEGORICAL[0],
        height=0.66,
    )
    axes[0].set_xscale("log")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("condition number (log scale)")
    axes[0].set_title("Conditioning")

    axes[1].barh(
        positions, [diagnostics[n].effective_rank for n in names], color=CATEGORICAL[2], height=0.66
    )
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("effective rank")
    axes[1].set_title("How many directions it really distinguishes")
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
