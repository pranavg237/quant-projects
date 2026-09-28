"""Charts and a Markdown report whose tables are also saved as CSV files."""
from __future__ import annotations

import functools
import math
import re
import textwrap
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import font_manager
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, PercentFormatter

from .attribution import Attribution
from .models import get_model
from .regression import RegressionResult

# Reference data-viz palette, light surface.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
# Each factor keeps its color in every chart.
ENTITY_COLORS = {
    "Mkt-RF": SERIES[0],
    "SMB": SERIES[1],
    "HML": SERIES[2],
    "RMW": SERIES[3],
    "CMA": SERIES[4],
    "MOM": SERIES[5],
    "alpha": SERIES[6],
}
# Diverging: red (negative) <- neutral gray -> blue (positive).
DIVERGING = LinearSegmentedColormap.from_list("ffmodel_diverging", [SERIES[7], "#f0efec", SERIES[0]])

_available_fonts = {f.name for f in font_manager.fontManager.ttflist}
_SANS = [f for f in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans") if f in _available_fonts] or ["DejaVu Sans"]

_RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 160,
    "font.family": "sans-serif",
    "font.sans-serif": _SANS,
    "font.size": 9.5,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "semibold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "xtick.color": BASELINE,
    "ytick.color": BASELINE,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}

PCT = PercentFormatter(1.0, decimals=None)


def _styled(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with matplotlib.rc_context(_RC):
            return func(*args, **kwargs)

    return wrapper


def _figure(width: float, height: float, title: str, subtitle: str = "", nrows: int = 1, ncols: int = 1, **subplot_kw):
    """A figure with a left-aligned title and subtitle above a plot area ``height`` inches tall."""
    lines = textwrap.wrap(subtitle, width=int(width * 15)) if subtitle else []
    header = 0.42 + 0.17 * len(lines) + (0.1 if lines else 0)
    total = height + header
    fig = Figure(figsize=(width, total), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0, 1, 1 - header / total))
    axes = fig.subplots(nrows, ncols, squeeze=False, **subplot_kw)
    fig.text(0.012, 1 - 0.14 / total, title, ha="left", va="top", fontsize=12.5, fontweight="semibold", color=INK)
    if lines:
        fig.text(0.012, 1 - 0.42 / total, "\n".join(lines), ha="left", va="top", fontsize=9, color=INK_2, linespacing=1.35)
    return fig, axes


def _date_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _end_label(ax, x, y, text: str) -> None:
    ax.annotate(text, (x, y), xytext=(5, 0), textcoords="offset points", va="center", ha="left",
                fontsize=8.5, color=INK_2, annotation_clip=False)


def _category_axis(ax, positions, labels) -> None:
    ax.set_yticks(positions, labels=labels)
    ax.tick_params(axis="y", labelcolor=INK_2, labelsize=9.5, length=0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)


def _hbar_with_labels(ax, labels, values, lows=None, highs=None, fmt=lambda v: f"{v:.2f}") -> None:
    """One-series horizontal bars from a zero baseline, value at the tip (past any interval)."""
    values = np.asarray(values, float)
    lows = values if lows is None else np.asarray(lows, float)
    highs = values if highs is None else np.asarray(highs, float)
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, height=0.34, color=SERIES[0], zorder=2)
    if lows is not values:
        ax.hlines(y, lows, highs, color=INK_2, linewidth=1.0, zorder=3)
    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)
    left, right = min(np.nanmin(lows), 0.0), max(np.nanmax(highs), 0.0)
    span = (right - left) or 1.0
    for yi, v, lo, hi in zip(y, values, lows, highs):
        x, ha = (hi + 0.02 * span, "left") if v >= 0 else (lo - 0.02 * span, "right")
        ax.text(x, yi, fmt(v), va="center", ha=ha, fontsize=9, color=INK)
    ax.set_xlim(left - 0.16 * span, right + 0.16 * span)
    _category_axis(ax, y, labels)


@_styled
def plot_loadings(result: RegressionResult) -> Figure:
    spec = get_model(result.model)
    ci = result.raw.conf_int().loc[result.factors]
    start, end = result.period
    fig, axes = _figure(
        7.2, 0.45 * len(result.factors) + 0.8,
        f"{result.name}: {spec.label} loadings",
        f"{start:%b %Y} to {end:%b %Y}, {result.nobs} observations. Lines are 95% confidence intervals "
        f"({result.cov_type.upper()} errors). Alpha {result.alpha_annual:+.2%} a year "
        f"(t = {result.tvalues['alpha']:.2f}), R² {result.rsquared:.2f}.",
    )
    ax = axes[0, 0]
    _hbar_with_labels(ax, result.factors, result.betas, ci[0], ci[1])
    ax.set_xlabel("Beta")
    return fig


@_styled
def plot_attribution(attribution: Attribution, name: str) -> Figure:
    summary = attribution.summary
    parts = summary.drop(index=["total", "residual"])["return (ann.)"]
    total = summary.loc["total", "return (ann.)"]
    explained = 1 - summary.loc["residual", "share of variance"]
    labels = ["Alpha" if k == "alpha" else k for k in parts.index]
    fig, axes = _figure(
        7.2, 0.45 * len(parts) + 0.8,
        f"{name}: where the excess return came from",
        f"Average excess return {total:+.2%} a year, split into alpha and beta × factor return. "
        f"The factors explain {explained:.0%} of the return variance.",
    )
    ax = axes[0, 0]
    _hbar_with_labels(ax, labels, parts, fmt=lambda v: f"{v:+.2%}")
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel("Contribution to annualized excess return")
    return fig


@_styled
def plot_cumulative_fit(attribution: Attribution, name: str) -> Figure:
    c = attribution.contributions
    actual = c.sum(axis=1).cumsum()
    explained = c.drop(columns=["alpha", "residual"]).sum(axis=1).cumsum()
    fig, axes = _figure(
        7.2, 3.4,
        f"{name}: actual vs. factor-explained excess return",
        "Running sum of per-period excess returns. The gap between the lines is alpha plus residual.",
    )
    ax = axes[0, 0]
    ax.plot(actual.index, actual.values, color=SERIES[0], label="Actual")
    ax.plot(explained.index, explained.values, color=SERIES[1], label="Factor-explained (β′f)")
    ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(loc="upper left")
    _date_axis(ax)
    low, high = ax.get_ylim()
    if abs(actual.iloc[-1] - explained.iloc[-1]) > 0.06 * (high - low):
        _end_label(ax, actual.index[-1], actual.iloc[-1], f"{actual.iloc[-1]:.0%}")
        _end_label(ax, explained.index[-1], explained.iloc[-1], f"{explained.iloc[-1]:.0%}")
    return fig


def _panel_grid(n: int) -> Tuple[int, int]:
    ncols = 1 if n == 1 else 2 if n <= 4 else 3
    return math.ceil(n / ncols), ncols


@_styled
def plot_rolling(rolled: pd.DataFrame, factors: Sequence[str], name: str, model: str, window: int,
                 periods_per_year: int) -> Figure:
    panels = ["alpha"] + list(factors)
    nrows, ncols = _panel_grid(len(panels))
    fig, axes = _figure(
        3.3 * ncols + 0.4, 2.0 * nrows,
        f"{name}: rolling {get_model(model).label} estimates",
        f"Re-estimated over a trailing {window}-period window. Alpha is annualized.",
        nrows, ncols, sharex=True,
    )
    for ax, column in zip(axes.flat, panels):
        series = rolled[column] * (periods_per_year if column == "alpha" else 1)
        ax.plot(series.index, series.values, color=ENTITY_COLORS.get(column, SERIES[0]))
        ax.axhline(0, color=BASELINE, linewidth=1.0, zorder=1)
        ax.set_title("Alpha (annualized)" if column == "alpha" else f"{column} beta")
        last = f"{series.iloc[-1]:.1%}" if column == "alpha" else f"{series.iloc[-1]:.2f}"
        _end_label(ax, series.index[-1], series.iloc[-1], last)
        if column == "alpha":
            ax.yaxis.set_major_formatter(PCT)
        _date_axis(ax)
    for k in range(len(panels), nrows * ncols):
        axes.flat[k].set_visible(False)
        axes.flat[k - ncols].tick_params(labelbottom=True)
    return fig


@_styled
def plot_pricing(realized: pd.Series, predicted: Mapping[str, pd.Series], pvalues: Mapping[str, float],
                 title: str) -> Figure:
    nrows, ncols = _panel_grid(len(predicted))
    fig, axes = _figure(
        3.9 * ncols + 0.4, 3.5 * nrows, title,
        "Each dot is one portfolio: the model's implied average excess return against the realized one, "
        "both annualized. Dots on the diagonal are priced exactly.",
        nrows, ncols, sharex=True, sharey=True,
    )
    values = np.concatenate([realized.to_numpy()] + [p.to_numpy() for p in predicted.values()])
    pad = 0.06 * (values.max() - values.min())
    lo, hi = values.min() - pad, values.max() + pad
    for ax, (label, pred) in zip(axes.flat, predicted.items()):
        ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.0, zorder=1)
        ax.scatter(pred.to_numpy(), realized.loc[pred.index].to_numpy(), s=46, color=SERIES[0],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.set_title(label)
        ax.set_title(f"GRS p = {pvalues[label]:.2g}", loc="right", fontsize=8.5, fontweight="normal", color=INK_2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_formatter(PCT)
        ax.yaxis.set_major_formatter(PCT)
    for k in range(len(predicted), nrows * ncols):
        axes.flat[k].set_visible(False)
        axes.flat[k - ncols].tick_params(labelbottom=True)
    fig.supxlabel("Model-implied average excess return", fontsize=9, color=INK_2)
    fig.supylabel("Realized average excess return", fontsize=9, color=INK_2)
    return fig


def _text_color_on(fill) -> str:
    rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(fill)]
    luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return INK if luminance > 0.179 else "#ffffff"


@_styled
def plot_alpha_heatmaps(alphas: Mapping[str, pd.Series], shape: Tuple[int, int], row_name: str, col_name: str,
                        periods_per_year: int, title: str) -> Figure:
    """Grid of annualized alphas laid out like the portfolio sort, one panel per model, one shared scale."""
    rows, cols = shape
    nrows, ncols = _panel_grid(len(alphas))
    fig, axes = _figure(
        3.3 * ncols + 1.0, 3.0 * nrows, title,
        f"Annualized time-series alpha of each portfolio. Rows sort on {row_name}, columns on {col_name}. "
        "Blue is positive, red negative; gray is zero.",
        nrows, ncols,
    )
    limit = max(float(np.nanmax(np.abs(a.to_numpy()))) for a in alphas.values()) * periods_per_year
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
    row_ticks = ["Small"] + [str(i) for i in range(2, rows)] + ["Big"] if row_name.startswith("size") else \
        ["Low"] + [str(i) for i in range(2, rows)] + ["High"]
    col_ticks = ["Low"] + [str(i) for i in range(2, cols)] + ["High"]
    gap = 0.03
    for ax, (label, alpha) in zip(axes.flat, alphas.items()):
        grid = alpha.to_numpy().reshape(rows, cols) * periods_per_year
        for i in range(rows):
            for j in range(cols):
                fill = DIVERGING(norm(grid[i, j]))
                ax.add_patch(Rectangle((j - 0.5 + gap, i - 0.5 + gap), 1 - 2 * gap, 1 - 2 * gap, color=fill, linewidth=0))
                ax.text(j, i, f"{grid[i, j]:+.1%}", ha="center", va="center", fontsize=8, color=_text_color_on(fill))
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xticks(range(cols), labels=col_ticks)
        ax.set_yticks(range(rows), labels=row_ticks)
        ax.tick_params(length=0, labelcolor=INK_2)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for k in range(len(alphas), nrows * ncols):
        axes.flat[k].set_visible(False)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=DIVERGING), ax=axes.ravel().tolist(), shrink=0.8,
                       format=PercentFormatter(1.0, decimals=0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, labelcolor=INK_2)
    bar.set_label("Alpha, annualized", color=INK_2, fontsize=9)
    return fig


@_styled
def plot_factor_growth(factors: pd.DataFrame, title: str) -> Figure:
    F = factors.drop(columns="RF", errors="ignore")
    growth = (1 + F).cumprod()
    fig, axes = _figure(
        7.6, 3.8, title,
        "Growth of $1 in each factor (log scale). Mkt-RF is the market's return over T-bills; "
        "the others are long-short spreads, compounded.",
    )
    ax = axes[0, 0]
    for column in growth.columns:
        ax.plot(growth.index, growth[column].values, color=ENTITY_COLORS.get(column, SERIES[0]), label=column)
    ax.set_yscale("log")
    ax.axhline(1.0, color=BASELINE, linewidth=1.0, zorder=1)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.4g}"))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=min(len(growth.columns), 6), borderaxespad=0)
    _date_axis(ax)
    return fig


# ---------------------------------------------------------------------------
# Tables and the Markdown report

_PCT_KEYS = ("alpha (ann.)", "alpha| (ann.)", "mean (ann.)", "vol (ann.)", "lambda (ann.)", "return", "period", "share")


def format_value(column, value, decimals: Optional[int] = None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, str):
        return value
    column = str(column)
    if column in ("N", "T"):
        return f"{int(value):,}"
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if any(key in column for key in _PCT_KEYS):
        return f"{value:.2%}"
    if column == "p-value":
        return f"{value:.3f}" if value >= 0.001 else f"{value:.1e}"
    if column.endswith("R2"):
        return f"{value:.3f}"
    return f"{value:.2f}"


def format_table(df: pd.DataFrame, decimals: Optional[int] = None) -> pd.DataFrame:
    """Strings for display: percentages for returns and alphas, 2 decimals for betas and t-stats."""
    index = [i.strftime("%Y-%m-%d") if isinstance(i, pd.Timestamp) else i for i in df.index]
    return pd.DataFrame(
        {c: [format_value(c, v, decimals) for v in df[c]] for c in df.columns}, index=index
    )


def markdown_table(df: pd.DataFrame, decimals: Optional[int] = None) -> str:
    shown = format_table(df, decimals)
    esc = lambda s: str(s).replace("|", "\\|")  # noqa: E731
    lines = [
        "| " + " | ".join([""] + [esc(c) for c in shown.columns]) + " |",
        "|" + "|".join([":---"] + ["---:"] * shown.shape[1]) + "|",
    ]
    lines += ["| " + " | ".join([esc(i)] + [esc(v) for v in row]) + " |" for i, row in zip(shown.index, shown.to_numpy())]
    return "\n".join(lines)


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", str(name)).strip("-").lower()


class Report:
    """Collects Markdown text, PNG figures and CSV tables in one directory."""

    def __init__(self, directory, title: str):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lines = [f"# {title}", ""]

    def heading(self, text: str, level: int = 2) -> None:
        self.lines += [f"{'#' * level} {text}", ""]

    def text(self, text: str) -> None:
        self.lines += [text, ""]

    def csv(self, df: pd.DataFrame, name: str) -> str:
        filename = f"{_slug(name)}.csv"
        df.to_csv(self.dir / filename)
        return filename

    def table(self, df: pd.DataFrame, name: str, decimals: Optional[int] = None) -> None:
        filename = self.csv(df, name)
        self.lines += [markdown_table(df, decimals), "", f"<sub>Data: [{filename}]({filename})</sub>", ""]

    def figure(self, fig: Figure, name: str, alt: str, data: Optional[pd.DataFrame] = None) -> None:
        filename = f"{_slug(name)}.png"
        fig.savefig(self.dir / filename)
        self.lines += [f"![{alt}]({filename})", ""]
        if data is not None:
            csv = self.csv(data, name)
            self.lines += [f"<sub>Chart data: [{csv}]({csv})</sub>", ""]

    def save(self) -> Path:
        path = self.dir / "report.md"
        path.write_text("\n".join(self.lines))
        return path
