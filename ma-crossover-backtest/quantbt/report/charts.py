"""Inline-SVG chart builders for the HTML tearsheet.

Every chart is a self-contained ``<svg>`` string that scales to its container and
renders identically in light and dark mode, because all colours are CSS custom
properties rather than literals. Charts carry a ``data-points`` attribute that the
page's hover script reads to draw a crosshair and tooltip; nothing here depends on an
external charting library, so a tearsheet is one file you can email.

Mark specs follow the house style: 2px lines, hairline solid gridlines one step off the
surface, >= 8px end markers with a 2px surface ring, a 2px surface gap between adjacent
fills, and labels in text tokens rather than the series colour.
"""

from __future__ import annotations

import html
import itertools
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Viewport used for every chart; the SVG scales to the container via viewBox.
WIDTH = 960
HEIGHT = 300
MARGIN = {"top": 16, "right": 76, "bottom": 28, "left": 56}


@dataclass(frozen=True)
class Series:
    """One line on a chart."""

    name: str
    values: pd.Series
    color_var: str = "series_1"
    fill: bool = False


def _plot_box() -> tuple[float, float, float, float]:
    x0 = MARGIN["left"]
    y0 = MARGIN["top"]
    x1 = WIDTH - MARGIN["right"]
    y1 = HEIGHT - MARGIN["bottom"]
    return x0, y0, x1, y1


def _nice_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Round tick values spanning ``[lo, hi]`` (1, 2, 2.5 or 5 times a power of ten)."""
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if raw <= step:
            break
    start = math.floor(lo / step) * step
    ticks = []
    value = start
    while value <= hi + step * 0.5:
        if value >= lo - step * 0.5:
            ticks.append(round(value, 10))
        value += step
    return ticks


def _fmt_pct(value: float, places: int = 0) -> str:
    return f"{value * 100:.{places}f}%"


def _fmt_num(value: float, places: int = 2) -> str:
    return f"{value:,.{places}f}"


def _esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def _date_ticks(index: pd.DatetimeIndex, count: int = 8) -> list[tuple[int, str]]:
    """Positions and labels for roughly ``count`` year boundaries."""
    if len(index) == 0:
        return []
    years = pd.DatetimeIndex(index).year
    boundaries = [0, *(i for i in range(1, len(index)) if years[i] != years[i - 1])]
    step = max(1, round(len(boundaries) / count))
    return [(i, str(years[i])) for i in boundaries[::step]]


def _last_finite(values: np.ndarray) -> float | None:
    for v in reversed(values):
        if math.isfinite(v):
            return float(v)
    return None


def _svg_open(kind: str, points_json: str, label: str) -> str:
    # The payload is JSON inside an HTML attribute: escape it, or a series named
    # "A & B" ends the document early. Browsers decode entities before the page
    # script reads the attribute, so JSON.parse still sees valid JSON.
    payload = html.escape(points_json, quote=True)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="chart" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'preserveAspectRatio="xMidYMid meet" aria-label="{_esc(label)}" '
        f'data-kind="{kind}" data-points="{payload}">'
    )


def _grid_and_axes(
    ticks: Sequence[float],
    y_of: Callable[[float], float],
    x0: float,
    x1: float,
    fmt: Callable[[float], str],
    zero_line: float | None = None,
) -> str:
    parts = []
    for tick in ticks:
        y = y_of(tick)
        parts.append(
            f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var(--grid)" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="var(--muted)">{_esc(fmt(tick))}</text>'
        )
    if zero_line is not None:
        y = y_of(zero_line)
        parts.append(
            f'<line x1="{x0}" x2="{x1}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="var(--axis)" stroke-width="1" />'
        )
    return "".join(parts)


def _x_axis(index: pd.DatetimeIndex, x_of: Callable[[float], float], y: float) -> str:
    """Year labels under the plot.

    The first and last are anchored inwards rather than centred, so the leftmost one
    does not sit on top of the bottom y-axis label.
    """
    ticks = _date_ticks(index)
    parts = []
    for k, (i, label) in enumerate(ticks):
        x = x_of(i)
        if k == 0:
            anchor = "start"
        elif k == len(ticks) - 1:
            anchor = "end"
        else:
            anchor = "middle"
        parts.append(
            f'<text x="{x:.1f}" y="{y + 18:.1f}" text-anchor="{anchor}" font-size="11" '
            f'fill="var(--muted)">{label}</text>'
        )
    return "".join(parts)


def time_series_chart(
    series: Sequence[Series],
    *,
    label: str,
    value_format: str = "percent",
    baseline: float | None = None,
    end_labels: bool = True,
) -> str:
    """Line chart (optionally area-filled) of one or more date-indexed series.

    ``value_format`` is ``"percent"``, ``"ratio"`` or ``"growth"`` and controls the axis
    labels and the tooltip. ``baseline`` draws an emphasised horizontal rule (0 for
    drawdowns and rolling Sharpe, 1 for a growth curve).
    """
    series = [s for s in series if len(s.values.dropna())]
    if not series:
        return '<p class="empty">No data.</p>'
    index = pd.DatetimeIndex(series[0].values.index)
    x0, y0, x1, y1 = _plot_box()
    n = len(index)

    lo = min(float(np.nanmin(s.values.to_numpy())) for s in series)
    hi = max(float(np.nanmax(s.values.to_numpy())) for s in series)
    if baseline is not None:
        lo, hi = min(lo, baseline), max(hi, baseline)
    if hi == lo:
        hi, lo = hi + 1, lo - 1
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    ticks = _nice_ticks(lo, hi)

    def x_of(i: float) -> float:
        return x0 + (x1 - x0) * (i / max(n - 1, 1))

    def y_of(v: float) -> float:
        return y1 - (y1 - y0) * ((v - lo) / (hi - lo))

    fmt: Callable[[float], str]
    if value_format == "percent":
        fmt = lambda v: _fmt_pct(v, 0)  # noqa: E731
    elif value_format == "growth":
        fmt = lambda v: f"{v:,.1f}x"  # noqa: E731
    else:
        fmt = lambda v: _fmt_num(v, 1)  # noqa: E731

    body = [_grid_and_axes(ticks, y_of, x0, x1, fmt, zero_line=baseline)]
    body.append(_x_axis(index, x_of, y1))

    # Direct end-labels work only while the series separate at the right edge. When they
    # converge, nudging labels apart detaches them from their lines and reads as noise,
    # so drop them for this chart and let the legend and the tooltip carry identity.
    finals = [_last_finite(s.values.to_numpy(dtype=float)) for s in series]
    label_ys = sorted(y_of(v) for v in finals if v is not None)
    if any(b - a < 14 for a, b in itertools.pairwise(label_ys)):
        end_labels = False

    for s in series:
        values = s.values.to_numpy(dtype=float)
        pts = [f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(values) if math.isfinite(v)]
        if not pts:
            continue
        color = f"var(--{s.color_var})"
        if s.fill:
            base_y = y_of(baseline if baseline is not None else lo)
            area = f"M{pts[0].split(',')[0]},{base_y:.1f} L" + " L".join(pts)
            area += f" L{pts[-1].split(',')[0]},{base_y:.1f} Z"
            body.append(f'<path d="{area}" fill="{color}" fill-opacity="0.10" stroke="none" />')
        body.append(
            f'<path d="M{" L".join(pts)}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round" />'
        )
        if end_labels:
            last_x, last_y = pts[-1].split(",")  # pts already skips non-finite points
            body.append(
                f'<circle cx="{last_x}" cy="{last_y}" r="4" fill="{color}" '
                f'stroke="var(--surface)" stroke-width="2" />'
            )
            body.append(
                f'<text x="{float(last_x) + 10:.1f}" y="{float(last_y) + 4:.1f}" font-size="11" '
                f'fill="var(--text_secondary)">{_esc(fmt(float(values[-1])))}</text>'
            )

    # hover layer: a crosshair line plus one dot per series, positioned by the page script
    body.append(
        f'<g class="crosshair" style="display:none">'
        f'<line y1="{y0}" y2="{y1}" stroke="var(--axis)" stroke-width="1" />'
        + "".join(
            f'<circle r="4" fill="var(--{s.color_var})" stroke="var(--surface)" stroke-width="2" />'
            for s in series
        )
        + "</g>"
    )
    body.append(
        f'<rect class="hit" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" '
        f'fill="transparent" />'
    )

    points = {
        "x0": x0,
        "x1": x1,
        "dates": [d.strftime("%Y-%m-%d") for d in index],
        "format": value_format,
        "series": [
            {
                "name": s.name,
                "color": s.color_var,
                "y": [
                    None if not math.isfinite(v) else round(y_of(v), 1)
                    for v in s.values.to_numpy(dtype=float)
                ],
                "v": [
                    None if not math.isfinite(v) else round(float(v), 6)
                    for v in s.values.to_numpy(dtype=float)
                ],
            }
            for s in series
        ],
    }
    return _svg_open("timeseries", json.dumps(points), label) + "".join(body) + "</svg>"


def monthly_heatmap(returns: pd.Series, *, label: str) -> str:
    """Year-by-month table of compounded returns, diverging blue (up) to red (down)."""
    monthly = (1.0 + returns.fillna(0.0)).resample("ME").prod() - 1.0
    if monthly.empty:
        return '<p class="empty">No data.</p>'
    month_index = pd.DatetimeIndex(monthly.index)
    frame = pd.DataFrame(
        {"year": month_index.year, "month": month_index.month, "value": monthly.to_numpy()}
    )
    table = frame.pivot(index="year", columns="month", values="value")
    year_key = pd.DatetimeIndex(returns.index).year
    yearly = (1.0 + returns.fillna(0.0)).groupby(year_key).prod() - 1.0

    years = list(table.index)
    cell, gap = 46.0, 2.0
    left, top = 52.0, 26.0
    width = left + 13 * (cell + gap) + 24
    height = top + len(years) * (cell + gap) + 12
    scale = float(np.nanmax(np.abs(table.to_numpy(dtype=float)))) or 1.0

    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    body = []
    for m, name in enumerate(names, start=1):
        x = left + (m - 1) * (cell + gap) + cell / 2
        body.append(
            f'<text x="{x:.1f}" y="{top - 8:.1f}" text-anchor="middle" font-size="11" '
            f'fill="var(--muted)">{name}</text>'
        )
    body.append(
        f'<text x="{left + 12 * (cell + gap) + cell / 2:.1f}" y="{top - 8:.1f}" '
        f'text-anchor="middle" font-size="11" fill="var(--text_secondary)">Year</text>'
    )

    cells: list[str] = []
    for row, year in enumerate(years):
        y = top + row * (cell + gap)
        body.append(
            f'<text x="{left - 10:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="11" fill="var(--muted)">{year}</text>'
        )
        for m in range(1, 13):
            raw = table.loc[year, m] if m in table.columns else np.nan
            value = float(raw) if pd.notna(raw) else float("nan")
            x = left + (m - 1) * (cell + gap)
            if not math.isfinite(value):
                body.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" rx="3" '
                    f'fill="var(--neutral)" fill-opacity="0.4" />'
                )
                continue
            v = value
            intensity = min(abs(v) / scale, 1.0) ** 0.7
            color = "var(--positive)" if v >= 0 else "var(--negative)"
            body.append(
                f'<rect class="cell" x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" '
                f'rx="3" fill="{color}" fill-opacity="{max(intensity, 0.06):.3f}" '
                f'data-label="{year}-{m:02d}" data-value="{_fmt_pct(v, 1)}" />'
            )
            if abs(v) >= 0.02:
                body.append(
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                    f'text-anchor="middle" font-size="10" fill="var(--text_secondary)" '
                    f'pointer-events="none">{v * 100:.0f}</text>'
                )
        total_raw = yearly.get(year, float("nan"))
        total = float(total_raw) if pd.notna(total_raw) else float("nan")
        x = left + 12 * (cell + gap)
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" rx="3" '
            f'fill="var(--neutral)" fill-opacity="0.5" />'
        )
        if math.isfinite(total):
            body.append(
                f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" text-anchor="middle" '
                f'font-size="10" font-weight="600" fill="var(--text)" '
                f'pointer-events="none">{total * 100:.0f}</text>'
            )
    body.extend(cells)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="chart heatmap" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'preserveAspectRatio="xMidYMid meet" aria-label="{_esc(label)}" data-kind="heatmap">'
        + "".join(body)
        + "</svg>"
    )


def factor_bars(table: pd.DataFrame, *, label: str, model: str = "ff5") -> str:
    """Horizontal betas with their t-statistics, blue for positive and red for negative."""
    values: dict[str, float] = {
        str(k): float(v) for k, v in table.loc[model].astype(float).to_dict().items()
    }
    betas = {
        name[len("beta_") :]: value
        for name, value in values.items()
        if name.startswith("beta_") and not math.isnan(value)
    }
    tstats = {k: values.get(f"t_{k}", float("nan")) for k in betas}
    if not betas:
        return '<p class="empty">No factor loadings.</p>'

    names = list(betas)
    bar_h, gap = 22.0, 14.0
    left, right = 92.0, 150.0
    top = 12.0
    width = 960.0
    height = top + len(names) * (bar_h + gap) + 8
    span = max(max(abs(v) for v in betas.values()), 0.2) * 1.15
    mid = left + (width - left - right) / 2

    def x_of(v: float) -> float:
        return mid + (width - left - right) / 2 * (v / span)

    body = [
        f'<line x1="{mid:.1f}" x2="{mid:.1f}" y1="{top:.1f}" y2="{height - 8:.1f}" '
        f'stroke="var(--axis)" stroke-width="1" />'
    ]
    for i, name in enumerate(names):
        beta = betas[name]
        y = top + i * (bar_h + gap)
        x_end = x_of(beta)
        x_start = min(mid, x_end)
        bar_w = abs(x_end - mid)
        color = "var(--positive)" if beta >= 0 else "var(--negative)"
        body.append(
            f'<text x="{left - 12:.1f}" y="{y + bar_h / 2 + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="var(--text_secondary)">{_esc(name)}</text>'
        )
        body.append(
            f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{max(bar_w, 1):.1f}" height="{bar_h}" '
            f'rx="3" fill="{color}" />'
        )
        t = tstats[name]
        significance = "significant" if abs(t) >= 2 else "not significant"
        body.append(
            f'<text x="{width - right + 12:.1f}" y="{y + bar_h / 2 + 4:.1f}" font-size="11" '
            f'fill="var(--text_secondary)">{beta:+.2f}'
            f'<tspan fill="var(--muted)">  t = {t:+.1f} ({significance})</tspan></text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="chart" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'preserveAspectRatio="xMidYMid meet" aria-label="{_esc(label)}" data-kind="bars">'
        + "".join(body)
        + "</svg>"
    )


def legend(series: Sequence[Series]) -> str:
    """Swatch-and-name legend; identity never rests on hue alone."""
    if len(series) < 2:
        return ""
    items = "".join(
        f'<span class="legend-item"><span class="swatch" '
        f'style="background: var(--{s.color_var})"></span>{_esc(s.name)}</span>'
        for s in series
    )
    return f'<div class="legend">{items}</div>'
