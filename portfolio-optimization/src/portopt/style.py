"""House chart style: one validated palette, applied everywhere.

Colour is assigned by the *job* it does, not by taste:

* **Categorical** (identity -- model vs market, CRR vs Leisen-Reimer): the fixed slot
  order ``blue, orange, aqua, violet``. Slots are assigned in order and never cycled.
  This four-colour set clears every all-pairs gate of the palette validator in light mode
  (worst colour-vision-deficient Delta E 9.2, worst normal-vision Delta E 16.3).
* **Sequential / ordinal** (magnitude or an ordered set -- expiries, step counts): a
  single blue ramp, light to dark. Expiries are *ordered*, so giving each one an
  arbitrary categorical hue would throw that ordering away.
* **Diverging** (polarity -- signed fit errors): blue/red poles around a neutral grey
  midpoint. Never a rainbow, never a hue at the midpoint.

Grid and axes are deliberately recessive, no chart uses two y-scales, and every chart with
more than one series carries a legend so identity is never colour-alone.
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import numpy as np
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "CATEGORICAL",
    "DIVERGING",
    "GRID",
    "INK",
    "INK_MUTED",
    "INK_SECONDARY",
    "SEQUENTIAL_BLUE",
    "SURFACE",
    "apply_house_style",
    "ordinal_colors",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e6e5e1"

#: Fixed categorical slot order. Assign in order; never cycle past the end.
CATEGORICAL: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7")

#: Blue ramp steps 250 -> 700, light to dark. Starts at 250 so even the lightest step
#: clears 2:1 contrast against the light surface.
_BLUE_STEPS = (
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list("house_blue", _BLUE_STEPS)

#: Blue <-> red through a neutral grey. Warm/cool poles read as opposite; the grey
#: midpoint reads as "nothing", which aqua or green would not.
DIVERGING = LinearSegmentedColormap.from_list(
    "house_diverging", ("#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a3a2", "#e34948", "#8f2423")
)


def ordinal_colors(n: int, start: float = 0.12, stop: float = 0.95) -> list[str]:
    """``n`` colours from the blue ramp, light to dark, for an ordered set of series.

    Args:
        n: Number of series.
        start: Position on the ramp for the first (lightest) series.
        stop: Position for the last (darkest) series.
    """
    if n <= 0:
        return []
    if n == 1:
        return [mpl.colors.to_hex(SEQUENTIAL_BLUE(stop))]
    positions = np.linspace(start, stop, n)
    return [mpl.colors.to_hex(SEQUENTIAL_BLUE(p)) for p in positions]


def apply_house_style() -> None:
    """Install the house style into Matplotlib's global rcParams."""
    params: dict[str, Any] = {
        "figure.facecolor": SURFACE,
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=list(CATEGORICAL)),
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_SECONDARY,
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "figure.titlesize": 13,
        "figure.titleweight": "semibold",
    }
    # rcParams is typed with a Literal key union; a plain dict is the idiomatic way to
    # pass a style and mypy cannot narrow it.
    mpl.rcParams.update(params)  # type: ignore[arg-type]
