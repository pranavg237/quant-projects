"""Chart palette and chrome tokens.

The values come from a palette validated for colour-vision deficiency separation and
surface contrast in both light and dark mode (OKLab delta-E >= 8 between adjacent
categorical slots, >= 3:1 against the surface). Series identity is carried by a legend
and direct end-labels as well as by hue, so nothing depends on colour alone.
"""

from __future__ import annotations

from typing import Final

LIGHT: Final[dict[str, str]] = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "series_1": "#2a78d6",  # strategy
    "series_2": "#eb6834",  # benchmark
    "positive": "#2a78d6",  # diverging: cool pole
    "negative": "#d03b3b",  # diverging: warm pole
    "neutral": "#f0efec",  # diverging midpoint
    "good": "#0ca30c",
    "critical": "#d03b3b",
}

DARK: Final[dict[str, str]] = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "positive": "#3987e5",
    "negative": "#d03b3b",
    "neutral": "#383835",
    "good": "#0ca30c",
    "critical": "#e66767",
}

FONT_STACK: Final[str] = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def css_variables() -> str:
    """``:root`` custom properties for both modes, with a data-theme override."""
    light = "\n".join(f"      --{k}: {v};" for k, v in LIGHT.items())
    dark = "\n".join(f"      --{k}: {v};" for k, v in DARK.items())
    return f"""    :root {{
      color-scheme: light;
{light}
    }}
    @media (prefers-color-scheme: dark) {{
      :root:not([data-theme="light"]) {{
        color-scheme: dark;
{dark}
      }}
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
{dark}
    }}"""
