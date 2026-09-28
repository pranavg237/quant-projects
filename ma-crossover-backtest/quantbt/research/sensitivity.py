"""Parameter-sensitivity surface for the MA crossover: is the chosen point a plateau or a spike?

This is a **robustness diagnostic, not a parameter selection.** Every cell is scored on
the full sample, so the surface is in-sample by construction. Nothing downstream picks
parameters from it; the walk-forward in :mod:`quantbt.validation.walkforward` is the only
place parameters are chosen. What the surface can show is whether good Sharpe ratios come
from a broad region (a plateau, where small parameter changes barely matter) or from an
isolated cell (a spike, which is what a lucky fit looks like).

The grid is scored with the vectorised long/flat backtest, which reproduces the
event-driven engine to floating-point precision (``scripts/verify_port.py``) and is fast
enough for a dense grid. Sharpe is in excess of the risk-free rate and net of costs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.data import PriceData
from quantbt.vectorized import backtest_ma_crossover


def ma_sharpe_grid(
    data: PriceData,
    fast: Sequence[int],
    slow: Sequence[int],
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    rf: pd.Series | float | None = None,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
) -> pd.DataFrame:
    """Excess Sharpe, net of costs, for every ``fast < slow`` pair; one row per pair.

    Columns: ``fast, slow, sharpe, cagr, max_drawdown, turnover, trades``. Pairs with
    ``fast >= slow`` are not valid crossovers and are left out.
    """
    rows: list[dict[str, float]] = []
    for f in fast:
        for s in slow:
            if f >= s:
                continue
            res = backtest_ma_crossover(
                data,
                f,
                s,
                start=start,
                end=end,
                slippage_bps=slippage_bps,
                commission_bps=commission_bps,
                rf=rf,
            )
            r = res.returns
            summ = res.summary(rf=rf)
            rows.append(
                {
                    "fast": f,
                    "slow": s,
                    "sharpe": metrics.sharpe(r, rf=rf),
                    "cagr": float(summ["cagr"]),
                    "max_drawdown": float(summ["max_drawdown"]),
                    "turnover": float(summ["turnover"]),
                    "trades": float(res.frame["trade"].abs().sum()),
                }
            )
    if not rows:
        raise ValueError("no valid (fast < slow) pairs in the grid")
    return pd.DataFrame(rows)


def neighbourhood_sharpe(table: pd.DataFrame, fast: int, slow: int) -> pd.Series:
    """Sharpe of ``(fast, slow)`` and of its immediate neighbours on the grid.

    Neighbours are the adjacent grid values in each direction (up to 8 cells, fewer at
    an edge or where ``fast >= slow``). A point on a plateau has neighbours about as good
    as itself; a spike has neighbours that are much worse.
    """
    fasts = sorted(table["fast"].unique())
    slows = sorted(table["slow"].unique())
    if fast not in fasts or slow not in slows:
        raise ValueError(f"({fast}, {slow}) is not on the grid")
    pivot = table.pivot(index="fast", columns="slow", values="sharpe")
    grid = pivot.reindex(index=fasts, columns=slows).to_numpy(dtype=float)
    i, j = fasts.index(fast), slows.index(slow)
    values: list[float] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            a, b = i + di, j + dj
            if 0 <= a < len(fasts) and 0 <= b < len(slows):
                v = float(grid[a, b])
                if np.isfinite(v):
                    values.append(v)
    centre = float(grid[i, j])
    return pd.Series(
        {
            "sharpe": centre,
            "n_neighbours": float(len(values)),
            "neighbour_mean": float(np.mean(values)) if values else float("nan"),
            "neighbour_min": float(np.min(values)) if values else float("nan"),
        }
    )


def surface_summary(
    table: pd.DataFrame,
    benchmark_sharpe: float,
    points: Iterable[tuple[int, int]],
) -> pd.DataFrame:
    """One row per point of interest: its Sharpe, its neighbours', and its grid percentile."""
    all_sharpe = table["sharpe"].to_numpy(dtype=float)
    rows = []
    for f, s in points:
        nb = neighbourhood_sharpe(table, f, s)
        rows.append(
            {
                "fast": f,
                "slow": s,
                **nb.to_dict(),
                "percentile_in_grid": float((all_sharpe < nb["sharpe"]).mean()),
                "beats_benchmark": bool(nb["sharpe"] > benchmark_sharpe),
            }
        )
    return pd.DataFrame(rows)
