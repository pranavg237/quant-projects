"""Helpers shared by the built-in strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantbt.strategy import Context


def is_new_month(ctx: Context) -> bool:
    """True on the first bar of a calendar month (decisions made here fill on bar two)."""
    i = ctx.position_index
    if i == 0:
        return True
    index = ctx.data.index
    return bool(index[i].month != index[i - 1].month)


def is_rebalance_bar(ctx: Context, rebalance: str | int) -> bool:
    """``"daily"``, ``"monthly"`` or an integer number of bars between rebalances."""
    if rebalance == "daily":
        return True
    if rebalance == "monthly":
        return is_new_month(ctx)
    if isinstance(rebalance, int) and rebalance > 0:
        return ctx.position_index % rebalance == 0
    raise ValueError(f"rebalance must be 'daily', 'monthly' or a positive int, got {rebalance!r}")


def realized_vol(closes: pd.DataFrame, lookback: int, ppy: int = 252) -> pd.Series:
    """Annualised standard deviation of the last ``lookback`` simple returns per column."""
    rets = closes.pct_change().iloc[-lookback:]
    vol: pd.Series = rets.std(ddof=1) * np.sqrt(ppy)
    return vol


def changed(new: dict[str, float], old: dict[str, float], tol: float = 1e-9) -> bool:
    keys = set(new) | set(old)
    return any(abs(new.get(k, 0.0) - old.get(k, 0.0)) > tol for k in keys)
