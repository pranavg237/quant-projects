"""Performance metrics on periodic returns and equity curves.

All functions take *simple* (not log) returns as a ``pd.Series`` indexed by date, or an
equity curve in currency units. ``rf`` is the risk-free rate as a per-period series
aligned to the returns, or an annual decimal that is converted per period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def periods_per_year(index: pd.Index) -> int:
    """Infer the sampling frequency from the median spacing of ``index``."""
    if len(index) < 2:
        return TRADING_DAYS
    idx = pd.DatetimeIndex(index)
    days = float(np.median(np.diff(idx.values).astype("timedelta64[s]").astype(float)) / 86400.0)
    if days <= 4:
        return TRADING_DAYS
    if days <= 10:
        return 52
    if days <= 45:
        return 12
    if days <= 120:
        return 4
    return 1


def _per_period_rf(rf: pd.Series | float | None, returns: pd.Series, ppy: int) -> pd.Series:
    if rf is None:
        return pd.Series(0.0, index=returns.index)
    if isinstance(rf, pd.Series):
        aligned = rf.reindex(returns.index)
        if aligned.isna().any():
            raise ValueError("rf series does not cover every return date")
        return aligned.astype(float)
    return pd.Series((1.0 + float(rf)) ** (1.0 / ppy) - 1.0, index=returns.index)


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Compound ``returns`` into an equity curve starting at ``initial`` (NaN treated as 0)."""
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def years_spanned(index: pd.Index) -> float:
    idx = pd.DatetimeIndex(index)
    return float((idx[-1] - idx[0]).days) / 365.25


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate over the calendar span of the curve."""
    years = years_spanned(equity.index)
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def annualized_return(returns: pd.Series, ppy: int | None = None) -> float:
    """Geometric mean return per year (period-count based, unlike calendar-based CAGR)."""
    ppy = ppy or periods_per_year(returns.index)
    r = returns.dropna()
    if len(r) == 0:
        return float("nan")
    growth = float(np.prod(1.0 + r.to_numpy()))
    return float(growth ** (ppy / len(r)) - 1.0)


def annualized_vol(returns: pd.Series, ppy: int | None = None) -> float:
    ppy = ppy or periods_per_year(returns.index)
    r = returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ppy))


def sharpe(
    returns: pd.Series, rf: pd.Series | float | None = None, ppy: int | None = None
) -> float:
    """Annualised Sharpe ratio: mean excess return over its standard deviation."""
    ppy = ppy or periods_per_year(returns.index)
    excess = (returns - _per_period_rf(rf, returns, ppy)).dropna()
    if len(excess) < 2:
        return float("nan")
    sd = float(excess.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ppy))


def sortino(
    returns: pd.Series, rf: pd.Series | float | None = None, ppy: int | None = None
) -> float:
    """Annualised Sortino ratio using the full-sample downside deviation (target = rf)."""
    ppy = ppy or periods_per_year(returns.index)
    excess = (returns - _per_period_rf(rf, returns, ppy)).dropna()
    if len(excess) < 2:
        return float("nan")
    downside = np.minimum(excess.to_numpy(), 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd == 0.0:
        return float("inf") if excess.mean() > 0 else float("nan")
    return float(excess.mean() / dd * np.sqrt(ppy))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak (0 at a new high, negative below it)."""
    peak = equity.cummax()
    return equity / peak - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """Most negative drawdown, as a (negative) fraction."""
    return float(drawdown_series(equity).min())


@dataclass(frozen=True)
class DrawdownInfo:
    depth: float
    peak: pd.Timestamp
    trough: pd.Timestamp
    recovery: pd.Timestamp | None
    duration_bars: int
    duration_days: int
    recovered: bool


def max_drawdown_info(equity: pd.Series) -> DrawdownInfo:
    """Depth and duration of the *longest* underwater stretch plus where the deepest one sits.

    Duration is measured peak-to-recovery for the longest underwater period (the classic
    "max drawdown duration"). If the curve ends underwater the open period counts up to
    the last bar and ``recovered`` is ``False``.
    """
    dd = drawdown_series(equity)
    under = dd < 0
    idx = equity.index
    # deepest point
    trough_pos = int(np.argmin(dd.to_numpy()))
    trough = pd.Timestamp(idx[trough_pos])
    peak_pos = int(np.argmax(equity.to_numpy()[: trough_pos + 1]))
    peak = pd.Timestamp(idx[peak_pos])
    # longest underwater stretch
    best_len, best_start, best_end, best_recovered = 0, 0, 0, True
    i = 0
    n = len(dd)
    while i < n:
        if not under.iloc[i]:
            i += 1
            continue
        start = i - 1 if i > 0 else 0
        j = i
        while j < n and under.iloc[j]:
            j += 1
        recovered = j < n
        end = j if recovered else n - 1
        if end - start > best_len:
            best_len, best_start, best_end, best_recovered = end - start, start, end, recovered
        i = j
    recovery: pd.Timestamp | None
    if best_len == 0:
        return DrawdownInfo(0.0, peak, trough, None, 0, 0, True)
    recovery = pd.Timestamp(idx[best_end]) if best_recovered else None
    days = int((pd.Timestamp(idx[best_end]) - pd.Timestamp(idx[best_start])).days)
    return DrawdownInfo(
        depth=float(dd.min()),
        peak=peak,
        trough=trough,
        recovery=recovery,
        duration_bars=best_len,
        duration_days=days,
        recovered=best_recovered,
    )


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0.0:
        return float("nan")
    return cagr(equity) / abs(mdd)


def turnover(weights: pd.DataFrame | pd.Series, ppy: int | None = None) -> float:
    """Annualised one-way turnover: sum of |weight changes| / 2 per year.

    A weight of 1.0 moved in and out once a year is a turnover of 1.0.
    """
    w = weights.to_frame() if isinstance(weights, pd.Series) else weights
    ppy = ppy or periods_per_year(w.index)
    w = w.fillna(0.0)
    change = w.diff().abs().sum(axis=1)
    change.iloc[0] = w.iloc[0].abs().sum()
    if len(w) == 0:
        return float("nan")
    return float(change.sum() / 2.0 / len(w) * ppy)


def exposure(weights: pd.DataFrame | pd.Series) -> float:
    """Average gross exposure (sum of |weights|) across all bars."""
    w = weights.to_frame() if isinstance(weights, pd.Series) else weights
    return float(w.abs().sum(axis=1).mean())


def time_in_market(weights: pd.DataFrame | pd.Series) -> float:
    """Fraction of bars with any non-zero position."""
    w = weights.to_frame() if isinstance(weights, pd.Series) else weights
    return float((w.abs().sum(axis=1) > 0).mean())


def hit_rate(pnls: pd.Series | np.ndarray | list[float]) -> float:
    """Fraction of closed trades with strictly positive P&L."""
    arr = np.asarray(pnls, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float((arr > 0).mean())


def profit_factor(pnls: pd.Series | np.ndarray | list[float]) -> float:
    arr = np.asarray(pnls, dtype=float)
    losses = -arr[arr < 0].sum()
    gains = arr[arr > 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def summary(
    returns: pd.Series,
    *,
    rf: pd.Series | float | None = None,
    weights: pd.DataFrame | pd.Series | None = None,
    trade_pnls: pd.Series | np.ndarray | list[float] | None = None,
    benchmark: pd.Series | None = None,
) -> pd.Series:
    """The standard metric set as a Series (name -> value)."""
    ppy = periods_per_year(returns.index)
    eq = equity_curve(returns)
    dd = max_drawdown_info(eq)
    growth = float(eq.iloc[-1])
    # The first return accrues over the period *ending* on the first date, so the curve
    # spans one period more than the index does.
    years = years_spanned(returns.index) + 1.0 / ppy
    out: dict[str, Any] = {
        "start": pd.Timestamp(returns.index[0]),
        "end": pd.Timestamp(returns.index[-1]),
        "periods": len(returns),
        "total_return": growth - 1.0,
        "cagr": growth ** (1.0 / years) - 1.0 if growth > 0 else float("nan"),
        "ann_vol": annualized_vol(returns, ppy),
        "sharpe": sharpe(returns, rf, ppy),
        "sortino": sortino(returns, rf, ppy),
        "calmar": calmar(eq),
        "max_drawdown": dd.depth,
        "max_dd_duration_days": dd.duration_days,
        "max_dd_recovered": dd.recovered,
        "best_period": float(returns.max()),
        "worst_period": float(returns.min()),
        "positive_periods": float((returns > 0).mean()),
    }
    if weights is not None:
        out["turnover"] = turnover(weights, ppy)
        out["exposure"] = exposure(weights)
        out["time_in_market"] = time_in_market(weights)
    if trade_pnls is not None:
        out["trades"] = len(np.asarray(trade_pnls))
        out["hit_rate"] = hit_rate(trade_pnls)
        out["profit_factor"] = profit_factor(trade_pnls)
    if benchmark is not None:
        b = benchmark.reindex(returns.index).fillna(0.0)
        out["benchmark_total_return"] = float(equity_curve(b).iloc[-1]) - 1.0
        out["benchmark_sharpe"] = sharpe(b, rf, ppy)
        active = returns - b
        out["excess_return_ann"] = annualized_return(returns, ppy) - annualized_return(b, ppy)
        te = annualized_vol(active, ppy)
        out["information_ratio"] = out["excess_return_ann"] / te if te and te > 0 else float("nan")
    return pd.Series(out)


def format_summary(s: pd.Series) -> str:
    """Human-readable rendering of :func:`summary`."""
    pct = {
        "total_return",
        "cagr",
        "ann_vol",
        "max_drawdown",
        "best_period",
        "worst_period",
        "positive_periods",
        "exposure",
        "time_in_market",
        "hit_rate",
        "benchmark_total_return",
        "excess_return_ann",
    }
    lines = []
    for k, v in s.items():
        key = str(k)
        if key in pct and isinstance(v, (float, np.floating)):
            lines.append(f"{key:<24}{v:>12.2%}")
        elif isinstance(v, (float, np.floating)):
            lines.append(f"{key:<24}{v:>12.3f}")
        elif isinstance(v, pd.Timestamp):
            lines.append(f"{key:<24}{v.date()!s:>12}")
        else:
            lines.append(f"{key:<24}{v!s:>12}")
    return "\n".join(lines)
