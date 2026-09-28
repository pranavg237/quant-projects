"""Vectorised single-asset long/flat backtest with next-open fills and explicit costs.

This is the corrected version of the original script and the reference that the
event-driven engine (``quantbt.engine``) is checked against. The model:

* A signal ``s[t] in {0, 1}`` is computed from data up to and including the close of
  bar ``t``.
* The position is changed at the **open of bar t+1**, paying ``slippage_bps`` against
  the open and ``commission_bps`` of the traded notional.
* While invested the strategy earns the total-return close-to-close move; while flat it
  earns the per-period risk-free rate ``rf`` (default zero). Cash held overnight into an
  entry bar earns that bar's rate too; cash raised at an exit earns nothing that bar.
* Evaluation starts at ``start``; bars before it are warm-up only.

The return of bar ``t+1`` on an entry day is
``close[t+1] / (open[t+1] * (1 + slip) * (1 + comm)) - 1``; on an exit day it is
``open[t+1] * (1 - slip) * (1 - comm) / close[t] - 1``; on a holding day
``close[t+1] / close[t] - 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantbt.data import PriceData
from quantbt.metrics import periods_per_year, summary


def ma_crossover_signal(close: pd.Series, short_window: int, long_window: int) -> pd.Series:
    """1 when the short MA is strictly above the long MA, else 0. NaN warm-up maps to 0.

    Scale-invariant, so it is safe on total-return-adjusted prices (see ``quantbt.data``).
    """
    if not 0 < short_window < long_window:
        raise ValueError("need 0 < short_window < long_window")
    fast = close.rolling(short_window).mean()
    slow = close.rolling(long_window).mean()
    return (fast > slow).astype(int)


@dataclass(frozen=True)
class VectorizedResult:
    """Output of :func:`backtest_long_flat`. ``frame`` has one row per evaluation bar."""

    frame: pd.DataFrame
    slippage_bps: float
    commission_bps: float

    @property
    def returns(self) -> pd.Series:
        return self.frame["strategy_return"]

    @property
    def equity(self) -> pd.Series:
        return self.frame["equity"]

    @property
    def benchmark_returns(self) -> pd.Series:
        return self.frame["benchmark_return"]

    @property
    def position(self) -> pd.Series:
        return self.frame["position"]

    @property
    def trade_pnls(self) -> pd.Series:
        """P&L (as a fraction of equity at entry) of each completed round trip."""
        pos = self.frame["position"].to_numpy()
        eq = self.frame["equity"].to_numpy()
        pnls: list[float] = []
        entry_eq: float | None = None
        for i in range(len(pos)):
            prev = pos[i - 1] if i > 0 else 0
            if pos[i] == 1 and prev == 0:
                entry_eq = eq[i - 1] if i > 0 else eq[0]
            elif pos[i] == 0 and prev == 1 and entry_eq is not None:
                pnls.append(eq[i] / entry_eq - 1.0)
                entry_eq = None
        return pd.Series(pnls, dtype=float, name="trade_pnl")

    def summary(self, rf: pd.Series | float | None = None) -> pd.Series:
        return summary(
            self.returns,
            rf=rf,
            weights=self.position.astype(float),
            trade_pnls=self.trade_pnls,
            benchmark=self.benchmark_returns,
        )


def backtest_long_flat(
    data: PriceData,
    signal: pd.Series,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    slippage_bps: float = 5.0,
    commission_bps: float = 1.0,
    rf: pd.Series | float | None = None,
) -> VectorizedResult:
    """Run a long/flat signal on a single-symbol :class:`PriceData` with next-open fills."""
    close = data.single("close")
    open_ = data.single("open")
    if not signal.index.equals(close.index):
        raise ValueError("signal index must equal the data index")
    if not signal.isin([0, 1]).all():
        raise ValueError("signal must be 0/1")
    if slippage_bps < 0 or commission_bps < 0:
        raise ValueError("costs must be non-negative")

    slip = slippage_bps / 1e4
    comm = commission_bps / 1e4
    ppy = periods_per_year(close.index)
    if rf is None:
        rf_series = pd.Series(0.0, index=close.index)
    elif isinstance(rf, pd.Series):
        rf_series = rf.reindex(close.index).fillna(0.0).astype(float)
    else:
        rf_series = pd.Series((1.0 + rf) ** (1.0 / ppy) - 1.0, index=close.index)

    # position[t] is what is held from the open of t (decided at the close of t-1).
    # The evaluation window starts flat: a position cannot pre-date the first bar on
    # which the strategy was allowed to act, so an in-force signal at ``start`` is
    # entered at the *next* open, exactly as the event-driven engine does.
    position = signal.shift(1).fillna(0).astype(int)
    first_eval = position.loc[start:end].index[0] if len(position.loc[start:end]) else None
    if first_eval is None:
        raise ValueError("no bars in the evaluation window")
    position = position.copy()
    position.loc[first_eval] = 0
    position.loc[: first_eval - pd.Timedelta(days=1)] = 0  # never "in" before the window
    prev_position = position.shift(1).fillna(0).astype(int)
    prev_close = close.shift(1)

    enter = (position == 1) & (prev_position == 0)
    exit_ = (position == 0) & (prev_position == 1)
    hold = (position == 1) & (prev_position == 1)
    flat = (position == 0) & (prev_position == 0)

    buy_fill = open_ * (1.0 + slip)
    sell_fill = open_ * (1.0 - slip)
    ret = pd.Series(np.nan, index=close.index, dtype=float)
    # Commission is charged on the notional: buying costs fill * (1 + comm) per unit of
    # equity, selling returns fill * (1 - comm). This is exactly how the event-driven
    # engine sizes a target-weight order, so the two agree to floating-point precision.
    # Cash held overnight into the entry bar still earns that bar's rate.
    ret[enter] = ((1.0 + rf_series) * close / (buy_fill * (1.0 + comm)) - 1.0)[enter]
    ret[exit_] = (sell_fill * (1.0 - comm) / prev_close - 1.0)[exit_]
    ret[hold] = (close / prev_close - 1.0)[hold]
    ret[flat] = rf_series[flat]
    # exit day: cash after the sale also earns nothing that day (fill happens at the open)

    fill_price = pd.Series(np.nan, index=close.index, dtype=float)
    fill_price[enter] = buy_fill[enter]
    fill_price[exit_] = sell_fill[exit_]

    benchmark = close.pct_change()

    frame = pd.DataFrame(
        {
            "close": close,
            "open": open_,
            "signal": signal,
            "position": position,
            "trade": position.diff().fillna(position.iloc[0]).astype(int),
            "fill_price": fill_price,
            "strategy_return": ret,
            "benchmark_return": benchmark,
            "rf": rf_series,
        }
    )
    frame = frame.loc[start:end].copy()
    frame.loc[frame.index[0], "benchmark_return"] = 0.0
    # The first evaluation bar cannot be an entry/exit decided before the window in a way
    # that the caller did not intend; we keep it as computed (warm-up is the caller's job).
    frame["strategy_return"] = frame["strategy_return"].fillna(0.0)
    frame["equity"] = (1.0 + frame["strategy_return"]).cumprod()
    frame["benchmark_equity"] = (1.0 + frame["benchmark_return"]).cumprod()
    return VectorizedResult(frame=frame, slippage_bps=slippage_bps, commission_bps=commission_bps)


def backtest_ma_crossover(
    data: PriceData,
    short_window: int = 50,
    long_window: int = 200,
    **kwargs: object,
) -> VectorizedResult:
    """Convenience wrapper: MA crossover signal on ``close`` then :func:`backtest_long_flat`."""
    signal = ma_crossover_signal(data.single("close"), short_window, long_window)
    return backtest_long_flat(data, signal, **kwargs)  # type: ignore[arg-type]
