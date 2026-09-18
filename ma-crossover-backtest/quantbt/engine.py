"""The event loop.

Per bar ``t`` (in order):

1. accrue one period of interest on cash (held from the previous close),
2. fill the orders submitted at the close of ``t - 1`` against bar ``t``,
3. force-liquidate positions in symbols that left the universe (at bar ``t``'s price),
4. mark every position at ``close[t]`` and record equity/positions/weights,
5. if ``t`` is inside the evaluation window, call ``strategy.on_bar`` and queue its orders.

Orders queued at the final bar are never filled; they are reported as rejections.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd

from quantbt.data import PriceData
from quantbt.execution import ExecutionSimulator, Fill, Order, OrderKind, Rejection
from quantbt.portfolio import Portfolio
from quantbt.results import BacktestResult
from quantbt.strategy import Context, Strategy
from quantbt.universe import PointInTimeUniverse, Universe, from_data


def _rf_series(rf: pd.Series | float | None, index: pd.DatetimeIndex, ppy: int) -> pd.Series:
    if rf is None:
        return pd.Series(0.0, index=index)
    if isinstance(rf, pd.Series):
        return rf.reindex(index).fillna(0.0).astype(float)
    return pd.Series((1.0 + float(rf)) ** (1.0 / ppy) - 1.0, index=index)


def run_backtest(
    data: PriceData,
    strategy: Strategy,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    initial_capital: float = 1_000_000.0,
    execution: ExecutionSimulator | None = None,
    rf: pd.Series | float | None = None,
    universe: Universe | None = None,
    benchmark: str | None = None,
    params: Mapping[str, Any] | None = None,
) -> BacktestResult:
    """Run ``strategy`` over ``data`` and return a :class:`BacktestResult`.

    Bars before ``start`` are visible as history but no decisions are made on them.
    ``universe`` defaults to :func:`quantbt.universe.from_data` (a symbol is tradable
    only while it has prices). ``rf`` is an annual rate or a per-period series.
    """
    execution = execution or ExecutionSimulator()
    index = data.index
    if len(index) == 0:
        raise ValueError("empty data")
    start_ts = pd.Timestamp(start) if start is not None else index[0]
    end_ts = pd.Timestamp(end) if end is not None else index[-1]
    eval_mask = (index >= start_ts) & (index <= end_ts)
    if not eval_mask.any():
        raise ValueError("no bars in the evaluation window")
    first_eval = int(eval_mask.argmax())
    if first_eval < strategy.warmup:
        warnings.warn(
            f"{strategy.name}: only {first_eval} warm-up bars available before "
            f"{start_ts.date()} but the strategy declares warmup={strategy.warmup}",
            stacklevel=2,
        )
    last_eval = int(len(index) - 1 - eval_mask[::-1].argmax())

    uni: Universe = universe if universe is not None else from_data(data)
    unknown = [s for s in uni.all_symbols if s not in data.symbols]
    if unknown:
        raise ValueError(f"universe symbols without data: {unknown}")

    from quantbt.metrics import periods_per_year

    ppy = periods_per_year(index)
    rf_series = _rf_series(rf, index, ppy)

    portfolio = Portfolio(cash=float(initial_capital))
    fill_field = data.open if execution.fill_at == "open" else data.close
    close_values = data.close.to_numpy()
    fill_values = fill_field.to_numpy()
    volume_values = data.volume.to_numpy()
    col = {s: i for i, s in enumerate(data.symbols)}

    pending: list[Order] = []
    rejections: list[Rejection] = []
    equity_hist: list[float] = []
    cash_hist: list[float] = []
    pos_hist: list[dict[str, float]] = []
    weight_hist: list[dict[str, float]] = []
    record_hist: list[dict[str, float]] = []
    eval_index: list[pd.Timestamp] = []
    started = False
    strategy_params = dict(params or strategy.params())

    # Nothing can happen before the first evaluated bar (no orders exist), so the
    # portfolio starts there: warm-up bars are history, not time in the market.
    for i in range(first_eval, last_eval + 1):
        now = pd.Timestamp(index[i])
        members = uni.members(now)

        # 1. interest on cash held since the previous close
        portfolio.accrue_interest(float(rf_series.iloc[i]))

        # 2. fills for orders queued at the previous close
        if pending:
            for order in pending:
                c = col[order.symbol]
                price = float(fill_values[i, c])
                volume = float(volume_values[i, c])
                pre_equity = _equity_at(portfolio, fill_values[i], col)
                outcome = execution.fill(
                    order, now, price, volume, pre_equity, portfolio.quantity(order.symbol)
                )
                if isinstance(outcome, Fill):
                    portfolio.apply_fill(outcome)
                else:
                    rejections.append(outcome)
            pending = []

        # 3. symbols that left the universe are liquidated at this bar's fill price
        for symbol, pos in list(portfolio.positions.items()):
            if pos.quantity != 0 and symbol not in members:
                price = float(fill_values[i, col[symbol]])
                if not (math.isfinite(price) and price > 0):
                    price = pos.last_price  # last mark; a delisting with no exit print
                outcome = execution.fill(
                    Order(symbol, OrderKind.QUANTITY, -pos.quantity, now),
                    now,
                    price,
                    float(volume_values[i, col[symbol]]),
                    portfolio.equity,
                    pos.quantity,
                )
                if isinstance(outcome, Fill):
                    portfolio.apply_fill(outcome)

        # 4. mark to market
        prices = {s: float(close_values[i, c]) for s, c in col.items()}
        portfolio.mark(prices)

        # 5. strategy decision on evaluated bars
        if i >= first_eval:
            ctx = Context(
                data=data,
                portfolio=portfolio,
                now=now,
                position_index=i,
                symbols=members,
                params=strategy_params,
            )
            if not started:
                strategy.on_start(ctx)
                started = True
            strategy.on_bar(ctx)
            if i < last_eval:
                pending = list(ctx._orders)
            else:
                rejections.extend(
                    Rejection(o, now, "submitted on the final bar") for o in ctx._orders
                )
            eval_index.append(now)
            equity_hist.append(portfolio.equity)
            cash_hist.append(portfolio.cash)
            pos_hist.append({s: p.quantity for s, p in portfolio.positions.items()})
            weight_hist.append(portfolio.weights())
            record_hist.append(dict(ctx._records))

    idx = pd.DatetimeIndex(eval_index, name="date")
    symbols = data.symbols
    positions = pd.DataFrame(pos_hist, index=idx).reindex(columns=symbols).fillna(0.0)
    weights = pd.DataFrame(weight_hist, index=idx).reindex(columns=symbols).fillna(0.0)
    records = pd.DataFrame(record_hist, index=idx)

    bench: pd.Series | None = None
    if benchmark is not None:
        if benchmark not in data.symbols:
            raise ValueError(f"benchmark {benchmark!r} not in data")
        b = data.close[benchmark].pct_change().loc[idx[0] : idx[-1]]
        b = b.copy()
        b.iloc[0] = 0.0
        bench = b.rename("benchmark_return")

    return BacktestResult(
        strategy_name=strategy.name,
        params=strategy_params,
        equity=pd.Series(equity_hist, index=idx, name="equity"),
        cash=pd.Series(cash_hist, index=idx, name="cash"),
        positions=positions,
        weights=weights,
        fills=list(portfolio.fills),
        rejections=rejections,
        round_trips=list(portfolio.round_trips),
        records=records,
        benchmark_returns=bench,
        rf=rf_series.loc[idx],
        initial_capital=float(initial_capital),
        total_commission=portfolio.total_commission,
        total_slippage=portfolio.total_slippage,
    )


def _equity_at(portfolio: Portfolio, prices_row: Any, col: Mapping[str, int]) -> float:
    """Equity with every position valued at the given row of fill prices (NaN -> last mark)."""
    total = portfolio.cash
    for symbol, pos in portfolio.positions.items():
        if pos.quantity == 0:
            continue
        p = float(prices_row[col[symbol]])
        if not (math.isfinite(p) and p > 0):
            p = pos.last_price
        total += pos.quantity * p
    return total


__all__ = ["PointInTimeUniverse", "run_backtest"]
