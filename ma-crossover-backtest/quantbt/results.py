"""Results layer: everything a backtest produced, plus derived views and the metric summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.execution import Fill, Rejection
from quantbt.portfolio import RoundTrip


@dataclass(frozen=True)
class BacktestResult:
    """Output of :func:`quantbt.engine.run_backtest`.

    All frames cover the evaluation window only (warm-up bars are excluded).
    """

    strategy_name: str
    params: dict[str, Any]
    equity: pd.Series  # marked at each close
    cash: pd.Series
    positions: pd.DataFrame  # share quantities
    weights: pd.DataFrame  # market value / equity, at each close
    fills: list[Fill]
    rejections: list[Rejection]
    round_trips: list[RoundTrip]
    records: pd.DataFrame  # strategy diagnostics via ctx.record
    benchmark_returns: pd.Series | None
    rf: pd.Series  # per-period cash rate used
    initial_capital: float
    total_commission: float
    total_slippage: float

    @property
    def returns(self) -> pd.Series:
        """Simple periodic returns; the first period is measured from ``initial_capital``."""
        prev = self.equity.shift(1)
        prev.iloc[0] = self.initial_capital
        return (self.equity / prev - 1.0).rename("strategy_return")

    @property
    def growth(self) -> pd.Series:
        return (self.equity / self.initial_capital).rename("growth")

    @property
    def drawdown(self) -> pd.Series:
        return metrics.drawdown_series(self.growth)

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "symbol",
                    "quantity",
                    "price",
                    "commission",
                    "reference_price",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "timestamp": f.timestamp,
                    "symbol": f.symbol,
                    "quantity": f.quantity,
                    "price": f.price,
                    "commission": f.commission,
                    "reference_price": f.reference_price,
                }
                for f in self.fills
            ]
        )

    def trades_frame(self) -> pd.DataFrame:
        cols = [
            "symbol",
            "entry",
            "exit",
            "direction",
            "quantity",
            "entry_price",
            "exit_price",
            "pnl",
            "return_pct",
            "bars_held",
        ]
        if not self.round_trips:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([{c: getattr(t, c) for c in cols} for t in self.round_trips])

    def trade_pnls(self) -> np.ndarray:
        return np.array([t.pnl for t in self.round_trips], dtype=float)

    def summary(self) -> pd.Series:
        s = metrics.summary(
            self.returns,
            rf=self.rf,
            weights=self.weights,
            trade_pnls=self.trade_pnls(),
            benchmark=self.benchmark_returns,
        )
        s["total_commission"] = self.total_commission
        s["total_slippage"] = self.total_slippage
        s["cost_drag_pct"] = (self.total_commission + self.total_slippage) / self.initial_capital
        s["rejected_orders"] = len(self.rejections)
        return s

    def to_frame(self) -> pd.DataFrame:
        """Equity, cash, returns, drawdown and (if any) benchmark, one row per bar."""
        frame = pd.DataFrame(
            {
                "equity": self.equity,
                "cash": self.cash,
                "return": self.returns,
                "drawdown": self.drawdown,
                "gross_exposure": self.weights.abs().sum(axis=1),
            }
        )
        if self.benchmark_returns is not None:
            frame["benchmark_return"] = self.benchmark_returns
        return frame
