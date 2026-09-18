"""Dual moving-average crossover, long/flat, one symbol."""

from __future__ import annotations

from quantbt.strategy import Context, Strategy


class MACrossover(Strategy):
    """Long when the ``short_window`` MA is above the ``long_window`` MA, otherwise in cash.

    Orders are sent only when the desired state changes, so the position is never
    "topped up" as equity drifts. This is the framework port of
    :func:`quantbt.vectorized.backtest_ma_crossover`.
    """

    name = "ma_crossover"

    def __init__(self, symbol: str, short_window: int = 50, long_window: int = 200) -> None:
        if not 0 < short_window < long_window:
            raise ValueError("need 0 < short_window < long_window")
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self.warmup = long_window
        self._target = 0

    def on_bar(self, ctx: Context) -> None:
        close = ctx.history("close", self.long_window, [self.symbol])[self.symbol]
        if len(close) < self.long_window or close.isna().any():
            desired, spread = 0, float("nan")
        else:
            fast = float(close.iloc[-self.short_window :].mean())
            slow = float(close.mean())
            desired, spread = (1 if fast > slow else 0), fast / slow - 1.0
        ctx.record(signal=desired, fast_over_slow=spread)
        if desired != self._target:
            ctx.order_target_weight(self.symbol, float(desired))
            self._target = desired
