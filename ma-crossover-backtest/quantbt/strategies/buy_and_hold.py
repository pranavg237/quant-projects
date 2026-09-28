from __future__ import annotations

from quantbt.strategy import Context, Strategy


class BuyAndHold(Strategy):
    """Equal-weight buy-and-hold of the universe, bought once at the first bar's next open."""

    name = "buy_and_hold"

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight
        self._done = False

    def on_bar(self, ctx: Context) -> None:
        if self._done:
            return
        tradable = [s for s in ctx.symbols if ctx.has_price(s)]
        if not tradable:
            return
        w = self.weight / len(tradable)
        for s in tradable:
            ctx.order_target_weight(s, w)
        self._done = True
