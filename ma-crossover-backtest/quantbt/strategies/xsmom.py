"""Cross-sectional (relative) momentum: long past winners, short past losers."""

from __future__ import annotations

from quantbt.strategies._common import is_rebalance_bar
from quantbt.strategy import Context, Strategy


class CrossSectionalMomentum(Strategy):
    """Rank the universe by ``t - lookback`` to ``t - skip`` return; long the top
    ``top_frac``, short the bottom ``top_frac`` (dollar neutral) or long-only.

    Names need ``lookback + 1`` bars of history to be ranked, so a stock enters the
    ranking a year after it lists; nothing is ever ranked on data it did not have.
    """

    name = "xsmom"

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        top_frac: float = 0.2,
        long_only: bool = False,
        rebalance: str | int = "monthly",
        gross: float = 1.0,
        min_names: int = 5,
    ) -> None:
        if lookback <= skip or not 0 < top_frac <= 0.5:
            raise ValueError("need lookback > skip and 0 < top_frac <= 0.5")
        self.lookback = lookback
        self.skip = skip
        self.top_frac = top_frac
        self.long_only = long_only
        self.rebalance = rebalance
        self.gross = gross
        self.min_names = min_names
        self.warmup = lookback + 1
        self._targets: dict[str, float] = {}

    def on_bar(self, ctx: Context) -> None:
        if not is_rebalance_bar(ctx, self.rebalance):
            return
        closes = ctx.history("close", self.lookback + 1)
        scores: dict[str, float] = {}
        for s in ctx.symbols:
            c = closes[s]
            if c.notna().sum() < self.lookback + 1:
                continue
            scores[s] = float(c.iloc[-1 - self.skip] / c.iloc[0] - 1.0)
        weights: dict[str, float] = {}
        if len(scores) >= self.min_names:
            ranked = sorted(scores, key=scores.__getitem__)
            # Cap at half the names: rounding top_frac=0.5 up on an odd count would put
            # the same symbol in both legs, and the short weight would silently win.
            n = max(1, min(round(self.top_frac * len(ranked)), len(ranked) // 2))
            longs, shorts = ranked[-n:], ranked[:n]
            if self.long_only:
                weights = dict.fromkeys(longs, self.gross / n)
            else:
                weights = dict.fromkeys(longs, 0.5 * self.gross / n)
                weights.update(dict.fromkeys(shorts, -0.5 * self.gross / n))
        ctx.record(n_ranked=len(scores), n_positions=len(weights))
        # Periodic strategies rebalance back to target on every rebalance bar, even when
        # the targets are unchanged, so price drift does not accumulate between them.
        ctx.order_target_weights(weights, close_others=True)
        self._targets = weights
