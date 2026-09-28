"""Time-series momentum (Moskowitz, Ooi & Pedersen 2012) with volatility scaling."""

from __future__ import annotations

import numpy as np

from quantbt.strategies._common import is_rebalance_bar, realized_vol
from quantbt.strategy import Context, Strategy


class TimeSeriesMomentum(Strategy):
    """Long assets with positive trailing return, short the rest, each scaled to ``vol_target``.

    Signal: total return from ``t - lookback`` to ``t - skip`` (``skip`` days are excluded
    to sidestep short-term reversal). Weight: ``sign * vol_target / realised_vol / n``
    with ``n`` the number of eligible assets, capped per asset and in gross total.
    """

    name = "tsmom"

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        vol_lookback: int = 60,
        vol_target: float = 0.10,
        max_weight: float = 0.5,
        max_gross: float = 1.5,
        rebalance: str | int = "monthly",
        long_only: bool = False,
    ) -> None:
        if lookback <= skip or vol_lookback < 5:
            raise ValueError("need lookback > skip and vol_lookback >= 5")
        self.lookback = lookback
        self.skip = skip
        self.vol_lookback = vol_lookback
        self.vol_target = vol_target
        self.max_weight = max_weight
        self.max_gross = max_gross
        self.rebalance = rebalance
        self.long_only = long_only
        self.warmup = max(lookback, vol_lookback) + 1
        self._targets: dict[str, float] = {}

    def on_bar(self, ctx: Context) -> None:
        if not is_rebalance_bar(ctx, self.rebalance):
            return
        # Enough bars for both the momentum window and the volatility window: asking for
        # `lookback + 1` alone silently shortens the vol estimate whenever vol_lookback
        # is the longer of the two.
        bars = max(self.lookback, self.vol_lookback) + 1
        closes = ctx.history("close", bars)
        # Eligibility is judged on the momentum window itself, not on the (longer) frame:
        # a name with gaps early in the frame must still have a complete signal window.
        signal_window = closes.iloc[-(self.lookback + 1) :]
        eligible = [s for s in ctx.symbols if signal_window[s].notna().all()]
        weights: dict[str, float] = {}
        if eligible:
            vols = realized_vol(closes[eligible], self.vol_lookback)
            n = len(eligible)
            for s in eligible:
                c = closes[s]
                past = float(c.iloc[-1 - self.skip] / c.iloc[-1 - self.lookback] - 1.0)
                vol = float(vols[s])
                if not np.isfinite(past) or not np.isfinite(vol) or vol <= 0:
                    continue
                sign = 1.0 if past > 0 else (0.0 if self.long_only else -1.0)
                if sign == 0.0:
                    continue
                w = sign * min(self.vol_target / vol, self.max_weight * n) / n
                weights[s] = float(np.clip(w, -self.max_weight, self.max_weight))
            gross = sum(abs(w) for w in weights.values())
            if gross > self.max_gross:
                weights = {s: w * self.max_gross / gross for s, w in weights.items()}
        ctx.record(n_eligible=len(eligible), gross=sum(abs(w) for w in weights.values()))
        # Periodic strategies rebalance back to target on every rebalance bar, even when
        # the targets are unchanged, so price drift does not accumulate between them.
        ctx.order_target_weights(weights, close_others=True)
        self._targets = weights
