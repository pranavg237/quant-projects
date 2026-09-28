"""Bollinger / z-score mean reversion across a universe."""

from __future__ import annotations

import numpy as np

from quantbt.strategies._common import changed
from quantbt.strategy import Context, Strategy


class MeanReversion(Strategy):
    """Buy names whose close is more than ``entry_z`` standard deviations below the
    ``window``-bar mean; sell when the z-score recovers to ``-exit_z``. With
    ``long_only=False`` the mirror image is shorted. At most ``max_positions`` names
    are held, each at ``1 / max_positions`` of equity, lowest z-scores first.
    """

    name = "mean_reversion"

    def __init__(
        self,
        window: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        max_positions: int = 5,
        long_only: bool = True,
        max_hold: int | None = None,
    ) -> None:
        if window < 5 or entry_z <= exit_z or max_positions < 1:
            raise ValueError("need window >= 5, entry_z > exit_z, max_positions >= 1")
        self.window = window
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_positions = max_positions
        self.long_only = long_only
        self.max_hold = max_hold
        self.warmup = window
        self._targets: dict[str, float] = {}
        self._entered: dict[str, int] = {}

    def on_bar(self, ctx: Context) -> None:
        closes = ctx.history("close", self.window)
        mean = closes.mean()
        std = closes.std(ddof=1)
        last = closes.iloc[-1]
        z = ((last - mean) / std).replace([np.inf, -np.inf], np.nan)
        weight = 1.0 / self.max_positions
        new: dict[str, float] = {}

        # keep positions that have not hit their exit
        for s, w in self._targets.items():
            if s not in ctx.symbols or s not in z.index or not np.isfinite(z[s]):
                continue  # left the universe or no data: drop it (engine liquidates)
            held = ctx.position_index - self._entered.get(s, ctx.position_index)
            if self.max_hold is not None and held >= self.max_hold:
                continue
            if (w > 0 and z[s] < -self.exit_z) or (w < 0 and z[s] > self.exit_z):
                new[s] = w

        # add new entries, most stretched first
        candidates = [s for s in ctx.symbols if s not in new and s in z.index and np.isfinite(z[s])]
        longs = sorted((s for s in candidates if z[s] <= -self.entry_z), key=lambda s: z[s])
        shorts = (
            []
            if self.long_only
            else sorted((s for s in candidates if z[s] >= self.entry_z), key=lambda s: -z[s])
        )
        for s in longs:
            if len(new) >= self.max_positions:
                break
            new[s] = weight
        for s in shorts:
            if len(new) >= self.max_positions:
                break
            new[s] = -weight

        ctx.record(n_positions=len(new), min_z=float(z.min()) if len(z) else float("nan"))
        if changed(new, self._targets):
            for s in new:
                if s not in self._targets:
                    self._entered[s] = ctx.position_index
            for s in list(self._entered):
                if s not in new:
                    del self._entered[s]
            ctx.order_target_weights(new, close_others=True)
            self._targets = new
