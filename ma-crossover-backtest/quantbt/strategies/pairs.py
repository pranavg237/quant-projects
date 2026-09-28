"""Pairs trading with Engle-Granger cointegration selection.

Every ``refit_every`` bars the candidate pairs are tested on the trailing ``formation``
window of log prices. Pairs whose Engle-Granger p-value is below ``pvalue`` are kept
(best first, at most ``max_pairs``); the hedge ratio and the spread's mean and standard
deviation are frozen from that window. Entries and exits are then made on the
z-score of the live spread against the frozen statistics, so nothing in the trading
rule uses data past the current bar.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from quantbt.strategies._common import changed
from quantbt.strategy import Context, Strategy


@dataclass(frozen=True)
class PairModel:
    a: str
    b: str
    hedge: float  # log(a) ~ alpha + hedge * log(b)
    intercept: float
    mean: float
    std: float
    pvalue: float

    def spread(self, log_a: float, log_b: float) -> float:
        return log_a - self.hedge * log_b - self.intercept

    def zscore(self, log_a: float, log_b: float) -> float:
        return (self.spread(log_a, log_b) - self.mean) / self.std if self.std > 0 else float("nan")


def fit_pair(log_a: pd.Series, log_b: pd.Series, a: str, b: str) -> PairModel:
    """OLS hedge ratio plus Engle-Granger cointegration p-value on a formation window."""
    x = np.column_stack([np.ones(len(log_b)), log_b.to_numpy()])
    coef, *_ = np.linalg.lstsq(x, log_a.to_numpy(), rcond=None)
    intercept, hedge = float(coef[0]), float(coef[1])
    spread = log_a.to_numpy() - hedge * log_b.to_numpy() - intercept
    _, pvalue, _ = coint(log_a.to_numpy(), log_b.to_numpy(), trend="c")
    return PairModel(
        a=a,
        b=b,
        hedge=hedge,
        intercept=intercept,
        mean=float(spread.mean()),
        std=float(spread.std(ddof=1)),
        pvalue=float(pvalue),
    )


class PairsTrading(Strategy):
    name = "pairs"

    def __init__(
        self,
        pairs: Sequence[tuple[str, str]],
        formation: int = 252,
        refit_every: int = 63,
        pvalue: float = 0.05,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 4.0,
        max_pairs: int = 5,
        weight_per_pair: float = 0.2,
    ) -> None:
        if formation < 60 or refit_every < 1 or not 0 <= exit_z < entry_z < stop_z:
            raise ValueError("bad parameters")
        self.pairs = list(pairs)
        self.formation = formation
        self.refit_every = refit_every
        self.pvalue = pvalue
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_pairs = max_pairs
        self.weight_per_pair = weight_per_pair
        self.warmup = formation
        self._models: list[PairModel] = []
        self._state: dict[tuple[str, str], int] = {}
        self._targets: dict[str, float] = {}
        self._bars_since_fit = refit_every  # refit on the first bar

    @property
    def models(self) -> list[PairModel]:
        return list(self._models)

    def _refit(self, ctx: Context) -> None:
        closes = ctx.history("close", self.formation)
        fitted: list[PairModel] = []
        for a, b in self.pairs:
            if a not in ctx.symbols or b not in ctx.symbols:
                continue
            window = closes[[a, b]].dropna()
            if len(window) < self.formation:
                continue
            log_a, log_b = np.log(window[a]), np.log(window[b])
            model = fit_pair(log_a, log_b, a, b)
            if model.pvalue < self.pvalue and model.std > 0:
                fitted.append(model)
        fitted.sort(key=lambda m: m.pvalue)
        self._models = fitted[: self.max_pairs]
        keep = {(m.a, m.b) for m in self._models}
        self._state = {k: v for k, v in self._state.items() if k in keep}

    def on_bar(self, ctx: Context) -> None:
        if self._bars_since_fit >= self.refit_every:
            self._refit(ctx)
            self._bars_since_fit = 0
        self._bars_since_fit += 1

        weights: dict[str, float] = {}
        zs: dict[str, float] = {}
        for m in self._models:
            key = (m.a, m.b)
            if not (ctx.has_price(m.a) and ctx.has_price(m.b)):
                self._state[key] = 0
                continue
            z = m.zscore(np.log(ctx.price(m.a)), np.log(ctx.price(m.b)))
            zs[f"z_{m.a}_{m.b}"] = z
            state = self._state.get(key, 0)
            if not np.isfinite(z):
                state = 0
            elif state == 0:
                if z <= -self.entry_z:
                    state = 1  # spread too low: long a, short b
                elif z >= self.entry_z:
                    state = -1
            elif abs(z) <= self.exit_z or abs(z) >= self.stop_z:
                state = 0
            self._state[key] = state
            if state != 0:
                half = 0.5 * self.weight_per_pair
                weights[m.a] = weights.get(m.a, 0.0) + state * half
                weights[m.b] = weights.get(m.b, 0.0) - state * half
        weights = {s: w for s, w in weights.items() if abs(w) > 1e-12}
        ctx.record(
            n_pairs=len(self._models), n_open=sum(1 for v in self._state.values() if v), **zs
        )
        if changed(weights, self._targets):
            ctx.order_target_weights(weights, close_others=True)
            self._targets = weights
