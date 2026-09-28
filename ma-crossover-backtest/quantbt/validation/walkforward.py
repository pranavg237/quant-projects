"""Grid search and walk-forward optimisation.

Walk-forward is the only place in this package where parameters get chosen, and it
chooses them on a training window that ends *before* the window on which they are
scored. The stitched out-of-sample series is what a strategy "would have done" if the
parameter search had been re-run at each step with only the data then available.

In-sample results are returned alongside for the honest comparison: the gap between
the in-sample best and the out-of-sample stitched result is the overfitting penalty.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.data import PriceData
from quantbt.engine import run_backtest
from quantbt.results import BacktestResult
from quantbt.strategy import Strategy

StrategyFactory = Callable[[Mapping[str, Any]], Strategy]
Objective = Callable[[BacktestResult], float]


def expand_grid(
    grid: Mapping[str, Sequence[Any]],
    constraint: Callable[[Mapping[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    """Cartesian product of a parameter grid, optionally filtered by ``constraint``."""
    keys = list(grid)
    combos = [dict(zip(keys, values, strict=True)) for values in itertools.product(*grid.values())]
    if constraint is not None:
        combos = [c for c in combos if constraint(c)]
    if not combos:
        raise ValueError("parameter grid is empty after applying the constraint")
    return combos


def sharpe_objective(result: BacktestResult) -> float:
    value = metrics.sharpe(result.returns, rf=result.rf)
    return float(value) if np.isfinite(value) else -np.inf


OBJECTIVES: dict[str, Objective] = {
    "sharpe": sharpe_objective,
    "sortino": lambda r: float(metrics.sortino(r.returns, rf=r.rf)),
    "calmar": lambda r: float(metrics.calmar(r.growth)),
    "total_return": lambda r: float(r.growth.iloc[-1] - 1.0),
}


def _objective(objective: str | Objective) -> Objective:
    if callable(objective):
        return objective
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {list(OBJECTIVES)}")
    return OBJECTIVES[objective]


@dataclass(frozen=True)
class GridResult:
    """Every configuration's in-sample score and return series over one window."""

    table: pd.DataFrame  # one row per combo: params..., objective, plus summary metrics
    returns: pd.DataFrame  # T x n_combos, column i = combo i
    combos: list[dict[str, Any]]
    best_index: int

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self.combos[self.best_index])


def grid_search(
    data: PriceData,
    factory: StrategyFactory,
    grid: Mapping[str, Sequence[Any]],
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    objective: str | Objective = "sharpe",
    constraint: Callable[[Mapping[str, Any]], bool] | None = None,
    run_kwargs: Mapping[str, Any] | None = None,
) -> GridResult:
    """Run every combination in ``grid`` over ``[start, end]`` and rank by ``objective``."""
    combos = expand_grid(grid, constraint)
    score = _objective(objective)
    kwargs = dict(run_kwargs or {})
    rows: list[dict[str, Any]] = []
    series: dict[int, pd.Series] = {}
    for i, params in enumerate(combos):
        result = run_backtest(data, factory(params), start=start, end=end, params=params, **kwargs)
        s = result.summary()
        row: dict[str, Any] = {**params, "objective": score(result)}
        for key in (
            "total_return",
            "cagr",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "turnover",
            "trades",
        ):
            row[key] = s.get(key, np.nan)
        rows.append(row)
        series[i] = result.returns
    table = pd.DataFrame(rows)
    best = int(table["objective"].to_numpy().argmax())
    return GridResult(table=table, returns=pd.DataFrame(series), combos=combos, best_index=best)


@dataclass(frozen=True)
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict[str, Any]
    train_objective: float
    test_objective: float
    train_table: pd.DataFrame = field(repr=False)


@dataclass(frozen=True)
class WalkForwardResult:
    folds: list[Fold]
    oos_returns: pd.Series  # stitched out-of-sample returns
    oos_weights: pd.DataFrame
    oos_rf: pd.Series
    is_best_params: dict[str, Any]  # best on the full period, for comparison only
    is_returns: pd.Series  # full-period returns with is_best_params (in-sample)
    objective_name: str

    def fold_table(self) -> pd.DataFrame:
        rows = []
        for f in self.folds:
            rows.append(
                {
                    "train_start": f.train_start.date(),
                    "train_end": f.train_end.date(),
                    "test_start": f.test_start.date(),
                    "test_end": f.test_end.date(),
                    **f.params,
                    "train_obj": f.train_objective,
                    "test_obj": f.test_objective,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> pd.DataFrame:
        """Side-by-side metrics: stitched out-of-sample vs. in-sample-optimised."""
        oos = metrics.summary(self.oos_returns, rf=self.oos_rf, weights=self.oos_weights)
        is_ = metrics.summary(
            self.is_returns, rf=self.oos_rf.reindex(self.is_returns.index).fillna(0.0)
        )
        frame = pd.DataFrame({"out_of_sample": oos, "in_sample_best": is_})
        return frame

    @property
    def parameter_stability(self) -> pd.DataFrame:
        """How the chosen parameters moved from fold to fold (a proxy for robustness)."""
        return pd.DataFrame(
            [f.params for f in self.folds], index=[f.test_start for f in self.folds]
        )


def walk_forward(
    data: PriceData,
    factory: StrategyFactory,
    grid: Mapping[str, Sequence[Any]],
    *,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str | None = None,
    train_years: float = 3.0,
    test_years: float = 1.0,
    anchored: bool = False,
    objective: str | Objective = "sharpe",
    constraint: Callable[[Mapping[str, Any]], bool] | None = None,
    run_kwargs: Mapping[str, Any] | None = None,
) -> WalkForwardResult:
    """Rolling (or anchored) walk-forward optimisation.

    The first test window begins ``train_years`` after ``start``; each fold trains on the
    preceding ``train_years`` (or everything since ``start`` if ``anchored``), picks the
    best grid point by ``objective`` and scores it on the next ``test_years``. Bars
    before ``start`` are warm-up history only.
    """
    index = data.index
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end is not None else index[-1]
    if start_ts >= end_ts:
        raise ValueError("start must be before end")
    score = _objective(objective)
    obj_name = objective if isinstance(objective, str) else getattr(objective, "__name__", "custom")
    kwargs = dict(run_kwargs or {})

    def offset(years: float) -> pd.DateOffset:
        return pd.DateOffset(months=round(years * 12))

    folds: list[Fold] = []
    oos_parts: list[pd.Series] = []
    weight_parts: list[pd.DataFrame] = []
    rf_parts: list[pd.Series] = []
    test_start = start_ts + offset(train_years)
    if test_start >= end_ts:
        raise ValueError("not enough data for one training window")
    while test_start < end_ts:
        test_end = min(test_start + offset(test_years) - pd.Timedelta(days=1), end_ts)
        train_start = start_ts if anchored else test_start - offset(train_years)
        train_end = test_start - pd.Timedelta(days=1)
        if not ((index >= test_start) & (index <= test_end)).any():
            break
        gr = grid_search(
            data,
            factory,
            grid,
            start=train_start,
            end=train_end,
            objective=score,
            constraint=constraint,
            run_kwargs=kwargs,
        )
        params = gr.best_params
        test_result = run_backtest(
            data, factory(params), start=test_start, end=test_end, params=params, **kwargs
        )
        folds.append(
            Fold(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                params=params,
                train_objective=float(gr.table["objective"].iloc[gr.best_index]),
                test_objective=score(test_result),
                train_table=gr.table,
            )
        )
        oos_parts.append(test_result.returns)
        weight_parts.append(test_result.weights)
        rf_parts.append(test_result.rf)
        test_start = test_start + offset(test_years)

    oos = pd.concat(oos_parts).rename("oos_return")
    if oos.index.has_duplicates:  # pragma: no cover - guarded by construction
        raise RuntimeError("overlapping test windows")

    full = grid_search(
        data,
        factory,
        grid,
        start=folds[0].test_start,
        end=end_ts,
        objective=score,
        constraint=constraint,
        run_kwargs=kwargs,
    )
    return WalkForwardResult(
        folds=folds,
        oos_returns=oos,
        oos_weights=pd.concat(weight_parts).fillna(0.0),
        oos_rf=pd.concat(rf_parts),
        is_best_params=full.best_params,
        is_returns=full.returns[full.best_index].rename("is_return"),
        objective_name=str(obj_name),
    )
