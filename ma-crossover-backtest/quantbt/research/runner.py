"""Research pipeline: walk-forward, overfitting diagnostics, factor regression, benchmark.

Used by ``scripts/run_strategies.py`` for every strategy so the results table in
``RESULTS.md`` is produced by one code path with one set of assumptions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.data import PriceData, load_yahoo
from quantbt.factors import compare_models, factor_regression, load_factors
from quantbt.strategy import Strategy
from quantbt.universe import from_data
from quantbt.validation import grid_search, overfit_report, walk_forward


@dataclass(frozen=True)
class StrategySpec:
    name: str
    symbols: Sequence[str]
    factory: Callable[[Mapping[str, Any]], Strategy]
    grid: Mapping[str, Sequence[Any]]
    start: str
    benchmark: str = "SPY"
    constraint: Callable[[Mapping[str, Any]], bool] | None = None
    description: str = ""
    caveats: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class SpecResult:
    name: str
    row: pd.Series
    oos_returns: pd.Series
    oos_weights: pd.DataFrame
    benchmark_returns: pd.Series
    rf: pd.Series
    folds: pd.DataFrame
    grid: pd.DataFrame
    factors: pd.DataFrame
    overfit_is: pd.Series
    overfit_oos: pd.Series
    out_dir: Path


def load_rf(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """Daily French risk-free rate (decimal per day) over the span."""
    return load_factors("ff3", "daily", start=start, end=end)["RF"].rename("RF")


def run_spec(
    spec: StrategySpec,
    *,
    end: str,
    out_root: str | Path = "reports/strategies",
    train_years: float = 5.0,
    test_years: float = 1.0,
    warmup_years: int = 2,
    rf: pd.Series | None = None,
    n_bootstrap: int = 1000,
    log: Callable[[str], None] = print,
) -> SpecResult:
    t0 = time.time()
    out_dir = Path(out_root) / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_start = pd.Timestamp(spec.start) - pd.DateOffset(years=warmup_years)
    tradable = list(spec.symbols)
    # The benchmark is loaded so its returns can be compared, but it is kept out of the
    # tradable universe unless the strategy trades it: a strategy must not be able to
    # buy the thing it is being measured against by accident.
    to_load = tradable + ([spec.benchmark] if spec.benchmark not in tradable else [])
    data: PriceData = load_yahoo(to_load, start=data_start, end=end)
    universe = from_data(data.select(tradable))
    rf_series = rf if rf is not None else load_rf(data_start, end)
    run_kwargs: dict[str, Any] = {
        "rf": rf_series,
        "benchmark": spec.benchmark,
        "universe": universe,
    }
    log(f"[{spec.name}] {len(tradable)} tradable symbols, {len(data)} bars, walk-forward...")

    wf = walk_forward(
        data,
        spec.factory,
        spec.grid,
        start=spec.start,
        end=end,
        train_years=train_years,
        test_years=test_years,
        constraint=spec.constraint,
        run_kwargs=run_kwargs,
    )
    oos = wf.oos_returns
    log(f"[{spec.name}] {len(wf.folds)} folds, OOS {oos.index[0].date()}..{oos.index[-1].date()}")

    grid = grid_search(
        data,
        spec.factory,
        spec.grid,
        start=wf.folds[0].test_start,
        end=end,
        constraint=spec.constraint,
        run_kwargs=run_kwargs,
    )
    # Every Sharpe-like statistic below is in excess of the risk-free rate, matching the
    # summary metrics; a mostly-in-cash strategy otherwise looks excellent on raw returns.
    grid_rf = rf_series.reindex(grid.returns.index).fillna(0.0)
    rep_is = overfit_report(
        grid.returns[grid.best_index],
        rf=grid_rf,
        trials_sharpe_annual=grid.table["sharpe"].to_numpy(),
        grid_returns=grid.returns,
        n_samples=n_bootstrap,
    ).to_series()
    rep_oos = overfit_report(oos, rf=wf.oos_rf, n_samples=n_bootstrap).to_series()

    factors = load_factors("ff6", "daily", start=oos.index[0], end=oos.index[-1])
    ff5 = factor_regression(oos, factors, "ff5", name=spec.name)
    ff6 = factor_regression(oos, factors, "ff6", name=spec.name)
    table = compare_models(oos, factors, ("capm", "ff3", "carhart", "ff5", "ff6"))

    bench = data.close[spec.benchmark].pct_change().reindex(oos.index).fillna(0.0)
    bench.iloc[0] = 0.0
    oos_rf = wf.oos_rf
    s_oos = metrics.summary(oos, rf=oos_rf, weights=wf.oos_weights, benchmark=bench)
    s_is = metrics.summary(wf.is_returns, rf=oos_rf.reindex(wf.is_returns.index).fillna(0.0))
    s_bench = metrics.summary(bench, rf=oos_rf)

    row = pd.Series(
        {
            "strategy": spec.name,
            "oos_start": oos.index[0].date().isoformat(),
            "oos_end": oos.index[-1].date().isoformat(),
            "folds": len(wf.folds),
            "n_symbols": len(tradable),
            "cagr": s_oos["cagr"],
            "ann_vol": s_oos["ann_vol"],
            "sharpe": s_oos["sharpe"],
            "sortino": s_oos["sortino"],
            "max_drawdown": s_oos["max_drawdown"],
            "max_dd_days": s_oos["max_dd_duration_days"],
            "turnover": s_oos["turnover"],
            "exposure": s_oos["exposure"],
            "is_sharpe": s_is["sharpe"],
            "is_params": json.dumps(wf.is_best_params),
            "bench_cagr": s_bench["cagr"],
            "bench_sharpe": s_bench["sharpe"],
            "bench_max_dd": s_bench["max_drawdown"],
            "psr": rep_oos["psr"],
            "boot_ci_lo": rep_oos["bootstrap_ci_lo"],
            "boot_ci_hi": rep_oos["bootstrap_ci_hi"],
            "pbo": rep_is["pbo"],
            "dsr_is": rep_is["dsr"],
            "ff5_alpha": ff5.alpha_annual,
            "ff5_alpha_t": ff5.alpha_tstat,
            "ff5_r2": ff5.rsquared,
            "beta_mkt": float(ff5.betas["Mkt-RF"]),
            "beta_mom": float(ff6.betas["MOM"]),
            "mom_t": float(ff6.tvalues["MOM"]),
            "ff6_alpha": ff6.alpha_annual,
            "ff6_alpha_t": ff6.alpha_tstat,
            "runtime_s": round(time.time() - t0, 1),
        }
    )

    # persist everything a tearsheet or a reader needs
    oos.to_csv(out_dir / "oos_returns.csv")
    wf.oos_weights.to_csv(out_dir / "oos_weights.csv")
    bench.rename("benchmark_return").to_csv(out_dir / "benchmark_returns.csv")
    oos_rf.rename("rf").to_csv(out_dir / "rf.csv")
    wf.fold_table().to_csv(out_dir / "folds.csv", index=False)
    grid.table.to_csv(out_dir / "grid.csv", index=False)
    table.to_csv(out_dir / "factors.csv")
    rep_is.to_csv(out_dir / "overfit_is.csv")
    rep_oos.to_csv(out_dir / "overfit_oos.csv")
    wf.summary().to_csv(out_dir / "oos_vs_is.csv")
    row.to_json(out_dir / "summary.json", indent=2)
    (out_dir / "spec.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "description": spec.description,
                "symbols": list(spec.symbols),
                "grid": {k: list(v) for k, v in spec.grid.items()},
                "start": spec.start,
                "end": end,
                "benchmark": spec.benchmark,
                "train_years": train_years,
                "test_years": test_years,
                "caveats": list(spec.caveats),
            },
            indent=2,
            default=str,
        )
    )
    log(
        f"[{spec.name}] done in {row['runtime_s']}s: OOS Sharpe {row['sharpe']:.2f}, "
        f"FF5 alpha {row['ff5_alpha']:+.1%} (t={row['ff5_alpha_t']:.2f}), PBO {row['pbo']:.2f}"
    )
    return SpecResult(
        name=spec.name,
        row=row,
        oos_returns=oos,
        oos_weights=wf.oos_weights,
        benchmark_returns=bench,
        rf=oos_rf,
        folds=wf.fold_table(),
        grid=grid.table,
        factors=table,
        overfit_is=rep_is,
        overfit_oos=rep_oos,
        out_dir=out_dir,
    )


def results_table(rows: Sequence[pd.Series]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows)).set_index("strategy")
    return frame.replace({np.nan: None})
