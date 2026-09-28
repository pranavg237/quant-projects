"""Deflated Sharpe ratios with an honest trial count: every configuration of all five strategies.

Reads the pipeline's output in reports/strategies/ (run scripts/run_strategies.py first)
and writes reports/multiple_testing/:

* trials.csv            every configuration tried and its full-span excess Sharpe
* dsr.csv               PSR and DSR of each strategy's walk-forward out-of-sample series,
                        and of the single best in-sample configuration, at N = all trials
* dsr_sensitivity.csv   the DSR of the MA crossover at a range of trial counts

The in-sample best configuration's return series is not stored by the pipeline, so it is
recomputed here with the vectorised backtest (SPY data from the local cache) and checked
against the Sharpe the engine wrote to grid.csv.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.data import add_live_data_flag, load_yahoo, use_live_data
from quantbt.research.multiple_testing import collect_trials, dsr_row, dsr_sensitivity
from quantbt.research.runner import load_rf
from quantbt.validation import sharpe_std_error
from quantbt.vectorized import backtest_ma_crossover

STRATEGIES = ["ma_crossover", "tsmom", "xsmom", "mean_reversion", "pairs"]
HEATMAP_GRID = Path("reports/sensitivity/ma_sharpe_grid.csv")  # scripts/ma_sensitivity.py


def _oos_excess(root: Path, name: str) -> pd.Series:
    oos = pd.read_csv(root / name / "oos_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    rf = pd.read_csv(root / name / "rf.csv", index_col=0, parse_dates=True).iloc[:, 0]
    return (oos - rf.reindex(oos.index).fillna(0.0)).rename(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="reports/strategies")
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--out", default="reports/multiple_testing")
    add_live_data_flag(parser)
    args = parser.parse_args()
    use_live_data(args.live_data)
    t0 = time.time()
    root = Path(args.root)

    results = pd.read_csv(root / "results.csv", index_col="strategy")
    if sorted(results.index) != sorted(STRATEGIES):
        raise SystemExit(f"results.csv has {list(results.index)}, expected {STRATEGIES}")

    trials = collect_trials(root, STRATEGIES)
    n_trials = len(trials)
    var_all = float(np.var(trials["sharpe"], ddof=1))
    per_strategy = trials.groupby("strategy")["sharpe"].agg(["count", "min", "max"])
    print(f"{n_trials} configurations across {len(STRATEGIES)} strategies:")
    print(per_strategy.loc[STRATEGIES].to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"cross-trial variance of annualised Sharpe: {var_all:.4f} (sd {np.sqrt(var_all):.3f})")

    rows = []
    for name in STRATEGIES:
        row = dsr_row(_oos_excess(root, name), n_trials, var_all)
        rows.append({"series": f"{name} walk-forward OOS", **row})

    # The textbook DSR: the single best configuration of all trials, over the span on
    # which it was the best. It must be an MA crossover config to be recomputed here.
    best = trials.iloc[int(trials["sharpe"].to_numpy().argmax())]
    if best["strategy"] != "ma_crossover":
        raise SystemExit(f"best trial is {best['strategy']}; recompute its returns by hand")
    params = dict(kv.split("=") for kv in str(best["params"]).split(", "))
    short, long = int(params["short"]), int(params["long"])
    start = results.loc["ma_crossover", "oos_start"]
    data = load_yahoo("SPY", start="1998-01-01", end=args.end)
    rf = load_rf("1998-01-01", args.end)
    vec = backtest_ma_crossover(data, short, long, start=start, end=args.end, rf=rf)
    best_excess = vec.returns - rf.reindex(vec.returns.index).fillna(0.0)
    gap = abs(metrics.sharpe(best_excess) - float(best["sharpe"]))
    print(
        f"in-sample best: MA {short}/{long}, Sharpe {best['sharpe']:.4f} (recomputed gap {gap:.1e})"
    )
    if gap > 1e-4:
        raise SystemExit("recomputed in-sample best does not match grid.csv")
    rows.append(
        {"series": f"ma_crossover {short}/{long} in-sample best of all trials",
         **dsr_row(best_excess, n_trials, var_all)}
    )  # fmt: skip
    ma_grid = trials.loc[trials["strategy"] == "ma_crossover", "sharpe"]
    old = dsr_row(best_excess, len(ma_grid), float(np.var(ma_grid, ddof=1)))
    rows.append({"series": f"ma_crossover {short}/{long}, previous method (MA grid only)", **old})
    table = pd.DataFrame(rows)

    # If the heatmap had been used to choose parameters, every one of its cells would be a
    # trial too. Its cells include the 17 MA grid points, so they are not counted twice.
    heat = pd.read_csv(HEATMAP_GRID)
    ma_keys = {
        (int(r["short"]), int(r["long"]))
        for r in pd.read_csv(root / "ma_crossover" / "grid.csv").to_dict("records")
    }
    new_cells = sum(
        (int(f), int(s)) not in ma_keys for f, s in zip(heat["fast"], heat["slow"], strict=True)
    )
    n_heat = n_trials + new_cells
    counts = {
        "N=1 (no correction: the PSR)": 1,
        "N=5 (one per strategy)": 5,
        f"N={len(ma_grid)} (MA grid only, previous count)": len(ma_grid),
        f"N={n_trials} (all five grids)": n_trials,
        f"N={n_heat} (... plus every heatmap cell)": n_heat,
    }
    ppy = metrics.periods_per_year(best_excess.index)
    var_ma = float(np.var(ma_grid, ddof=1))
    var_se = (sharpe_std_error(best_excess) * np.sqrt(ppy)) ** 2
    variances = {
        f"V={var_ma:.4f} (MA grid only, previous)": var_ma,
        f"V={var_se:.4f} (sampling variance of one Sharpe)": var_se,
        f"V={var_all:.4f} (all {n_trials} trials)": var_all,
    }
    sens = pd.concat(
        [
            dsr_sensitivity(_oos_excess(root, "ma_crossover"), counts, variances).assign(
                series="ma_crossover walk-forward OOS"
            ),
            dsr_sensitivity(best_excess, counts, variances).assign(
                series=f"ma_crossover {short}/{long} in-sample best"
            ),
        ],
        ignore_index=True,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trials.to_csv(out / "trials.csv", index=False, float_format="%.6f")
    table.to_csv(out / "dsr.csv", index=False, float_format="%.6f")
    sens.to_csv(out / "dsr_sensitivity.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 200)
    cols = ["series", "sharpe", "psr", "n_trials", "sr0_annual", "dsr"]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for series, part in sens.groupby("series", sort=False):
        print(f"\nDSR of {series}: rows = trial count, columns = cross-trial variance")
        pivot = part.pivot(index="n_assumption", columns="var_assumption", values="dsr")
        pivot = pivot.reindex(index=list(counts), columns=list(variances))
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"written to {out}/ in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
