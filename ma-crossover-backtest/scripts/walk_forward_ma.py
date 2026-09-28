"""Walk-forward optimisation and overfitting diagnostics for the SPY MA crossover.

Rolling 5-year training windows pick (short, long) by Sharpe; the next year is scored
out of sample. The full-period in-sample best is shown for contrast, and the grid's
Sharpe distribution feeds the deflated Sharpe ratio and the PBO estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantbt import metrics
from quantbt.data import add_live_data_flag, load_yahoo, use_live_data
from quantbt.strategies import MACrossover
from quantbt.validation import grid_search, overfit_report, walk_forward

GRID = {"short": [10, 20, 50, 100], "long": [50, 100, 150, 200, 250]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--train-years", type=float, default=5.0)
    parser.add_argument("--test-years", type=float, default=1.0)
    parser.add_argument("--anchored", action="store_true")
    parser.add_argument("--out", default="reports/walk_forward_ma")
    add_live_data_flag(parser)
    args = parser.parse_args()
    use_live_data(args.live_data)

    data = load_yahoo(
        args.ticker, start=pd.Timestamp(args.start) - pd.DateOffset(years=2), end=args.end
    )
    factory = lambda p: MACrossover(args.ticker, p["short"], p["long"])  # noqa: E731
    constraint = lambda p: p["short"] < p["long"]  # noqa: E731

    wf = walk_forward(
        data,
        factory,
        GRID,
        start=args.start,
        end=args.end,
        train_years=args.train_years,
        test_years=args.test_years,
        anchored=args.anchored,
        constraint=constraint,
        run_kwargs={"benchmark": args.ticker},
    )
    pd.set_option("display.width", 160)
    print(
        f"{args.ticker} MA crossover walk-forward ({'anchored' if args.anchored else 'rolling'} "
        f"{args.train_years}y train / {args.test_years}y test)"
    )
    print(wf.fold_table().to_string(index=False))
    print(f"\nOut-of-sample (stitched) vs full-period in-sample best {wf.is_best_params}:")
    print(
        wf.summary()
        .loc[["total_return", "cagr", "ann_vol", "sharpe", "max_drawdown", "turnover"]]
        .to_string()
    )

    # overfitting diagnostics using the full-period grid
    grid = grid_search(
        data, factory, GRID, start=wf.folds[0].test_start, end=args.end, constraint=constraint
    )
    trial_sharpes = grid.table["sharpe"].to_numpy()
    rep = overfit_report(
        grid.returns[grid.best_index],
        trials_sharpe_annual=trial_sharpes,
        grid_returns=grid.returns,
        n_samples=1000,
    )
    print("\nOverfitting diagnostics for the in-sample best configuration:")
    print(rep.to_series().to_string())
    rep_oos = overfit_report(wf.oos_returns, n_samples=1000)
    print("\nBootstrap for the walk-forward out-of-sample series:")
    print(rep_oos.to_series().to_string())

    bench = data.close[args.ticker].pct_change().loc[wf.oos_returns.index]
    print("\nBuy-and-hold over the same out-of-sample span:")
    print(
        metrics.format_summary(
            metrics.summary(bench)[["total_return", "cagr", "sharpe", "max_drawdown"]]
        )
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wf.fold_table().to_csv(out / "folds.csv", index=False)
    wf.summary().to_csv(out / "oos_vs_is.csv")
    grid.table.to_csv(out / "grid.csv", index=False)
    rep.to_series().to_csv(out / "overfit_is.csv")
    rep_oos.to_series().to_csv(out / "overfit_oos.csv")
    wf.oos_returns.to_csv(out / "oos_returns.csv")
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    main()
