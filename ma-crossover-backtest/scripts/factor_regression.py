"""Regress a daily return series on Fama-French factors (CAPM, FF3, Carhart, FF5, FF6).

Usage: python scripts/factor_regression.py reports/walk_forward_ma/oos_returns.csv [--name NAME]
The CSV needs a date index and one column of simple daily returns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quantbt.data import add_live_data_flag, use_live_data
from quantbt.factors import compare_models, factor_regression, load_factors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--name", default=None)
    parser.add_argument("--out", default=None, help="CSV path for the comparison table")
    add_live_data_flag(parser)
    args = parser.parse_args()
    use_live_data(args.live_data)

    frame = pd.read_csv(args.csv, index_col=0, parse_dates=True)
    returns = frame.iloc[:, 0].rename(args.name or Path(args.csv).stem)
    factors = load_factors("ff6", "daily", start=returns.index[0], end=returns.index[-1])
    missing = returns.index.difference(factors.index)
    if len(missing):
        print(f"note: {len(missing)} return dates have no factor data and are dropped")

    for model in ("ff3", "ff5"):
        print(factor_regression(returns, factors, model).summary())
        print()
    table = compare_models(returns, factors, ("capm", "ff3", "carhart", "ff5", "ff6"))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    if args.out:
        table.to_csv(args.out)


if __name__ == "__main__":
    main()
