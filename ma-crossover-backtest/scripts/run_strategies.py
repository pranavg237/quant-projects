"""Run every strategy through walk-forward, overfitting diagnostics and factor regression.

Writes reports/strategies/<name>/ for each strategy and reports/strategies/results.csv.
Use --only NAME to run one strategy.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import pandas as pd

from quantbt.data import add_live_data_flag, use_live_data
from quantbt.research.runner import StrategySpec, load_rf, results_table, run_spec
from quantbt.research.universes import (
    ASSET_CLASS_ETFS,
    LARGE_CAP_STOCKS,
    PAIR_CANDIDATES,
    PAIR_SYMBOLS,
    SECTOR_ETFS,
)
from quantbt.strategies import (
    CrossSectionalMomentum,
    MACrossover,
    MeanReversion,
    PairsTrading,
    TimeSeriesMomentum,
)

SPECS: list[StrategySpec] = [
    StrategySpec(
        name="ma_crossover",
        symbols=["SPY"],
        factory=lambda p: MACrossover("SPY", p["short"], p["long"]),
        grid={"short": [10, 20, 50, 100], "long": [50, 100, 150, 200, 250]},
        constraint=lambda p: p["short"] < p["long"],
        start="2000-01-01",
        description="Long SPY when the short MA is above the long MA, else cash at the "
        "T-bill rate.",
    ),
    StrategySpec(
        name="tsmom",
        symbols=ASSET_CLASS_ETFS,
        factory=lambda p: TimeSeriesMomentum(
            lookback=p["lookback"], skip=p["skip"], vol_target=p["vol_target"]
        ),
        grid={"lookback": [63, 126, 252], "skip": [0, 21], "vol_target": [0.08, 0.12]},
        start="2005-01-01",
        description="Long/short each of 13 asset-class ETFs by the sign of its trailing return, "
        "volatility-scaled, rebalanced monthly.",
        borrow_rate=0.003,  # 30 bp p.a.: liquid ETFs trade close to general collateral
        caveats=["ETF list chosen today (selection bias)."],
    ),
    StrategySpec(
        name="xsmom",
        symbols=LARGE_CAP_STOCKS,
        factory=lambda p: CrossSectionalMomentum(
            lookback=p["lookback"], skip=p["skip"], top_frac=p["top_frac"]
        ),
        grid={"lookback": [126, 252], "skip": [0, 21], "top_frac": [0.2, 0.3]},
        start="2005-01-01",
        description="Dollar-neutral: long the top and short the bottom fraction of 70 large caps "
        "by 12-1 momentum, rebalanced monthly.",
        borrow_rate=0.005,  # 50 bp p.a.: large-cap general collateral
        caveats=[
            "SURVIVORSHIP BIAS: the 70 stocks are today's large caps; every failure since "
            "2005 is missing.",
            "Borrow is charged at a flat 50 bp; hard-to-borrow names cost far more.",
        ],
    ),
    StrategySpec(
        name="mean_reversion",
        symbols=[*SECTOR_ETFS, "SPY", "QQQ", "IWM"],
        factory=lambda p: MeanReversion(
            window=p["window"], entry_z=p["entry_z"], max_positions=p["max_positions"]
        ),
        grid={"window": [10, 20, 40], "entry_z": [1.5, 2.0, 2.5], "max_positions": [3, 5]},
        start="2005-01-01",
        description="Buy sector/index ETFs more than entry_z standard deviations below their "
        "window-day mean, exit at the mean; long only.",
        caveats=["ETF list chosen today (selection bias)."],
    ),
    StrategySpec(
        name="pairs",
        symbols=PAIR_SYMBOLS,
        factory=lambda p: PairsTrading(
            PAIR_CANDIDATES,
            formation=p["formation"],
            entry_z=p["entry_z"],
            exit_z=p["exit_z"],
        ),
        grid={"formation": [126, 252], "entry_z": [1.5, 2.0], "exit_z": [0.25, 0.5]},
        start="2005-01-01",
        description="Engle-Granger cointegration test on 17 candidate pairs every quarter; "
        "trade the z-score of the frozen spread, dollar neutral.",
        borrow_rate=0.005,
        caveats=[
            "Candidate pairs chosen by hand today (selection bias).",
            "Borrow is charged at a flat 50 bp; hard-to-borrow names cost far more.",
        ],
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--end", default="2025-08-29")
    parser.add_argument("--train-years", type=float, default=5.0)
    parser.add_argument("--test-years", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", default="reports/strategies")
    add_live_data_flag(parser)
    args = parser.parse_args()
    use_live_data(args.live_data)

    specs = [s for s in SPECS if not args.only or s.name in args.only]
    rf = load_rf("1998-01-01", args.end)
    rows = []
    for spec in specs:
        try:
            res = run_spec(
                spec,
                end=args.end,
                out_root=args.out,
                train_years=args.train_years,
                test_years=args.test_years,
                rf=rf,
                n_bootstrap=args.bootstrap,
            )
            rows.append(res.row)
        except Exception:
            print(f"[{spec.name}] FAILED", file=sys.stderr)
            traceback.print_exc()
    if rows:
        table = results_table(rows)
        pd.set_option("display.width", 250)
        print(table.T.to_string())
        table.to_csv(f"{args.out}/results.csv")


if __name__ == "__main__":
    main()
