"""Check that the event-driven MACrossover port reproduces the vectorised Phase 2 result on SPY."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from quantbt.data import add_live_data_flag, load_yahoo, use_live_data
from quantbt.engine import run_backtest
from quantbt.execution import ExecutionSimulator, FixedBpsSlippage, PercentageCommission
from quantbt.strategies import MACrossover
from quantbt.vectorized import backtest_ma_crossover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_live_data_flag(parser)
    use_live_data(parser.parse_args().live_data)
    start, end, short, long = "2018-01-01", "2023-12-31", 50, 200
    data = load_yahoo("SPY", start=pd.Timestamp(start) - pd.offsets.BDay(long + 10), end=end)
    vec = backtest_ma_crossover(
        data, short, long, start=start, end=end, slippage_bps=5.0, commission_bps=1.0
    )
    res = run_backtest(
        data,
        MACrossover("SPY", short, long),
        start=start,
        end=end,
        execution=ExecutionSimulator(PercentageCommission(1e-4), FixedBpsSlippage(5.0)),
        benchmark="SPY",
    )
    diff = np.abs(res.growth.to_numpy() - vec.equity.to_numpy()).max()
    print(f"bars: {len(res.equity)}  fills: {len(res.fills)}  round trips: {len(res.round_trips)}")
    print(f"vectorised total return: {vec.equity.iloc[-1] - 1:.4%}")
    print(f"event-driven total return: {res.growth.iloc[-1] - 1:.4%}")
    print(f"max |equity difference|: {diff:.2e}")
    print(f"commission paid: {res.total_commission:,.2f}  slippage paid: {res.total_slippage:,.2f}")
    assert diff < 1e-9, "port does not match the vectorised reference"
    print("OK: port matches the vectorised reference to floating-point precision")


if __name__ == "__main__":
    main()
