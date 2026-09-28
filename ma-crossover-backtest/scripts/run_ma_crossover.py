"""Corrected MA crossover backtest on SPY (replaces the original backtest.py).

Usage: python scripts/run_ma_crossover.py [--short 50] [--long 200] [--start 2018-01-01] ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from quantbt.data import load_yahoo
from quantbt.metrics import format_summary
from quantbt.vectorized import backtest_ma_crossover


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2018-01-01", help="first evaluation date")
    parser.add_argument("--end", default="2023-12-31", help="last evaluation date (inclusive)")
    parser.add_argument("--short", type=int, default=50)
    parser.add_argument("--long", type=int, default=200)
    parser.add_argument("--slippage-bps", type=float, default=5.0)
    parser.add_argument("--commission-bps", type=float, default=1.0)
    parser.add_argument("--rf", type=float, default=0.0, help="annual cash yield while flat")
    parser.add_argument("--out", default="reports/ma_crossover.png")
    args = parser.parse_args()

    # Warm-up: pull enough history before --start so the long MA is defined on day one.
    warmup_start = pd.Timestamp(args.start) - pd.tseries.offsets.BDay(args.long + 10)
    data = load_yahoo(args.ticker, start=warmup_start, end=args.end)
    res = backtest_ma_crossover(
        data,
        args.short,
        args.long,
        start=args.start,
        end=args.end,
        slippage_bps=args.slippage_bps,
        commission_bps=args.commission_bps,
        rf=args.rf,
    )
    print(f"{args.ticker} MA({args.short}/{args.long}) {args.start} to {args.end}")
    print(
        f"fills at next open, slippage {args.slippage_bps} bps, "
        f"commission {args.commission_bps} bps"
    )
    print(format_summary(res.summary(rf=args.rf)))

    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    f = res.frame
    ax1.plot(f["close"], label="Total-return close", alpha=0.7)
    ax1.plot(f["close"].rolling(args.short).mean(), label=f"MA {args.short}", alpha=0.8)
    ax1.plot(f["close"].rolling(args.long).mean(), label=f"MA {args.long}", alpha=0.8)
    buys = f[f["trade"] == 1]
    sells = f[f["trade"] == -1]
    ax1.scatter(
        buys.index, buys["fill_price"], marker="^", color="green", zorder=5, label="Buy (next open)"
    )
    ax1.scatter(
        sells.index,
        sells["fill_price"],
        marker="v",
        color="red",
        zorder=5,
        label="Sell (next open)",
    )
    ax1.set_title(f"{args.ticker} MA crossover ({args.short}/{args.long})")
    ax1.legend()
    ax2.plot(f["benchmark_equity"], label="Buy & hold")
    ax2.plot(f["equity"], label="MA strategy (net of costs)")
    ax2.set_title("Growth of 1")
    ax2.legend()
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"chart saved to {args.out}")


if __name__ == "__main__":
    main()
