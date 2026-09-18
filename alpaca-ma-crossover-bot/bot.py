"""
Runs one decision cycle of the MA crossover strategy against Alpaca's PAPER
trading API: pulls recent daily bars, computes the signal, compares to the
current position, and submits a market order if a change is needed.

This is meant to be run once a day (e.g. via a scheduled task shortly after
market close, using daily bars) - it is NOT a continuously-running intraday
loop. Re-running it when nothing has changed is a safe no-op.

Usage:
  cp .env.example .env        # fill in your paper keys
  set -a; source .env; set +a
  python3 bot.py --symbol SPY --short 50 --long 200 --risk-fraction 0.20 --dry-run
  python3 bot.py --symbol SPY --short 50 --long 200 --risk-fraction 0.20   # actually submits orders
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from alpaca_client import AlpacaClient
from strategy import compute_signal, position_size, decide_order, Signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("bot")


def run_once(client: AlpacaClient, symbol: str, short_window: int, long_window: int,
             risk_fraction: float, max_position_fraction: float, dry_run: bool) -> None:
    account = client.get_account()
    equity = float(account["equity"])
    if account["status"] != "ACTIVE" or account.get("trading_blocked"):
        raise RuntimeError(f"Account not tradable: status={account['status']}, "
                            f"trading_blocked={account.get('trading_blocked')}")

    lookback_start = (dt.date.today() - dt.timedelta(days=long_window * 3)).isoformat()
    bars = client.get_daily_bars(symbol, start=lookback_start)
    if len(bars) < long_window:
        raise RuntimeError(f"Only got {len(bars)} bars for {symbol}, need >= {long_window}. "
                            f"(Market data may lag on a free/paper account.)")

    closes = pd.Series([b["c"] for b in bars])
    last_price = closes.iloc[-1]
    signal = compute_signal(closes, short_window, long_window)

    position = client.get_position(symbol)
    current_qty = position.qty if position else 0.0

    target_qty = position_size(equity, last_price, risk_fraction, max_position_fraction)
    order = decide_order(current_qty, signal, target_qty)

    log.info(f"{symbol}: signal={signal.value}  last_price={last_price:.2f}  "
             f"current_qty={current_qty}  target_qty={target_qty}  equity=${equity:,.2f}")

    if order is None:
        log.info("No action needed - already in the correct state.")
        return

    side, qty = order
    if dry_run:
        log.info(f"[DRY RUN] Would submit: {side} {qty} {symbol}")
        return

    result = client.submit_market_order(symbol, qty, side)
    log.info(f"Submitted order: {side} {qty} {symbol} -> order id {result.get('id')}, "
              f"status={result.get('status')}")


def main():
    p = argparse.ArgumentParser(description="MA crossover paper trading bot (one decision cycle)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--short", type=int, default=50, dest="short_window")
    p.add_argument("--long", type=int, default=200, dest="long_window")
    p.add_argument("--risk-fraction", type=float, default=0.20,
                    help="Fraction of equity to allocate when entering a position (default 0.20)")
    p.add_argument("--max-position-fraction", type=float, default=0.25,
                    help="Hard cap on fraction of equity in one name (default 0.25)")
    p.add_argument("--dry-run", action="store_true", help="Compute the decision but don't submit any order")
    args = p.parse_args()

    client = AlpacaClient()
    run_once(client, args.symbol, args.short_window, args.long_window,
              args.risk_fraction, args.max_position_fraction, args.dry_run)


if __name__ == "__main__":
    main()
