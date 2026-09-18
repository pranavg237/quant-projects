# Alpaca MA Crossover Bot (Paper Trading)

Takes the moving-average crossover signal from
[ma-crossover-backtest](../ma-crossover-backtest) and wires it to Alpaca's
**paper** trading API, so the strategy runs against a live, real-time paper
account instead of only historical backtests. This is the "close the loop"
project from the original idea list (Alpaca-based live trading).

## How it works

One decision cycle (`bot.py`), meant to run once per day, ideally shortly
after market close:

1. Pull recent daily bars for the symbol from Alpaca's market data API
2. Compute the MA crossover signal (short MA vs. long MA - same logic as the
   backtest, pulled into `strategy.py` so it's shared/testable)
3. Compare to the current position
4. If the signal calls for a change, size the position (capped at
   `--max-position-fraction` of equity, default 25%) and submit a market
   order - otherwise do nothing

It intentionally does **not** run continuously - it's a scheduled job, not
an event loop. Re-running it when nothing changed is always a safe no-op
(that's what the unit tests in `test_strategy.py` for `decide_order` lock down).

## Setup

```bash
cp .env.example .env
# edit .env with your free paper keys from
# https://app.alpaca.markets/paper/dashboard/overview
set -a; source .env; set +a

python3 -m unittest discover -s tests -v   # 11 tests, no network needed

python3 bot.py --symbol SPY --short 50 --long 200 --risk-fraction 0.20 --dry-run
```

Drop `--dry-run` once you've checked the dry-run output looks right, and it
will actually submit orders - to the **paper** account, not real money,
as long as `ALPACA_BASE_URL` is left unset/pointed at
`paper-api.alpaca.markets`.

## Files

- `alpaca_client.py` - minimal REST wrapper (account, positions, bars,
  market orders) - no `alpaca-py` SDK dependency
- `strategy.py` - `compute_signal()`, `position_size()`, `decide_order()` -
  pure functions, fully unit-tested without hitting the network
- `bot.py` - orchestrates one decision cycle end-to-end
- `.env.example` - copy to `.env`, never commit the real one

## Risk controls actually in here

- Position sizing is capped at `--max-position-fraction` of account equity
  (default 25%) regardless of what `--risk-fraction` asks for
- Won't submit a 0-share order if there isn't enough buying power for even
  one share
- Refuses to run if the account isn't `ACTIVE` or is `trading_blocked`
- Defaults to the paper endpoint; going live requires deliberately setting
  `ALPACA_BASE_URL`

## Known simplifications (roadmap)

- No stop-loss / max-drawdown circuit breaker - it only ever compares to the
  MA signal, so a strategy that's simply wrong keeps riding it out. A real
  next step: add a hard stop (e.g. exit if position is down >X% regardless
  of signal)
- No slippage/commission modeling (paper fills are usually close to
  frictionless, which live trading is not)
- Daily bars only - no intraday rebalancing
- Single symbol at a time - running against a basket means running this
  once per symbol (cron-friendly, but not portfolio-aware: it won't know
  that buying AAPL and TSLA both at 20% risk-fraction can add up to more
  concentration than intended)
