# Alpaca MA Crossover Bot (Paper Trading)

Runs the 50/200-day moving-average crossover from
[ma-crossover-backtest](../ma-crossover-backtest) against Alpaca's **paper** trading
API. It's a daily job that decides once, after the close, whether to be long SPY or in
cash, and submits the order if the position needs to change.

```bash
pip install -r requirements.txt
cp .env.example .env          # add your free paper keys; bot.py loads .env itself
python bot.py --symbol SPY --dry-run
python -m pytest              # 24 tests, no network or keys needed
```

## Problem

A backtest shows what a rule *would* have done. Running it live shows the engineering
problems a backtest hides: partial bars, corporate actions in the price feed, buying
power, and re-running safely. This project is the smallest honest version of that.

## Method

One decision cycle (`bot.py`), meant to run once a day after the 4pm close:

1. Pull about 600 calendar days of daily bars from Alpaca, **split- and
   dividend-adjusted** (`adjustment=all`).
2. **Drop today's bar if the session hasn't closed.** Otherwise a mid-day run would
   treat the last trade as a close.
3. Compute the signal: LONG if the 50-day MA is above the 200-day MA, else FLAT
   (`strategy.py`, the same rule as the backtest).
4. Compare with the current position. If a change is needed, size the buy as
   `min(risk fraction × equity, max position fraction × equity, buying power)` in whole
   shares, and submit a market order. Otherwise do nothing, so re-running is always safe.

## Results

**There is no live track record.** Nothing here has been run against a funded account,
and no paper-trading P&L is reported. What the bot trades has been backtested in
`ma-crossover-backtest`, with the same rule, next-open fills, 5 bp slippage, 1 bp
commission, and cash earning the T-bill rate:

| SPY, 2005-01-03 to 2025-08-29 | CAGR | Sharpe | Max drawdown | Turnover/yr | Trades |
|---|---|---|---|---|---|
| 50/200 MA crossover | 8.48% | 0.54 | -33.7% | 0.51 | 10 |
| Buy and hold | 10.50% | 0.53 | -55.2% | 0 | 1 |

Reproduce with, in `../ma-crossover-backtest`:

```bash
python scripts/run_ma_crossover.py --start 2005-01-03 --end 2025-08-29 --french-rf
```

The data is Yahoo Finance daily bars, downloaded 2026-09-28. The rule gives up about 2%
a year of return for a much shallower worst drawdown, at the same Sharpe ratio. Ten
round trips in 20 years is far too few to call that edge. The walk-forward study in
[RESULTS.md](../ma-crossover-backtest/RESULTS.md) reaches the same conclusion: no
significant alpha, and most of the return is market and momentum exposure.

## Bugs found and fixed

These are the interesting part for a reviewer, because each is invisible in a backtest:

- **Unadjusted prices.** Alpaca's bars endpoint defaults to raw prices. A 4-for-1 split
  looks like a 75% crash and can flip the MA signal on a corporate action rather than a
  trend. The bot now requests `adjustment=all`.
- **Partial daily bar.** Run at 11am, the "daily bar" for today closes at whatever the last
  trade was, so the signal depended on when the job ran. `completed_bars()` drops it
  until 4pm ET.
- **Buying power.** The old README said the bot checked buying power, but it sized on
  equity only. It now takes the minimum of the two.

## Risk controls

- Position capped at `--max-position-fraction` of equity (default 25%), whatever
  `--risk-fraction` asks for.
- Never submits a zero-share order.
- Refuses to run if the account isn't `ACTIVE` or is `trading_blocked`.
- Uses the paper endpoint unless `ALPACA_BASE_URL` is deliberately changed.
- Credentials come only from the environment or `.env` (gitignored). No key has ever been
  committed to this repository.

## Limitations

- **No stop-loss or drawdown circuit breaker.** A rule that is simply wrong keeps riding.
- **Market orders at the next open** have real slippage. Paper fills are optimistic
  about it.
- **One symbol per run, not portfolio-aware.** Running several symbols at 20% each can
  add up to more exposure than intended.
- **Early-close days** (1pm) are handled conservatively: today's bar is ignored until
  4pm, so a run between 1pm and 4pm uses the previous close.
- **Not verified against the live API in this repository's tests.** The tests use a
  fake client that mirrors the endpoints' responses. The first real run should be
  `--dry-run`.

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper keys)

python bot.py --symbol SPY --short 50 --long 200 --risk-fraction 0.20 --dry-run
python bot.py --symbol SPY                     # submits to the paper account
python -m pytest
```

To schedule it, run it once a day after the close, e.g. a cron entry at 16:15 ET on
weekdays.

## Files

- `strategy.py`: `compute_signal()`, `completed_bars()`, `position_size()`,
  `decide_order()`. These are pure functions and fully unit-tested.
- `alpaca_client.py`: a minimal REST wrapper built on `requests`, with no SDK dependency.
- `bot.py`: runs one decision cycle.
- `tests/`: strategy unit tests, plus end-to-end `run_once` tests against a fake client.
