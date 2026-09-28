# Alpaca MA Crossover Bot (Paper Trading)

Runs the 50/200-day moving-average crossover from
[ma-crossover-backtest](../ma-crossover-backtest) against Alpaca's **paper** trading
API. It's a daily job that decides once, after the close, whether to be long SPY or in
cash, and submits the order if the position needs to change.

```bash
pip install -r requirements.txt
cp .env.example .env          # add your free paper keys; bot.py loads .env itself
python bot.py --symbol SPY --dry-run
python -m pytest              # 121 tests, no network or keys needed
```

## Problem

A backtest shows what a rule *would* have done. Running it live shows the engineering
problems a backtest hides: partial bars, corporate actions in the price feed, buying
power, and re-running safely. This project is the smallest honest version of that.

## Method

One decision cycle (`bot.py`), meant to run once a day after the 4pm close:

1. Read the account. Refuse to run if it isn't `ACTIVE` or is `trading_blocked`.
2. Pull about 600 calendar days of daily bars from Alpaca, **split- and
   dividend-adjusted** (`adjustment=all`).
3. **Drop today's bar if the session hasn't closed.** Otherwise a mid-day run would
   treat the last trade as a close.
4. **Refuse unusable data:** too few bars, a missing or non-positive close, or a newest
   bar more than 5 calendar days old (a stopped feed).
5. Compute the signal: LONG if the 50-day MA is above the 200-day MA, else FLAT
   (`strategy.py`, the same rule as the backtest).
6. Compare with the current position. If a change is needed, size the buy as
   `min(risk fraction × equity, max position fraction × equity, buying power)` in whole
   shares.
7. **Idempotency checks:** skip if any order for the symbol is still open, or if an order
   with this bar's `client_order_id` already exists.
8. **Pre-trade risk checks** (`risk.py`). Any failure blocks the order.
9. Submit one market order, exactly once. Then re-read it and log whether it filled, is
   still open, or ended partially filled.

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
- **Duplicate orders on re-run.** An order submitted after the close waits at the broker
  as `accepted` until the next open, so the position is still 0. The bot only checked
  the position, so running it twice that evening would have bought twice. See
  [Idempotency](#idempotency-running-twice-never-buys-twice).
- **Permanently underweight after a partial fill.** The bot only traded when the
  position was zero, so a buy that filled 50 of 143 shares and was then cancelled left
  it at 50 shares until the next crossover, possibly years later. It now tops up when the
  position is below half the target.

## Risk controls

`risk.py` is a small pure module. `evaluate_order()` takes the proposed order, the
current position, an account snapshot and the limits. It returns every check with a
pass/fail/skip status and a reason. All checks run, so the log shows every reason an order
was blocked, not just the first.

| Check | Applies to | Default | Blocks when |
|---|---|---|---|
| Kill switch | **every** order, buys and sells | off | `BOT_KILL_SWITCH` is set to anything but `0/false/no/off`, or the file `KILL_SWITCH` exists |
| Order sanity | every order | | side isn't buy/sell, qty ≤ 0 or NaN, price ≤ 0 |
| No short | sells | | selling more than the current long position |
| Daily loss | risk-increasing orders | 2% | `(last_equity − equity) / last_equity ≥ limit`, or `last_equity` missing (fails closed) |
| Position % of equity | risk-increasing orders | 25% | post-trade position value / equity > limit |
| Position notional | risk-increasing orders | $50,000 | post-trade position value > limit |
| Buying power | buys | | order value > buying power |

You can set these with CLI flags or environment variables. Flags win, and
`.env.example` lists the variables:

| Flag | Environment variable |
|---|---|
| `--max-position-fraction` | `BOT_MAX_POSITION_PCT` |
| `--max-position-notional` | `BOT_MAX_POSITION_NOTIONAL` |
| `--max-daily-loss` | `BOT_MAX_DAILY_LOSS_PCT` |
| `--kill-switch-file` | `BOT_KILL_SWITCH_FILE` |

The $50k notional default is deliberately small. On a paper account larger than $250k,
buys will be blocked until you raise it, which is the safe direction to fail.

Three design decisions worth explaining:

- **The loss limit never blocks a sell.** `last_equity` is Alpaca's equity at the
  previous trading day's close. A breach blocks new buys, but the bot can still sell. A
  breaker that stopped you exiting would trap you in the losing position.
- **The kill switch halts everything but does not flatten.** You pull a kill switch when
  something looks wrong, such as bad data, a bug or a broker incident. Selling at market
  automatically in those conditions means trading on state you've just declared
  untrustworthy, possibly into a price gap or a trading halt. For a long-only daily
  strategy the overnight position is the risk it is meant to take, not a runaway one.
  Flattening stays a deliberate human action (the Alpaca dashboard, or
  `DELETE /v2/positions`). The env var fails safe: a typo like `BOT_KILL_SWITCH=ture`
  engages it.
- **Sizing and checking are separate on purpose.** `position_size()` already caps the buy
  at 25% of equity and at buying power. `risk.py` then re-checks the *post-trade*
  position against the same limit, so a bug in sizing, or a partial-fill top-up, can't
  slip past it.

These controls are unchanged from before:

- The bot never submits a zero-share order.
- It uses the paper endpoint unless `ALPACA_BASE_URL` is deliberately changed.
- Credentials come only from the environment or `.env` (gitignored). No key has ever been
  committed to this repository.

## Idempotency: running twice never buys twice

The dangerous case is an order submitted after the close. It waits at the broker as
`accepted` until the next open, so the position is still 0. The old bot checked only the
position, so a second run that evening, or a cron retry, would have bought again. Three
layers now prevent that:

1. **Open orders.** If any order for the symbol is still open, the run logs
   `pending_order` and does nothing.
2. **A deterministic `client_order_id`**, built as `macx-{symbol}-{bar date}-{side}`
   (e.g. `macx-SPY-20260925-buy`). Before submitting, the bot looks the id up. If an
   order with that id exists in any state, even cancelled, the bot has already acted on
   this bar and stops.
3. **The broker.** Alpaca rejects a second order with the same `client_order_id`, so a
   race between two runs produces a rejection, not a duplicate.

**Order submission is never retried**, and the same id makes failed submissions safe. If
`POST /v2/orders` times out or returns a 5xx, the order may or may not have reached the
broker. So the bot looks it up by `client_order_id` rather than sending it again:

- If the order is there, the run carries on as submitted.
- If not, the run exits with code 1, and the next scheduled run tries again with the same
  id.

Read-only calls (account, bars, positions, lookups) are retried up to 3 times with
exponential backoff on timeouts, connection errors, 429 and 5xx. Other 4xx errors (bad
keys, insufficient buying power, an invalid order) are never retried, because repeating
them can't help.

## Failure modes

Each row is covered by a test in `tests/test_scenarios.py` or `tests/test_client_http.py`,
run against an HTTP-level fake of the Alpaca API.

| Situation | What the bot does | Exit |
|---|---|---|
| API 5xx / 429 / timeout on a read | Retries 3× with backoff. If still failing, logs `run_failed` and places no order | 1 |
| API 401/403/422 on a read | No retry, logs `run_failed`. A 404 on the position just means flat | 1 |
| Timeout / 5xx on order submission | No retry. Looks the order up by `client_order_id`: carries on if found, else logs `order_not_placed` | 0 / 1 |
| Broker rejects with 403 insufficient buying power | Logs `order_submit_error` with Alpaca's error code. No retry | 1 |
| Order partially filled, then cancelled or expired | Logs `order_incomplete` with filled and unfilled qty. On the next bar, tops up if the position is below half the target | 0 |
| Order still open when the run ends (normal after the close) | Logs `order_open_at_end_of_run` with the next open. The next run sees it as pending | 0 |
| Market closed (after the close, weekend) | The `day` order queues for the next open. Re-runs over the weekend do nothing | 0 |
| Same-day re-run | `pending_order` or `already_submitted`, never a second order | 0 |
| Stale, missing or malformed bars | `run_refused`, no order | 1 |
| Intraday run | Drops today's partial bar. The signal and order id use yesterday's bar | 0 |
| Risk check or kill switch fails | `order_blocked`, listing every failing reason | 2 |

## Logging

Every step writes one JSON line to stderr with:

- a timestamp and level
- a sortable `run_id`, e.g. `20260925T201500Z-3f9a1c2e`
- an event name, such as `decision`, `risk_check`, `order_submitted`,
  `order_submit_error` (with the HTTP status and Alpaca error code) or `run_finished`

Redirect it to a file to keep a history, and filter with `jq`. Below is a real run
against the offline fake, with the account down 3.5% on the day. Timestamps and some
checks are elided.

```json
{"ts": "...", "level": "info", "run_id": "20260925T201500Z-3f9a1c2e", "event": "decision", "symbol": "SPY", "signal": "long", "bar_date": "2026-09-25", "last_price": 139.0, "current_qty": 0.0, "target_qty": 138, "decision": "buy 138"}
{"ts": "...", "level": "info", "run_id": "20260925T201500Z-3f9a1c2e", "event": "risk_check", "symbol": "SPY", "side": "buy", "qty": 138, "approved": false, "risk_increasing": true, "checks": [{"check": "kill_switch", "status": "pass", "reason": "not engaged"}, ..., {"check": "daily_loss", "status": "fail", "reason": "daily loss limit breached: equity 96,500.00 vs last_equity 100,000.00 (-3.50%), limit -2.00%"}, {"check": "position_pct", "status": "pass", "reason": "post-trade 19.88% of equity, limit 25.00%"}, ...]}
{"ts": "...", "level": "warning", "run_id": "20260925T201500Z-3f9a1c2e", "event": "order_blocked", "symbol": "SPY", "side": "buy", "qty": 138, "reasons": ["daily loss limit breached: equity 96,500.00 vs last_equity 100,000.00 (-3.50%), limit -2.00%"]}
```

**No secrets in logs.** There are two layers:

1. The client strips its own key values from any error text it raises, since an error
   page could echo the request headers.
2. As a safety net, the formatter redacts any field whose name looks like a credential,
   and any occurrence of the configured key values.

`tests/test_logging.py` runs full invocations, including a server that echoes the headers
back, with and without the safety net. It asserts the keys never appear.

The exit codes are:

- `0`: ran cleanly, including "nothing to do" and "order pending".
- `1`: something failed, or the broker rejected the order.
- `2`: a risk check or the kill switch blocked an order.

## Limitations

- **No stop-loss.** The daily-loss check only stops the bot *adding* risk on a bad day. It
  doesn't exit a losing position. A rule that is simply wrong keeps riding until the
  crossover flips.
- **Market orders at the next open** have real slippage. Paper fills are optimistic
  about it.
- **One symbol per run, not portfolio-aware.** Running several symbols at 20% each can
  add up to more exposure than intended. The per-symbol limits don't add up across runs.
- **The risk checks price the order at the last close.** A market order fills at the
  next open, which can gap. The broker's own buying-power check is the backstop.
- **Early-close days** (1pm) are handled conservatively: today's bar is ignored until
  4pm, so a run between 1pm and 4pm uses the previous close.
- **Staleness is measured in calendar days, not with an exchange calendar.** The 5-day
  window allows for a weekend plus a holiday. A longer market closure would make the bot
  refuse to trade until fresh bars arrive.
- **Not verified against the live API in this repository's tests.** The fake server
  follows Alpaca's documented response shapes and error codes, e.g. 403 / 40310000 for
  insufficient buying power and 422 for a duplicate `client_order_id`. It has not been
  checked against real responses. The first real run should be `--dry-run`.

## How to run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY (paper keys)

python bot.py --symbol SPY --short 50 --long 200 --risk-fraction 0.20 --dry-run
mkdir -p logs
python bot.py --symbol SPY 2>> logs/bot.jsonl   # submits to the paper account
touch KILL_SWITCH                                # halts all order submission until removed
python -m pytest
```

To schedule it, run it once a day after the close, e.g. a cron entry at 16:15 ET on
weekdays.

## Files

- `strategy.py`: `compute_signal()`, `completed_bars()`, `validate_bars()`,
  `position_size()`, `decide_order()`. These are pure functions and fully unit-tested.
- `risk.py`: the pre-trade checks (`evaluate_order()`, pure) and `read_kill_switch()`.
- `alpaca_client.py`: a minimal REST wrapper built on `requests`, with no SDK dependency.
  Its docstring describes the retry policy.
- `runlog.py`: JSON-lines logging with run ids and secret redaction.
- `bot.py`: runs one decision cycle and maps the outcome to an exit code.
- `tests/`:
  - strategy and risk unit tests
  - HTTP retry tests
  - end-to-end failure scenarios against `tests/fake_alpaca.py`, an in-memory Alpaca at
    the HTTP boundary
  - logging and secret-leak tests
  - the original `run_once` tests against a simpler fake client
