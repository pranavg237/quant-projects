# Quant Projects

Personal quant finance / Python projects. Each subfolder is its own
self-contained git repo (own venv, own README) so they can be worked on,
tested, and eventually pushed to GitHub independently.

## Current projects

| Folder | What it does | Status |
|---|---|---|
| [`ma-crossover-backtest`](./ma-crossover-backtest) | Original MA crossover backtest on SPY (yfinance) | existing |
| [`options-pricing-toolkit`](./options-pricing-toolkit) | Black-Scholes pricer, Greeks, implied vol solver, 0-2 DTE theta/gamma decay visualizer | new - built out |
| [`fama-french-factor-model`](./fama-french-factor-model) | 3-factor regression (market/size/value) + rolling beta, real data from Yahoo + Ken French's data library | new - built out |
| [`alpaca-ma-crossover-bot`](./alpaca-ma-crossover-bot) | Wires the MA crossover signal to Alpaca's paper trading API | new - built out |

All three new projects were built and tested against **live data sources**
(Yahoo Finance chart API, Ken French's Dartmouth data library) - not
placeholder/sample data - except the Alpaca bot, which needs your own paper
API keys to actually place orders (the strategy logic itself is unit-tested
without any network dependency).

## Getting each one running

All three new projects use only `numpy`, `pandas`, `scipy`, `matplotlib`,
`requests` - nothing needs installing beyond what's likely already on your
machine from the backtest project:

```bash
pip3 install numpy pandas scipy matplotlib requests
```

Then `cd` into whichever project and follow its own README. Each has its
own `tests/` directory runnable with `python3 -m unittest discover -s tests -v`.

## Why these three

They were picked to (a) directly support what you're already doing -
short-dated options (the pricing toolkit) and the MA strategy (the Alpaca
bot) - and (b) close out the two items that were already on the to-do list
(Fama-French, Alpaca live trading), rather than starting from a blank slate
of unrelated ideas.

## Roadmap - ideas not yet built

Rough next candidates, roughly in order of how directly they build on what's
already here vs. how much new ground they'd cover:

- **Pairs trading / stat-arb backtester** - cointegration test (e.g.
  Engle-Granger via `statsmodels`, once installed) on a pair of correlated
  names, backtest a mean-reversion spread trade. Natural next step after the
  factor model, and squarely in the "market-neutral quant" territory that
  Citadel/DRW-style firms care about.
- **Options chain IV surface** - pull a real chain (Alpaca's options data
  API, or a free chain source), run `implied_vol()` from the pricing toolkit
  across every strike/expiry, and plot the smile/skew instead of a single
  flat vol input.
- **Order book / market microstructure simulator** - a simplified
  limit-order-book simulator (price-time priority matching engine) to
  reason about queue position, adverse selection, and market-making P&L -
  most relevant to Jane Street/Akuna/IMC/DRW-style market-making roles
  specifically (as opposed to the more discretionary-trading flavor of the
  projects here so far).
- **Monte Carlo VaR / risk engine** - simulate a portfolio's P&L
  distribution (e.g. bootstrapped historical returns or a fitted
  distribution) and report VaR/CVaR at a few confidence levels.
- **Walk-forward optimization for the MA crossover** - right now the
  short/long windows are hand-picked; a walk-forward (train on one window,
  test out-of-sample on the next, roll forward) would test whether any
  particular window choice is actually robust or just overfit to one backtest
  period.

## Repo hygiene

Each project folder is its own git repo. `.env`/secrets are gitignored where
relevant (Alpaca bot only). None of these are pushed to GitHub yet except
`ma-crossover-backtest` - `git init` was run locally for the three new ones,
so `git remote add origin <url>` + `git push` whenever you're ready to put
them up.
