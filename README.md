# Quant Projects

[![CI](https://github.com/pranavg237/quant-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/pranavg237/quant-projects/actions/workflows/ci.yml)

Quantitative finance projects in Python, built from scratch. Every number in these READMEs
comes from running the code in this repository, and each README says what data and dates
produced it. Where a strategy loses to buying and holding the index, the README says so.

| Project | What it is | Headline result |
|---|---|---|
| [**options-pricing-engine**](./options-pricing-engine) | Black-Scholes, binomial lattices, Monte Carlo and Heston, cross-validated against each other and calibrated to a real SPY option chain | Heston fits the SPY surface to **2.51 vol points** RMSE against 10.61 for Black-Scholes with one vol per expiry (2.38 out of sample). It fails in the short-dated put wing (5.92), which needs jumps. Put-call parity "violations" are mostly American early exercise: 44.5% of pairs breach the spread, 10.6% once the premium is priced |
| [**market-making-simulator**](./market-making-simulator) | A price-time-priority limit order book and matching engine, plus Avellaneda-Stoikov optimal quoting with informed order flow, markouts and a parameter-sensitivity study | Reproduces the paper's Table 1 (profit 64.98 vs 65.0). In the full book, inventory skew cuts PnL volatility 4x at statistically identical PnL. Informed-fill markouts match the model's prediction; A-S's edge under informed flow comes from how it unwinds, not from avoiding it |
| [**ma-crossover-backtest**](./ma-crossover-backtest) | An event-driven backtester built to be hard to fool yourself with: next-bar fills, costs, walk-forward, bootstrap, deflated Sharpe over every configuration tried, PBO, a parameter-sensitivity heatmap, factor regressions | Five strategies, all out of sample and net of costs: **none beats buy-and-hold SPY** on Sharpe, and none has a significant Fama-French alpha. Deflated for all 63 configurations tried, the MA crossover's DSR is 0.08 out of sample |
| [**portfolio-optimization**](./portfolio-optimization) | Markowitz, Ledoit-Wolf, risk parity, HRP and Black-Litterman, walk-forward tested on 15 ETFs, with per-method trading costs and an estimation-error study | No method's Sharpe differs significantly from 1/N (all p ≥ 0.20). Unconstrained leverage loses 95% in one month, and a cap fixes it |
| [**fama-french-factor-model**](./fama-french-factor-model) | CAPM to FF6 regressions with Newey-West errors (checked against statsmodels), rolling betas with HC3 bands, attribution, GRS and Fama-MacBeth tests. The example runs offline from a dated data snapshot | Recovers textbook results: HML is redundant given the other FF5 factors (alpha t = 0.75), and GRS rejects every model on the 25 size/value portfolios. Of 30 tests of whether an exposure changed over time, only 2 reject at 5% |
| [**alpaca-ma-crossover-bot**](./alpaca-ma-crossover-bot) | The 50/200 SPY crossover as a daily Alpaca paper-trading job, with pre-trade risk checks, a kill switch, duplicate-proof re-runs and JSON logs, tested against a simulated API | No live track record. Backtest 2005-2025: Sharpe 0.54 vs 0.53 for buy-and-hold, max drawdown -34% vs -55% |

## How the results are kept honest

- **Out of sample, net of costs.** Backtests fill on the bar *after* the signal, charge
  slippage, commission and short-borrow fees, choose parameters on past data only, and
  report against buy-and-hold over identical dates.
- **Sharpe ratios are on excess returns** over the T-bill rate, and they come with
  confidence intervals or significance tests. Point estimates alone are not treated as
  evidence.
- **Reproducible offline.** The market data behind every headline number is committed as
  a dated, hash-checked snapshot, and the analysis scripts read it by default. An agent
  that had not seen the work re-ran every project from a clean clone to check the README
  numbers (see [CHANGES.md](CHANGES.md)). Where a result is fragile, the README says so:
  one backtest's Sharpe moves by 0.02 when a single price changes in its seventh digit.
- **Tested.** TESTCOUNT tests across the six projects, all offline, run by CI on every push.
  The four main packages also pass strict `mypy` and `ruff`.

[PLAN.md](PLAN.md) is the audit and plan behind two rounds of work, and
[CHANGES.md](CHANGES.md) explains every change they led to.

## Running a project

Each project installs and runs on its own (Python 3.12):

```bash
cd <project>
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

Each project's README gives the one command that regenerates its results.
