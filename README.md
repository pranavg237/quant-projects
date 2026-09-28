# Quant Projects

Quantitative finance projects in Python, each built from scratch and tested. Every
folder is a self-contained project with its own README, `requirements.txt` and tests.
Each project's full commit history is preserved: `git log -- <folder>` shows it.

| Project | What it does |
|---|---|
| [`options-pricing-engine`](./options-pricing-engine) | Black-Scholes, binomial trees, Monte Carlo and Heston, cross-validated against each other and calibrated to a live SPY option chain. Heston fits the surface to 2.51 vol points RMSE vs 10.61 for per-expiry Black-Scholes. |
| [`market-making-simulator`](./market-making-simulator) | A limit order book and matching engine from scratch, plus Avellaneda-Stoikov optimal quoting. Reproduces the paper's Table 1, then measures what inventory control is worth with informed flow. |
| [`portfolio-optimization`](./portfolio-optimization) | Markowitz, Ledoit-Wolf shrinkage, risk parity, HRP and Black-Litterman, walk-forward tested. Shows naive Markowitz fails because of leverage, not the optimiser. *(Tests and docs in progress.)* |
| [`fama-french-factor-model`](./fama-french-factor-model) | CAPM through FF6 regressions with Newey-West errors, return attribution, GRS and Fama-MacBeth tests, on Ken French's library data. |
| [`ma-crossover-backtest`](./ma-crossover-backtest) | An event-driven backtesting framework designed to be hard to fool yourself with, including an audit of how a naive MA backtest overstated its results. |
| [`alpaca-ma-crossover-bot`](./alpaca-ma-crossover-bot) | Runs the MA crossover signal against Alpaca's paper-trading API. |
| [`options-pricing-toolkit`](./options-pricing-toolkit) | An earlier, compact Black-Scholes pricer with Greeks, implied vol and a 0-2 DTE theta/gamma decay visualizer. |

## Running a project

```bash
cd <project>
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e ".[dev]" where there is a pyproject
python -m pytest -q
```

See each project's README for its analysis scripts and results. [`STATUS.md`](STATUS.md)
tracks what is finished and what is next.
