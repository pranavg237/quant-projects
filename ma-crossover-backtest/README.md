# quantbt (work in progress)

Started as a 60-line moving-average crossover script; being rebuilt into a bias-safe
backtesting framework. See `AUDIT.md` for what was wrong with the original and
`DECISIONS.md` for the judgement calls made along the way.

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest            # tests + coverage gate
.venv/bin/ruff check . && .venv/bin/mypy
.venv/bin/python scripts/run_ma_crossover.py --short 50 --long 200
```

Corrected SPY 50/200 result, 2018-01-02 to 2023-12-29, fills at the next open with
5 bps slippage and 1 bp commission, proper 200-bar warm-up: total return 51.3%,
Sharpe 0.51, max drawdown -33.7%, versus buy-and-hold 95.5% / Sharpe 0.65.
The original script reported 48.2% with same-bar fills, no costs and no warm-up.
