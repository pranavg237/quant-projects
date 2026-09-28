# Build status

Resume notes. Everything below is committed; nothing is in a broken state.

**Shared environment:** `.venv-shared/` at this level has the whole stack (numpy, scipy,
pandas, matplotlib, yfinance, scikit-learn, pytest, ruff, mypy). Each project has its own
git repo and its own `requirements.txt`. Run anything with:

```bash
cd <project> && PYTHONPATH=src "../.venv-shared/bin/python" -m pytest -q
```

---

## 1. options-pricing-engine — **COMPLETE**

4 commits. 373 tests, 99% coverage, ruff + mypy clean. README, DECISIONS, REVIEW all
written. Analysis script runs offline from a committed SPY snapshot.

Headline: Heston fits the real SPY surface to 2.51 vol points RMSE (1.09 in the body,
5.92 in the short-dated put wing) against 10.61 for one-vol-per-expiry Black-Scholes;
out-of-sample 2.38. Reproduces Hull's textbook values and the Black-Scholes limit to 1e-10.

## 2. market-making-simulator — **COMPLETE**

5 commits. 124 tests, 98% coverage, ruff + mypy clean. README, DECISIONS, REVIEW written.

Headline: reproduces Avellaneda & Stoikov (2008) Table 1 (profit 64.98 vs 65.0, std 6.62
vs 6.6). In a full order book with informed flow, A-S gets Sharpe 11.7 against 2.8 for
symmetric quoting at statistically identical PnL.

## 3. portfolio-optimization — **CORE DONE, tests and README outstanding**

3 commits. All source modules written, ruff + mypy clean, and
`python scripts/run_analysis.py` runs end to end and writes 11 figures plus
`results/results.json`.

**Done:** `types`, `data`, `covariance` (Ledoit-Wolf from scratch, matches sklearn to
1e-17), `optimizers` (closed-form frontier + constrained SLSQP + convex max-Sharpe),
`riskparity`, `hrp` (leaf ordering verified against scipy), `blacklitterman`, `backtest`
(walk-forward, drift, costs, lookahead guard), `metrics`, `strategies`, `plotting`,
`scripts/run_analysis.py`.

**Still to do:**
1. `tests/` — nothing written yet. Target the same standard as projects 1 and 2:
   closed-form checks (frontier scalars A/B/C/D, GMV variance = 1/A), sklearn agreement
   for Ledoit-Wolf, risk contributions equal to ~1e-8, HRP order vs `scipy.leaves_list`,
   Black-Litterman no-views identity, and a synthetic-data lookahead test on `walk_forward`.
2. `README.md`, `DECISIONS.md`, `REVIEW.md`.

**The headline result is already measured and is better than the textbook claim.** Naive
Markowitz does not fail because of the optimiser, it fails because of leverage:

| gross leverage cap | Sharpe | turnover/yr | max DD | mean leverage |
|---|---|---|---|---|
| 1.0 (long-only) | 1.07 | 192% | −24% | 1.0 |
| 1.5x | **1.17** | 321% | −21% | 1.5 |
| 2x | 1.15 | 449% | −18% | 2.0 |
| 3x | 1.02 | 671% | −18% | 2.8 |
| 5x | 0.88 | 866% | −22% | 3.5 |
| uncapped | 0.27 | 11,712% | −98% | 9.1 (peak 553x) |

At a 36-month window the uncapped version is ruined outright: −827% in one month, 638x
peak leverage. Long-only Markowitz on the same data and the same estimator beats 1/N at
every lookback tested. A bug was found and fixed along the way: `max_leverage` was being
silently ignored by the max-Sharpe solver.

## 4-6. Not started

`stat-arb-kalman`, `ml-alpha-research`, `probability-and-interview-lab`.

## Final step (not started)

Top-level `README.md` indexing every project. Projects 1 and 2 already have their
`REVIEW.md`; projects 3-6 still need theirs.
