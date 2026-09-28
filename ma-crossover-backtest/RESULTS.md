# Results

Five strategies, each run through the same pipeline: walk-forward parameter selection,
bootstrap and overfitting diagnostics, and a Fama-French factor regression. Nothing here
was chosen after seeing the out-of-sample numbers, and nothing that failed has been
removed. Four of the five do not beat buying and holding the index.

Rebuild everything with:

```bash
python scripts/run_strategies.py      # ~25 minutes, writes reports/strategies/
python scripts/build_tearsheets.py    # writes reports/tearsheets/
```

## How these numbers were produced

| | |
|---|---|
| Fills | the open of the bar **after** the signal bar |
| Costs | 5 bps slippage per side, 1 bp commission on notional, and a stock-loan fee on shorts |
| Borrow | 30 bp p.a. on short ETF value (`tsmom`), 50 bp on short stock value (`xsmom`, `pairs`) |
| Cash | earns the Ken French daily risk-free rate; Sharpe is on excess returns |
| Parameters | re-chosen every year on the preceding 5 years, by Sharpe, in-sample only |
| Reported series | the stitched out-of-sample windows, never the training windows |
| Benchmark | buy and hold SPY over the identical dates |
| Data | Yahoo Finance daily bars, split- and dividend-adjusted, through 2025-08-29; downloaded 2026-09-28 |

## The table

Out-of-sample, net of costs. `ma_crossover` starts in 2005 because it needs less history;
the other four start in 2010 after their first 5-year training window.

| Strategy | CAGR | Vol | **Sharpe** | Sortino | Max DD | DD days | Turnover | Exposure |
|---|---|---|---|---|---|---|---|---|
| SPY buy & hold (2010-2025) | 13.8% | 17.3% | **0.76** | 1.07 | -33.7% | 709 | 0.0 | 100% |
| ma_crossover | 8.2% | 12.0% | **0.58** | 0.79 | -22.4% | 808 | 2.0 | 78% |
| mean_reversion | 6.3% | 13.8% | **0.42** | 0.60 | -39.6% | 636 | 10.6 | 32% |
| xsmom | 2.5% | 8.5% | **0.18** | 0.24 | -16.1% | 799 | 5.9 | 100% |
| tsmom | 0.7% | 6.4% | **-0.06** | -0.08 | -19.9% | 2,051 | 3.8 | 91% |
| pairs | 1.0% | 1.5% | **-0.17** | -0.24 | -3.8% | 2,291 | 4.7 | 14% |

Is any of it real? `psr` is the probabilistic Sharpe ratio (P that the true Sharpe exceeds
zero); the bootstrap interval is a 95% stationary-bootstrap range for the out-of-sample
Sharpe; `pbo` is the probability of backtest overfitting from combinatorially symmetric
cross-validation, where anything near or above 0.5 means the in-sample best configuration
is a coin flip out of sample.

| Strategy | Sharpe | Bootstrap 95% CI | PSR | PBO | FF5 alpha (ann.) | t | R² | Momentum beta | t |
|---|---|---|---|---|---|---|---|---|---|
| ma_crossover | 0.58 | 0.21 to 1.01 | 0.995 | 0.54 | +2.4% | 1.22 | 0.41 | +0.14 | 5.7 |
| mean_reversion | 0.42 | 0.00 to 0.96 | 0.950 | 0.71 | -2.3% | -0.89 | 0.57 | +0.04 | 1.5 |
| xsmom | 0.18 | -0.30 to 0.61 | 0.757 | 0.91 | +1.0% | 0.50 | 0.07 | +0.39 | 21.1 |
| tsmom | -0.06 | -0.52 to 0.44 | 0.407 | 0.43 | -2.2% | -1.25 | 0.12 | +0.13 | 7.7 |
| pairs | -0.17 | -0.63 to 0.27 | 0.257 | 0.00 | -0.3% | -0.80 | 0.01 | -0.00 | -1.0 |

**Not one strategy has a statistically significant alpha.** Every t-statistic on the
Fama-French five-factor intercept is below 2 in absolute value. The largest (the MA
crossover, t = 1.22) is exactly what you would expect to see by chance from five tries.

## Strategy by strategy

### ma_crossover: the least bad, and still not good

Long SPY when the short moving average is above the long one, otherwise in Treasury bills.
Sharpe 0.58 against 0.53 for buy and hold over the same 2005-2025 window, with a much
smaller drawdown (-22% against -55%): it sat out most of 2008. That drawdown reduction is
the real result, and it is worth something; the return improvement is not.

What it is *not* is alpha. The five-factor regression leaves +2.4% a year with a
t-statistic of 1.22, and the momentum loading of +0.14 (t = 5.7) says most of what the
rule does is buy a diluted, trend-following version of the market. 16 of 21 folds had a
positive out-of-sample Sharpe, but the chosen parameters covered 10 distinct settings out
of a 17-point grid and touched every value of both windows, which is what a flat, noisy
objective surface looks like. PBO of 0.54 confirms it: pick the in-sample best and you are
below the median out of sample about half the time.

### mean_reversion: buys dips, and buys them all the way down

Sharpe 0.42 with a -39.6% drawdown, worse than the index it is trading, on only 32%
average exposure. Turnover of 10.6x a year is the highest here, so it is also the most
cost-sensitive: it is the one strategy whose ranking would change materially under a
harsher cost model. Its market beta is 0.60 with R² of 0.57 and a *negative* five-factor
alpha, which is the signature of an expensive way to be long. 13 of 16 folds were
positive, but PBO of 0.71 says the parameter choice does not generalise.

### xsmom: momentum, with the losers removed from history

Long the top 20% and short the bottom 20% of 70 large-cap stocks by 12-1 momentum. Sharpe
0.18 with a momentum beta of +0.39 and a t-statistic of 21: it is a momentum factor
tracker with a 0.07 R² against everything else, and the factor loading eats the whole
return. PBO of 0.91, the worst here, means the in-sample best configuration is almost
always below median out of sample.

This is also the most biased result on the page. The 70 stocks are the large caps that
exist *today*. Every company that was large in 2010 and then failed, was acquired or fell
out of the index is missing, and momentum strategies are exactly the kind that would have
been short many of them. Read 0.18 as a ceiling, not an estimate.

### tsmom: the textbook strategy that stopped working

Long assets with positive trailing returns, short the rest, volatility-scaled, across 13
asset-class ETFs. Sharpe -0.06: it lost money net of costs over 16 years. Time-series
momentum is one of the best-documented anomalies in the literature (Moskowitz, Ooi and
Pedersen 2012), which makes it a useful control: the implementation is faithful, the data
is clean, and it still does not work after 2010. Either the anomaly decayed after
publication, or the 2010-2025 sample, with its single dominant equity trend and two
violent reversals, is hostile to it. The longest underwater stretch was 2,051 days, more
than eight years.

### pairs: cointegration finds pairs, not profits

Engle-Granger tests on 17 candidate pairs each quarter, trading the z-score of the frozen
spread. Sharpe -0.17, and the -3.8% max drawdown only looks good because average gross
exposure is 14%; scaled to the same risk as the others it would be a much larger loss.
The PBO of 0.00 is *not* good news here: no configuration was better than the others,
so the in-sample "best" was never above the median for anything. The R² of 0.01 against
the factors confirms it is genuinely market-neutral. It is uncorrelated with everything,
including with making money.

A candidate pair had to be dropped mid-project: Walgreens was taken private in 2025 and
its history disappeared from the data source. That is survivorship bias happening live.

## Reading this honestly

The single most useful comparison is Sharpe against buy and hold, and only one strategy
beats it. The second most useful is the gap between the in-sample-optimised Sharpe and the
walk-forward one, which is the overfitting penalty:

| Strategy | In-sample best | Walk-forward OOS | Penalty |
|---|---|---|---|
| ma_crossover | 0.68 | 0.58 | -0.10 |
| mean_reversion | 0.51 | 0.42 | -0.09 |
| tsmom | 0.37 | -0.06 | -0.42 |
| xsmom | 0.13 | 0.18 | +0.05 |
| pairs | 0.21 | -0.17 | -0.38 |

Even the in-sample numbers, which had the benefit of choosing parameters with full
hindsight over the test period, only reach 0.68 at best. There was no version of these
five strategies that was worth trading and got ruined by honest accounting. They were
never that good.

Two caveats that would move these numbers if fixed, both in the optimistic direction:

1. **Universes are chosen from instruments that exist today.** The stock list is
   survivorship-biased; the ETF lists are selection-biased.
2. **Costs are flat.** 5 bps of slippage is reasonable for SPY and optimistic for a
   small-cap short in 2010, and the borrow fee is a single number for every name and
   every year when the real one is a distribution with a long right tail. A volume-share
   impact model is implemented but not used here.

A third caveat was closed in the final review. The first version of this table charged
nothing to borrow stock, which makes a short leg free. Adding a flat fee cost the three
shorting strategies roughly what you would expect and changed no conclusion:

| Strategy | Borrow | Sharpe before | Sharpe after | CAGR before | CAGR after |
|---|---|---|---|---|---|
| tsmom | 30 bp | -0.05 | -0.06 | 0.79% | 0.71% |
| xsmom | 50 bp | 0.21 | 0.18 | 2.72% | 2.46% |
| pairs | 50 bp | -0.14 | -0.17 | 1.08% | 1.05% |

Two implementation bugs in `tsmom` and `xsmom` were fixed in the same pass. Neither was
reachable at the settings used here — re-running the fixed code with the borrow charge
switched off reproduces the previous table to six decimals — so the movement above is the
borrow fee and nothing else. Both are written up in [REVIEW.md](REVIEW.md).

## Reproducibility check (2026-09-28)

The whole pipeline was re-run from a fresh Yahoo download on 2026-09-28, with the same
end date (2025-08-29) and code. Four of the five strategies reproduced the previous table
to at least five significant figures. **`mean_reversion` did not**: Sharpe 0.44 → 0.42,
longest drawdown 1,102 → 636 days, turnover 10.3 → 10.6. The tables above are the new run.

The likely cause is that five of its twelve ETFs (XLB, XLE, XLK, XLU, XLY) split 2-for-1 on
2025-12-05, after the original download, and Yahoo re-adjusted their whole history. The
z-score signal is unaffected by a constant rescaling of prices, but whole-share order
rounding and the volume cap are not, so the fills differ slightly. The drawdown-duration
change is large because duration is fragile: it depends on whether equity gets back to a
previous high, and a small P&L difference can decide that. This was not isolated further,
because the original raw download was not kept (`data/cache/` is gitignored).

The lesson is the one in `data/README.md`: a result tied to a data vendor's current
adjustment is only reproducible against a stored snapshot.
