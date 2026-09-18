# Results

Five strategies, each run through the same pipeline: walk-forward parameter selection,
bootstrap and overfitting diagnostics, and a Fama-French factor regression. Nothing here
was chosen after seeing the out-of-sample numbers, and nothing that failed has been
removed. Four of the five do not beat buying and holding the index.

Rebuild everything with:

```bash
python scripts/run_strategies.py      # ~7 minutes, writes reports/strategies/
python scripts/build_tearsheets.py    # writes reports/tearsheets/
```

## How these numbers were produced

| | |
|---|---|
| Fills | the open of the bar **after** the signal bar |
| Costs | 5 bps slippage per side plus 1 bp commission on notional |
| Cash | earns the Ken French daily risk-free rate; Sharpe is on excess returns |
| Parameters | re-chosen every year on the preceding 5 years, by Sharpe, in-sample only |
| Reported series | the stitched out-of-sample windows, never the training windows |
| Benchmark | buy and hold SPY over the identical dates |
| Data | Yahoo Finance daily bars, split- and dividend-adjusted, snapshot of 2025-08-29 |

## The table

Out-of-sample, net of costs. `ma_crossover` starts in 2005 because it needs less history;
the other four start in 2010 after their first 5-year training window.

| Strategy | CAGR | Vol | **Sharpe** | Sortino | Max DD | DD days | Turnover | Exposure |
|---|---|---|---|---|---|---|---|---|
| SPY buy & hold (2010-2025) | 13.8% | 17.3% | **0.76** | 1.07 | -33.7% | 709 | 0.0 | 100% |
| ma_crossover | 8.2% | 12.0% | **0.58** | 0.79 | -22.4% | 808 | 2.0 | 78% |
| mean_reversion | 6.6% | 13.9% | **0.44** | 0.62 | -39.6% | 1,102 | 10.3 | 32% |
| xsmom | 2.7% | 8.5% | **0.21** | 0.28 | -16.1% | 799 | 5.9 | 100% |
| tsmom | 0.8% | 6.4% | **-0.05** | -0.06 | -19.9% | 2,051 | 3.8 | 91% |
| pairs | 1.1% | 1.5% | **-0.14** | -0.21 | -3.8% | 2,259 | 4.7 | 14% |

Is any of it real? `psr` is the probabilistic Sharpe ratio (P that the true Sharpe exceeds
zero); the bootstrap interval is a 95% stationary-bootstrap range for the out-of-sample
Sharpe; `pbo` is the probability of backtest overfitting from combinatorially symmetric
cross-validation, where anything near or above 0.5 means the in-sample best configuration
is a coin flip out of sample.

| Strategy | Sharpe | Bootstrap 95% CI | PSR | PBO | FF5 alpha (ann.) | t | R² | Momentum beta | t |
|---|---|---|---|---|---|---|---|---|---|
| ma_crossover | 0.58 | 0.21 to 1.01 | 0.995 | 0.54 | +2.4% | 1.22 | 0.41 | +0.14 | 5.7 |
| mean_reversion | 0.44 | 0.01 to 0.98 | 0.956 | 0.71 | -2.0% | -0.78 | 0.57 | +0.03 | 1.0 |
| xsmom | 0.21 | -0.27 to 0.64 | 0.791 | 0.93 | +1.3% | 0.62 | 0.07 | +0.39 | 21.1 |
| tsmom | -0.05 | -0.51 to 0.46 | 0.427 | 0.44 | -2.1% | -1.20 | 0.12 | +0.13 | 7.7 |
| pairs | -0.14 | -0.60 to 0.29 | 0.288 | 0.00 | -0.2% | -0.69 | 0.01 | -0.00 | -1.0 |

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

Sharpe 0.44 with a -39.6% drawdown, worse than the index it is trading, on only 32%
average exposure. Turnover of 10.3x a year is the highest here, so it is also the most
cost-sensitive: it is the one strategy whose ranking would change materially under a
harsher cost model. Its market beta is 0.60 with R² of 0.57 and a *negative* five-factor
alpha, which is the signature of an expensive way to be long. 13 of 16 folds were
positive, but PBO of 0.71 says the parameter choice does not generalise.

### xsmom: momentum, with the losers removed from history

Long the top 20% and short the bottom 20% of 70 large-cap stocks by 12-1 momentum. Sharpe
0.21 with a momentum beta of +0.39 and a t-statistic of 21: it is a momentum factor
tracker with a 0.07 R² against everything else, and the factor loading eats the whole
return. PBO of 0.93, the worst here, means the in-sample best configuration is almost
always below median out of sample.

This is also the most biased result on the page. The 70 stocks are the large caps that
exist *today*. Every company that was large in 2010 and then failed, was acquired or fell
out of the index is missing, and momentum strategies are exactly the kind that would have
been short many of them. Read 0.21 as a ceiling, not an estimate.

### tsmom: the textbook strategy that stopped working

Long assets with positive trailing returns, short the rest, volatility-scaled, across 13
asset-class ETFs. Sharpe -0.05: it lost money net of costs over 16 years. Time-series
momentum is one of the best-documented anomalies in the literature (Moskowitz, Ooi and
Pedersen 2012), which makes it a useful control: the implementation is faithful, the data
is clean, and it still does not work after 2010. Either the anomaly decayed after
publication, or the 2010-2025 sample, with its single dominant equity trend and two
violent reversals, is hostile to it. The longest underwater stretch was 2,051 days, more
than eight years.

### pairs: cointegration finds pairs, not profits

Engle-Granger tests on 17 candidate pairs each quarter, trading the z-score of the frozen
spread. Sharpe -0.14, and the -3.8% max drawdown only looks good because average gross
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
| mean_reversion | 0.51 | 0.44 | -0.07 |
| tsmom | 0.38 | -0.05 | -0.42 |
| xsmom | 0.16 | 0.21 | +0.05 |
| pairs | 0.23 | -0.14 | -0.37 |

Even the in-sample numbers, which had the benefit of choosing parameters with full
hindsight over the test period, only reach 0.68 at best. There was no version of these
five strategies that was worth trading and got ruined by honest accounting. They were
never that good.

Three caveats that would move these numbers if fixed, all of them in the optimistic
direction:

1. **Universes are chosen from instruments that exist today.** The stock list is
   survivorship-biased; the ETF lists are selection-biased.
2. **Short positions pay no borrow cost.** `tsmom`, `xsmom` and `pairs` all short.
3. **Costs are flat.** 5 bps of slippage is reasonable for SPY and optimistic for a
   small-cap short in 2010. A volume-share impact model is implemented but not used here.
