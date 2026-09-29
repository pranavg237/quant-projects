# Results

Five strategies, each run through the same pipeline: walk-forward parameter selection,
bootstrap and overfitting diagnostics, and a Fama-French factor regression. Nothing here
was chosen after seeing the out-of-sample numbers, and nothing that failed has been
removed. Four of the five do not beat buying and holding the index.

Rebuild everything with:

```bash
python scripts/run_strategies.py      # ~25 minutes, writes reports/strategies/
python scripts/build_tearsheets.py    # writes reports/tearsheets/
python scripts/multiple_testing.py    # ~3 seconds, reads reports/strategies/, writes reports/multiple_testing/
python scripts/ma_sensitivity.py      # ~10 seconds, writes reports/sensitivity/
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
| Data | Yahoo Finance daily bars, split- and dividend-adjusted, through 2025-08-29; downloaded 2026-09-28 and committed in `data/snapshot-2026-09-28/` (sha256 per file), together with the Ken French daily factor files |

## The table

Out-of-sample, net of costs. `ma_crossover` starts in 2005 because it needs less history;
the other four start in 2010 after their first 5-year training window.

| Strategy | CAGR | Vol | **Sharpe** | Sortino | Max DD | DD days | Turnover | Exposure |
|---|---|---|---|---|---|---|---|---|
| SPY buy & hold (2010-2025) | 13.8% | 17.3% | **0.76** | 1.07 | -33.7% | 709 | 0.0 | 100% |
| ma_crossover | 8.2% | 12.0% | **0.58** | 0.79 | -22.4% | 808 | 2.0 | 78% |
| mean_reversion | 6.6% | 13.9% | **0.44** | 0.62 | -39.6% | 1,102 | 10.4 | 32% |
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
| mean_reversion | 0.44 | 0.01 to 0.98 | 0.956 | 0.71 | -2.0% | -0.78 | 0.57 | +0.03 | 1.0 |
| xsmom | 0.18 | -0.30 to 0.61 | 0.757 | 0.91 | +1.0% | 0.50 | 0.07 | +0.39 | 21.1 |
| tsmom | -0.06 | -0.52 to 0.44 | 0.407 | 0.43 | -2.2% | -1.25 | 0.12 | +0.13 | 7.7 |
| pairs | -0.17 | -0.63 to 0.27 | 0.257 | 0.00 | -0.3% | -0.80 | 0.01 | -0.00 | -1.0 |

**Not one strategy has a statistically significant alpha.** Every t-statistic on the
Fama-French five-factor intercept is below 2 in absolute value. The largest (the MA
crossover, t = 1.22) is exactly what you would expect to see by chance from five tries.

## Correcting for the search: the deflated Sharpe ratio

The PSR column above asks whether one Sharpe ratio is distinguishable from zero. That is
the wrong question for a project that tried many things and is writing up the best one.
The deflated Sharpe ratio (Bailey & Lopez de Prado 2014) asks instead whether the Sharpe
beats what **the best of N tries would show if every try were pure noise**. That hurdle,
`SR0`, grows with the number of trials `N` and with how widely their Sharpe ratios are
spread (`V`). The DSR is the probability that the true Sharpe is above `SR0`.

**The trial count is every configuration of every strategy: N = 63** (17 `ma_crossover` +
12 `tsmom` + 8 `xsmom` + 18 `mean_reversion` + 8 `pairs`, read from each strategy's
`grid.csv`). Earlier versions of this pipeline deflated each strategy only by its own
grid, which for the MA crossover meant N = 17. The 63 Sharpe ratios have a standard
deviation of 0.38 (`V` = 0.145), so the expected best of 63 noise strategies is an
annualised Sharpe of **0.90**.

| Series | Sharpe | PSR | DSR, N = 63 |
|---|---|---|---|
| ma_crossover walk-forward OOS | 0.58 | 0.995 | 0.076 |
| mean_reversion walk-forward OOS | 0.44 | 0.956 | 0.036 |
| xsmom walk-forward OOS | 0.18 | 0.757 | 0.002 |
| tsmom walk-forward OOS | -0.06 | 0.407 | 0.000 |
| pairs walk-forward OOS | -0.17 | 0.257 | 0.000 |
| ma_crossover 10/200, in-sample best of all 63 | 0.68 | 0.999 | 0.160 |

The last row is the textbook use of the DSR: the single best configuration found, over the
2005-2025 span on which it was best. Its PSR of 0.999 says "almost certainly not zero".
Its DSR of 0.16 says the observed 0.68 is well inside what the luckiest of 63 noise
strategies would produce, far from the 0.95 a significant result needs. The same pipeline used to report 0.99 for this configuration (N = 17 and
the 17 MA Sharpes' own variance), which is the number REVIEW.md warned should not be
believed.

**How much does this depend on the assumptions?** A lot, and the direction matters, so
here is the whole grid for the in-sample best (walk-forward OOS in brackets). Rows are
trial counts, columns are the cross-trial variance `V`:

| N | V = 0.006 (17 MA configs only) | V = 0.050 (noise in one Sharpe) | V = 0.145 (all 63 configs) |
|---|---|---|---|
| 1 (no correction, = PSR) | 0.999 (0.995) | 0.999 (0.995) | 0.999 (0.995) |
| 5 (one per strategy) | 0.996 (0.986) | 0.965 (0.920) | 0.839 (0.715) |
| 17 (MA grid only) | 0.992 (0.977) | 0.881 (0.777) | 0.465 (0.303) |
| **63 (all five grids)** | 0.987 (0.965) | 0.741 (0.588) | **0.160 (0.076)** |
| 162 (+ every heatmap cell below) | 0.983 (0.955) | 0.623 (0.456) | 0.060 (0.023) |

How to read it:

* **The left column is not a credible assumption.** The 17 MA configurations have a
  small Sharpe spread (0.43 to 0.68) because they are near-copies of one another: they
  are all long SPY most of the time. Near-copies should be handled by counting fewer
  *effective* trials, not by plugging in their tiny spread as if it were the noise in a
  Sharpe estimate. The noise in one 20-year Sharpe estimate is a standard error of 0.22
  (`V` = 0.050, middle column).
* **The right column is the paper's recipe and is on the harsh side here.** Part of the
  0.38 spread across all 63 configurations is real: `pairs` really is a different, worse
  strategy (Sharpe down to -1.06 in-sample), not a noisy draw of the same one.
* **63 is a floor on the true count, not a ceiling.** It counts coded grid points. It does
  not count strategies considered and never coded, or earlier versions of the code.

The conclusion does not hinge on which cell you pick. The MA crossover's Sharpe clears
the conventional 0.95 bar only if the whole project is treated as about five independent
tries with noise-sized variance (0.965 in-sample, 0.920 out of sample), or with the
left-column variance that the previous paragraph explains away. Under the paper's own
recipe and an honest trial count it is 0.16 in-sample and 0.08 out of sample. **The
evidence that the MA crossover has any skill beyond selection luck is weak.** That agrees
with its PBO of 0.54 and its insignificant factor alpha, which were computed a different
way.

Caveats: the trial Sharpes come from slightly different windows (`ma_crossover` from
2005, the others from 2010), and the DSR's expected-maximum formula assumes the trials are
independent, which none of these grids are.

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

#### Plateau or spike? The parameter surface

The heatmap scores 116 fast/slow pairs (fast 5 to 150 days, slow 50 to 300) on SPY over the
same 2005-01-03 to 2025-08-29 window, net of the same costs, as excess Sharpe. **It is
in-sample by construction**: every cell has seen the whole period. It is a robustness
diagnostic and nothing is picked from it. Choosing the brightest cell here would be
exactly the overfitting the walk-forward exists to prevent.

![MA crossover Sharpe by fast and slow window](reports/sensitivity/ma_sharpe_heatmap.png)

Numbers behind it: [`reports/sensitivity/ma_sharpe_grid.csv`](reports/sensitivity/ma_sharpe_grid.csv)
and [`ma_sharpe_points.csv`](reports/sensitivity/ma_sharpe_points.csv).

| Point | Sharpe | Mean of its 8 neighbours | Worst neighbour | Reading |
|---|---|---|---|---|
| 10/75, the best cell on the surface | 0.73 | 0.52 | 0.41 | a spike |
| 10/200, in-sample best of the 17-point grid | 0.68 | 0.62 | 0.59 | a modest ridge |
| 50/200, the textbook rule | 0.54 | 0.57 | 0.54 | the level of buy-and-hold |
| Buy-and-hold SPY, same dates | 0.53 | | | |

What the surface says:

* **It is a broad, low plateau at roughly buy-and-hold's Sharpe, with noise on top.** The
  median cell is 0.55 against buy-and-hold's 0.53, and the middle half of all cells sits
  between 0.52 and 0.59. 80 of 116 cells beat buy-and-hold in-sample, by a median of
  0.05.
* **The best cell is a spike and would be the wrong thing to report.** 10/75 has a Sharpe
  of 0.73, but its neighbours average 0.52 and one is 0.41. Moving one step in any
  direction loses most of the edge, which is what a lucky fit looks like.
* **10/200 sits on a ridge, not a spike, but the ridge is not much higher than the
  plain.** A band of fast 10-20 and slow 175-225 scores 0.58-0.68. That is the most
  robust-looking region, and it is still only about 0.1 above buy-and-hold.
* **None of these differences are statistically meaningful.** The whole range across the
  surface, 0.39 to 0.73, is 1.5 standard errors of a single cell's Sharpe (0.22). The
  surface cannot tell these parameter choices apart. This is why the walk-forward picked
  10 different settings in 21 folds (outlined in the figure), and why 3 of those settings
  sit below buy-and-hold on the full sample.

### mean_reversion: buys dips, and buys them all the way down

Sharpe 0.44 with a -39.6% drawdown, worse than the index it is trading, on only 32%
average exposure. Turnover of 10.4x a year is the highest here, so it is also the most
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
| mean_reversion | 0.51 | 0.44 | -0.07 |
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

## Reproducibility

**Every number on this page is computed from the committed snapshot
`data/snapshot-2026-09-28/`** (see [data/README.md](data/README.md)): the Yahoo prices and
Ken French factor files as downloaded on 2026-09-28, with a sha256 per file. The loaders read
it by default and never touch the network. Two full runs of `scripts/run_strategies.py` from
it produced byte-identical reports, apart from the recorded runtime. So did two runs each of
the heatmap and deflated-Sharpe scripts.

The snapshot was needed because two fresh Yahoo downloads on the same day gave two
different `mean_reversion` results: Sharpe 0.42, longest drawdown 636 days, turnover 10.6 on
one, and 0.44, 1,102 days, 10.4 on the other. The other four strategies agreed to about 1e-5
in Sharpe. The snapshot is the second download, so the tables above changed from 0.42 to
0.44 when it was adopted.

**What caused the flip.** It was one exit decision. `mean_reversion` exits a position when
the z-score of the close against its 20-day mean recovers to 0. On 2021-04-27 XLE's z-score
in the snapshot is **+0.0000039**, so the strategy exits the next morning. Yahoo's prices are
float32 values, good to about 7 significant digits, and two downloads can differ in the last
of them. Two tests:

* Adding random noise of that size (6e-8 relative) to every price flipped exactly this
  XLE exit in two of three draws. The in-sample best configuration's Sharpe then moved from
  0.506795 to 0.510777; the earlier download's run had 0.510776.
* Lowering that one XLE close by 3e-7 (19.868521 → 19.868515), and changing nothing else,
  reproduces the earlier download's whole walk-forward result: Sharpe 0.4215 against 0.4219,
  longest drawdown 636 days, turnover 10.57 against 10.57, PBO 0.709 against 0.708. The
  small remaining gaps are the 1e-6-level differences every strategy shows between
  downloads.

A walk-forward amplifies the flip: a slightly different training Sharpe can change which
parameters win a fold, and the drawdown duration jumps because it depends on whether equity
regains an old high. The earlier explanation on this page, that December 2025 ETF splits
changed order rounding, was wrong. These runs use fractional shares and no volume cap, and
rescaling every price by a constant changes nothing.

The lesson is less about data plumbing than about the strategy. A result that moves by 0.02
of Sharpe when one price changes in its seventh digit is reporting its own noise floor.
`mean_reversion`'s 0.44 should be read as "somewhere around 0.42-0.44, and probably less
certain than that".
