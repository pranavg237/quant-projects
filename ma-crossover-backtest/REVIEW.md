# Review

A final pass over the finished codebase, read the way a skeptical senior quant would read
it: assume every number is wrong until the code says otherwise, and assume anything that
flatters the strategy is a bug.

Seven findings were worth fixing and are fixed. Eleven are documented and left, either
because fixing them needs data this project does not have, or because the honest fix is a
caveat rather than a code change. Everything below is ordered by how much it would move a
reported number.

---

## Fixed in this pass

### 1. Short positions were free

**Severity: high — it changed three of the five headline Sharpes.**

The engine credited cash interest on short sale proceeds (`Portfolio.accrue_interest`) and
never charged the offsetting stock-loan fee. A long/short strategy therefore ran its short
leg at zero carrying cost, which is the single largest free lunch available in a backtest:
`xsmom` holds a ~50% short book continuously for sixteen years.

`Portfolio.charge_borrow` now debits `borrow_rate / periods_per_year` of short market value
every bar, threaded through `run_backtest(..., borrow_rate=)` and `StrategySpec.borrow_rate`.
The rate defaults to 0, so nothing silently changed under existing tests; the research specs
set 30 bp for ETFs and 50 bp for single stocks. Measured effect:

| Strategy | Borrow | Sharpe before → after | CAGR before → after |
|---|---|---|---|
| tsmom | 30 bp | -0.05 → -0.06 | 0.79% → 0.71% |
| xsmom | 50 bp | 0.21 → 0.18 | 2.72% → 2.46% |
| pairs | 50 bp | -0.14 → -0.17 | 1.08% → 1.05% |

It changed no conclusion, which is itself the finding: the shorting strategies were not
losing to borrow costs, they were losing on their own. This is the only change in this pass
that moved a published number — findings 2 and 3 were latent, confirmed by re-running the
fixed code with `borrow_rate = 0` and reproducing the previous table exactly.

### 2. `tsmom` silently truncated its volatility window

**Severity: latent — no reported number changed, and that is luck rather than design.**

`on_bar` asked for `lookback + 1` bars of history and then measured realised volatility
over `vol_lookback` of them. Whenever `vol_lookback > lookback` the vol estimate quietly
used fewer observations than requested, so the same nominal parameters meant different
things at different grid points. Since the weight is `vol_target / realised_vol`, a shorter
vol window is a systematically different position size, not merely a noisier one.

It never fired in the reported run: `vol_lookback` is fixed at 60 and the grid's smallest
`lookback` is 63. Re-running the fixed code with `borrow_rate = 0` reproduces the previous
`tsmom` numbers to six decimals, which is how that was established rather than assumed.
Any grid that varied `vol_lookback`, or added a `lookback` below 60, would have hit it.

Fixed to request `max(lookback, vol_lookback) + 1` bars. Two further latent problems
surfaced while fixing it, both fixed and both reachable only once the frame is longer than
the momentum window: the signal was indexed from the *start of the frame* rather than from
`lookback` bars back, and a `NaN` trailing return fell through to the `else` branch and was
treated as negative, so a name with a data gap would have been shorted on missing data.
Regression tests: `test_tsmom_uses_the_full_volatility_window_*`,
`test_tsmom_momentum_window_is_the_lookback_not_the_frame`.

### 3. `xsmom` could hold the same name long and short

**Severity: latent — never fired at the reported settings, silent and material if it had.**

`n = max(1, round(top_frac * len(ranked)))` with `top_frac = 0.5` and an odd number of
ranked names rounds both legs up past half, so `ranked[-n:]` and `ranked[:n]` overlap. The
long weight was written first and the short weight overwrote it, so the name was silently
short and the book was not dollar-neutral. The reported run ranks 70 names at `top_frac` of
0.2 or 0.3, so it never fired, and re-running with `borrow_rate = 0` reproduces the previous
`xsmom` numbers exactly. Any universe small enough for `round` to push both legs past half
would have hit it, with no error and no warning. Now capped at `len(ranked) // 2`.
Regression test: `test_xsmom_legs_never_share_a_symbol`.

### 4. `research/runner.py` had no tests at all

**Severity: high as a process failure, even though no bug was found.**

The module that produces every number in RESULTS.md was at 0% coverage — the one file where
a silent error would be invisible *and* consequential. It now has 10 tests running fully
offline against a fake downloader and fake factor data, covering the pieces that matter:
that the benchmark is loaded but kept out of the tradable universe, that a strategy ordering
a non-universe symbol raises, that the persisted CSVs are the same series that were reported
on, that folds never overlap, that the in-sample and out-of-sample comparison covers the
same span, and that one risk-free series is used everywhere. `research/` is now at 100%;
the package total went from 94.4% to 98%.

### 5. RESULTS.md was hand-typed around machine-generated numbers

**Severity: medium — the exact failure mode that makes a portfolio look dishonest.**

Every table in RESULTS.md was transcribed by hand from `reports/strategies/results.csv`.
That arrangement goes stale the first time the pipeline is re-run, and a stale number in a
public results table is indistinguishable from a doctored one. `test_results_md_matches_the_generated_table`
now parses the markdown tables and checks all three of them — headline metrics, overfitting
diagnostics, in-sample-vs-out-of-sample penalty — against the CSV, within rounding. A second
test asserts that any strategy charging borrow states its rate in the document. It caught
one inconsistency immediately (a penalty column rounded from already-rounded inputs).

### 6. Leverage was invisible

`Portfolio` allows cash to go negative and gross exposure to exceed 1, by design — that is a
strategy constraint, not a bookkeeping one. But nothing *reported* it, so an accidentally
levered backtest read exactly like an unlevered one. `BacktestResult.summary()` now reports
`max_gross_exposure` and `min_cash_weight`, and `cost_drag_pct` includes borrow.

### 7. Dead branch in the event loop

`if i >= first_eval:` inside a loop that starts at `first_eval` is always true. Harmless, but
it invites the reader to believe there is a case where the strategy is not called, which is
exactly the kind of misdirection that hides a real bug later. Removed.

---

## Known and left, in order of how much they would move a number

### A. The universes are built from instruments that exist today

This is the largest remaining bias in the project and it cannot be fixed without a
point-in-time constituent file. The framework side is done — `PointInTimeUniverse` takes
membership intervals, the engine refuses orders for non-members and force-liquidates names
that leave — but `research/universes.py` is still a hand-written list of survivors. Every
company that was a large cap in 2010 and then failed is missing, and a momentum strategy is
precisely the kind that would have been short many of them. `xsmom`'s 0.18 is a ceiling.

The one honest datapoint: Walgreens was taken private mid-project, its history vanished from
the data source, and a pair had to be dropped. That is the bias happening in real time, on a
three-month horizon, in a single 70-name universe.

### B. The deflated Sharpe ratio counted the wrong number of trials (partly fixed)

The research pipeline deflated each strategy only by its own parameter grid: 17
configurations for `ma_crossover`, which gave its in-sample best a DSR of 0.99
(`dsr_is` in `results.csv`, never shown in RESULTS.md). The number that belongs there is
the number of configurations *ever tried*, across every strategy.

`scripts/multiple_testing.py` now does that: N = 63, every configuration of all five
grids, with the cross-trial variance of all 63 Sharpes. Under that count the MA
crossover's DSR is 0.16 in-sample and 0.08 out of sample, and RESULTS.md shows how it
moves under other reasonable assumptions (it clears 0.95 only at about five effective
trials). The formula itself is now pinned to the worked example in Bailey & Lopez de Prado
(2014) by a test.

What is still not fixed: 63 counts coded grid points only. Ideas considered and never
coded, and earlier versions of the code, are not in any file, so 63 remains a lower bound
on the true count. The per-strategy `dsr_is` column is still written by the pipeline for
continuity and should not be quoted.

### C. Pairs selection does not correct for multiple testing

`PairsTrading._refit` runs 17 Engle-Granger tests each quarter and keeps everything with
p < 0.05. At 17 tests that is roughly one false cointegration per refit by construction, and
the p-value is not adjusted. The right fix is a Bonferroni or FDR threshold; the reason it
is not applied is that `pairs` already loses money with the generous threshold, so tightening
it only makes a negative result more negative.

### D. Cost models are flat in size, name and time

5 bps of slippage is about right for SPY in 2024 and clearly optimistic for a small-cap
short in 2010. A `VolumeShareSlippage` with quadratic impact exists and is tested but is not
used for the reported runs, and `max_volume_share` is `None`, so nothing caps an order at a
fraction of the bar. At $1m of capital the positions are small enough that this does not
bind — `xsmom`'s largest single position over the whole out-of-sample period is 0.34% of
that bar's dollar volume, and the 99.9th percentile is 0.07% — but the results say nothing
about capacity, and a capacity claim is what the flat
model would break first. Likewise `borrow_rate` is one number per strategy; real borrow is a
per-name, time-varying distribution with a long right tail, and the names a momentum short
book wants are drawn disproportionately from that tail.

### E. Delisting has no haircut

When a symbol leaves the universe the engine liquidates it at that bar's fill price, or at
the last mark if there is no price. A real delisting is not a clean exit at the last quote;
CRSP's delisting returns average roughly -30% for performance-related deletions. With
`from_data` universes over liquid instruments this almost never fires, but it is the wrong
default to build a bankruptcy-sensitive strategy on.

### F. Cash borrowing is charged at the risk-free rate

`accrue_interest` multiplies cash by `1 + rf` whether cash is positive or negative, so a
margin loan costs exactly what T-bills pay. No broker offers that. None of the five reported
strategies runs negative cash materially — `min_cash_weight` now makes that checkable rather
than assumed — but the engine would happily run a 3x levered backtest at zero spread.

### G. Volume is not adjustment-consistent with price

Prices are total-return adjusted; `volume` is passed through as the vendor reports it. If
the vendor's volume is not split-adjusted on the same basis, then any rule comparing an
adjusted share count to a raw volume — `max_volume_share`, `VolumeShareSlippage` — is wrong
by the cumulative split factor. Nothing in the reported results uses volume, so no number
here is affected, but the first person to switch on the volume cap on a pre-split history
will get a silently wrong answer.

### H. Two strategies rebalance to target, two let weights drift

`tsmom` and `xsmom` re-issue target weights on every rebalance bar so price drift does not
accumulate. `mean_reversion` and `pairs` only trade when the target *set* changes, so an
open position drifts away from its nominal weight between signals. Both are defensible and
both are documented, but they are not the same policy, and comparing turnover across the two
groups compares slightly different things. `mean_reversion`'s 10.3x turnover would be higher
still under the drift-correcting policy, which makes it the strategy most sensitive to the
choice.

### I. The rebalance calendar depends on where the data starts

`is_rebalance_bar(ctx, n)` for an integer `n` tests `position_index % n`, and
`position_index` counts from the first bar in the `PriceData`, including warm-up. Change the
download start date and every integer-cadence strategy rebalances on different days. The
`"monthly"` path is calendar-based and immune. No reported strategy uses the integer path,
but it is a reproducibility trap.

### J. One data vendor, and it revises history

Everything comes from Yahoo Finance, which restates splits, dividends and adjusted closes
without notice, and which withdrew a symbol's entire history mid-project. `data/cache/MANIFEST.json`
records the snapshot dates, which makes the results reproducible from the cache but not
independently verifiable. A second source (Stooq, Tiingo, a vendor file) reconciled against
the first is the standard defence and is not implemented.

### K. Annualisation is not identical in every metric

`metrics.summary` computes CAGR over `years_spanned + 1/ppy`, counting the period the first
return accrued over; `metrics.cagr`, and therefore `calmar`, uses `years_spanned` alone. On a
sixteen-year series the two differ in the fourth decimal. Similarly, `turnover` halves the
initial build from cash, which is genuinely one-sided, understating annual turnover by about
`0.5 / years`. Both are immaterial at these horizons and both would matter on a one-year
backtest.

---

## What holds up

Stated plainly, because a review that only lists problems is not a review.

- **Lookahead is structural, not tested-for.** `Context` cannot address a bar past
  `position_index`, and `ExecutionSimulator` fills only against bars after submission. There
  is no code path that fills at the price that generated the signal. The bias tests
  (`test_bias_*`) verify this on synthetic data where a leak would produce an impossible
  Sharpe — and the test suite documents *why* the detector needs overnight gaps to fire,
  which is the kind of thing that is usually learned twice.
- **The engine and the vectorised reference agree to 4e-15** on the same strategy, which is
  a real cross-check: two independent implementations of the cost and sizing arithmetic
  producing the same equity curve.
- **Parameters are never chosen on scored data.** Walk-forward is the only place parameters
  are selected, and the in-sample-best is reported beside the out-of-sample result over the
  same dates, so the overfitting penalty is visible rather than inferred. A test splices a
  different price path in after one fold's training window and checks that every
  parameter choice up to that fold is unchanged; moving the training window forward by
  six months makes it fail.
- **The reporting is in excess terms end to end.** An earlier version computed the summary
  Sharpe on excess returns and the overfitting diagnostics on raw ones, which made the
  mostly-in-cash `pairs` strategy show a PSR of 0.999 alongside a Sharpe of -0.14. That is
  fixed, and `excess_returns` is now the single entry point.
- **The results are negative and are reported as negative.** Four of five strategies lose to
  buy-and-hold, no strategy has a significant five-factor alpha, and the document says so in
  its first paragraph rather than its last.

## What a desk would still reject

A point-in-time universe and a second data source are not optional, and without them the
cross-sectional results are not evidence. Beyond that: no risk model, so position sizing is
`1/n` and vol-scaling rather than anything that knows about covariance; no execution model
below the daily bar; no capacity analysis; and a single backtest per strategy rather than a
distribution over data-generating assumptions. The framework is sound enough that adding
each of these is a contained change, which is the most that can be claimed for it.
