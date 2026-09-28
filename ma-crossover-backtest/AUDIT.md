# Audit of `ma-crossover-backtest` (as of commit 7eba071)

Scope: everything in the repository at the initial commit: `backtest.py` (62 lines),
`main.py` (PyCharm boilerplate), `README.md.py`, `backtest.png`, `.idea/`, `.DS_Store`.
The sibling folder `project a` (an `ffmodel` Fama-French package) is a separate,
un-versioned project and is out of scope, although its French-library parser
informed the Phase 5 design.

Every number below was reproduced by running the original logic against Yahoo data on
2026-09-18 (SPY, 2018-01-01 to 2024-01-01). The original script reproduces its README
figures exactly: 48.24% (50/200) and 36.93% (20/50) versus 95.54% buy-and-hold.

## 1. Architecture

There is no architecture. `backtest.py` is a top-level script that:

1. downloads SPY with `yfinance` (`auto_adjust=True`),
2. computes two rolling means of the adjusted close,
3. sets `signal = 1` when the short MA is above the long MA,
4. computes `strategy_return = market_return * signal.shift(1)`,
5. prints two total-return numbers and saves a chart.

Nothing is a function, nothing is importable, nothing is testable, and every parameter
is a module-level constant. `main.py` is the untouched PyCharm template and is unrelated
to the project. The README is saved as `README.md.py`, so GitHub renders it as Python.

## 2. Code quality

| Issue | Where | Severity |
|---|---|---|
| Script-only, no functions, no `__main__` guard | `backtest.py` | high |
| No dependency manifest (`requirements.txt` / `pyproject.toml`), no pinned versions | repo | high |
| No tests, no linting, no type hints | repo | high |
| README has a `.py` extension | `README.md.py` | medium |
| PyCharm boilerplate committed | `main.py` | low |
| IDE metadata and OS junk committed | `.idea/`, `.DS_Store` | low |
| Generated artefact committed | `backtest.png` (230 KB) | low |
| `df[["Close"]]` on a modern `yfinance` returns MultiIndex columns; the script still works only because pandas happens to broadcast the single-ticker level. Any second ticker silently breaks it | `backtest.py:14` | medium |
| `position` column is really a *trade* indicator (`signal.diff()`), not a position | `backtest.py:26` | low (naming) |
| Only total return is reported; no annualisation, volatility, Sharpe, drawdown or trade count | `backtest.py:36-39` | high for a research artefact |
| `plt.show()` blocks in headless runs | `backtest.py:62` | low |
| Data is re-downloaded on every run; Yahoo's back-adjusted series changes every ex-dividend date, so the printed numbers drift over time and are not reproducible | `backtest.py:13` | medium |

## 3. Correctness issues, ranked by how much they inflate the backtest

Measured impact uses the 50/200 configuration unless stated. "Inflation" is the
difference between the reported figure and what the same rule would have earned when
the issue is corrected.

### 3.1 Same-bar fill: the trade is executed at the close that generated the signal (HIGH)

`strategy_return[t] = market_return[t] * signal[t-1]`. The signal at `t-1` is computed
from the rolling mean *including* `Close[t-1]`. The position then earns
`Close[t] / Close[t-1] - 1`, i.e. it was bought at `Close[t-1]`, the very price that
produced the signal. A real trader cannot observe a closing price and trade at it. The
README's claim that a "1-day lag" removes lookahead is only half true: it prevents
`signal[t]` from multiplying `return[t]` (pure lookahead) but it still fills at the
signal bar.

| Fill assumption | Total return | Sharpe |
|---|---|---|
| Same bar's close (original) | 48.24% | 0.50 |
| Next bar's open (realistic, what Phase 2 uses) | 43.65% | 0.46 |
| Next bar's close | 45.19% | 0.47 |

Inflation: about 4.6 percentage points of a 48-point return, roughly a tenth of the
whole result, from one shift.

*Re-checked 2026-09-28* with an independent script on a fresh Yahoo download (SPY,
2018-01-02 to 2023-12-29, no costs): same-bar close fill 57.80% / Sharpe 0.54, next-open
fill 52.93% / Sharpe 0.51. The levels differ from the table because the data was
re-downloaded and the original script's warm-up handling was not reconstructed exactly,
but the inflation (4.9 points) reproduces. Trend-following signals are systematically flattered by
this bug because the bar after a breakout tends to continue in the breakout direction.

### 3.2 No transaction costs or slippage (MEDIUM)

Zero commission, zero spread, zero market impact. For 7 round-trip legs on SPY this is
small (about 0.5 points at 5 bps per side) but it grows linearly with trade count: the
20/50 variant trades 25 times. Any parameter search without costs will drift toward
faster, more frequently trading settings, which is exactly the direction where costs
bite. Costs are also what makes "does not work" verdicts honest.

### 3.3 Unequal comparison window: the strategy is flat for the first 200 bars of the test period while buy-and-hold is invested from day one (MEDIUM, direction: deflates)

The download starts on the first day of the test window, so the 200-day MA is `NaN`
until roughly October 2018 and the strategy sits in cash while SPY rises. With a proper
warm-up (data from 2017, evaluation from 2018) the same rule returns 57.8% instead of
48.2%. This is a bias against the strategy, but it is still a correctness bug: the two
lines on the chart are not measured over the same window and the "why it underperforms"
narrative in the README is partly an artefact of it.

### 3.4 Cash earns nothing when the strategy is out of the market (LOW, direction: deflates)

The strategy is flat about 40% of the time and earns zero. T-bills paid 0 to 5% over
the period. This is a modelling choice rather than a bug, but it should be explicit and
configurable, and Sharpe must use the same risk-free rate.

### 3.5 In-sample parameter reporting (MEDIUM)

Two parameter sets (50/200 and 20/50) are run on the same six years and both are
reported. There is no train/test split, so "50/200 beats 20/50" is a statement about
the sample, not about the rule. The classical 50/200 pair is itself famous because it
worked on past US index data, so even a single run carries selection bias.

### 3.6 Survivorship and selection bias (LOW for this script, HIGH for any extension)

SPY is a survivor by construction (an index of survivors, rebalanced by a committee)
and a well-known long-run winner. A single-asset test cannot be survivorship-biased in
the usual sense, but the moment this script is pointed at a "universe of stocks" pulled
from today's index membership, every stock that was delisted, acquired or dropped is
missing and momentum/mean-reversion results become meaningless. The framework needs
point-in-time universe membership before any cross-sectional strategy is run.

### 3.7 Adjusted prices: correct here, fragile in general (LOW)

`auto_adjust=True` gives a total-return price series (splits and dividends). Using it
for P&L is right. Using it for the *signal* is right *only because* an MA crossover is
scale-invariant: the back-adjustment factor for all dates before `t` is a constant at
time `t`, so `MA_short > MA_long` is unchanged by future dividends. Any signal that is
not scale-invariant (price levels, dollar thresholds, ATR in dollars, round-lot sizing)
would be reading future dividend information. For reference, computing the same signal
on the *unadjusted* close (which has dividend drops in it) gives 37.8% rather than
48.2%, so the choice is not cosmetic. The fixed design keeps three series: raw close,
split-adjusted close (for anything level-based) and total-return close (for P&L).

### 3.8 Signal alignment / off-by-one details (LOW)

* `signal.diff()` marks the trade on the signal day. With the same-bar fill above that
  is consistent; with a next-open fill the marker is one bar early. The chart's buy and
  sell arrows are drawn at the *signal* close, not the fill price.
* `market_return` and both cumulative series start with `NaN`; `cumprod` skips it, so
  the result is right, but `market_cumulative.iloc[0]` is `NaN` rather than 1.
* Ties (`MA_short == MA_long`) map to flat, which is fine but undocumented.

### 3.9 Timezones and calendar (LOW)

Daily bars from Yahoo arrive tz-naive in exchange-local dates, and `end` is exclusive
(data ends 2023-12-29). Nothing in the script asserts either fact. Switching the
interval to intraday returns tz-aware UTC timestamps, and mixing the two raises in
pandas. The French factor files (Phase 5) are also naive exchange dates, so the fixed
code normalises every index to tz-naive `DatetimeIndex` at the data boundary and
refuses tz-aware input.

### 3.10 No risk or trade statistics (HIGH for interpretation, no inflation)

Total return alone says nothing. 48% with a 34% drawdown and 43% with a 12% drawdown
are different strategies. Sharpe, Sortino, max drawdown and its duration, exposure,
turnover and hit rate are all missing.

## 4. Ranked summary

| Rank | Issue | Direction | Measured effect (50/200) |
|---|---|---|---|
| 1 | Same-bar close fill (3.1) | inflates | +4.6 pts return, +0.04 Sharpe |
| 2 | Zero costs and slippage (3.2) | inflates | +0.5 pts at 5 bps; scales with trades |
| 3 | In-sample parameter choice (3.5) | inflates | unquantifiable without walk-forward |
| 4 | No risk metrics (3.10) | misleads | n/a |
| 5 | Unequal comparison window (3.3) | deflates strategy | -9.6 pts vs. proper warm-up |
| 6 | Zero cash yield (3.4) | deflates | roughly -3 pts over 2018-2023 |
| 7 | Survivorship risk for extensions (3.6) | inflates | n/a for SPY alone |
| 8 | Adjusted-price fragility (3.7) | either | 0 here; 10 pts if signal type changes |
| 9 | Alignment / chart markers (3.8) | cosmetic | 0 |
| 10 | Timezone assumptions (3.9) | latent | 0 |

## 5. What Phase 2 changes

* Signals are generated at the close of bar `t` and filled at the open of bar `t+1`
  with configurable slippage and commission. A test asserts that the strategy's return
  on the signal bar is zero.
* Costs default to 5 bps slippage plus a per-share commission and are always reported.
* Warm-up data is fetched before the evaluation window; the comparison is made over the
  same dates for strategy and benchmark.
* Cash can earn a configurable risk-free rate (default: French `RF`, which is also used
  for Sharpe).
* A full metrics set is computed and printed.
* Data is snapshotted to disk so results are reproducible.
* The repository gets a package layout, `pyproject.toml`, tests, `ruff` and `mypy`.
