# ffmodel — Fama-French factor models

A Python package and command-line tool for working with the Fama-French family of
factor models, using the official factor data from Kenneth French's data library.

| | |
|---|---|
| **Models** | CAPM, Fama-French 3-factor, Carhart 4-factor, Fama-French 5-factor, Fama-French 6-factor (FF5 + momentum) |
| **Data** | Downloads, parses and caches French library files (factors and any test-portfolio set, monthly or daily). Asset returns from Yahoo Finance or your own CSV. Dated snapshots for offline, exactly reproducible runs |
| **Time-series regressions** | Alpha and betas with Newey-West, White or classical standard errors, and a side-by-side comparison of the t-stats under each; Holm-adjusted alpha p-values across assets; annualized alpha, residual volatility, information ratio; rolling-window estimates with 95% bands and a test of whether exposures changed; side-by-side model comparison on a common sample |
| **Attribution** | Splits an asset's excess return into alpha, each factor's contribution and residual, and splits its variance the same way |
| **Asset pricing tests** | Gibbons-Ross-Shanken (GRS) test with the Fama-French (2015) summary statistics; two-pass Fama-MacBeth risk premia with Shanken and Newey-West corrections, full-sample or rolling betas |
| **Factor diagnostics** | Summary statistics, correlations and spanning regressions (is a factor explained by the others?) |
| **Reports** | Each command writes a Markdown report with charts (PNG) and every table as CSV |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # installs ffmodel and the `ffmodel` command
.venv/bin/python -m pytest                  # 61 offline tests
```

This is an analysis tool, not a trading strategy, so there is no Sharpe ratio, drawdown
or turnover of its own to report. Its outputs are factor loadings, alphas and asset-pricing
test statistics. The worked examples use French data through July 2026 (`--end 2026-07`),
and all three are built **offline from committed snapshots**. Each snapshot's `manifest.json`
records the download time, URLs, SHA-256 hashes, file sizes and sample periods, and the
commands below reproduce the reports byte for byte. Every file is checked against its hash
when read, including the Yahoo `returns.csv` (its hash was added to the 2026-09-28
manifest after the fact, from the file as committed with the snapshot).

- [example-analyze](reports/example-analyze/report.md) reads
  [data/snapshot-2026-09-28/](data/snapshot-2026-09-28/): the French monthly factor files
  exactly as downloaded, and Yahoo monthly returns for SPY, IWN, BRK-B and AAPL
  (Jan 2005 to Jul 2026), all downloaded on 2026-09-28. The snapshot exists because Yahoo
  revises adjusted prices: a fresh download on the same day moved a few printed t-stats in
  the second decimal (SPY's t(alpha) went from −1.99 to −1.98).
- [example-25-portfolios](reports/example-25-portfolios/report.md) and
  [example-factors](reports/example-factors/report.md) read
  [data/snapshot-2026-09-29/](data/snapshot-2026-09-29/): the French monthly 3-factor,
  5-factor (2x3) and momentum files and the 25 size/book-to-market portfolios
  (`25_Portfolios_5x5`), exactly as downloaded on 2026-09-29 at 02:10 UTC. The files run to
  Aug 2026 (factors from Jul 1926, Jul 1963 for the five-factor file; portfolios from Jul
  1926); both reports use Jul 1963 to Jul 2026, 757 months. The three factor files are
  byte-identical to the 2026-09-28 ones. This snapshot exists because French revises
  history in the last digit: these reports used to download live data, and regenerating them
  moved, for example, the CAPM GRS p-value from 7.5e-11 to 7.4e-11 and the Carhart
  Fama-MacBeth momentum premium from 23.67% to 23.49% a year. No test's conclusion changed.
  The snapshot is 581 kB (580,870 bytes), 549 kB of it the portfolio file.

French data is cached in `~/.cache/ffmodel` (override with `FFMODEL_CACHE`) and re-downloaded
when it is more than 7 days old or when you pass `--refresh`. `--data-dir DIR` (accepted by
`analyze`, `test-portfolios` and `factors`) reads it from a snapshot instead and never
downloads; a file whose SHA-256 differs from its manifest entry is refused, and a file missing
from the snapshot is an error, not a download.

## Command line

```bash
# Reproduce reports/example-analyze from the committed snapshot (no network, about 12 s)
ffmodel analyze --csv data/snapshot-2026-09-28/returns.csv --data-dir data/snapshot-2026-09-28 \
    --start 2005-01 --end 2026-07 --model ff5 --compare --rolling 36 --weights SPY=0.6,IWN=0.4 \
    --out reports/example-analyze

# Reproduce reports/example-25-portfolios and reports/example-factors (no network, about 4 s and 2 s)
ffmodel test-portfolios --dataset 25_Portfolios_5x5 --start 1963-07 --end 2026-07 --compare \
    --data-dir data/snapshot-2026-09-29 --out reports/example-25-portfolios
ffmodel factors --model ff6 --end 2026-07 --data-dir data/snapshot-2026-09-29 --out reports/example-factors

# The same analysis on live data, and saving new dated snapshots (need network)
ffmodel analyze --tickers SPY IWN BRK-B AAPL --start 2005-01 --model ff5 \
    --compare --rolling 36 --weights SPY=0.6,IWN=0.4
ffmodel snapshot --tickers SPY IWN BRK-B AAPL --start 2005-01 --end 2026-07 --out data/snapshot-YYYY-MM-DD
ffmodel snapshot --portfolios 25_Portfolios_5x5 --out data/snapshot-YYYY-MM-DD   # French files only

# Your own returns (first column dates, one column per asset)
ffmodel analyze --csv my_fund.csv --percent --model ff3
ffmodel analyze --csv prices.csv --prices --freq daily --model carhart

# Asset pricing tests on French test portfolios
ffmodel test-portfolios --dataset 25_Portfolios_5x5 --start 1963-07 --compare
ffmodel test-portfolios --dataset 10_Industry_Portfolios --model ff5 --beta-window 60

# The factors themselves
ffmodel factors --model ff6
```

(`python -m ffmodel ...` works too.) Common options: `--freq monthly|daily`, `--start`,
`--end`, `--out DIR` (default `reports/<command>-<timestamp>`), `--no-report`, `--refresh`,
`--data-dir DIR`.
Run `ffmodel <command> -h` for the rest (`--cov hac|robust|ols`, `--lags`, `--excess`,
`--weighting value|equal`, `--fm-lags`, ...).

Worked examples are in [reports/](reports/):
[example-analyze](reports/example-analyze/report.md) (snapshot 2026-09-28),
[example-25-portfolios](reports/example-25-portfolios/report.md) and
[example-factors](reports/example-factors/report.md) (snapshot 2026-09-29).

## Library

```python
import ffmodel as ff

factors = ff.load_factors("ff5", start="1963-07")            # Mkt-RF, SMB, HML, RMW, CMA, RF (decimals)
returns = ff.download_returns(["AAPL", "BRK-B"], start="2005-01")

res = ff.fit_factor_model(returns["AAPL"], factors, model="ff5")   # raw returns; RF is subtracted
print(res.summary())                  # coefficients, t-stats, CIs, alpha (ann.), R2, IR
res.alpha_annual, res.betas, res.raw  # raw = the statsmodels results object

attribution = ff.attribute_returns(res, factors)
attribution.summary                   # return and variance shares by component
rolling = ff.rolling_regression(returns["AAPL"], factors, "ff5", window=36, se="hc3")  # + se(...) columns
ff.stability_test(returns["AAPL"], factors, "ff5", block=36)   # did any exposure change across blocks?
ff.compare_standard_errors(res)       # t-stats under OLS, White, Newey-West (rule and 12 lags)

portfolios = ff.load_portfolios("25_Portfolios_5x5", start="1963-07")
excess = portfolios.sub(factors["RF"], axis=0)
ff5 = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
grs = ff.grs_test(excess, factors[ff5])
fm = ff.fama_macbeth(excess, factors[ff5])
fm.table()                            # premia with FM and Shanken t-stats vs. factor means
```

## Methods and conventions

- **Returns** are decimals. Monthly data is indexed by month-end dates. Regressions use
  `r − RF` unless you say the returns are already excess (`excess=True` / `--excess`).
  Alphas and premia are annualized arithmetically (×12 monthly, ×252 daily).
- **Factor files.** CAPM, FF3 and Carhart use the 3-factor file; FF5 and FF6 use the 5-factor
  (2x3) file. The two files build SMB differently, so SMB is not identical across models.
  Momentum comes from the separate momentum file.
- **Standard errors (full sample).** The default is Newey-West (HAC) with
  L = floor(4·(T/100)^(2/9)) lags, the Newey-West (1994) rule of thumb: 4 lags for the
  259-month examples (4 for T = 100–272, 5 for 273–620). I use the rule rather than a
  hand-picked lag so that the choice is fixed before seeing the results. It grows slowly
  with T, which suits monthly returns: they have little serial correlation, so at this
  frequency NW mostly buys robustness to heteroskedasticity plus a margin for mild
  autocorrelation. Each report also shows 12 lags as a sensitivity check. The estimator is
  statsmodels' `cov_type="HAC"`: Bartlett weights 1 − l/(L+1), no small-sample
  degrees-of-freedom correction, p-values from the normal distribution.
  [tests/test_standard_errors.py](tests/test_standard_errors.py) checks it against a
  from-scratch sandwich formula and against statsmodels called directly, for L = 0, 1, 4 and 12.
- **Standard errors (rolling windows).** Windows use HC3 (MacKinnon-White) errors, not
  Newey-West. A Monte Carlo on the real 2005–2026 FF5 factors
  ([scripts/window_se_simulation.py](scripts/window_se_simulation.py), 16 s, offline) shows
  that with 36 observations and six coefficients, nominal 95% NW intervals miss the true
  beta 13–14% of the time. HC3 intervals miss it 4–5% of the time. Over the full 259 months,
  NW is only slightly liberal (6.5–7.2% misses). Classical OLS misses 10.6% when residual
  volatility moves with the market.
- **Rolling estimates.** Each row is dated at the last month of its window, and the bands
  are ± 1.96 HC3 standard errors. Windows overlap, so the bands are pointwise, not joint.
  Wiggles in a rolling chart are not evidence that exposures changed. The report tests that
  separately: it splits the sample into non-overlapping 36-month blocks and runs, for each
  coefficient, a Wald chi-squared test that all blocks share one value. With HC3 the test
  rejects a truly constant beta 2–6% of the time at the 5% level. With NW it rejects 23–35%
  of the time, which is why NW is not used there.
- **GRS.** The finite-sample form in Campbell, Lo and MacKinlay (1997), which is exactly
  F(N, T−N−L) under normal errors. Also reports A|α|, A|α|/A|r̄| (Fama-French 2015) and
  SR(α) = sqrt(α′Σ⁻¹α).
- **Fama-MacBeth.** Betas come from full-sample time-series regressions, or with
  `beta_window` from the trailing window ending the period before, so no look-ahead. Each
  period's cross-section is regressed on those betas and the premia are the averages of the
  slopes. The Shanken (1992) errors-in-variables correction is applied with full-sample betas.
  Newey-West errors are optional.
- **Attribution.** The excess return in each period equals α + Σ βₖfₖ + ε exactly. The
  variance split is an Euler decomposition, βₖ·cov(fₖ, β′f)/var(r), so a factor that hedges
  the others can have a negative share.
- **Yahoo prices** are split- and dividend-adjusted closes. Monthly returns use month-end
  prices, and a trailing partial month is dropped.

## Sanity checks against known results (Jul 1963 – Jul 2026, French data downloaded 2026-09-29)

- SPY on FF5 (2005–2026): market beta 0.99, R² 0.997, alpha −0.36%/yr. That is a small
  shortfall and the right sign for fund costs, though larger than SPY's roughly 0.09% expense
  ratio (see the interpretation below).
- IWN (small-cap value ETF): SMB 0.82, HML 0.36. BRK-B: HML +0.43, SMB −0.36.
- Spanning regressions (`factors --model ff6`): HML's alpha given the other five FF6
  factors (market, SMB, RMW, CMA and momentum) is 0.9%/yr (t = 0.75). This is consistent
  with the Fama-French (2015) finding that HML is redundant in the five-factor model.
- 25 size/book-to-market portfolios: GRS rejects every model (p-values from 7.4e-11 for CAPM
  to 3.7e-06 for FF6). The small-growth portfolio has the largest FF3 alpha, −5.6%/yr
  (t = −5.1). In Fama-MacBeth the market premium is insignificant once there is an intercept
  (t from −1.79 to 0.31 across the five models; negative in four, +1.3%/yr under FF6), the
  classic flat security market line.

## Interpreting the example

All numbers below come from [reports/example-analyze](reports/example-analyze/report.md):
FF5 regressions on monthly excess returns, Jan 2005 to Jul 2026 (259 months), from the
2026-09-28 snapshot. t-stats and p-values use Newey-West errors with 4 lags. "Significant"
means p < 0.05 for that one test; the multiple-comparison caveat is further down.

| Asset | Significant exposures (NW t) | Alpha per year (t, p) | Reading |
|---|---|---|---|
| SPY | Mkt 0.99 (179), SMB −0.12 (−17.1), HML 0.02 (2.7), RMW 0.05 (3.3), CMA 0.02 (2.1) | −0.36% (−1.98, 0.047) | Large-cap market fund. The HML/RMW/CMA loadings are statistically "significant" but economically trivial. |
| IWN | Mkt 0.96 (72), SMB 0.82 (33), HML 0.36 (14) | −1.33% (−2.22, 0.027) | Small-cap value, as designed. RMW and CMA are indistinguishable from zero. |
| BRK-B | Mkt 0.68 (8.3), SMB −0.36 (−3.7), HML 0.43 (3.6) | +2.76% (1.03, 0.30) | Low-beta, large-cap value. Alpha not significant. |
| AAPL | Mkt 1.24 (11.7), HML −0.44 (−2.3), RMW 0.62 (3.1) | +15.83% (2.95, 0.003) | High-beta growth stock with a profitability tilt, plus a large alpha. |
| 60/40 SPY/IWN | Mkt 0.98, SMB 0.25, HML 0.16 (all \|t\| > 12), RMW 0.04 (2.3) | −0.75% (−2.58, 0.010) | Exactly the weighted average of its parts. |

**What the loadings mean.** A beta is the return sensitivity to a long-short factor
portfolio, so its sign says which side of the sort the asset looks like:

- **AAPL.** HML −0.44 means it co-moves with low book-to-market (growth) stocks. That is a
  growth tilt, not a claim that Apple is "expensive". RMW +0.62 means it behaves like
  high-profitability firms. CMA −0.43 would point to an aggressive-investment tilt, but it is
  not significant (t = −1.46, p = 0.14).
- **BRK-B.** Market beta 0.68 means it is defensive, SMB −0.36 means it behaves like a large
  cap, and HML +0.43 is a value tilt.
- **IWN.** Its HML loading is 0.52 under FF3 but 0.36 under FF5. HML and CMA have a
  correlation of 0.68 (1963–2026, [example-factors](reports/example-factors/report.md)), so
  part of what FF3 calls "value" is shared with the investment factor once CMA is in the
  model. A loading is only defined relative to the other factors in the regression.
- **SPY.** Its HML loading of 0.02 is "significant" (t = 2.7) because R² is 0.997 and the
  residuals are tiny, so even negligible loadings are estimated precisely. A 0.02 loading on
  a factor that earns a few percent a year is worth a few basis points. Statistical
  significance is not economic significance.

**Alphas, and why none of them is a free lunch.**

- **IWN and SPY.** Both have negative alphas, which is the expected sign for real funds
  measured against paper factor portfolios. The French factors have no fees, trading costs
  or cash drag. Small-value stocks are the most expensive to trade, and IWN's shortfall
  (−1.33%) is the largest. SPY's −0.36% is larger than its roughly 0.09% expense ratio. The
  factor "market" is all CRSP stocks, not the S&P 500, so part of the gap is benchmark
  mismatch rather than cost. SPY's alpha is also borderline: p = 0.047 under NW and 0.065
  under OLS.
- **AAPL.** Its alpha is large and survives every correction in the report. It is still not
  evidence of a strategy. AAPL was picked for this example because everyone knows it did
  well. Regressing one famous winner, chosen with hindsight, will always find a big alpha.
  The relevant family of tests is every stock one could have picked in 2005, not the five
  rows in this table. In the rolling chart, most of the alpha comes from the early windows.
  The 36-month block alphas range from −2.9% to +44.3% a year, but the block test cannot
  reject a constant alpha (p = 0.46), so even the apparent decay is not established.
- **The 60/40 portfolio.** Its loadings and alpha are 0.6×SPY + 0.4×IWN (for example,
  0.6×(−0.36%) + 0.4×(−1.33%) = −0.75%), because OLS is linear. That makes it a useful
  check of the code, but not independent evidence. Its t-stat is larger than either
  component's only because the two residuals partly diversify.

**Newey-West vs OLS: what changed.** Each asset's section in the report has a table of
t-stats under OLS, White and NW with 4 and 12 lags. For the alphas, NW barely matters here:
every alpha t-stat moves by less than 0.15 between OLS and NW, and 12 lags give almost the
same answer as 4. Only one conclusion flips at 5%: SPY's alpha goes from t = −1.85
(p = 0.065) under OLS to −1.98 (p = 0.047) under NW. A result that flips with the choice of
standard error was never robust, so treat it as "borderline" under either. The factor
loadings of the high-R² funds move more. SPY's RMW t falls from 5.59 to 3.33 and HML's from
3.69 to 2.73, and the 60/40 portfolio's HML falls from 16.3 to 12.7. Part of each drop
already appears with White errors, which allow for heteroskedasticity. The rest comes from
the lag terms, which allow for autocorrelation. No factor loading changes significance.

**Rolling betas (36-month windows, 224 windows ending Dec 2007 to Jul 2026).** The charts
show pointwise 95% HC3 bands. For the diversified funds the bands are narrow (IWN's market
beta stays between 0.91 and 1.08). For single stocks they are wide. AAPL's rolling CMA beta
ranges from −3.6 to +1.2, and a typical AAPL band is ±1.7 for RMW and ±1.9 for CMA
(median half-widths). The
block test (7 non-overlapping 36-month blocks, Aug 2005 to Jul 2026) asks whether the
movements exceed noise. Of 30 coefficient tests (5 series × 6 coefficients), only two have
p < 0.05: SPY's market beta (blocks 0.97 to 1.02, p = 0.024) and BRK-B's market beta (0.16
to 1.11, p = 0.016). With 30 tests, about 1.5 such results are expected by chance, so the
honest summary is that 36-month windows cannot distinguish most of the apparent drift from
estimation noise. BRK-B's market beta is the one change that is also economically large.

**Multiple comparisons, and what "not significant" means.**

- The report tests 30 coefficients, so at a 5% threshold about 1.5 false positives are
  expected even if every true coefficient were zero. For the five alphas, the report adds
  Holm-adjusted p-values, which control the chance of any false rejection across the family.
  After adjustment AAPL (0.016) and the 60/40 portfolio (0.039) stay below 5%, and IWN
  (0.080) and SPY (0.094) do not. The deeper multiple-testing problem is the one Holm cannot
  fix: the choice of which assets to put in the table (see AAPL above).
- "Not significant" means the data cannot rule out zero. It does not mean the effect is
  zero. BRK-B's alpha is +2.76% a year with t = 1.03, and its 95% interval runs from about
  −2.5% to +8.0% a year. That is consistent with no skill and also with substantial skill.
  A rough rule is t(alpha) ≈ information ratio × √years. BRK-B's IR is 0.20 over 21.6 years,
  which gives about 0.9. At that IR it would take roughly (2 / 0.2)² = 100 years of data to
  reach t = 2. Twenty years of monthly data can detect only large alphas.
- Newey-West fixes the standard error for heteroskedasticity and autocorrelation. It does
  not fix look-ahead in the choice of asset, survivorship in Yahoo data, or errors in the
  model itself.

## Limitations

- The factors are the published ones, not rebuilt from CRSP/Compustat (that needs WRDS
  access). To test your own factors, pass any DataFrame of factor returns to the library
  functions.
- Yahoo Finance data has survivorship bias and occasional errors. Use your own CSV for
  anything that matters.
- GRS assumes i.i.d. normal errors. Fama-MacBeth with rolling betas has no Shanken correction.
- Full-sample Newey-West t-stats are slightly liberal at T = 259 (simulated 95% intervals miss
  6.5–7.2% of the time), so a p-value just under 0.05 is weaker evidence than it looks.
- The block stability test treats the blocks as independent and tests each coefficient
  separately, not all of them jointly.
- The snapshots hold monthly data only, and only the 25 size/book-to-market portfolios among
  the test-portfolio sets; other `--dataset` choices (such as `10_Industry_Portfolios`) need
  the network or a new snapshot.

## Layout

```
ffmodel/
  data.py           French library download, cache and parser
  models.py         model definitions
  regression.py     time-series regressions, standard-error comparison, rolling, stability test
  asset_pricing.py  GRS test, Fama-MacBeth
  attribution.py    return and variance attribution
  factor_stats.py   factor summary statistics, spanning regressions
  assets.py         Yahoo / CSV returns, portfolios
  report.py         charts and Markdown/CSV report
  snapshot.py       save / read dated data snapshots for offline runs
  cli.py            command-line interface
data/               dated input snapshots (French zips as downloaded, Yahoo returns, manifest.json)
scripts/            window_se_simulation.py: which standard errors to trust in short windows
tests/              offline tests (synthetic data with known parameters)
```
