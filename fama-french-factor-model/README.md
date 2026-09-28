# ffmodel — Fama-French factor models

A Python package and command-line tool for working with the Fama-French family of
factor models, using the official factor data from Kenneth French's data library.

| | |
|---|---|
| **Models** | CAPM, Fama-French 3-factor, Carhart 4-factor, Fama-French 5-factor, Fama-French 6-factor (FF5 + momentum) |
| **Data** | Downloads, parses and caches French library files (factors and any test-portfolio set, monthly or daily). Asset returns from Yahoo Finance or your own CSV |
| **Time-series regressions** | Alpha and betas with Newey-West, White or classical standard errors; annualized alpha, residual volatility, information ratio; rolling-window estimates; side-by-side model comparison on a common sample |
| **Attribution** | Splits an asset's excess return into alpha, each factor's contribution and residual, and splits its variance the same way |
| **Asset pricing tests** | Gibbons-Ross-Shanken (GRS) test with the Fama-French (2015) summary statistics; two-pass Fama-MacBeth risk premia with Shanken and Newey-West corrections, full-sample or rolling betas |
| **Factor diagnostics** | Summary statistics, correlations and spanning regressions (is a factor explained by the others?) |
| **Reports** | Each command writes a Markdown report with charts (PNG) and every table as CSV |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest          # 26 offline tests
```

French data is cached in `~/.cache/ffmodel` (override with `FFMODEL_CACHE`) and re-downloaded
when it is more than 7 days old or when you pass `--refresh`.

## Command line

```bash
# Factor regressions for stocks/funds, plus a 60/40 portfolio of two of them
ffmodel analyze --tickers SPY IWN BRK-B AAPL --start 2005-01 --model ff5 \
    --compare --rolling 36 --weights SPY=0.6,IWN=0.4

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
`--end`, `--out DIR` (default `reports/<command>-<timestamp>`), `--no-report`, `--refresh`.
Run `ffmodel <command> -h` for the rest (`--cov hac|robust|ols`, `--lags`, `--excess`,
`--weighting value|equal`, `--fm-lags`, ...).

Worked examples from live data are in [reports/](reports/):
[example-analyze](reports/example-analyze/report.md),
[example-25-portfolios](reports/example-25-portfolios/report.md),
[example-factors](reports/example-factors/report.md).

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
rolling = ff.rolling_regression(returns["AAPL"], factors, "ff5", window=36)

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
- **Standard errors.** The default is Newey-West with the lag rule floor(4·(T/100)^(2/9)).
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

## Sanity checks against known results (live data, Jul 1963 – Jul 2026)

- SPY on FF5 (2005–2026): market beta 0.99, R² 0.997, alpha −0.36%/yr, about the fund's
  cost drag.
- IWN (small-cap value ETF): SMB 0.82, HML 0.36. BRK-B: HML +0.43, SMB −0.36.
- Spanning regressions: HML's alpha given the other FF5 factors is 0.9%/yr (t = 0.75). This
  is the Fama-French (2015) finding that HML is redundant in the five-factor model.
- 25 size/book-to-market portfolios: GRS rejects every model. The small-growth portfolio has
  the largest FF3 alpha, −5.6%/yr (t = −5.1). In Fama-MacBeth the market premium is negative
  and insignificant once there is an intercept, the classic flat security market line.

## Limitations

- The factors are the published ones, not rebuilt from CRSP/Compustat (that needs WRDS
  access). To test your own factors, pass any DataFrame of factor returns to the library
  functions.
- Yahoo Finance data has survivorship bias and occasional errors. Use your own CSV for
  anything that matters.
- GRS assumes i.i.d. normal errors. Fama-MacBeth with rolling betas has no Shanken correction.

## Layout

```
ffmodel/
  data.py           French library download, cache and parser
  models.py         model definitions
  regression.py     time-series regressions, rolling, model comparison
  asset_pricing.py  GRS test, Fama-MacBeth
  attribution.py    return and variance attribution
  factor_stats.py   factor summary statistics, spanning regressions
  assets.py         Yahoo / CSV returns, portfolios
  report.py         charts and Markdown/CSV report
  cli.py            command-line interface
tests/              offline tests (synthetic data with known parameters)
```
