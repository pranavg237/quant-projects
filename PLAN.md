# Plan

Written after auditing every project on 2026-09-28, before changing any code. Part 1 is
the audit and Part 2 ranks the fixes by recruiter impact per hour of work.

Environment for the audit: Python 3.12, numpy 2.5, pandas 3.0, scipy 1.18. Every
suite was run.

## Part 1: Audit

### Repo-wide

| Check | Finding |
|---|---|
| Secrets | **None committed, in the working tree or in history.** I searched every commit's diff for key patterns (`API_KEY`, `SECRET`, `PK…`, `AKIA…`, `sk-…`, and long quoted tokens). The only matches are the placeholders in `alpaca-ma-crossover-bot/.env.example` and code that reads environment variables. The alpaca `.env` is gitignored in its subfolder, but the root `.gitignore` does not ignore `.env`. |
| CI | **No CI runs at all.** `ma-crossover-backtest/.github/workflows/ci.yml` exists, but GitHub only reads workflows from the repo-root `.github/`, so it has never run since the projects were merged into one repo. |
| One-command setup | **Broken for 3 of 7 projects.** In the src-layout projects (`options-pricing-engine`, `market-making-simulator`, `portfolio-optimization`), `pip install -r requirements.txt && pytest` fails with `ModuleNotFoundError` because `requirements.txt` does not install the package itself. The tests only pass with `PYTHONPATH=src`. `ma-crossover-backtest`, `fama-french-factor-model`, `alpaca-ma-crossover-bot` and `options-pricing-toolkit` have no `requirements.txt`. |
| `STATUS.md` | Stale and misleading. It says each project is its own git repo, refers to a `.venv-shared/` that doesn't exist, and lists planned projects (`stat-arb-kalman`, `ml-alpha-research`, …) that were never started. |
| Stale claims in READMEs | options engine says 373 tests (337 exist). ma-crossover says 100 tests / 94% coverage (125 / 98%). |

### Per project

Test results are from running each suite in this environment.

#### options-pricing-engine

- **What it does:** Black-Scholes with 10 Greeks, CRR/JR/Leisen-Reimer lattices (European and American), Monte Carlo with antithetic and control variates, safeguarded-Newton implied vol, and a Heston model calibrated to a committed SPY chain snapshot (2026-09-18).
- **Runs:** yes. 337 / 337 tests pass (with `PYTHONPATH=src`). Analysis runs offline from the committed snapshot.
- **Code quality:** high. Strict mypy and ruff configs, full type hints and docstrings.
- **Methodology:** it's a pricing and calibration project, not a trading strategy, so lookahead, survivorship and transaction costs don't apply. The limitations are already documented in REVIEW.md: one snapshot of one underlying; SPY options are American but are inverted as European; the Feller condition is violated.
- **Issues:** the README's test count is wrong (373 vs 337). Not one-command runnable.

#### market-making-simulator

- **What it does:** a price-time-priority limit order book, Avellaneda-Stoikov quoting, and two simulation engines (idealised and full-book with informed flow). Reproduces the paper's Table 1.
- **Runs:** yes. 124 / 124 tests pass (with `PYTHONPATH=src`).
- **Code quality:** high.
- **Methodology:**
  - The headline "Sharpe 11.7" is the mean over the standard deviation of per-session PnL across 200 simulated sessions. It is not annualised and is not comparable to a strategy Sharpe. Section 8 says so, but the headline table doesn't, so a recruiter skimming will read "Sharpe 11.7" as a red flag.
  - Maker fees default to 0.
  - There's no latency and no competing maker (documented).
- **Issues:** not one-command runnable.

#### portfolio-optimization

- **What it does:** Ledoit-Wolf shrinkage, Markowitz (closed-form and constrained), risk parity, HRP and Black-Litterman on 15 ETFs (monthly, snapshot to 2026-09-18). Walk-forward backtest with weight drift and turnover-based costs, plus a leverage sweep.
- **Runs:** the analysis script runs. **There are zero tests** (the config points at a `tests/` folder that doesn't exist). **There's no README.**
- **Code quality:** high in the source, with type hints and docstrings.
- **Methodology flaws:**
  1. **Sharpe uses a 0% risk-free rate.** `evaluate()` defaults to `risk_free=0.0` and the script never passes one. Every reported Sharpe is therefore return/vol, not excess return/vol. This inflates all of them, and inflates the low-volatility strategies (min variance, HRP) the most.
  2. **The long-short leverage sweep pays nothing to short.** Shorting ETFs costs a borrow fee, so every leveraged row is overstated by an amount the current code can't measure.
  3. **There's no market benchmark.** 1/N is a good hurdle but not buy-and-hold of the market.
  4. **Weak evidence behind "Markowitz beats 1/N".** The comparison is 1.07 vs 0.92, but the Sharpe confidence intervals overlap almost completely (0.59–1.56 vs 0.43–1.40). The paired difference is never tested, and the claim was chosen after trying several lookbacks and estimators.
  5. **Selection and prior bias.** The 15-ETF universe was chosen today. The Black-Litterman market-cap prior is a static set of today's weights, which is mild lookahead in the prior.
  6. No lookahead in the walk-forward itself: the window ends at t and weights apply to t+1, enforced by a runtime check.

#### fama-french-factor-model

- **What it does:** CAPM through FF6 time-series regressions with Newey-West errors, return and variance attribution, GRS and Fama-MacBeth tests, and a CLI that writes reports. Uses Ken French's data.
- **Runs:** yes. 26 / 26 tests pass, 92% coverage. `assets.py` (Yahoo and CSV loading) is at 55%.
- **Code quality:** good, but typing is inconsistent. There are 77 missing parameter or return annotations, mostly in `cli.py`/`report.py`, and old-style `Optional`/`Dict`.
- **Methodology:** it's an analysis tool, not a strategy, so backtest biases mostly don't apply. Fama-MacBeth rolling betas are lagged (no lookahead). Survivorship in Yahoo tickers is documented. `--weights` portfolios are rebalanced every period at no cost, which is fine for attribution but should be stated.
- **Issues:** no `requirements.txt`.

#### ma-crossover-backtest (`quantbt`)

- **What it does:** an event-driven backtester with next-bar fills, costs on by default (5 bp slippage and 1 bp commission), short borrow fees, point-in-time universes, walk-forward optimisation, stationary bootstrap, PSR/deflated Sharpe/PBO, and FF5 regressions. It runs five strategies against SPY buy-and-hold.
- **Runs:** yes. 125 / 125 tests pass, 98% coverage. Data is downloaded from Yahoo and cached (gitignored). I'm re-running the full pipeline to check the numbers.
- **Code quality:** high. It has strict mypy config. Many public methods lack docstrings (mostly properties and small helpers).
- **Methodology:** the strongest in the repo, and its own AUDIT.md and REVIEW.md already list the remaining issues honestly:
  - `xsmom` has survivorship bias (today's 70 large caps).
  - The ETF lists are selection-biased.
  - Costs are flat.
  - The deflated Sharpe undercounts trials.
  - Pairs selection has no multiple-testing correction.
- **Issues:** stale test counts in the README. The CI workflow never runs (see repo-wide).

#### alpaca-ma-crossover-bot

- **What it does:** runs one daily decision cycle of the 50/200 MA crossover against Alpaca's **paper** API.
- **Runs:** tests yes (11 / 11). The bot itself needs paper keys, which I don't have, so its live path is untested. No `requirements.txt`.
- **Methodology bugs:**
  1. **It requests unadjusted bars.** Alpaca's bars endpoint defaults to `adjustment=raw`, so a stock split puts a cliff in the price series and can flip the MA signal on a split rather than a trend. (Not an issue for SPY in practice, but it is for any stock.)
  2. **It can trade on a partial bar.** If run during market hours, today's still-forming daily bar is used as if it were a close.
  3. The README says it won't order without enough buying power, but it sizes on equity, not buying power.
- **Secrets:** handled correctly via environment variables. `.env` isn't loaded automatically: the README asks you to `source` it.

#### options-pricing-toolkit

- **What it does:** a compact Black-Scholes pricer, Greeks, implied vol via Brent, and a 0–2 DTE theta/gamma plot.
- **Runs:** yes. 10 / 10 tests pass.
- **Issues:**
  - **It's a strict subset of `options-pricing-engine`.** It adds breadth without depth, which is the opposite of what the repo should signal.
  - Its implied-vol no-arbitrage check uses `intrinsic·e^(−qT)` instead of the correct lower bound `max(0, S·e^(−qT) − K·e^(−rT))`.
  - It has 53 missing annotations and no `requirements.txt`.

## Part 2: Ranked plan

Ranked by (impact on a 2-minute skim plus how well it holds up under a deep read) ÷ hours.

| # | Change | Why it matters to a reviewer | Est. hours |
|---|---|---|---|
| 1 | **Root CI workflow** running every project's tests on push, plus a badge | A green badge is the first thing seen. It also proves the "tests pass" claims. | 0.5 |
| 2 | **One-command runnability**: `pythonpath=src` in the pytest configs, `-e .` in `requirements.txt`, a `requirements.txt` for every project | An engineer who clones and gets `ModuleNotFoundError` stops reading. | 0.5 |
| 3 | **Root README as a portfolio index**, one line per project with a verified headline number, and delete stale `STATUS.md` | This is the 2-minute skim. | 0.5 |
| 4 | **portfolio-optimization: fix the Sharpe risk-free rate, add a short-borrow cost, add a SPY buy-and-hold benchmark, re-run** | These are real methodology flaws in a headline result, and a quant interviewer would catch the rf=0 Sharpe. | 2 |
| 5 | **portfolio-optimization: tests** (closed-form frontier, Ledoit-Wolf vs sklearn, risk parity, HRP, BL identity, walk-forward lookahead and cost accounting) | A project with no tests stands out in a repo where the others have hundreds. | 2 |
| 6 | **portfolio-optimization: README** with results from the re-run | It's currently undocumented. | 1 |
| 7 | **Re-run ma-crossover-backtest end to end and reconcile RESULTS.md** | The rule is that every number must come from running code. | 1 |
| 8 | **market-making: label the Sharpe as per-session and un-annualised** in the headline | Removes the "Sharpe 11.7 = bug" first impression. | 0.25 |
| 9 | **alpaca bot: split-adjusted bars, drop the partial bar, size on min(equity, buying power), load `.env`, tests** | These are real live-trading bugs, and each is a good interview story. | 1 |
| 10 | **Secrets hygiene**: root `.gitignore` covers `.env` everywhere, and the bot loads `.env` itself | There's no leak to fix, but this closes the gap. | 0.25 |
| 11 | **Type hints and docstrings** on public functions (fama-french, toolkit, alpaca, quantbt's missing docstrings) | Only noticed on a deep read, and the work is mostly mechanical. | 2 |
| 12 | **fama-french: tests for `assets.py`** (55% covered) | Lowest-covered module in the repo. | 0.5 |
| 13 | **options-pricing-toolkit: fix the IV bound and label it as superseded** | Honest positioning. The user should decide whether to delete it. | 0.5 |
| 14 | Fix stale test counts in the READMEs | Small, but a wrong number found on a deep read costs trust. | 0.25 |

**Not doing, on purpose.** No new strategies, ML models or dashboards. The repo already has
enough breadth, and the rule is depth over breadth.

**Recommendation for the owner (not executed):** delete or archive
`options-pricing-toolkit`. Everything it does, `options-pricing-engine` does better, and a
reviewer who opens it second sees the weaker version of work they've already seen.

## Status

All 14 items above were executed. [CHANGES.md](CHANGES.md) describes what each one changed,
including findings the audit did not anticipate: the options engine ignored its snapshot,
the market-making defaults didn't match its README, and fama-french's `--end` was broken.
The owner has since accepted the recommendation: `options-pricing-toolkit` was deleted in
round 2 (below).

## Round 2: from "correct and tested" to "shows quant judgment"

Round 1 made every project run, test and report honestly. Round 2 deepens each one where
a quant interviewer would push next. Depth over breadth: a task that couldn't be done
properly is skipped and the README says why.

| Project | Work |
|---|---|
| repo | Delete `options-pricing-toolkit` (the options engine replaces it). |
| portfolio-optimization | Keep "nothing beats 1/N" as the headline. Verify Ledoit-Wolf and the walk-forward, report turnover and cost drag per method, and explain why estimation error defeats mean-variance (DeMiguel, Garlappi & Uppal 2009). |
| ma-crossover-backtest | Verify the walk-forward, add a parameter-sensitivity heatmap, and make the multiple-testing correction (deflated Sharpe) count every trial. |
| options-pricing-engine | Report Greek-vs-finite-difference errors, test put-call parity on the SPY snapshot, and plot the implied-vol smile and surface. |
| market-making-simulator | Verify the Avellaneda-Stoikov formulas, add a markout-based adverse-selection analysis, and a parameter-sensitivity study. Every Sharpe stays labelled per session. |
| fama-french-factor-model | Rolling betas, Newey-West alpha errors checked against statsmodels, and a plain interpretation of which exposures are significant. |
| alpaca-ma-crossover-bot | No keys, so no live or paper trading. Add pre-trade risk checks, a kill switch, structured logging, and tests that simulate API errors and partial fills. |

After merging, an agent that had not seen the work re-ran every project from a clean
clone and checked every README number. [CHANGES.md](CHANGES.md) records the outcome.

## Round 3: closing round 2's open items

| Project | Work |
|---|---|
| options-pricing-engine | Build the American early-exercise correction into the surface and forward, keep the European surface for comparison, and re-run the calibration. |
| fama-french-factor-model | Commit snapshots for the factor and 25-portfolio examples so every example runs offline. |
| alpaca-ma-crossover-bot | Pass ruff and strict mypy, and turn on those checks in CI. |

As before, a fresh agent re-ran the changed projects from a clean clone and checked every
number; [CHANGES.md](CHANGES.md) records the outcome.

## Round 4: loose ends from round 3

| Project | Work |
|---|---|
| options-pricing-engine | Test whether SPY's discrete quarterly dividends explain the long-dated call/put overshoot. |
| fama-french-factor-model | Hash-check the snapshot's Yahoo returns file. |

A fresh agent re-ran the changed projects from a clean clone; [CHANGES.md](CHANGES.md)
records the outcome.
