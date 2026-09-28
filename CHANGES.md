# Changes

What changed in this round, and why, in plain language. Each item says what was wrong,
what I did about it, and what it changed. The audit that led to these is in
[PLAN.md](PLAN.md). The git log has one commit per item.

## The short version

- **Two results changed materially once they were measured properly.**
  - Portfolio optimisation's Sharpe ratios were computed against a 0% risk-free rate. They
    fall by about 0.15 each against real T-bills, and "Markowitz beats 1/N" turns out not
    to be statistically supported.
  - One of the five backtest strategies moved slightly after the data vendor re-adjusted
    prices for a stock split.
- **Five real bugs fixed**, each with a regression test:
  - the options engine ignored its own committed data snapshot;
  - the Alpaca bot used unadjusted prices and could trade on a half-finished daily bar;
  - an implied-vol solver rejected valid option prices;
  - a CLI date option never worked.
- **Engineering basics:** every project installs and tests with one command, CI runs all
  731 tests on every push, and there's a README for every project with numbers taken from
  real runs.
- **No secrets were ever committed.** I checked the full git history.

---

## 1. Secrets

**What I checked.** I searched every commit in the repository's history for API-key
patterns (Alpaca, AWS, OpenAI-style keys, anything named `SECRET` or `TOKEN`, and long
random-looking strings). The only hits were the placeholder values in
`alpaca-ma-crossover-bot/.env.example`.

**What I changed.**
- The root `.gitignore` now ignores `.env` files in every project. Before, only the bot's
  own folder did.
- The bot loads `.env` itself (`python-dotenv`), instead of asking you to `source` it.
  Variables already set in your shell still win.

**How to explain it:** "Keys live in environment variables or a gitignored `.env`; the repo
ships a `.env.example` with placeholders. I audited the whole history, not just the latest
commit, because a key deleted in a later commit is still public."

## 2. One-command setup, and CI

**Problem 1.** In three projects, `pip install -r requirements.txt && pytest` failed with
`ModuleNotFoundError`. Those projects use a `src/` layout, and `requirements.txt` never
installed the package itself. The tests only passed if you knew to set `PYTHONPATH=src`.
Three other projects had no `requirements.txt` at all.

**Fix.** Every `requirements.txt` now ends with `-e .` (install this project in editable
mode), and pytest is configured to find `src/`. I checked each project in a brand-new
virtual environment.

**Problem 2.** No CI had run since the projects were merged into one repo. The one
workflow file lived at `ma-crossover-backtest/.github/workflows/`, and GitHub only reads
workflows from the repository root.

**Fix.** There's a root workflow (`.github/workflows/ci.yml`) with one job per project.
Each job installs from that project's `requirements.txt`, exactly as its README says. For
the four main packages it also runs `ruff` (lint) and strict `mypy` (types), and then it
runs the tests. I confirmed every suite runs with the network blocked, so CI can't depend
on Yahoo being up.

## 3. Methodology fixes

### portfolio-optimization: the Sharpe ratio used a 0% risk-free rate

**What was wrong.** Sharpe ratio = (return − risk-free rate) / volatility. The code's
docstring said it used excess returns, but the risk-free rate defaulted to 0 and the
analysis never passed one. From 2009 to 2026 T-bills averaged 1.75% a year, so every
Sharpe was too high. The inflation is biggest for low-volatility portfolios, because the
same 1.75% is divided by a smaller volatility.

**Fix.** The analysis now uses the 13-week T-bill yield (`^IRX`). The rate used for month
*m* is the one observed at the end of month *m − 1*, so it's known in advance: no
lookahead.

**Effect.** I measured each fix separately:
- Markowitz: 1.07 → 0.93. 1/N: 0.92 → 0.80.
- The risk-free fix accounts for about −0.15 of that.

### portfolio-optimization: shorting was free

**What was wrong.** The leverage experiment allows short positions, but borrowing a
security to short it costs a fee.

**Fix.** A 50 bp a year stock-loan fee is charged each month on the short positions
actually held.

**Effect.** −0.015 Sharpe at 1.5x leverage, −0.05 at 3x. The conclusion doesn't change:
unconstrained leverage is what ruins Markowitz. But the numbers are now honest.

### portfolio-optimization: an 18-day "month"

**What was wrong.** The price data ends on 18 September. Monthly resampling treated those
18 days as a full month.

**Fix.** Incomplete final months are dropped. The effect was tiny (+0.005 Sharpe), but it's
the kind of thing an interviewer probes.

### portfolio-optimization: "Markowitz beats 1/N" was not tested

**What was wrong.** The headline compared point estimates (1.07 vs 0.92) whose confidence
intervals overlapped almost completely. The claim was also chosen after trying several
lookbacks and estimators.

**Fix.** I added a paired test of equal Sharpe ratios: Jobson-Korkie with Memmel's
correction. "Paired" matters here. Two strategies trading the same assets are correlated,
so their estimation errors partly cancel. Comparing their separate intervals is too
pessimistic, and eyeballing point estimates is too optimistic. I also added buy-and-hold
SPY as a market benchmark.

**Effect.** None of the nine methods differs from 1/N at the 5% level (all p > 0.15). SPY
buy-and-hold (Sharpe 0.91) is level with the best of them. The README now leads with that.
One more claim didn't survive: "long-only Markowitz beats 1/N at every lookback". It's
false at the 24- and 36-month windows.

**How to explain it:** "With 17 years of monthly data the standard error of a Sharpe ratio
is about 0.25, so a 0.1 difference is noise. I tested the difference instead of ranking
point estimates, and the answer was that nothing beats equal weight with confidence."

### ma-crossover-backtest: re-ran everything, one strategy moved

**What I did.** I re-downloaded all 100 symbols and re-ran the full walk-forward pipeline
(about 25 minutes) with the same 2025-08-29 end date.

**Result.** Four of the five strategies reproduced to at least five significant figures.
`mean_reversion` moved: Sharpe 0.44 → 0.42, and its longest drawdown went from 1,102 to
636 days.

**Why, as far as I could tell.** Five of the ETFs it trades split 2-for-1 on 2025-12-05,
after the original download, so Yahoo re-adjusted their whole price history. The strategy's
signal (a z-score) doesn't change when all prices are rescaled. Whole-share order rounding
and volume caps do change, so fills differ slightly. I couldn't confirm this further,
because the original raw data was never saved.

**What I changed.**
- RESULTS.md shows the new numbers and explains the change.
- The README now has a results table against buy-and-hold. A test checks that table
  against the pipeline's output, so it can't go stale silently.

**Also:**
- I independently re-checked AUDIT.md's claim that the original "same-bar fill" bug
  inflated returns by 4.6 points. I got 4.9 points on fresh data.
- The single-strategy script can now use the T-bill rate for cash and for the Sharpe
  hurdle (`--french-rf`). Before, it assumed 0%.

### alpaca-ma-crossover-bot: three bugs a backtest can't show

1. **Unadjusted prices.** Alpaca returns raw prices by default, so a 4-for-1 split looks
   like a 75% crash and can flip the moving-average signal. The bot now requests adjusted
   prices.
2. **Half-finished bar.** Run at 11am, the bot treated the latest trade as today's close,
   so its decision depended on what time it ran. Today's bar is now ignored until 4pm ET.
3. **Buying power.** The README said orders were limited by buying power, but the code only
   looked at equity. It now uses whichever is smaller.

I added tests for each, plus end-to-end tests of a full decision cycle against a fake
Alpaca client (11 → 24 tests).

The README states plainly that there's **no live track record**. It reports the backtest of
exactly what the bot trades: SPY 50/200, 2005-2025, net of costs. Sharpe was 0.54 against
0.53 for buy-and-hold, and max drawdown −34% against −55%, from only 10 trades.

### options-pricing-toolkit: the implied-vol solver rejected valid prices

**What was wrong.** The solver rejected any price below intrinsic value (K − S for a put).
That's wrong for European options. A deep in-the-money European put is worth *less* than
K − S, because you receive the strike later, not now. The solver refused a price its own
model had just produced (40.49 for a put with intrinsic value 50).

**Fix.** It now checks the correct no-arbitrage bounds, which discount the strike. I added
a regression test.

## 4. Bugs found while verifying the READMEs

**options-pricing-engine ignored its committed data.** The README said the analysis
re-runs offline from a saved SPY option chain. The code only used a saved chain from
*today*. Otherwise it downloaded a live one, so results changed with the date you ran it.
On a weekend it crashed outright, because 94% of live quotes are one-sided when the market
is closed.
- Fix: it now uses the saved snapshot unless you pass `--refresh`, and it uses the interest
  rate curve from the same day as the options data.
- After the fix, every headline number reproduced exactly.
- The README also had the test count wrong (373 vs 337) and one price mis-copied
  (6.0900 vs 6.0909).

**market-making-simulator's README couldn't be reproduced with its own command.** Its
numbers came from 2,000 / 200 / 500 simulated sessions, but the script defaulted to
1,000 / 120 / 300.
- Fix: the defaults now match the README.
- Re-running reproduced everything except one policy's PnL (11.31 → 11.34), which moved
  in a way I couldn't trace to a cause. The README says so.
- I also labelled every Sharpe ratio in the headline as *per simulated session, not
  annualised*. Someone skimming "Sharpe 11.7" would otherwise assume a bug.

**fama-french-factor-model's `--end` option never worked for stock data.** The CLI help
says to write `--end 2024-12`, but that string went straight to Yahoo, which requires a
full date.
- Fix: a month now means "through the end of that month".
- I re-ran all three documented examples, and every number the README quotes reproduced.

**All three analysis scripts crashed** if you pointed `--out` at a folder outside the
project. The cause was a path-printing line. Fixed.

## 5. Tests

- **portfolio-optimization: 0 → 67 tests.** Each method is checked against math with a
  known answer:
  - Ledoit-Wolf matches scikit-learn.
  - Minimum variance equals 1/A.
  - Risk parity contributions are equal.
  - HRP's ordering matches scipy.
  - Black-Litterman with no views returns the market.

  The backtest tests include one where a strategy that would need tomorrow's data tries to
  profit and can't, which proves there's no lookahead. The new significance test rejects
  about 5% of the time when there's truly no difference, which is what a correctly
  calibrated test should do.
- **fama-french-factor-model: 26 → 37 tests.** Its data-loading module went from 55% to
  95% coverage.
- **options-pricing-engine: 2 new tests** covering the snapshot bug.
- **Others:** alpaca 11 → 24, toolkit 10 → 14, ma-crossover 125 → 126.

## 6. Type hints and docstrings

Every function parameter and return value in `fama-french-factor-model`,
`options-pricing-toolkit` and the Alpaca bot now has a type annotation. Public functions
and classes that had no docstring now have one (including 28 in `ma-crossover-backtest`).
The three larger src-layout packages were already fully typed.

## 7. READMEs

- **New:** portfolio-optimization (it had none), the root portfolio index, and the Alpaca
  bot (rewritten).
- **Corrected:** every other README, against fresh runs.

The pricing and factor-model projects say explicitly that they're not trading strategies,
so Sharpe, drawdown and turnover don't apply to them. `options-pricing-toolkit` is labelled
as earlier work that `options-pricing-engine` supersedes. I'd recommend deleting it. I left
that decision to you.

## What I did not do, on purpose

- **No new strategies or features.** The brief was depth over breadth.
- **I did not re-derive every number in the older design documents** (`AUDIT.md`,
  `REVIEW.md`, `DECISIONS.md`). They record what was true when they were written. Every
  number in a README or RESULTS.md was checked, apart from one exception: the AUDIT.md
  bug-impact figure, which I re-checked separately and describe in section 3.
- **I didn't make `fama-french-factor-model` pass strict mypy.** It never used it, and the
  remaining errors are mostly limitations of the pandas and matplotlib type stubs, not
  bugs.
- **I haven't verified the Alpaca bot against the live API.** I had no paper keys. The
  first real run should use `--dry-run`.
