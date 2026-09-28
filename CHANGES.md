# Changes

What changed, and why, in plain language. Each item says what was wrong, what I did about
it, and what it changed. The plans behind both rounds are in [PLAN.md](PLAN.md).

- [Round 2](#round-2-from-correct-and-tested-to-shows-quant-judgment): deepen every project.
- [Round 1](#round-1-make-everything-run-test-and-report-honestly): make everything run,
  test and report honestly.

# Round 2: from "correct and tested" to "shows quant judgment"

## The short version

- **No new winners, on purpose.** Every headline null result survived: nothing beats 1/N
  in portfolio optimisation, and nothing beats buy-and-hold SPY in the backtests. What
  changed is how well each result is explained and defended.
- **Four real bugs or wrong explanations, found by checking rather than assuming:**
  - portfolio-optimization's Ledoit-Wolf shrinkage (constant-correlation target) was
    missing a term from the paper and shrank about twice as hard as it should;
  - the Alpaca bot could place the same order twice if run twice in one evening;
  - ma-crossover's `mean_reversion` result differed between two downloads, and the
    round 1 explanation (a stock split) was wrong: it is one exit decision on a knife-edge;
  - market-making's README said Avellaneda-Stoikov loses less to informed traders. It
    doesn't; it recovers more afterwards.
- **Multiple testing taken seriously.** The MA crossover's deflated Sharpe ratio, counting
  all 63 configurations actually tried, is 0.08 out of sample. Before, it was deflated
  only for its own 17-point grid and looked like 0.99.
- **Everything reproduces offline.** ma-crossover and fama-french now commit dated,
  hash-checked data snapshots, so a rerun gives the same digits. An agent that had not
  seen the work re-ran every project from a clean clone (results below).
- **`options-pricing-toolkit` deleted.** `options-pricing-engine` does everything it did,
  better.
- **Tests: 731 → 904** (the old count included the deleted toolkit's 14).

---

## Project by project

### portfolio-optimization: a shrinkage bug fixed, no look-ahead proven, costs reported, and why 1/N is hard to beat

The headline did not change: none of the nine methods has a Sharpe ratio statistically
distinguishable from 1/N (smallest p = 0.20).

- **Fixed a real bug in Ledoit-Wolf shrinkage.** The backtest shrinks the covariance
  matrix towards a "constant correlation" target (Ledoit & Wolf 2003). The formula for
  *how much* to shrink was missing the paper's ρ̂ term, which corrects for the target
  being estimated from the same data. Without it the code shrank about twice as hard as
  it should (0.54 vs 0.29 on the first window; 0.20 on average after the fix). The fix is
  tested against a line-by-line transcription of the paper and against a simulation where
  the right answer is known; the old formula fails the simulation test. Shrunk max-Sharpe
  moved 0.83 → 0.88 and shrunk min-variance 0.46 → 0.51. Neither is significantly
  different from 1/N, so the headline stands.
- **Proved the backtest can't see the future.** A new test replaces every return after a
  decision date with garbage and reruns all nine strategies. Nothing chosen up to that
  date changes, and weights after it do, so the test can't pass vacuously.
- **Reported what trading costs.** A new table gives, per method, turnover per rebalance
  and per year, cost drag in bp/yr, and Sharpe before and after costs. Even the busiest
  optimiser (about 200% turnover a year) loses only about 19 bp a year, or 0.02 of Sharpe.
  What separates the methods is estimation error, not trading.
- **Explained why 1/N is hard to beat (DeMiguel, Garlappi & Uppal 2009).** Mean-variance
  weights are roughly the inverse covariance times the estimated means. A 60-month mean is
  only known to about ±6.7% a year, and inverting a covariance matrix with condition
  number 668 amplifies that noise. A simulation calibrated to these 15 ETFs shows that
  here mean-variance should need far fewer months than DGU's ~3000 to beat 1/N, because
  this universe's 1/N is mostly equity and sits well inside the frontier; the README
  flags that this is a best case for mean-variance (IID returns). On real data, with every
  estimation window scored on the same 141 months, short windows lose to 1/N and long
  ones gain, which is the direction the theory predicts, but every gap is within about one
  standard error (p 0.48–0.60). Detecting even a 0.10 Sharpe edge at that noise level would
  take roughly 2,500–3,200 months. The honest conclusion is a null result: 17 years of
  monthly data cannot show that any of these methods beats 1/N.

### ma-crossover-backtest: a robustness heatmap, an honest trial count, and a direct look-ahead test

- **Parameters are proven not to peek at the future.** Walk-forward already chose
  parameters on past data only. A new test now proves it directly: it swaps every price
  after one training window for a completely different price history and checks that
  every parameter chosen up to that point stays exactly the same. I also broke the code on
  purpose (letting training see six months ahead) to confirm the test catches it.
- **A parameter heatmap asks "plateau or lucky spike?"** It shows the MA crossover's
  excess Sharpe, after costs, for 116 fast/slow window pairs on SPY, 2005–2025. It is
  in-sample and labelled as a diagnostic that selects nothing. The picture is a broad,
  low plateau around buy-and-hold's 0.53. The single best cell (10/75, 0.73) is an
  isolated spike whose neighbours average 0.52, which is what luck looks like. The whole
  range spans only 1.5 standard errors of one Sharpe estimate, so the data can't tell
  these settings apart, which is also why walk-forward kept switching parameters.
- **The deflated Sharpe ratio now counts every configuration tried.** The formula is
  pinned by a test to the worked example in Bailey & López de Prado (2014). The real
  change is the trial count: before, each strategy was deflated only for its own 17-point
  grid, which gave the MA crossover a reassuring DSR of 0.99 (it was never quoted in the
  results page). Five strategies and 63 configurations were actually tried. The best of
  63 pure-noise strategies would be expected to show a Sharpe of about 0.90, and against
  that hurdle the MA crossover's DSR falls to 0.16 in-sample and 0.08 out of sample. A
  table shows how the answer moves with the assumptions: it only passes the usual 95% bar
  if the whole project counts as about five independent tries. The takeaway: a Sharpe of
  0.58 that looks significant alone (PSR 0.995) is not significant once you admit how
  many things were tried.
- **A reproducibility finding.** A second fresh Yahoo download on the same day
  reproduced four of the five strategies, but `mean_reversion` came back at its *original*
  numbers (Sharpe 0.44 vs the committed 0.42). That contradicts the earlier explanation
  (a stock-split re-adjustment), and RESULTS.md now says so. The root cause is that this
  project has no committed price snapshot; see "Open items" below.
- **Every number now reproduces offline from frozen inputs.** The project now commits a
  12.6 MB snapshot (`data/snapshot-2026-09-28/`) of exactly the prices and French factor
  files the pipeline reads, with a sha256 per file. The code reads it by default and
  refuses to download or to quietly return less data than asked for (`--live-data` opts
  back in). Two full reruns gave byte-identical results. Building it explained the
  `mean_reversion` flip: the whole 0.42-vs-0.44 difference is one exit decision. On
  2021-04-27 XLE's z-score was 0.0000039 above the exit threshold, so a change in the
  seventh digit of one price (about the precision of Yahoo's data) moves the exit by a
  day. Nudging that single price reproduces the other download's result. The old
  explanation (stock splits) was wrong. The committed numbers now come from the snapshot:
  `mean_reversion` Sharpe 0.44 (was 0.42 in the README). The lesson is about the strategy
  as much as the data: a Sharpe that moves 0.02 when one price changes in its seventh
  digit is telling you how uncertain it is.

### options-pricing-engine: a real Greeks audit, put-call parity on SPY, bid-ask smiles

- **The Greeks check now covers what the README claimed.** It used to test two parameter
  points, and three second-order Greeks were checked against the analytic delta and vega
  rather than the price. Now all nine Greeks, calls and puts, are compared with finite
  differences of the *price* on 168 cases per option type, from one day to three years and
  ±3 standard deviations in strike. Step sizes come from a small error analysis: the price
  is a difference of two terms of size S, so its round-off is about ε·S, and the textbook
  step loses about 10x accuracy on one-day options. The tests also plant typical unit
  bugs (vega per vol point, theta per day, a sign flip) and require the check to catch
  them. No formula was wrong. The worst relative error is 6e-8 for first-order Greeks and
  7e-4 for second-order. The README's "10 Greeks" was corrected to 9.
- **Put-call parity on the real SPY chain, with its circularity stated up front.** The
  project infers the forward from parity, so parity can't be tested at its *level*, only
  for consistency: one forward per expiry must explain every strike within the bid-ask
  spread. Under European parity, 44.5% of pairs fail beyond the spread, in a systematic
  pattern. The cause is that SPY options are American: with rates near 4%, a deep
  in-the-money American put is worth roughly rKτ more than a European one. Pricing that
  early-exercise premium on a binomial tree (nothing fitted to the residuals) cuts
  failures to 10.6% and shows the pipeline's forward is biased low by up to 85 bp at 21
  months. That bias, not noisy data, is what caused a call/put vol mismatch the README had
  blamed on free quotes. A control test shows the same adjustment makes a truly European
  chain *worse*, so the SPY result isn't a free fit. The remaining out-of-sample failures
  are stale quotes, found with model-free arbitrage checks: 72 options are quoted below
  their immediate-exercise value. The corrected forward is measured but not yet built into
  the surface, because doing that properly changes every calibration number and deserves
  its own review.
- **Smiles with uncertainty.** The surface now records implied vol at the bid and at the
  ask, and a new per-expiry smile chart shows that band. It is a fraction of a vol point
  almost everywhere, so Heston's 5–10 point misses in the short-dated put wing are model
  failure, not quote noise.

### market-making-simulator: adverse selection measured by counterparty, sensitivity with error bars

- **The formulas are now checked against hand-worked numbers.** The Avellaneda-Stoikov
  reservation price r = s − qγσ²(T−t) and total spread γσ²(T−t) + (2/γ)ln(1+γ/k) were
  already implemented correctly and reproduce the paper's Table 1. But the old test
  compared the formula with itself. A new test uses values worked out by hand, in both the
  "reservation price ± half-spread" form and the paper's per-side form.
- **Markout split by counterparty.** Every fill now records whether the other side was an
  informed trader. For each fill we measure the *edge* (what the maker earned against the
  mid at the time), the *markout* (how the mid moved 1, 5, 20 and 100 steps later), and the
  *realised spread* (edge + markout). Because the simulation releases an informed order's
  price impact at a known rate, we know exactly what an informed fill *should* lose,
  −J(1−(1−v)^h), and the measurement lands on it (−1.24 vs −1.28 ticks at 20 steps, −1.89
  vs −1.99 at 100). Fills against uninformed traders lose essentially nothing. That is a
  check that the measurement is right, not just a number.
- **A corrected explanation.** The README used to say A-S does better under informed flow
  because it avoids being picked off. The split shows its informed losses are the same as
  the naive maker's (0.72 vs 0.71 per session). The difference is *how it unwinds*: after
  being picked off it skews its quotes and buys back from uninformed traders while the
  price is still moving its way, recovering about 43% of what the informed flow took. One
  thing is flagged as unexplained: the symmetric maker's position-reducing fills mark out
  at −0.21 ticks, 2.5 standard errors from zero.
- **Sensitivity with standard errors.** Risk aversion γ, fill decay κ, volatility σ,
  arrival rate A and the informed fraction are each swept for all three policies on the
  same seeds, with bootstrap standard errors, a CSV and a figure. A-S's advantage grows
  when inventory is riskier relative to spread income (higher κ, σ, more informed flow),
  nearly vanishes at low volatility, and with too little risk aversion (γ = 0.01) a plain
  position limit beats it.
- **Honesty about noise.** The old informed-flow table (66 sessions, no error bars) showed
  A-S 12.9 vs symmetric 12.2 with no informed flow. The rebuilt table uses 100 sessions on
  a different seed block and reads 10.5 ± 0.8 vs 8.8 ± 0.6; on matched subsets the two
  seed blocks differ by about 2 standard errors. A per-session Sharpe near 10 carries a
  standard error of roughly 0.8 at this sample size, which is why every table now shows
  one. Every Sharpe remains labelled per simulated session, not annualised, including the
  figure axes.

### fama-french-factor-model: reproducible data, checked standard errors, honest rolling betas

- **Frozen inputs.** The example report now runs offline from a committed, dated snapshot
  (`data/snapshot-2026-09-28/`): the French factor files as downloaded, Yahoo returns, and
  a manifest with download time, URLs, hashes and sample periods. New commands:
  `ffmodel snapshot` and `--data-dir`. A same-day re-download had shifted several printed
  t-stats because Yahoo revises adjusted prices (SPY alpha t −1.99 → −1.98), so the old
  claim that reruns reproduce every digit wasn't true until the inputs were frozen.
- **Newey-West checked, and compared with OLS.** A new test checks the Newey-West errors
  against a hand-written sandwich formula and against statsmodels' HAC. The lag rule,
  floor(4(T/100)^(2/9)) (4 lags here), is now explained: fixed in advance, grows slowly
  with T, fine for monthly returns that have little autocorrelation. Each report shows
  t-stats under OLS, White and Newey-West side by side. Only one conclusion flips: SPY's
  alpha goes from p = 0.065 (OLS) to 0.047 (NW), which means it was never robust.
- **Rolling betas with honest uncertainty.** Rolling 36-month betas now have 95% bands,
  and a Wald test asks whether exposures actually changed across seven non-overlapping
  3-year blocks. My first version used Newey-West errors inside the windows and found
  lots of "significant" drift. That looked too good, so I simulated it: with only 36
  observations, Newey-West intervals miss the true beta 13–14% of the time instead of 5%,
  and a truly constant beta is "rejected" up to 35% of the time. Rolling analysis
  therefore uses HC3 errors (4–5% misses). With correct errors, only 2 of 30 stability
  tests are below 0.05 — about what chance alone would produce.
- **Interpretation section.** Which loadings are significant and what they mean
  economically (AAPL HML −0.44 = growth tilt, RMW +0.62 = profitability tilt), why AAPL's
  alpha is hindsight in choosing a known winner rather than a strategy, Holm-adjusted
  p-values across the five alphas, and why "not significant" is not "zero": BRK-B's alpha
  95% interval is about −2.5% to +8% a year.

### alpaca-ma-crossover-bot: risk checks, safe re-runs, and failure testing

The bot already made the right trading decision. What it lacked was protection for when
something goes wrong: the broker, the network, the data or the operator. This round adds
that protection without changing the strategy; the backtest numbers are untouched.

**Pre-trade risk checks (`risk.py`).** Before any order is sent, a small pure function
checks the post-trade position against 25% of equity and $50k notional, the day's loss
against Alpaca's previous-close equity (2% limit), buying power, that a sell can't open a
short, and a kill switch. Every check is logged with its reason, so a blocked order
explains itself. Two judgement calls:
- The daily-loss limit only blocks *buys*. Stopping the bot from selling would trap it in
  a losing position.
- The kill switch (an env var or a `KILL_SWITCH` file) halts everything but does not sell
  automatically. You pull it when you don't trust the system, which is exactly when an
  automatic market sell is most dangerous.

**Idempotency bug found and fixed.** An order placed after the close waits at the broker
overnight while the position is still zero. The old bot only looked at positions, so
running it twice the same evening would have bought twice. Now it refuses to act while any
order is open, it gives each order a fixed `client_order_id` built from symbol, bar date
and side and checks whether that id already exists, and the broker itself rejects a
repeated id, so even two runs racing each other can't double-buy.

**Retries done carefully.** Reads are retried with backoff on timeouts, 429 and 5xx. Order
submission is never retried: if it times out, the order may already exist. The bot looks
it up by its fixed id instead of sending it again.

**Partial fills.** A buy that only half-filled used to leave the bot underweight until the
next crossover, possibly years later. It now tops up on the next bar if the position is
below half the target. The threshold is loose on purpose, so normal price moves don't
make it trade every day.

**Logging and tests.** Every step writes a JSON log line with a run id. API keys are kept
out of the logs two ways, and a test proves it even when a fake server echoes the keys
back in an error body. Tests went from 24 to 121. They drive the real client against an
in-memory fake of Alpaca's HTTP API: errors, timeouts, partial fills, a closed market, bad
data and re-runs. The fake follows Alpaca's documented responses but has not been checked
against the live API, and there is still no live or paper track record.

## Repository-wide, verification and open items

### Repository-wide

- **Deleted `options-pricing-toolkit`**: its folder, its CI job and its row in the root
  README. It was a strict subset of `options-pricing-engine`.
- **CI** sets `OPENBLAS_NUM_THREADS=1`. On a shared machine, oversubscribed BLAS threads
  slowed the options suite about 15x.
- **How the work was done.** Each project was deepened by a separate agent in its own git
  worktree, allowed to touch only its own folder, then merged here. Shared files (root
  README, this file, PLAN.md, CI) were edited only on the integration branch, so no
  merge conflicts were possible.

### Independent verification

After merging, an agent that had seen none of this work cloned the branch fresh, installed
each project from its `requirements.txt`, ran exactly what CI runs, reran every results
pipeline (offline ones with the network deliberately cut), and checked each number in
every README against the regenerated output: roughly 770 numbers.

- **Code and results: clean.** All six projects pass lint, types and tests (904 tests).
  Every committed result reproduced. The only files that changed on the rerun were runtime
  fields, optimiser noise below 1e-5 in two portfolio JSON files, and font rendering in
  some PNGs. Nothing claimed to be offline touched the network.
- **Documents: it found real problems, since fixed.**
  - The root README said no backtest strategy "beats" buy-and-hold, while RESULTS.md said
    four of five lose. The truth is in between: the MA crossover's Sharpe is 0.58 against
    0.53 for buy-and-hold over the same dates, but the gap is well inside the noise and
    doesn't survive the deflated Sharpe. Both READMEs now say exactly that.
  - Market-making claimed a hard position limit beats A-S for γ ≤ 0.04. In the data it
    only does at γ = 0.01. Fixed, along with three one-digit rounding slips and a results
    table whose Sharpe column wasn't labelled per session.
  - The options engine quoted a few numbers that no script produced (for example a −1.15
    vol-point gap where the parity output says −1.08). Each is now generated by the
    pipeline, or removed.
  - Stale statements in AUDIT.md, REVIEW.md and DECISIONS.md (defaults, speed-ups, an
    implied-vol tolerance that had since been fixed) were corrected against the code.
  - The root README's "offline" claim now names its exception: fama-french's factor and
    test-portfolio examples download live, and French's small data revisions move those
    reports in the last digit (every quoted figure still holds).

### Open items

- **No live or paper trading record** for the Alpaca bot, and its fake API follows Alpaca's
  documentation but hasn't been checked against the real service.
- **The options surface still uses the European-parity forward.** The American-exercise
  bias is measured (up to 85 bp at 21 months) but not yet removed, because removing it
  changes every calibration number.
- **fama-french's live examples** are not frozen; only the main `analyze` example is.

---

# Round 1: make everything run, test and report honestly

This section describes the repository as it was at the end of round 1. Test counts, and
references to `options-pricing-toolkit` (deleted in round 2), are round 1 figures.

## The short version

- **Two results changed materially once they were measured properly.**
  - Portfolio optimisation's Sharpe ratios were computed against a 0% risk-free rate. They
    fall by about 0.15 each against real T-bills, and "Markowitz beats 1/N" turns out not
    to be statistically supported.
  - One of the five backtest strategies moved slightly on a fresh data download. (Round 1
    blamed a stock-split re-adjustment; round 2 found the real cause.)
- **Five real bugs fixed**, each with a regression test:
  - the options engine ignored its own committed data snapshot;
  - the Alpaca bot used unadjusted prices, and separately could trade on a half-finished
    daily bar;
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

> **Corrected in round 2:** the explanation below was wrong. The move comes from a single
> exit decision sitting on a knife-edge (XLE's z-score was 0.0000039 above zero on
> 2021-04-27), so seventh-digit noise between downloads flips it. See the round 2
> ma-crossover-backtest section.

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

### options-pricing-toolkit: the implied-vol solver rejected valid prices (project deleted in round 2)

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
that decision to you. (You did: it was deleted in round 2.)

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
