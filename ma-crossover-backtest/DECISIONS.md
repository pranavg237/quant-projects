# Decisions log

Running log of judgement calls made while working autonomously through the eight
phases. Newest entries at the bottom.

## Phase 1: Audit

1. **Scope is the `ma-crossover-backtest` git repository.** The sibling `project a`
   folder (an `ffmodel` Fama-French package) is not under version control and is a
   separate, already-tested project. It is left untouched. Phase 5's factor module is
   written fresh inside this repo so the repo is self-contained, borrowing the French
   library parsing approach.
2. **Python 3.14 (Homebrew) with a project-local `.venv`.** It is the only interpreter
   on the machine besides the 3.9 venv inside `project a`. `requires-python` is set to
   `>=3.11` so modern typing syntax can be used.
3. **Baseline numbers were re-derived, not trusted.** The original script was run
   verbatim; it reproduces the README (48.24% / 36.93% / 95.54%), so audit measurements
   are made against a live reproduction.
4. **Adjusted prices for signals were judged correct for scale-invariant rules** after
   checking that the back-adjustment factor is a constant across all dates before `t`.
   The audit records this rather than "fixing" it, because the naive fix (raw close)
   is worse. The framework carries raw, split-adjusted and total-return series so the
   right one can be chosen per signal.
5. **Commit directly to `main`, never push.** The user asked for a commit per phase and
   did not ask for pushing or branching; the remote is left alone.

## Phase 2: Fix and test

6. **Package name `quantbt`; the repository keeps its name.** Renaming the GitHub repo is
   the user's call.
7. **Fills happen at the next bar's open, sized on the pre-fill equity.** Same-bar close
   fills are the main inflation source in the audit. Next-open is the most realistic
   assumption available with daily bars; next-close is offered as an alternative in the
   engine (Phase 3) but is not the default.
8. **Default costs: 5 bps slippage per side, 1 bp commission on notional.** Round
   numbers in the right range for a liquid US ETF; every result prints the assumption.
   Commission is charged on the notional and included in sizing, so a 100% target
   leaves cash at exactly zero. This makes the vectorised and event-driven results
   agree to floating-point precision rather than "within tolerance".
9. **Cash earns a configurable rate, default zero.** Sharpe uses the same rate. Runs
   against real data will pass the French `RF` series once Phase 5 lands.
10. **Total-return prices for P&L and scale-invariant signals; split-adjusted prices
    kept for level-based rules; as-traded prices reconstructed.** See the module
    docstring of `quantbt/data.py` and audit item 3.7.
11. **tz-aware indices are rejected, not converted.** Silent conversion is how UTC
    midnight bars end up on the wrong trading day. Daily Yahoo bars are naive
    exchange dates already; the downloader strips a tz only if Yahoo adds one.
12. **Raw downloads are cached on disk and git-ignored, with a manifest.** Committing
    price data would bloat the repo; the manifest records the snapshot dates so numbers
    in the README can be tied to a data vintage.
13. **CAGR in `summary()` is returns-based and counts the first period.** The
    curve-based `cagr(equity)` helper stays for callers who already have a curve.
14. **Python 3.12 typing baseline.** numpy's stubs use the `type` statement, so mypy
    cannot target 3.11.
15. **Synthetic lookahead detector needs overnight gaps.** With `open[t+1] == close[t]`
    a next-open fill *is* a same-bar-close fill, so gap-free synthetic data cannot expose
    the bug. `synthetic_prices` therefore models close-to-open and open-to-close moves
    separately, and the detector uses a signal that "knows" the overnight gap.

## Phase 3: Event-driven framework

16. **Bar-level event loop rather than tick-level queues.** With daily bars a full
    message-queue architecture adds overhead without changing any result; the loop
    keeps the essential property (orders submitted at close ``t`` fill on bar ``t+1``)
    and stays fast enough for walk-forward grids. Order/Fill/Rejection are still
    explicit objects so an intraday extension has somewhere to plug in.
17. **Target-weight orders are sized at the fill price, net of commission.** A
    strategy expressing "100% long" gets exactly that with zero residual cash, which
    is what makes the port agree with the vectorised model to 1e-15 rather than to a
    tolerance. Share-count orders are available for strategies that size themselves.
18. **The evaluation window starts flat.** Both the vectorised reference and the engine
    enter an in-force signal at the first opportunity *inside* the window instead of
    assuming the position already existed. The SPY 50/200 figure moved from 51.3% to
    51.1% as a result.
19. **Interest accrues on cash before fills each bar**, i.e. on the balance held from
    the previous close; cash raised at an exit earns nothing until the next bar.
20. **Symbols that leave the universe are force-liquidated on the next bar** at that
    bar's fill price, or at the last mark if no print exists. There is no delisting
    haircut; a real delisting typically loses more than that, so this is a known
    optimistic simplification recorded in REVIEW.md later.
21. **Cost basis and round trips use average cost, FIFO-free.** Simpler, and hit
    rate/profit factor do not depend on lot matching.
22. **Strategies raise on orders for symbols outside the universe** rather than
    silently ignoring them, so survivorship-biased code fails loudly.

## Phase 4: Metrics and validation

23. **Turnover is one-way, annualised: sum |Δw| / 2 per year.** Hit rate is per closed
    round trip; "positive periods" is reported separately for bar-level win rate.
24. **Walk-forward windows are calendar-based (years), rolling by default.** Anchored is
    an option. The first fold's training window starts at ``start``; the caller supplies
    warm-up history before it. Fold boundaries never overlap, and the stitched OOS
    series is checked for duplicate dates.
25. **The objective defaults to Sharpe on the training window with the same cost model
    as the test window**, so parameter choice already "sees" costs; cost-free selection
    is what drives grids to over-trade.
26. **Overfitting diagnostics: Mertens standard error, PSR, DSR with the expected-max
    correction, stationary bootstrap (Politis-Romano) with n^(1/3) expected block length,
    and CSCV-PBO.** Each is a published estimator with known limitations; the report
    prints flags rather than a single verdict.
27. **First real result (SPY, 2005-2025 OOS, rolling 5y/1y, 17-point grid):** stitched
    walk-forward Sharpe 0.67 vs in-sample-best 0.79, PBO 0.50, chosen parameters wander
    across the whole grid. The strategy is not robustly better than buy-and-hold
    (Sharpe 0.62 over the same span); the framework says so rather than hiding it.

## Phase 5: Factor analysis

28. **The French loader is a fresh implementation inside this repo** (the same parsing
    idea as the `ffmodel` package in `project a`, which stays untouched). Cached zips
    live under `data/cache/french/` with a 30-day refresh, since French updates monthly.
29. **Newey-West lag defaults to the 1994 rule of thumb `4 (T/100)^(2/9)`**, i.e. 9
    lags for ~5,000 daily observations; callers can fix it. Daily regressions are the
    default because strategies are daily; monthly aggregation is available via the
    `frequency` argument for anyone who prefers the conventional monthly alphas.
30. **Regressions take raw strategy returns and subtract French `RF`.** A strategy that
    sits in cash earning zero therefore shows a *negative* excess return while flat,
    which is correct: the honest fix is to let cash earn `RF` in the backtest (Phase 6
    runs do exactly that).
31. **First factor result: the walk-forward MA crossover on SPY (2005-2025) has FF5
    alpha of +1.8% p.a. with t = 0.93 and a momentum loading of 0.15 (t = 6.2).**
    It is a diluted market-plus-momentum exposure, not alpha.
