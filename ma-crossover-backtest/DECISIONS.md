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

## Phase 7: Reporting and docs

32. **Tearsheets are hand-rolled inline SVG in a single HTML file**, with no charting
    library and no external requests. One file per strategy can be emailed or opened
    from disk, which matters more for a portfolio piece than fancy interactions.
33. **Colours are CSS custom properties, never literals in the markup**, so light and
    dark mode are both first-class. The two-series palette was validated for
    colour-vision-deficiency separation (delta-E 24.7 light, 26.8 dark) and for at
    least 3:1 contrast against both surfaces. Identity is carried by a legend and
    direct labels as well as hue.
34. **The tearsheet leads with a verdict, not with the equity curve.** Four of the five
    strategies lose to buy and hold, and a reader should not have to work that out from
    a chart. The verdict is computed from the out-of-sample Sharpe against the
    benchmark and the probabilistic Sharpe ratio.
35. **Rolling Sharpe is computed on excess returns and blanked when annualised
    volatility is under 1%.** A long/flat strategy sitting in Treasury bills for a year
    has a tiny, nearly constant return and therefore an enormous raw Sharpe; plotting
    that spike flattens every other year on the axis and says "this was cash", not
    "this was brilliant". Found by rendering the chart and looking at it.
36. **Direct end-labels are dropped when they would collide** rather than nudged apart,
    because a nudged label is detached from its line and reads as noise. The legend and
    the hover tooltip carry identity in that case.
37. **Generated tearsheets and result CSVs are committed; raw price data is not.**
    The reports are the deliverable; the price cache is large and gets revised by the
    vendor. Per-bar weight files are also ignored, being big and reproducible.
38. **The README's code examples are executed by the test suite**
    (`tests/test_docs.py`), so they cannot rot silently.
39. **Chart data is thinned to about 1.5 points per horizontal pixel, keeping each
    bucket's minimum and maximum across all series.** A 20-year daily series is 5,000
    points drawn into 960 pixels. Plain every-kth sampling would be wrong because it can
    drop the bottom of a drawdown; min/max bucketing keeps every extreme, and each kept
    point retains its real date and value so tooltips stay truthful. Tearsheets went from
    about 1 MB to about 350 KB.
40. **Per-bar weight files are not committed.** The cross-sectional momentum run writes a
    4 MB weights file; it is regenerated by `scripts/run_strategies.py`.
41. **Short borrow is charged, at a flat rate per strategy, and defaults to zero.**
    30 bp a year on short ETF value and 50 bp on short stock value are general-collateral
    numbers; the names a momentum short book actually wants cost more. A per-name,
    time-varying borrow curve is the right model and the data for it is not free, so the
    flat rate is stated in RESULTS.md rather than hidden. The default of zero keeps every
    existing result and test reproducible, and makes the charge an explicit choice at the
    call site instead of a silent change of meaning.
42. **The borrow fee accrues on marked short value each bar, alongside cash interest,
    not at trade time.** It is a carrying cost: a short held for a year through a doubling
    stock costs roughly twice what one held flat costs, which is what a stock loan does.
43. **RESULTS.md is verified against `results.csv` by a test.** Hand-transcribed numbers
    around generated ones go stale on the first re-run, and a stale number in a public
    results table is indistinguishable from a doctored one. `tests/test_docs.py` parses
    all three markdown tables and checks them against the CSV within rounding. It found a
    rounding inconsistency the moment it was written.
44. **Bug fixes that could have moved a published number were checked by re-running with
    the new behaviour switched off.** Two latent bugs in `tsmom` and `xsmom` were fixed in
    the same pass as the borrow charge. Rather than asserting they were unreachable, the
    pipeline was re-run with `borrow_rate = 0`; it reproduced the previous table to six
    decimals, which is what licenses the claim in RESULTS.md that the movement is the
    borrow fee alone.
45. **`summary()` reports max gross exposure and minimum cash weight.** The portfolio
    deliberately permits negative cash and gross exposure above 1 — that is a strategy
    constraint, not a bookkeeping one — but an unreported leverage is indistinguishable
    from none. Reporting it costs two lines and makes the permissive default safe.

## Phase 8: Robustness and multiple testing

46. **The deflated Sharpe ratio counts every configuration of every strategy (N = 63).**
    The per-strategy grid (17 for the MA crossover) understates the search, because the
    write-up reports the best of five strategies. 63 is still a lower bound: it cannot
    count ideas that were never coded. The DSR is reported as a grid over the trial count
    and the cross-trial Sharpe variance rather than as one number, because in this project
    the variance assumption moves the answer at least as much as the count does.
47. **The cross-trial variance of the 17 MA configurations is not used as the headline.**
    Those configurations are near-copies, so their Sharpe ratios barely differ (variance
    0.006). Plugging that into the DSR treats near-duplicates as if their tiny spread were
    the noise in a Sharpe estimate, which is what produced the old DSR of 0.99. The honest
    options are the paper's recipe (variance of all 63 trials) or the sampling variance of
    one Sharpe estimate; both are shown.
48. **The parameter heatmap is in-sample and is labelled that way on the figure itself.**
    It exists to show whether good parameters form a plateau or a spike, not to pick
    parameters. Nothing reads from it. It uses the vectorised backtest because the dense
    grid would take the event-driven engine about 8 minutes (about 4 s a cell) against a
    few seconds vectorised. The script cross-checks the 17 shared cells against the
    engine's `grid.csv` and refuses to write if they disagree.
49. **Heatmap cells are colored relative to buy-and-hold, not to zero.** The question the
    figure answers is "does any region beat simply holding SPY?", so buy-and-hold's Sharpe
    over the same dates is the diverging midpoint (white).

## Phase 9: Frozen inputs

50. **Every reported number is computed from a committed snapshot, read by default.**
    Entries 12, 28 and 37 are superseded: raw downloads still go to the git-ignored
    `data/cache/`, but `data/snapshot-2026-09-28/` is what the scripts read unless
    `--live-data` is passed. The rule this serves is that every number in the README and
    RESULTS.md must come back when someone runs the code, and a vendor that restates its
    history makes that impossible without frozen inputs. Two same-day downloads gave two
    different `mean_reversion` results, which settled it.
51. **The snapshot holds what the pipeline reads, at full precision, and nothing else.**
    Yahoo's raw Open, Close, Adj Close, Volume, Dividends and Stock Splits are kept exactly
    (float64 text, read back with round-trip parsing), so the snapshot reproduces the
    cache run bit for bit. High and Low are dropped because no code reads them. They load
    as NaN, and `Context.history` refuses to serve them rather than handing a strategy
    NaNs. Rounding prices to 8-9 significant digits was tried and saved only 1.5-2 MB, not
    worth giving up exactness. History starts at the earliest date any committed script
    loads (1998 for SPY, 2003 for the rest) and runs to the download date, because splits
    after the analysis end still feed the as-traded price reconstruction. The total is
    12.6 MB.
52. **The loaders fail loudly instead of degrading.** A missing symbol, a hash mismatch, a
    later end date or a start before the trim date raises. The alternatives are a silent
    download, which brings back the drift, or a silently shorter series, which changes the
    warm-up. Both would produce a number that looks fine and is wrong.

