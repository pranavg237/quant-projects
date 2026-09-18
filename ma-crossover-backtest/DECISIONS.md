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
