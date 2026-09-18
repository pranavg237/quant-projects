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
