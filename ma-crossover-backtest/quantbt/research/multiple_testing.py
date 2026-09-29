"""Deflated Sharpe ratios for the whole research family, not one strategy at a time.

Five strategies were run, each over its own parameter grid. The number of trials that
belongs in the deflated Sharpe ratio (Bailey & Lopez de Prado 2014) is every
configuration whose result could have ended up in the write-up, which is the sum of all
five grids, not the size of one strategy's grid. Using one grid's size (17 for the MA
crossover) understates the search by almost 4x.

Two things about the count are worth saying out loud:

* it is a **lower bound** on the true number of trials: ideas that were considered and
  never coded, and earlier versions of the code, are not in any grid file;
* it is also **conservative in another way**: the DSR's expected-maximum formula
  assumes independent trials, and the 17 MA configurations are highly correlated with
  each other, so the *effective* number of independent trials is smaller than 63.

The two errors push in opposite directions, which is why the script reports the DSR over
a range of trial counts rather than one number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.validation.bootstrap import (
    _moments,
    deflated_sharpe_from_stats,
    expected_max_sharpe,
    psr_from_stats,
)

_METRIC_COLUMNS = (
    "objective",
    "total_return",
    "cagr",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "turnover",
    "trades",
)


def collect_trials(root: str | Path, strategies: Sequence[str]) -> pd.DataFrame:
    """Every configuration tried, one row each: ``strategy, params, sharpe``.

    Reads ``<root>/<strategy>/grid.csv`` as written by the research pipeline. The Sharpe
    ratios there are annualised, in excess of the risk-free rate and net of costs, over
    each strategy's full out-of-sample span (in-sample for the grid, by construction).
    """
    rows: list[dict[str, object]] = []
    for name in strategies:
        grid = pd.read_csv(Path(root) / name / "grid.csv")
        params = [c for c in grid.columns if c not in _METRIC_COLUMNS]
        for _, r in grid.iterrows():
            rows.append(
                {
                    "strategy": name,
                    "params": ", ".join(f"{p}={r[p]:g}" for p in params),
                    "sharpe": float(r["sharpe"]),
                }
            )
    if not rows:
        raise ValueError("no trials found")
    return pd.DataFrame(rows)


def dsr_row(
    excess_returns: pd.Series, n_trials: int, var_trials_sharpe_annual: float
) -> dict[str, float]:
    """Sharpe, PSR, the noise hurdle and the DSR for one excess-return series.

    ``sr0_annual`` is the Sharpe that the best of ``n_trials`` pure-noise configurations
    with the given cross-trial variance would be expected to show; the DSR is the
    probability that the true Sharpe exceeds it.
    """
    r = excess_returns.dropna()
    ppy = metrics.periods_per_year(r.index)
    n, skew, kurt = _moments(r)
    sd = float(r.std(ddof=1))
    sr = float(r.mean() / sd) if sd > 0 else float("nan")
    var_period = var_trials_sharpe_annual / ppy
    return {
        "sharpe": sr * np.sqrt(ppy),
        "n_obs": float(n),
        "skew": skew,
        "kurtosis": kurt,
        "psr": psr_from_stats(sr, 0.0, n, skew, kurt),
        "n_trials": float(n_trials),
        "sr0_annual": expected_max_sharpe(n_trials, var_period) * np.sqrt(ppy),
        "dsr": deflated_sharpe_from_stats(sr, n, skew, kurt, n_trials, var_period),
    }


def dsr_sensitivity(
    excess_returns: pd.Series,
    trial_counts: Mapping[str, int],
    variances: Mapping[str, float],
) -> pd.DataFrame:
    """The DSR of one series for every (trial count, cross-trial variance) assumption.

    The DSR's noise hurdle grows with both the number of trials ``N`` and the variance
    ``V`` of their Sharpe ratios, and in practice ``V`` moves it at least as much as
    ``N``. Reporting one number hides that, so this returns the full grid.
    """
    rows = []
    for n_label, n in trial_counts.items():
        for v_label, v in variances.items():
            rows.append(
                {
                    "n_assumption": n_label,
                    "var_assumption": v_label,
                    "var_trials_sharpe_annual": v,
                    **dsr_row(excess_returns, n, v),
                }
            )
    return pd.DataFrame(rows)
