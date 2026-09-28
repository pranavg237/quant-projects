"""Descriptive statistics and spanning regressions for the factors themselves."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .regression import _order_columns, infer_periods_per_year, newey_west_lags


def factor_summary(factors: pd.DataFrame, periods_per_year: Optional[int] = None) -> pd.DataFrame:
    """Annualized mean, volatility and Sharpe ratio of each factor, with the t-stat of its mean."""
    F = factors.drop(columns="RF", errors="ignore")
    ppy = periods_per_year or infer_periods_per_year(F.index)
    mean, sd, n = F.mean(), F.std(), F.count()
    return pd.DataFrame(
        {
            "mean (ann.)": mean * ppy,
            "vol (ann.)": sd * np.sqrt(ppy),
            "Sharpe (ann.)": mean / sd * np.sqrt(ppy),
            "t(mean)": mean / sd * np.sqrt(n),
            "worst period": F.min(),
            "best period": F.max(),
            "N": n,
        }
    )


def spanning_regressions(
    factors: pd.DataFrame, periods_per_year: Optional[int] = None, lags: Optional[int] = None
) -> pd.DataFrame:
    """Regress each factor on all the others (Newey-West errors).

    A factor whose intercept is indistinguishable from zero is spanned by the
    rest: it adds nothing to the model's explanation of average returns (the
    Fama-French 2015 finding for HML in the five-factor model).
    """
    F = factors.drop(columns="RF", errors="ignore").dropna()
    if F.shape[1] < 2:
        raise ValueError("spanning regressions need at least two factors")
    ppy = periods_per_year or infer_periods_per_year(F.index)
    maxlags = newey_west_lags(len(F)) if lags is None else lags
    rows = {}
    for factor in F.columns:
        others = [c for c in F.columns if c != factor]
        fit = sm.OLS(F[factor], sm.add_constant(F[others], has_constant="add")).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        rows[factor] = {
            "alpha (ann.)": fit.params["const"] * ppy,
            "t(alpha)": fit.tvalues["const"],
            **fit.params[others].to_dict(),
            "R2": fit.rsquared,
        }
    return _order_columns(pd.DataFrame.from_dict(rows, orient="index"))
