"""Time-series factor regressions with Newey-West (HAC) standard errors.

``r_t - rf_t = alpha + beta' f_t + e_t``. Alpha is reported per period and annualised;
t-statistics use HAC standard errors with the Newey-West (1994) automatic lag unless
``lags`` is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from quantbt import metrics
from quantbt.factors.french import MODELS


def newey_west_lags(nobs: int) -> int:
    """Newey-West (1994) rule of thumb: ``floor(4 * (T / 100) ** (2 / 9))``."""
    return int(np.floor(4.0 * (nobs / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class FactorRegression:
    """Alpha, factor loadings and fit statistics of one factor regression."""

    name: str
    model: str
    factors: list[str]
    params: pd.Series
    std_errors: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    rsquared: float
    rsquared_adj: float
    nobs: int
    lags: int
    periods_per_year: int
    resid: pd.Series
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def alpha(self) -> float:
        return float(self.params["alpha"])

    @property
    def alpha_annual(self) -> float:
        return self.alpha * self.periods_per_year

    @property
    def alpha_tstat(self) -> float:
        return float(self.tvalues["alpha"])

    @property
    def betas(self) -> pd.Series:
        return self.params[self.factors]

    @property
    def residual_vol_annual(self) -> float:
        return float(self.resid.std(ddof=1) * np.sqrt(self.periods_per_year))

    @property
    def information_ratio(self) -> float:
        rv = self.residual_vol_annual
        return self.alpha_annual / rv if rv > 0 else float("nan")

    def table(self) -> pd.DataFrame:
        """Coefficients with HAC standard errors, t-stats and p-values."""
        return pd.DataFrame(
            {
                "coef": self.params,
                "std_err": self.std_errors,
                "t": self.tvalues,
                "p": self.pvalues,
            }
        )

    def to_series(self) -> pd.Series:
        out: dict[str, Any] = {
            "model": self.model,
            "alpha_annual": self.alpha_annual,
            "alpha_t": self.alpha_tstat,
            "alpha_p": float(self.pvalues["alpha"]),
        }
        for f in self.factors:
            out[f"beta_{f}"] = float(self.params[f])
            out[f"t_{f}"] = float(self.tvalues[f])
        out["r2"] = self.rsquared
        out["r2_adj"] = self.rsquared_adj
        out["info_ratio"] = self.information_ratio
        out["nobs"] = self.nobs
        out["nw_lags"] = self.lags
        return pd.Series(out)

    def summary(self) -> str:
        lines = [
            f"{self.name}: {self.model.upper()} regression, {self.nobs} obs "
            f"({self.start.date()} to {self.end.date()}), Newey-West lags={self.lags}",
            f"alpha {self.alpha_annual:+.2%} p.a. (t={self.alpha_tstat:.2f}), "
            f"R2={self.rsquared:.3f}, IR={self.information_ratio:.2f}",
        ]
        for f in self.factors:
            lines.append(f"  {f:<7} beta={self.params[f]:+.3f}  t={self.tvalues[f]:+.2f}")
        return "\n".join(lines)


def factor_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    *,
    excess: bool = False,
    lags: int | None = None,
    name: str | None = None,
    factor_names: Sequence[str] | None = None,
) -> FactorRegression:
    """Regress ``returns`` (raw, unless ``excess=True``) on the factors of ``model``.

    ``factors`` must contain the model's factor columns and ``RF`` (as from
    :func:`quantbt.factors.load_factors`). Rows are inner-joined on date; a monthly
    strategy series against daily factors will simply align on the common dates, so
    resample first.
    """
    cols = list(factor_names) if factor_names is not None else list(MODELS[model])
    missing = [c for c in [*cols, "RF"] if c not in factors.columns]
    if missing:
        raise ValueError(f"factors frame lacks columns {missing}")
    joined = pd.concat([returns.rename("r"), factors[[*cols, "RF"]]], axis=1, join="inner").dropna()
    if len(joined) < len(cols) + 10:
        raise ValueError(f"only {len(joined)} overlapping observations")
    y = joined["r"] if excess else joined["r"] - joined["RF"]
    x = sm.add_constant(joined[cols]).rename(columns={"const": "alpha"})
    nlags = newey_west_lags(len(joined)) if lags is None else int(lags)
    fit = sm.OLS(y.to_numpy(), x.to_numpy()).fit(cov_type="HAC", cov_kwds={"maxlags": nlags})
    idx = pd.Index(["alpha", *cols])
    ppy = metrics.periods_per_year(joined.index)
    return FactorRegression(
        name=name or str(returns.name or "strategy"),
        model=model,
        factors=cols,
        params=pd.Series(np.asarray(fit.params), index=idx),
        std_errors=pd.Series(np.asarray(fit.bse), index=idx),
        tvalues=pd.Series(np.asarray(fit.tvalues), index=idx),
        pvalues=pd.Series(np.asarray(fit.pvalues), index=idx),
        rsquared=float(fit.rsquared),
        rsquared_adj=float(fit.rsquared_adj),
        nobs=int(fit.nobs),
        lags=nlags,
        periods_per_year=ppy,
        resid=pd.Series(np.asarray(fit.resid), index=joined.index),
        start=pd.Timestamp(joined.index[0]),
        end=pd.Timestamp(joined.index[-1]),
    )


def compare_models(
    returns: pd.Series,
    factors: pd.DataFrame,
    models: Sequence[str] = ("capm", "ff3", "ff5"),
    **kwargs: Any,
) -> pd.DataFrame:
    """One row per model: alpha, t-stat, betas, R2 (columns are the union of factors)."""
    rows = []
    for m in models:
        available = [f for f in MODELS[m] if f in factors.columns]
        if len(available) != len(MODELS[m]):
            continue
        rows.append(factor_regression(returns, factors, m, **kwargs).to_series())
    if not rows:
        raise ValueError("no model could be estimated with the supplied factor columns")
    return pd.DataFrame(rows).set_index("model")
