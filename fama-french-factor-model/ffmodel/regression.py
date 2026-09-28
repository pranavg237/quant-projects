"""Time-series factor regressions: r - rf = alpha + beta'f + e."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

from .models import ALL_FACTORS, get_model

COV_TYPES = ("hac", "robust", "ols")


def infer_periods_per_year(index: pd.DatetimeIndex) -> int:
    """Guess the sampling frequency from the median spacing of the dates."""
    if len(index) < 2:
        raise ValueError("need at least two dates to infer the frequency")
    days = float(np.median(np.diff(index.values) / np.timedelta64(1, "D")))
    if days <= 4:
        return 252
    if days <= 10:
        return 52
    if days <= 45:
        return 12
    if days <= 120:
        return 4
    return 1


def newey_west_lags(nobs: int) -> int:
    """Newey-West (1994) rule of thumb: floor(4 * (T/100)^(2/9))."""
    return int(4 * (nobs / 100.0) ** (2.0 / 9.0))


@dataclass
class RegressionResult:
    """A fitted factor regression. ``raw`` is the underlying statsmodels result."""

    name: str
    model: str
    factors: List[str]
    raw: object = field(repr=False)
    periods_per_year: int
    cov_type: str

    @property
    def params(self) -> pd.Series:
        return self.raw.params

    @property
    def bse(self) -> pd.Series:
        return self.raw.bse

    @property
    def tvalues(self) -> pd.Series:
        return self.raw.tvalues

    @property
    def pvalues(self) -> pd.Series:
        return self.raw.pvalues

    @property
    def alpha(self) -> float:
        return float(self.params["alpha"])

    @property
    def alpha_annual(self) -> float:
        return self.alpha * self.periods_per_year

    @property
    def betas(self) -> pd.Series:
        return self.params[self.factors]

    @property
    def rsquared(self) -> float:
        return float(self.raw.rsquared)

    @property
    def rsquared_adj(self) -> float:
        return float(self.raw.rsquared_adj)

    @property
    def nobs(self) -> int:
        return int(self.raw.nobs)

    @property
    def resid(self) -> pd.Series:
        return self.raw.resid

    @property
    def fitted(self) -> pd.Series:
        return self.raw.fittedvalues

    @property
    def resid_vol_annual(self) -> float:
        return float(np.sqrt(self.raw.mse_resid * self.periods_per_year))

    @property
    def information_ratio(self) -> float:
        return self.alpha_annual / self.resid_vol_annual

    @property
    def period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        return self.resid.index[0], self.resid.index[-1]

    def table(self, level: float = 0.95) -> pd.DataFrame:
        ci = self.raw.conf_int(alpha=1 - level)
        return pd.DataFrame(
            {
                "coef": self.params,
                "std err": self.bse,
                "t": self.tvalues,
                "p-value": self.pvalues,
                f"ci low ({level:.0%})": ci[0],
                f"ci high ({level:.0%})": ci[1],
            }
        )

    def summary_row(self) -> pd.Series:
        row = {"alpha (ann.)": self.alpha_annual, "t(alpha)": float(self.tvalues["alpha"])}
        row.update(self.betas.to_dict())
        row.update(
            {
                "R2": self.rsquared,
                "adj R2": self.rsquared_adj,
                "resid vol (ann.)": self.resid_vol_annual,
                "IR": self.information_ratio,
                "N": self.nobs,
            }
        )
        return pd.Series(row, name=self.name)

    def summary(self) -> str:
        start, end = self.period
        lines = [
            f"{self.name} | {get_model(self.model).label} | {start:%Y-%m-%d} to {end:%Y-%m-%d} | "
            f"N={self.nobs} | {self.cov_type.upper()} standard errors",
            self.table().to_string(float_format=lambda x: f"{x: .4f}"),
            f"alpha (annualized) {self.alpha_annual:.2%} | R2 {self.rsquared:.3f} | adj R2 {self.rsquared_adj:.3f} | "
            f"resid vol {self.resid_vol_annual:.2%} | information ratio {self.information_ratio:.2f}",
        ]
        return "\n".join(lines)


def _prepare(returns: pd.Series, factors: pd.DataFrame, factor_names, excess: bool) -> pd.DataFrame:
    missing = [f for f in factor_names if f not in factors.columns]
    if missing:
        raise KeyError(f"factor data is missing columns {missing}")
    y = returns.astype(float)
    if not excess:
        if "RF" not in factors.columns:
            raise ValueError("factors has no 'RF' column; pass excess=True if returns are already excess returns")
        y = y - factors["RF"]
    return pd.concat([y.rename("excess_return"), factors[list(factor_names)]], axis=1, join="inner").dropna()


def _design(data: pd.DataFrame, factor_names) -> pd.DataFrame:
    X = sm.add_constant(data[list(factor_names)], has_constant="add")
    return X.rename(columns={"const": "alpha"})


def fit_factor_model(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff5",
    excess: bool = False,
    cov: str = "hac",
    lags: Optional[int] = None,
    periods_per_year: Optional[int] = None,
    name: Optional[str] = None,
) -> RegressionResult:
    """Regress an asset's excess return on the factors of ``model``.

    ``returns`` are raw returns (RF is subtracted) unless ``excess=True``.
    ``cov`` is "hac" (Newey-West, default lag rule unless ``lags`` is given),
    "robust" (HC1) or "ols".
    """
    if lags is not None and lags < 0:
        raise ValueError("lags must be >= 0 (or None for the automatic Newey-West rule)")
    spec = get_model(model)
    data = _prepare(returns, factors, spec.factors, excess)
    if len(data) < len(spec.factors) + 3:
        raise ValueError(f"only {len(data)} overlapping observations for {name or returns.name}")
    y, X = data["excess_return"], _design(data, spec.factors)

    cov = cov.lower()
    if cov == "hac":
        raw = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(y)) if lags is None else lags})
    elif cov == "robust":
        raw = sm.OLS(y, X).fit(cov_type="HC1")
    elif cov == "ols":
        raw = sm.OLS(y, X).fit()
    else:
        raise ValueError(f"cov must be one of {COV_TYPES}")

    return RegressionResult(
        name=str(name or returns.name or "asset"),
        model=spec.name,
        factors=list(spec.factors),
        raw=raw,
        periods_per_year=periods_per_year or infer_periods_per_year(data.index),
        cov_type=cov,
    )


def fit_many(returns: pd.DataFrame, factors: pd.DataFrame, model: str = "ff5", **kwargs) -> Dict[str, RegressionResult]:
    """Fit ``model`` to every column of ``returns``; columns with too little data are skipped."""
    results = {}
    for column in returns.columns:
        try:
            results[column] = fit_factor_model(returns[column], factors, model=model, name=column, **kwargs)
        except ValueError as exc:
            warnings.warn(f"skipping {column}: {exc}")
    return results


def summarize(results: Mapping[str, RegressionResult]) -> pd.DataFrame:
    """One row per regression: annualized alpha, its t-stat, betas and fit statistics."""
    table = pd.DataFrame([r.summary_row() for r in results.values()], index=list(results))
    return _order_columns(table)


def _order_columns(table: pd.DataFrame) -> pd.DataFrame:
    head = [c for c in ("alpha (ann.)", "t(alpha)") if c in table]
    factor_cols = [f for f in ALL_FACTORS if f in table]
    tail = [c for c in table.columns if c not in head and c not in factor_cols]
    return table[head + factor_cols + tail]


def compare_models(
    returns: pd.Series,
    factor_sets: Mapping[str, pd.DataFrame],
    excess: bool = False,
    common_sample: bool = True,
    **kwargs,
) -> Tuple[pd.DataFrame, Dict[str, RegressionResult]]:
    """Fit several models to one asset. ``factor_sets`` maps model name -> factor data.

    With ``common_sample`` every model is fit on the same dates so the fit
    statistics are comparable (FF5 data starts in 1963, for instance).
    """
    if common_sample:
        index = returns.dropna().index
        for factors in factor_sets.values():
            index = index.intersection(factors.dropna().index)
        returns = returns.loc[index]
    results = {m: fit_factor_model(returns, f, model=m, excess=excess, **kwargs) for m, f in factor_sets.items()}
    table = pd.DataFrame([r.summary_row() for r in results.values()], index=list(results))
    table["BIC"] = [r.raw.bic for r in results.values()]
    return _order_columns(table), results


def rolling_regression(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff5",
    window: int = 36,
    excess: bool = False,
) -> pd.DataFrame:
    """Alpha, betas and R2 re-estimated over a rolling window of ``window`` periods."""
    spec = get_model(model)
    data = _prepare(returns, factors, spec.factors, excess)
    if len(data) < window:
        raise ValueError(f"{len(data)} observations is fewer than the rolling window of {window}")
    fitted = RollingOLS(data["excess_return"], _design(data, spec.factors), window=window).fit()
    out = fitted.params.copy()
    out["R2"] = fitted.rsquared
    return out.dropna(how="all")
