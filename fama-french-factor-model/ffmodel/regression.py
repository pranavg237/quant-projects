"""Time-series factor regressions: r - rf = alpha + beta'f + e."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
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
    """Newey-West (1994) rule of thumb: floor(4 * (T/100)^(2/9)).

    The lag grows very slowly with the sample: 3 lags for T = 28 to 99, 4 for 100 to 272,
    5 for 273 to 620, 6 for 621 to 1,240. That suits monthly returns, which have little
    autocorrelation: a few lags guard against it (and against conditional
    heteroskedasticity) without making the variance estimate itself noisy.
    """
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
    def hac_lags(self) -> Optional[int]:
        """Newey-West lags used for the standard errors (None unless ``cov_type`` is "hac")."""
        return int(self.raw.cov_kwds["maxlags"]) if self.cov_type == "hac" else None

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


def _prepare(returns: pd.Series, factors: pd.DataFrame, factor_names: Sequence[str], excess: bool) -> pd.DataFrame:
    missing = [f for f in factor_names if f not in factors.columns]
    if missing:
        raise KeyError(f"factor data is missing columns {missing}")
    y = returns.astype(float)
    if not excess:
        if "RF" not in factors.columns:
            raise ValueError("factors has no 'RF' column; pass excess=True if returns are already excess returns")
        y = y - factors["RF"]
    return pd.concat([y.rename("excess_return"), factors[list(factor_names)]], axis=1, join="inner").dropna()


def _design(data: pd.DataFrame, factor_names: Sequence[str]) -> pd.DataFrame:
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
    if cov not in COV_TYPES:
        raise ValueError(f"cov must be one of {COV_TYPES}")
    raw = _fit_ols(y, X, cov, lags)

    return RegressionResult(
        name=str(name or returns.name or "asset"),
        model=spec.name,
        factors=list(spec.factors),
        raw=raw,
        periods_per_year=periods_per_year or infer_periods_per_year(data.index),
        cov_type=cov,
    )


def _fit_ols(y: pd.Series, X: pd.DataFrame, cov: str, lags: Optional[int]):
    """OLS with the chosen covariance.

    "hac" is statsmodels' Newey-West estimator: Bartlett weights 1 - l/(L+1) for
    l = 1..L, no small-sample degrees-of-freedom correction, and p-values from the
    normal distribution. L defaults to ``newey_west_lags(T)``. "hc3" (MacKinnon-White
    jackknife errors) is meant for short windows, where Newey-West is too narrow.
    """
    if cov == "hac":
        maxlags = newey_west_lags(len(y)) if lags is None else lags
        return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    if cov == "robust":
        return sm.OLS(y, X).fit(cov_type="HC1")
    if cov == "hc3":
        return sm.OLS(y, X).fit(cov_type="HC3")
    if cov == "ols":
        return sm.OLS(y, X).fit()
    raise ValueError(f"cov must be one of {COV_TYPES}")


def compare_standard_errors(result: RegressionResult, extra_lags: Sequence[int] = (12,),
                            level: float = 0.05) -> pd.DataFrame:
    """t-statistics of the same point estimates under classical OLS, White (HC1) and Newey-West errors.

    The coefficients (``result.params``) do not change, only their standard errors. Newey-West is shown with
    the default lag rule and with each lag in ``extra_lags`` as a sensitivity check. The
    last column says whether significance at ``level`` differs between OLS and the default
    Newey-West errors.
    """
    y = pd.Series(result.raw.model.endog, index=result.resid.index)
    X = pd.DataFrame(result.raw.model.exog, index=y.index, columns=result.params.index)
    lags = newey_west_lags(len(y))
    fits = {"OLS": _fit_ols(y, X, "ols", None), "White HC1": _fit_ols(y, X, "robust", None),
            f"NW, {lags} lags": _fit_ols(y, X, "hac", lags)}
    for extra in extra_lags:
        if extra != lags:
            fits[f"NW, {extra} lags"] = _fit_ols(y, X, "hac", extra)
    out = pd.DataFrame(index=result.params.index)
    for label, fit in fits.items():
        out[f"t ({label})"] = fit.tvalues
    ols_p, nw_p = fits["OLS"].pvalues, fits[f"NW, {lags} lags"].pvalues
    out["p (OLS)"] = ols_p
    out[f"p (NW, {lags} lags)"] = nw_p
    out["significance changes"] = np.where((ols_p < level) != (nw_p < level), "yes", "")
    return out


def holm_adjust(pvalues: pd.Series) -> pd.Series:
    """Holm (1979) step-down adjusted p-values: family-wise error control, valid under any dependence."""
    p = pvalues.astype(float)
    order = p.sort_values().index
    m = len(p)
    adjusted = (p[order] * (m - np.arange(m))).cummax().clip(upper=1.0)
    return adjusted.reindex(p.index)


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
    se: Optional[str] = None,
    lags: Optional[int] = None,
) -> pd.DataFrame:
    """Alpha, betas and R2 re-estimated over a rolling window of ``window`` periods.

    Each row is dated at the last period of its window. With ``se`` ("hc3", "hac",
    "robust" or "ols") the standard error of every coefficient in every window is added
    as a column "se(<name>)". Use "hc3" for short windows: in a simulation with 36-month
    windows of real FF5 factor data and six coefficients, nominal 95% Newey-West intervals
    (3 lags) missed the true beta 13-14% of the time, HC3 intervals 4-5%
    (scripts/window_se_simulation.py). These are
    pointwise errors for one window at a time; the windows overlap, so neighbouring
    estimates are strongly dependent.
    """
    spec = get_model(model)
    data = _prepare(returns, factors, spec.factors, excess)
    if len(data) < window:
        raise ValueError(f"{len(data)} observations is fewer than the rolling window of {window}")
    y, X = data["excess_return"], _design(data, spec.factors)
    fitted = RollingOLS(y, X, window=window).fit()
    out = fitted.params.copy()
    out["R2"] = fitted.rsquared
    out = out.dropna(how="all")
    if se is not None:
        se = se.lower()
        errors = pd.DataFrame(
            [_fit_ols(y.iloc[i - window:i], X.iloc[i - window:i], se, lags).bse.to_numpy()
             for i in range(window, len(y) + 1)],
            index=y.index[window - 1:], columns=[f"se({c})" for c in X.columns],
        )
        out = out.join(errors)
    return out


def stability_test(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff5",
    block: int = 36,
    excess: bool = False,
    se: str = "hc3",
    lags: Optional[int] = None,
) -> pd.DataFrame:
    """Do the coefficients differ between non-overlapping blocks of ``block`` periods?

    Rolling estimates always wiggle, so wiggles alone are not evidence that exposures
    changed. This splits the sample into consecutive non-overlapping blocks (aligned to
    the end of the sample; the oldest leftover periods are dropped), fits each block on
    its own and, for every coefficient, runs a Wald test that all blocks share one value:
    chi2 = sum_k (b_k - b_pooled)^2 / se_k^2, with b_pooled the inverse-variance weighted
    mean. It is chi-squared with (blocks - 1) degrees of freedom if the block estimates
    are independent. Standard errors within each block follow ``se`` and ``lags``.
    The default, HC3, matters: with 36-period blocks, Newey-West errors are too small
    and the test rejected a truly constant beta 23-35% of the time at the 5% level in
    simulation, against 2-6% with HC3 and 6-9% with classical errors
    (scripts/window_se_simulation.py).
    """
    spec = get_model(model)
    data = _prepare(returns, factors, spec.factors, excess)
    n_blocks = len(data) // block
    if n_blocks < 2:
        raise ValueError(f"{len(data)} observations give fewer than two blocks of {block}")
    data = data.iloc[len(data) - n_blocks * block:]
    y, X = data["excess_return"], _design(data, spec.factors)
    fits = [_fit_ols(y.iloc[k * block:(k + 1) * block], X.iloc[k * block:(k + 1) * block], se.lower(), lags)
            for k in range(n_blocks)]
    est = pd.DataFrame([f.params for f in fits])
    weights = 1.0 / pd.DataFrame([f.bse ** 2 for f in fits])
    pooled = (weights * est).sum() / weights.sum()
    chi2 = (weights * (est - pooled) ** 2).sum()
    table = pd.DataFrame({
        "lowest block": est.min(),
        "highest block": est.max(),
        "chi2": chi2,
        "p-value": stats.chi2.sf(chi2, n_blocks - 1),
    })
    table.attrs.update(blocks=n_blocks, block=block, start=data.index[0], end=data.index[-1])
    return table
