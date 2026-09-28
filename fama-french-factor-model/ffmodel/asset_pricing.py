"""Cross-sectional asset pricing tests: Gibbons-Ross-Shanken and Fama-MacBeth."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.regression.rolling import RollingOLS

from .regression import infer_periods_per_year, newey_west_lags


def _panel(excess_returns: pd.DataFrame, factors: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.concat({"R": excess_returns, "F": factors}, axis=1, join="inner")
    return data["R"], data["F"]


@dataclass
class GRSResult:
    """Gibbons-Ross-Shanken test that every intercept is zero, plus Fama-French (2015) summaries."""

    statistic: float
    pvalue: float
    df_num: int
    df_den: int
    alphas: pd.Series
    alpha_tstats: pd.Series
    rsquared: pd.Series
    mean_abs_alpha: float
    rel_mean_abs_alpha: float
    sharpe_alpha: float
    nobs: int
    periods_per_year: int

    def summary(self) -> pd.Series:
        return pd.Series(
            {
                "GRS": self.statistic,
                "p-value": self.pvalue,
                "A|alpha| (ann.)": self.mean_abs_alpha * self.periods_per_year,
                "A|alpha| / A|r|": self.rel_mean_abs_alpha,
                "SR(alpha) (ann.)": self.sharpe_alpha * np.sqrt(self.periods_per_year),
                "avg R2": float(self.rsquared.mean()),
                "T": self.nobs,
                "N": len(self.alphas),
            }
        )


def grs_test(excess_returns: pd.DataFrame, factors: pd.DataFrame, periods_per_year: Optional[int] = None) -> GRSResult:
    """Gibbons, Ross and Shanken (1989) test that all N time-series alphas are zero.

    Uses the finite-sample form in Campbell, Lo and MacKinlay (1997, eq. 6.3.28),
    exact under normal errors: (T-N-L)/N * a'S^-1 a / (1 + mu'O^-1 mu) ~ F(N, T-N-L),
    with S and O the MLE residual and factor covariance matrices. Rows with any
    missing value are dropped. Also reports the Fama-French (2015) summary
    statistics A|a| and A|a|/A|r|, where r is each portfolio's average excess
    return minus the cross-sectional average of those averages.
    """
    R, F = _panel(excess_returns, factors)
    keep = R.notna().all(axis=1) & F.notna().all(axis=1)
    R, F = R[keep], F[keep]
    T, N = R.shape
    L = F.shape[1]
    if T - N - L < 1:
        raise ValueError(f"GRS needs T > N + L; got T={T}, N={N}, L={L}")

    r, f = R.to_numpy(float), F.to_numpy(float)
    X = np.column_stack([np.ones(T), f])
    B, *_ = np.linalg.lstsq(X, r, rcond=None)
    alpha = B[0]
    E = r - X @ B
    sigma = E.T @ E / T
    mu = f.mean(axis=0)
    omega = (f - mu).T @ (f - mu) / T

    quad_alpha = float(alpha @ np.linalg.solve(sigma, alpha))
    statistic = (T - N - L) / N * quad_alpha / (1.0 + float(mu @ np.linalg.solve(omega, mu)))

    se_alpha = np.sqrt(np.diag(E.T @ E) / (T - L - 1) * np.linalg.inv(X.T @ X)[0, 0])
    rbar = r.mean(axis=0)
    rsq = 1 - (E**2).sum(axis=0) / ((r - rbar) ** 2).sum(axis=0)

    return GRSResult(
        statistic=statistic,
        pvalue=float(stats.f.sf(statistic, N, T - N - L)),
        df_num=N,
        df_den=T - N - L,
        alphas=pd.Series(alpha, index=R.columns),
        alpha_tstats=pd.Series(alpha / se_alpha, index=R.columns),
        rsquared=pd.Series(rsq, index=R.columns),
        mean_abs_alpha=float(np.abs(alpha).mean()),
        rel_mean_abs_alpha=float(np.abs(alpha).mean() / np.abs(rbar - rbar.mean()).mean()),
        sharpe_alpha=float(np.sqrt(quad_alpha)),
        nobs=T,
        periods_per_year=periods_per_year or infer_periods_per_year(R.index),
    )


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth factor risk premia with plain, Shanken and optional Newey-West errors."""

    lambdas: pd.Series
    se: pd.Series
    tstats: pd.Series
    se_shanken: Optional[pd.Series]
    tstats_shanken: Optional[pd.Series]
    factor_means: pd.Series
    betas: pd.DataFrame
    pricing_errors: pd.Series
    cs_rsquared: float
    nobs: int
    periods_per_year: int
    lambda_series: pd.DataFrame = field(repr=False)

    def table(self) -> pd.DataFrame:
        ppy = self.periods_per_year
        out = pd.DataFrame({"lambda (ann.)": self.lambdas * ppy, "t (FM)": self.tstats})
        if self.tstats_shanken is not None:
            out["t (Shanken)"] = self.tstats_shanken
        out["factor mean (ann.)"] = (self.factor_means * ppy).reindex(out.index)
        return out


def fama_macbeth(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    intercept: bool = True,
    beta_window: Optional[int] = None,
    lags: Optional[int] = 0,
    periods_per_year: Optional[int] = None,
) -> FamaMacBethResult:
    """Two-pass Fama-MacBeth (1973) estimates of factor risk premia.

    Pass 1 estimates each asset's betas: over the full sample by default, or,
    with ``beta_window``, from the ``beta_window`` periods before each date
    (using only data available at the time). Pass 2 regresses each period's
    cross-section of excess returns on those betas; the premia are the time
    averages of the period-by-period slopes.

    ``lags=0`` gives the classic Fama-MacBeth standard errors; ``lags=None``
    or a positive integer uses Newey-West on the slope series. Shanken (1992)
    errors-in-variables corrected standard errors are reported for
    full-sample betas.
    """
    if lags is not None and lags < 0:
        raise ValueError("lags must be >= 0 (or None for the automatic Newey-West rule)")
    R, F = _panel(excess_returns, factors)
    F = F.dropna()
    R = R.loc[F.index]
    assets, names = list(R.columns), list(F.columns)
    r, f = R.to_numpy(float), F.to_numpy(float)
    T, N = r.shape
    L = len(names)

    full_betas = _full_sample_betas(r, f)
    if beta_window is None:
        B = np.broadcast_to(full_betas, (T, N, L))
    else:
        B = _rolling_betas(R, F, beta_window)

    k = L + int(intercept)
    lam = np.full((T, k), np.nan)
    resid = np.full((T, N), np.nan)
    for t in range(T):
        ok = np.isfinite(r[t]) & np.isfinite(B[t]).all(axis=1)
        if ok.sum() <= k:
            continue
        Z = B[t][ok]
        if intercept:
            Z = np.column_stack([np.ones(len(Z)), Z])
        coef, *_ = np.linalg.lstsq(Z, r[t, ok], rcond=None)
        lam[t] = coef
        resid[t, ok] = r[t, ok] - Z @ coef

    columns = (["const"] if intercept else []) + names
    lam_df = pd.DataFrame(lam, index=R.index, columns=columns).dropna()
    n = len(lam_df)
    if n < 2:
        raise ValueError("not enough periods with a usable cross-section")
    lambdas = lam_df.mean()
    if lags == 0:
        var = lam_df.var(ddof=1) / n
    else:
        nw = newey_west_lags(n) if lags is None else lags
        var = pd.Series({c: _hac_variance_of_mean(lam_df[c].to_numpy(), nw) for c in columns})
    se = np.sqrt(var)

    se_shanken = t_shanken = None
    if beta_window is None:
        sigma_f = np.atleast_2d(np.cov(F.loc[lam_df.index].to_numpy(float), rowvar=False))
        lf = lambdas[names].to_numpy()
        c = float(lf @ np.linalg.solve(sigma_f, lf))
        extra = pd.Series(0.0, index=columns)
        extra[names] = np.diag(sigma_f) / n
        se_shanken = np.sqrt((1 + c) * var + extra)
        t_shanken = lambdas / se_shanken

    used = np.isfinite(lam).all(axis=1)
    pricing_errors = pd.Series(np.nanmean(resid[used], axis=0), index=assets)
    rbar = pd.Series(np.nanmean(r[used], axis=0), index=assets)

    return FamaMacBethResult(
        lambdas=lambdas,
        se=se,
        tstats=lambdas / se,
        se_shanken=se_shanken,
        tstats_shanken=t_shanken,
        factor_means=F.loc[lam_df.index].mean(),
        betas=pd.DataFrame(full_betas, index=assets, columns=names),
        pricing_errors=pricing_errors,
        cs_rsquared=float(1 - pricing_errors.var() / rbar.var()),
        nobs=n,
        periods_per_year=periods_per_year or infer_periods_per_year(R.index),
        lambda_series=lam_df,
    )


def _full_sample_betas(r: np.ndarray, f: np.ndarray) -> np.ndarray:
    T, N = r.shape
    X = np.column_stack([np.ones(T), f])
    betas = np.full((N, f.shape[1]), np.nan)
    for j in range(N):
        ok = np.isfinite(r[:, j])
        if ok.sum() > X.shape[1] + 1:
            coef, *_ = np.linalg.lstsq(X[ok], r[ok, j], rcond=None)
            betas[j] = coef[1:]
    return betas


def _rolling_betas(R: pd.DataFrame, F: pd.DataFrame, window: int) -> np.ndarray:
    """Betas at date t estimated from the ``window`` periods ending at t-1."""
    X = sm.add_constant(F, has_constant="add")
    out = np.full((len(R), R.shape[1], F.shape[1]), np.nan)
    for j, asset in enumerate(R.columns):
        params = RollingOLS(R[asset], X, window=window, missing="skip").fit().params
        out[:, j, :] = params[list(F.columns)].shift(1).to_numpy()
    return out


def _hac_variance_of_mean(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett kernel) variance of the sample mean of ``x``."""
    x = x - x.mean()
    T = len(x)
    s = x @ x / T
    for j in range(1, lags + 1):
        s += 2 * (1 - j / (lags + 1)) * (x[j:] @ x[:-j]) / T
    return s / T
