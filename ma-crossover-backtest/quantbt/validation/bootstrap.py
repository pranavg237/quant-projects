"""Bootstrap and overfitting diagnostics for a return series or a grid of them.

* :func:`stationary_bootstrap` resamples the realised returns in random-length blocks
  (Politis & Romano 1994) to get a distribution of Sharpe, CAGR and drawdown that
  respects short-range autocorrelation.
* :func:`probabilistic_sharpe_ratio` and :func:`deflated_sharpe_ratio` (Bailey &
  Lopez de Prado 2012, 2014) ask whether the observed Sharpe is distinguishable from
  zero given the sample length, non-normality and, for the deflated version, how many
  configurations were tried to find it.
* :func:`pbo_cscv` is the combinatorially symmetric cross-validation estimate of the
  probability of backtest overfitting (Bailey, Borwein, Lopez de Prado & Zhu 2014):
  how often the in-sample best configuration is below median out of sample.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from quantbt import metrics


def _moments(returns: pd.Series) -> tuple[int, float, float]:
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 4:
        return n, 0.0, 3.0
    return n, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))


def sharpe_std_error(returns: pd.Series, sr_period: float | None = None) -> float:
    """Standard error of the *per-period* Sharpe ratio, allowing for skew and kurtosis.

    Mertens (2002); reduces to ``sqrt((1 + sr^2 / 2) / (n - 1))`` for normal returns.
    """
    n, skew, kurt = _moments(returns)
    if sr_period is None:
        r = returns.dropna()
        sr_period = float(r.mean() / r.std(ddof=1)) if r.std(ddof=1) > 0 else 0.0
    var = (1.0 - skew * sr_period + (kurt - 1.0) / 4.0 * sr_period**2) / (n - 1)
    return float(np.sqrt(max(var, 0.0)))


def psr_from_stats(
    sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float
) -> float:
    """The PSR formula on summary statistics, all Sharpe ratios *per period* (not annual).

    ``PSR = Phi((sr - sr_benchmark) * sqrt(n - 1) / sqrt(1 - skew*sr + (kurt - 1)/4 * sr^2))``

    ``kurtosis`` is the raw (non-excess) kurtosis, 3 for a normal distribution. This is
    equation (2) of Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", with
    ``sr_benchmark`` = 0 for the PSR and = the expected maximum Sharpe for the DSR.
    """
    if n_obs < 2:
        return float("nan")
    var = (1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr**2) / (n_obs - 1)
    if var <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_benchmark) / np.sqrt(var)))


def probabilistic_sharpe_ratio(returns: pd.Series, benchmark_sharpe_annual: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given the sample: the PSR of Bailey & Lopez de Prado."""
    r = returns.dropna()
    ppy = metrics.periods_per_year(r.index)
    sd = float(r.std(ddof=1))
    if sd == 0 or len(r) < 4:
        return float("nan")
    sr = float(r.mean() / sd)
    sr_star = benchmark_sharpe_annual / np.sqrt(ppy)
    n, skew, kurt = _moments(r)
    return psr_from_stats(sr, sr_star, n, skew, kurt)


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum of ``n_trials`` independent Sharpe draws with variance ``var_sharpe``.

    Uses the extreme-value approximation from Bailey & Lopez de Prado (2014), eq. (1):
    ``sqrt(V) * ((1 - g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))`` with ``g`` the
    Euler-Mascheroni constant. It assumes the true Sharpe of every trial is zero, so the
    result is the Sharpe the *best* of ``N`` pure-noise configurations would show.
    """
    if n_trials <= 1:
        return 0.0
    euler = 0.5772156649015329
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sharpe) * ((1 - euler) * z1 + euler * z2))


def deflated_sharpe_from_stats(
    sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
    n_trials: int,
    var_trials_sharpe: float,
) -> float:
    """The DSR on summary statistics, every Sharpe input *per period* (not annualised).

    ``var_trials_sharpe`` is the variance of the per-period Sharpe ratios across the
    ``n_trials`` configurations tried. The DSR is the PSR measured against the expected
    maximum of ``n_trials`` noise Sharpes instead of against zero (Bailey & Lopez de
    Prado 2014, eq. 2). With their worked example (annual SR 2.5, daily, T = 1250,
    N = 100, annual V = 0.5, skew -3, kurtosis 10) this returns 0.9004.
    """
    sr_star = expected_max_sharpe(n_trials, var_trials_sharpe)
    return psr_from_stats(sr, sr_star, n_obs, skew, kurtosis)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    var_trials_sharpe_annual: float | None = None,
    trials_sharpe_annual: np.ndarray | pd.Series | None = None,
) -> float:
    """PSR against the Sharpe one would expect from the *best of* ``n_trials`` noise strategies.

    Pass either the variance of the annualised Sharpe ratios across trials or the
    trials' Sharpe ratios themselves. A DSR below 0.95 means the observed Sharpe is not
    distinguishable from what the search would have found in pure noise.
    """
    r = returns.dropna()
    ppy = metrics.periods_per_year(r.index)
    if trials_sharpe_annual is not None:
        arr = np.asarray(trials_sharpe_annual, dtype=float)
        arr = arr[np.isfinite(arr)]
        var_trials_sharpe_annual = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
        n_trials = max(n_trials, len(arr))
    if var_trials_sharpe_annual is None:
        raise ValueError("need var_trials_sharpe_annual or trials_sharpe_annual")
    # An annual Sharpe is sqrt(ppy) times the per-period one, so its variance is ppy times.
    var_period = var_trials_sharpe_annual / ppy
    sd = float(r.std(ddof=1))
    if sd == 0 or len(r) < 4:
        return float("nan")
    n, skew, kurt = _moments(r)
    return deflated_sharpe_from_stats(float(r.mean() / sd), n, skew, kurt, n_trials, var_period)


def min_track_record_length(
    returns: pd.Series, benchmark_sharpe_annual: float = 0.0, confidence: float = 0.95
) -> float:
    """Periods needed for PSR to reach ``confidence`` at the observed Sharpe, skew and kurtosis."""
    r = returns.dropna()
    ppy = metrics.periods_per_year(r.index)
    sd = float(r.std(ddof=1))
    if sd == 0 or len(r) < 4:
        return float("nan")
    sr = float(r.mean() / sd)
    sr_star = benchmark_sharpe_annual / np.sqrt(ppy)
    if sr <= sr_star:
        return float("inf")
    _, skew, kurt = _moments(r)
    z = stats.norm.ppf(confidence)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_star)) ** 2)


@dataclass(frozen=True)
class BootstrapResult:
    """Resampled performance statistics and their confidence intervals."""

    samples: pd.DataFrame  # one row per resample: sharpe, cagr, max_drawdown
    observed: pd.Series
    block_len: float

    def ci(self, column: str = "sharpe", level: float = 0.95) -> tuple[float, float]:
        values = self.samples[column].dropna()
        if values.empty:  # e.g. a constant return series, where Sharpe is undefined
            return float("nan"), float("nan")
        lo, hi = np.quantile(values, [(1 - level) / 2, 1 - (1 - level) / 2])
        return float(lo), float(hi)

    def p_value(self, column: str = "sharpe", threshold: float = 0.0) -> float:
        """Fraction of resamples at or below ``threshold`` (one-sided, e.g. P(Sharpe <= 0))."""
        values = self.samples[column].dropna()
        return float((values <= threshold).mean()) if not values.empty else float("nan")


def stationary_bootstrap(
    returns: pd.Series,
    n_samples: int = 1000,
    block_len: float | None = None,
    seed: int | None = 0,
) -> BootstrapResult:
    """Stationary (random block length) bootstrap of a return series.

    ``block_len`` is the *expected* block length in periods; the default is the Politis
    & White rule-of-thumb ``n ** (1/3)``. Each resample has the same length as the input.
    """
    r = returns.dropna().to_numpy()
    n = len(r)
    if n < 10:
        raise ValueError("need at least 10 observations")
    rng = np.random.default_rng(seed)
    if block_len is None:
        block_len = max(1.0, n ** (1.0 / 3.0))
    p = 1.0 / block_len
    ppy = metrics.periods_per_year(returns.dropna().index)
    idx_template = returns.dropna().index
    rows = []
    for _ in range(n_samples):
        # stationary bootstrap: with prob p start a new block at a random point, else continue
        starts = rng.integers(0, n, size=n)
        new_block = rng.random(n) < p
        new_block[0] = True
        pos = np.empty(n, dtype=int)
        for i in range(n):
            pos[i] = starts[i] if new_block[i] else (pos[i - 1] + 1) % n
        sample = pd.Series(r[pos], index=idx_template)
        eq = metrics.equity_curve(sample)
        rows.append(
            {
                "sharpe": metrics.sharpe(sample, ppy=ppy),
                "cagr": metrics.annualized_return(sample, ppy),
                "max_drawdown": metrics.max_drawdown(eq),
            }
        )
    obs = pd.Series(
        {
            "sharpe": metrics.sharpe(returns, ppy=ppy),
            "cagr": metrics.annualized_return(returns, ppy),
            "max_drawdown": metrics.max_drawdown(metrics.equity_curve(returns)),
        }
    )
    return BootstrapResult(samples=pd.DataFrame(rows), observed=obs, block_len=float(block_len))


def bootstrap_metrics(
    returns: pd.Series, n_samples: int = 1000, seed: int | None = 0
) -> pd.DataFrame:
    """Quantile table (5/25/50/75/95%) of bootstrapped Sharpe, CAGR and max drawdown."""
    res = stationary_bootstrap(returns, n_samples=n_samples, seed=seed)
    q = res.samples.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).T
    q.columns = [f"q{int(c * 100):02d}" for c in q.columns]
    q["observed"] = res.observed
    return q


def pbo_cscv(
    returns_matrix: pd.DataFrame, n_blocks: int = 16, metric: str = "sharpe"
) -> dict[str, float]:
    """Probability of backtest overfitting via combinatorially symmetric cross-validation.

    ``returns_matrix`` is T x N (one column per configuration). The rows are cut into
    ``n_blocks`` contiguous blocks; for every choice of half the blocks as "in-sample",
    the best in-sample configuration is located and its out-of-sample rank recorded.
    PBO is the fraction of splits where that rank is below the median.

    Returns PBO, the mean OOS rank-logit, and the slope of OOS-vs-IS performance
    (a negative slope means better in-sample scores predict *worse* out-of-sample ones).
    """
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks must be even and >= 4")
    m = returns_matrix.dropna(how="all").fillna(0.0)
    t, n = m.shape
    if n < 2:
        raise ValueError("need at least two configurations")
    if t < n_blocks * 2:
        raise ValueError("too few rows for the requested number of blocks")
    blocks = np.array_split(np.arange(t), n_blocks)
    values = m.to_numpy()
    ppy = metrics.periods_per_year(m.index)

    def score(rows: np.ndarray) -> np.ndarray:
        sub = values[rows]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(sd > 0, mu / sd * np.sqrt(ppy), -np.inf)
        return np.asarray(out, dtype=float)

    logits: list[float] = []
    pairs: list[tuple[float, float]] = []
    half = n_blocks // 2
    for train_blocks in itertools.combinations(range(n_blocks), half):
        train_rows = np.concatenate([blocks[b] for b in train_blocks])
        test_rows = np.concatenate([blocks[b] for b in range(n_blocks) if b not in train_blocks])
        is_scores = score(train_rows)
        oos_scores = score(test_rows)
        best = int(np.argmax(is_scores))
        rank = (oos_scores < oos_scores[best]).sum() + 0.5 * (
            (oos_scores == oos_scores[best]).sum() - 1
        )
        w = (rank + 0.5) / n  # relative rank in (0, 1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(float(np.log(w / (1 - w))))
        pairs.append((float(is_scores[best]), float(oos_scores[best])))
    logit_arr = np.array(logits)
    is_arr = np.array([p[0] for p in pairs])
    oos_arr = np.array([p[1] for p in pairs])
    finite = np.isfinite(is_arr) & np.isfinite(oos_arr)
    slope = (
        float(np.polyfit(is_arr[finite], oos_arr[finite], 1)[0])
        if finite.sum() > 2
        else float("nan")
    )
    return {
        "pbo": float((logit_arr <= 0).mean()),
        "mean_logit": float(logit_arr.mean()),
        "n_splits": float(len(logits)),
        "is_oos_slope": slope,
        "oos_below_zero": float((oos_arr[finite] < 0).mean()) if finite.any() else float("nan"),
    }


@dataclass(frozen=True)
class OverfitReport:
    """Probabilistic and deflated Sharpe ratios, with warning flags."""

    sharpe: float
    sharpe_se_annual: float
    psr: float
    dsr: float | None
    n_trials: int
    min_track_record_periods: float
    bootstrap_ci_95: tuple[float, float]
    bootstrap_p_sharpe_le_0: float
    pbo: float | None
    flags: list[str]

    def to_series(self) -> pd.Series:
        return pd.Series(
            {
                "sharpe": self.sharpe,
                "sharpe_se_annual": self.sharpe_se_annual,
                "psr": self.psr,
                "dsr": self.dsr if self.dsr is not None else np.nan,
                "n_trials": self.n_trials,
                "min_track_record_periods": self.min_track_record_periods,
                "bootstrap_ci_lo": self.bootstrap_ci_95[0],
                "bootstrap_ci_hi": self.bootstrap_ci_95[1],
                "bootstrap_p_sharpe_le_0": self.bootstrap_p_sharpe_le_0,
                "pbo": self.pbo if self.pbo is not None else np.nan,
                "flags": "; ".join(self.flags) if self.flags else "none",
            }
        )


def excess_returns(returns: pd.Series, rf: pd.Series | float | None) -> pd.Series:
    """``returns`` net of the risk-free rate, so every Sharpe-like statistic is excess.

    Mixing the two is a real reporting bug rather than a nicety: a strategy that sits in
    cash most of the time has a high *raw* Sharpe, because cash has almost no volatility,
    and a negative *excess* one.
    """
    if rf is None:
        return returns
    if isinstance(rf, pd.Series):
        return returns - rf.reindex(returns.index).fillna(0.0)
    ppy = metrics.periods_per_year(returns.index)
    per_period = (1.0 + float(rf)) ** (1.0 / ppy) - 1.0
    out: pd.Series = returns - per_period
    return out


def overfit_report(
    returns: pd.Series,
    *,
    rf: pd.Series | float | None = None,
    trials_sharpe_annual: np.ndarray | pd.Series | None = None,
    n_trials: int | None = None,
    grid_returns: pd.DataFrame | None = None,
    n_samples: int = 1000,
    seed: int | None = 0,
) -> OverfitReport:
    """One-stop overfitting check, computed on returns in excess of ``rf``.

    Supply the grid's Sharpe ratios (and, optionally, its return matrix) so the deflated
    Sharpe ratio and the probability of backtest overfitting account for the search.
    ``grid_returns`` is put in excess terms with the same rate.
    """
    r = excess_returns(returns, rf).dropna()
    ppy = metrics.periods_per_year(r.index)
    sr = metrics.sharpe(r, ppy=ppy)
    se = sharpe_std_error(r) * np.sqrt(ppy)
    psr = probabilistic_sharpe_ratio(r)
    trials = (
        n_trials
        if n_trials is not None
        else (len(trials_sharpe_annual) if trials_sharpe_annual is not None else 1)
    )
    dsr = None
    if trials_sharpe_annual is not None and trials > 1:
        dsr = deflated_sharpe_ratio(r, trials, trials_sharpe_annual=trials_sharpe_annual)
    boot = stationary_bootstrap(r, n_samples=n_samples, seed=seed)
    ci = boot.ci("sharpe")
    p0 = boot.p_value("sharpe", 0.0)
    pbo = None
    if grid_returns is not None and grid_returns.shape[1] > 1:
        # Put the whole grid in excess terms with the same rate before ranking.
        shift = grid_returns.iloc[:, 0] - excess_returns(grid_returns.iloc[:, 0], rf)
        pbo = pbo_cscv(grid_returns.sub(shift, axis=0))["pbo"]
    flags: list[str] = []
    if not np.isfinite(sr) or sr <= 0:
        flags.append("non-positive Sharpe")
    if np.isfinite(psr) and psr < 0.95:
        flags.append(f"PSR {psr:.2f} < 0.95: Sharpe not distinguishable from zero")
    if dsr is not None and dsr < 0.95:
        flags.append(f"DSR {dsr:.2f} < 0.95 after {trials} trials: likely selection effect")
    if np.isfinite(p0) and p0 > 0.05:
        flags.append(f"bootstrap P(Sharpe<=0) = {p0:.2f}")
    if pbo is not None and pbo > 0.5:
        flags.append(f"PBO {pbo:.2f} > 0.5: in-sample best is usually below median OOS")
    return OverfitReport(
        sharpe=float(sr),
        sharpe_se_annual=float(se),
        psr=float(psr),
        dsr=dsr,
        n_trials=int(trials),
        min_track_record_periods=min_track_record_length(r),
        bootstrap_ci_95=ci,
        bootstrap_p_sharpe_le_0=p0,
        pbo=pbo,
        flags=flags,
    )
