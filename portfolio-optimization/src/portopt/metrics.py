r"""Performance and risk metrics for a backtested portfolio.

Some conventions that are easy to get wrong and materially change the answer:

**Annualised return is geometric**, :math:`(\prod_t(1+r_t))^{P/T} - 1`, not the arithmetic
mean times :math:`P`. The two differ by roughly :math:`\sigma^2/2`, which for a 15%
volatility portfolio is 1.1% a year -- larger than most of the differences between the
strategies being compared here.

**Sharpe uses the excess return over the risk-free rate**, and the volatility of the excess
return. Reporting a "Sharpe" that is really return over volatility is fine when rates are
zero and badly misleading when they are 5%.

**Maximum drawdown is measured on the compounded equity curve**, peak to trough, and the
recovery date is reported alongside. A 30% drawdown that recovers in six months and one
that takes six years are not the same risk, and a single number cannot tell them apart.

**Turnover is one-way** -- the sum of absolute weight changes -- and is reported per year.
Doubling it for round-trip is a convention; both appear in the literature, so the choice is
stated rather than assumed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .backtest import BacktestResult
from .types import as_float, as_timestamp

__all__ = [
    "PerformanceMetrics",
    "drawdown_series",
    "evaluate",
    "max_drawdown",
    "sharpe_confidence_interval",
    "summarise",
]


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Fractional drawdown from the running peak of the compounded equity curve."""
    equity = (1.0 + returns).cumprod()
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """Worst peak-to-trough fall, with the peak and trough dates.

    Returns:
        ``(depth, peak_date, trough_date)`` with ``depth`` negative.
    """
    if returns.empty:
        return 0.0, None, None
    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    trough = drawdown.idxmin()
    peak = equity.loc[:trough].idxmax()
    return as_float(drawdown.min()), as_timestamp(peak), as_timestamp(trough)


def sharpe_confidence_interval(
    returns: pd.Series, periods_per_year: float, level: float = 0.95
) -> tuple[float, float]:
    r"""Confidence interval for an annualised Sharpe ratio.

    Uses Lo (2002): for IID returns the standard error of the *per-period* Sharpe is
    :math:`\sqrt{(1 + \hat S^2/2)/T}`, and annualising multiplies by
    :math:`\sqrt{P}`. Reported because Sharpe ratios estimated from 20 years of monthly
    data have standard errors around 0.2-0.25 -- so two strategies differing by 0.1 are
    indistinguishable, and a table of Sharpe ratios quoted to three decimals without error
    bars invites exactly the wrong conclusion.

    The IID assumption understates the error when returns are autocorrelated, so this is a
    lower bound on the true uncertainty.
    """
    clean = returns.dropna()
    n = len(clean)
    if n < 3 or clean.std(ddof=1) == 0:
        return float("nan"), float("nan")
    per_period = float(clean.mean() / clean.std(ddof=1))
    annual = per_period * np.sqrt(periods_per_year)
    standard_error = np.sqrt((1.0 + 0.5 * per_period**2) / n) * np.sqrt(periods_per_year)
    z = float(stats.norm.ppf(0.5 * (1.0 + level)))
    return float(annual - z * standard_error), float(annual + z * standard_error)


@dataclass(frozen=True)
class PerformanceMetrics:
    """Summary statistics for one backtested strategy.

    Attributes:
        strategy: Name.
        annual_return: Geometric annualised return, net of costs.
        annual_volatility: Annualised standard deviation of returns.
        sharpe: Annualised excess return over annualised volatility.
        sharpe_lower / sharpe_upper: 95% confidence bounds on the Sharpe (Lo, 2002).
        sortino: Like Sharpe but dividing by downside deviation only.
        max_drawdown: Worst peak-to-trough fall, negative.
        calmar: Annual return over the absolute maximum drawdown.
        annual_turnover: One-way turnover per year.
        annual_cost_drag: Return given up to transaction costs each year.
        gross_sharpe: Sharpe before costs, so the cost impact is visible.
        skew / excess_kurtosis: Shape of the return distribution. Portfolio returns are
            reliably left-skewed and fat-tailed, which is what makes Sharpe an incomplete
            summary.
        worst_period / best_period: Extremes.
        hit_rate: Fraction of periods with a positive return.
        mean_leverage / max_leverage: Gross exposure at rebalance, averaged and worst.
        is_ruined: Whether the portfolio ever lost 100% or more in a single period. When
            this is ``True`` every other ratio here is meaningless and the strategy should
            be read as "lost everything", not as its Sharpe.
        n_periods: Sample size.
    """

    strategy: str
    annual_return: float
    annual_volatility: float
    sharpe: float
    sharpe_lower: float
    sharpe_upper: float
    sortino: float
    max_drawdown: float
    calmar: float
    annual_turnover: float
    annual_cost_drag: float
    gross_sharpe: float
    skew: float
    excess_kurtosis: float
    worst_period: float
    best_period: float
    hit_rate: float
    mean_leverage: float
    max_leverage: float
    is_ruined: bool
    n_periods: int

    def as_dict(self) -> dict[str, float | str | int]:
        """Flat dict for building a DataFrame."""
        return asdict(self)


def evaluate(
    result: BacktestResult, risk_free: float = 0.0, level: float = 0.95
) -> PerformanceMetrics:
    """Compute :class:`PerformanceMetrics` for a backtest result.

    Args:
        result: A completed walk-forward run.
        risk_free: Annualised risk-free rate used for the Sharpe and Sortino ratios.
        level: Confidence level for the Sharpe interval.
    """
    returns = result.returns.dropna()
    periods = result.periods_per_year
    n = len(returns)
    if n == 0:  # pragma: no cover - a backtest always produces at least one period
        raise ValueError("cannot evaluate an empty backtest")

    total_growth = as_float((1.0 + returns).prod())
    # A portfolio that has been wiped out has no meaningful annualised return; reporting
    # -100% is the honest answer and everything downstream of it should be read as ruin.
    annual_return = float(total_growth ** (periods / n) - 1.0) if total_growth > 0 else -1.0
    annual_volatility = float(returns.std(ddof=1) * np.sqrt(periods))

    per_period_rf = (1.0 + risk_free) ** (1.0 / periods) - 1.0
    excess = returns - per_period_rf
    sharpe = (
        float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods))
        if excess.std(ddof=1) > 0
        else float("nan")
    )
    lower, upper = sharpe_confidence_interval(excess, periods, level)

    downside = excess[excess < 0]
    downside_deviation = (
        float(np.sqrt((downside**2).mean()) * np.sqrt(periods)) if len(downside) else 0.0
    )
    sortino = (
        float(excess.mean() * periods / downside_deviation)
        if downside_deviation > 0
        else float("nan")
    )

    depth, _, _ = max_drawdown(returns)
    calmar = float(annual_return / abs(depth)) if depth < 0 else float("nan")

    years = n / periods
    annual_turnover = float(result.turnover.sum() / years) if years > 0 else 0.0
    annual_cost_drag = float(result.costs.sum() / years) if years > 0 else 0.0

    gross = result.gross_returns.dropna() - per_period_rf
    gross_sharpe = (
        float(gross.mean() / gross.std(ddof=1) * np.sqrt(periods))
        if gross.std(ddof=1) > 0
        else float("nan")
    )

    return PerformanceMetrics(
        strategy=result.name,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        sharpe_lower=lower,
        sharpe_upper=upper,
        sortino=sortino,
        max_drawdown=depth,
        calmar=calmar,
        annual_turnover=annual_turnover,
        annual_cost_drag=annual_cost_drag,
        gross_sharpe=gross_sharpe,
        skew=float(stats.skew(returns.to_numpy(dtype=np.float64))),
        excess_kurtosis=float(stats.kurtosis(returns.to_numpy(dtype=np.float64))),
        worst_period=float(returns.min()),
        best_period=float(returns.max()),
        hit_rate=float((returns > 0).mean()),
        mean_leverage=float(result.leverage.mean()) if len(result.leverage) else 1.0,
        max_leverage=float(result.leverage.max()) if len(result.leverage) else 1.0,
        is_ruined=result.is_ruined,
        n_periods=n,
    )


def summarise(results: dict[str, BacktestResult], risk_free: float = 0.0) -> pd.DataFrame:
    """Evaluate several strategies and return one row each, sorted by Sharpe."""
    rows = [evaluate(result, risk_free).as_dict() for result in results.values()]
    frame = pd.DataFrame(rows)
    return frame.sort_values("sharpe", ascending=False).reset_index(drop=True)
