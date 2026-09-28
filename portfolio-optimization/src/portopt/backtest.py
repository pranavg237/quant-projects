r"""Walk-forward backtesting with transaction costs.

The only interesting question about a portfolio construction method is how it does on data
it has never seen. Everything here is arranged around making that question answerable
honestly.

**No lookahead, enforced structurally.** At each rebalance date :math:`t` the estimation
window is ``returns.loc[:t]`` *inclusive of* :math:`t` and nothing after. The weights
chosen at :math:`t` are then applied to the returns of :math:`(t, t+1]`. The split is done
by integer position on a sorted index, so an off-by-one is a test failure rather than a
silently inflated Sharpe ratio -- and :func:`walk_forward` asserts it on every step.

**Weights drift between rebalances.** A portfolio set to 60/40 does not stay 60/40: the
winner grows. Re-imposing the target every period without trading would be free
rebalancing, which is the second most common way to accidentally inflate a backtest. Drift
is simulated explicitly, and turnover is measured against the *drifted* weights, which is
what actually has to be traded.

**Costs are charged on turnover.** Cost per period is
:math:`c \times \sum_i |w_i^{\text{new}} - w_i^{\text{drifted}}|`, with :math:`c` a
one-way proportional rate in basis points. This is the honest minimum: it captures the
spread and commission but not market impact, which is negligible at ETF scale and
sub-$100m size and is *not* negligible for anything larger.

**Return convention.** Portfolio return in a period is the weighted average of simple asset
returns, then costs are subtracted. Equity compounds. Annualised return is geometric, not
arithmetic -- the two differ by about half the variance, which for a 15% vol portfolio is
1.1% a year, more than most of the differences being measured.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .types import Weights

__all__ = [
    "BacktestResult",
    "PortfolioBuilder",
    "compare_strategies",
    "equal_weight_builder",
    "rebalance_dates",
    "walk_forward",
]

#: A strategy: given the returns available *up to and including* the decision date,
#: produce target weights. It is handed only what it is allowed to see.
PortfolioBuilder = Callable[[pd.DataFrame], Weights]


def equal_weight_builder(returns: pd.DataFrame) -> Weights:
    """The benchmark that is genuinely hard to beat: 1/N.

    DeMiguel, Garlappi and Uppal (2009) found that no mean-variance strategy they tested
    reliably beat it out of sample across 14 datasets, because the estimation error in the
    optimised strategies outweighed their theoretical advantage. Any portfolio project that
    does not benchmark against 1/N is not being serious.
    """
    return pd.Series(1.0 / returns.shape[1], index=returns.columns, name="weight")


def rebalance_dates(index: pd.DatetimeIndex, frequency: str = "ME") -> pd.DatetimeIndex:
    """The last available observation date within each calendar period.

    Using the *last available* date rather than the calendar month end matters: a calendar
    month end is often a weekend, and reindexing to it would either drop the rebalance or
    silently pull in a price from the wrong side of it.
    """
    series = pd.Series(index, index=index)
    return pd.DatetimeIndex(series.resample(frequency).last().dropna().to_numpy())


@dataclass
class BacktestResult:
    """Output of a walk-forward backtest.

    Attributes:
        name: Strategy name.
        returns: Net-of-cost portfolio returns, indexed by period end.
        gross_returns: Before costs, so the cost drag is separable.
        weights: Target weights at each rebalance date.
        turnover: One-way turnover at each rebalance (sum of absolute weight changes).
        costs: Cost charged at each rebalance, in return units.
        leverage: Gross exposure (sum of absolute weights) at each rebalance. 1.0 for a
            fully invested long-only portfolio; anything above that is borrowed money, and
            it is the single best predictor of whether a mean-variance portfolio will
            survive out of sample.
        periods_per_year: For annualisation.
    """

    name: str
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    leverage: pd.Series
    periods_per_year: float = 12.0
    meta: dict[str, float] = field(default_factory=dict)

    @property
    def equity_curve(self) -> pd.Series:
        """Cumulative growth of one unit, net of costs."""
        return (1.0 + self.returns).cumprod()

    @property
    def is_ruined(self) -> bool:
        """Whether the portfolio ever lost 100% or more in a single period.

        A levered long-short portfolio can, and in this repo's backtests routinely does.
        Once it happens the equity curve is non-positive and every ratio computed from it
        -- Sharpe included -- is arithmetic noise rather than a performance measure. It
        must be reported as ruin, not as a number.
        """
        return bool((self.returns <= -1.0).any())

    def __str__(self) -> str:
        return f"{self.name}: {len(self.returns)} periods, {len(self.weights)} rebalances"


def _drift_weights(weights: Weights, period_returns: pd.Series) -> Weights:
    r"""Where a portfolio ends up after one period of returns, before any trading.

    :math:`w_i^{\text{drift}} = w_i(1+r_i) / \sum_j w_j(1+r_j)`. Ignoring this and assuming
    the portfolio stays at its target is free rebalancing, and it flatters every strategy
    -- most of all the ones with the highest turnover.
    """
    grown = weights * (1.0 + period_returns.reindex(weights.index).fillna(0.0))
    total = float(grown.sum())
    if abs(total) < 1e-12:  # pragma: no cover - total wipeout
        return weights
    return grown / total


def walk_forward(
    returns: pd.DataFrame,
    builder: PortfolioBuilder,
    name: str,
    lookback: int = 60,
    rebalance_every: int = 1,
    cost_bps: float = 10.0,
    periods_per_year: float = 12.0,
    min_lookback: int | None = None,
) -> BacktestResult:
    """Run a walk-forward backtest of one portfolio construction method.

    Args:
        returns: Asset returns at the rebalancing frequency, dates on the index.
        builder: Maps the visible return history to target weights.
        name: Strategy name for reporting.
        lookback: Estimation window length in periods. ``0`` means expanding window.
        rebalance_every: Rebalance every ``n`` periods. Between rebalances the portfolio
            drifts and no costs are charged.
        cost_bps: One-way proportional cost in basis points of turnover.
        periods_per_year: For annualisation.
        min_lookback: Minimum history before the first trade. Defaults to ``lookback``.

    Returns:
        A :class:`BacktestResult`.

    Raises:
        ValueError: if there is not enough history to produce a single out-of-sample period.
    """
    if not returns.index.is_monotonic_increasing:
        returns = returns.sort_index()
    minimum = min_lookback if min_lookback is not None else max(lookback, 2)
    n_periods = len(returns)
    if n_periods <= minimum:
        raise ValueError(
            f"need more than {minimum} periods of history; got {n_periods}. "
            "Shorten the lookback or lengthen the sample."
        )

    cost_rate = cost_bps / 10_000.0
    net: list[float] = []
    gross: list[float] = []
    dates: list[pd.Timestamp] = []
    weight_rows: dict[pd.Timestamp, Weights] = {}
    turnovers: dict[pd.Timestamp, float] = {}
    charges: dict[pd.Timestamp, float] = {}
    leverages: dict[pd.Timestamp, float] = {}

    current: Weights | None = None
    periods_since_trade = 0

    for i in range(minimum, n_periods):
        decision_date = returns.index[i - 1]
        realisation_date = returns.index[i]

        # The estimation window ends at i-1 inclusive. Slicing by position on a sorted
        # index makes the boundary explicit, and the invariant is checked every step with a
        # real raise rather than an `assert`, which `python -O` would strip out -- leaving
        # a silently lookahead-biased backtest in exactly the configuration you would use
        # to run it for real.
        start = max(0, i - 1 - lookback + 1) if lookback > 0 else 0
        window = returns.iloc[start:i]
        if window.index[-1] != decision_date or window.index[-1] >= realisation_date:
            raise RuntimeError(  # pragma: no cover - guards against a future refactor
                f"lookahead detected: window ends {window.index[-1]}, decision "
                f"{decision_date}, realisation {realisation_date}"
            )

        trade_now = current is None or periods_since_trade >= rebalance_every
        if trade_now:
            target = builder(window).reindex(returns.columns).fillna(0.0)
            previous = current if current is not None else pd.Series(0.0, index=returns.columns)
            turnover = float((target - previous).abs().sum())
            charge = cost_rate * turnover
            current = target
            periods_since_trade = 0
            weight_rows[decision_date] = target
            turnovers[decision_date] = turnover
            charges[decision_date] = charge
            leverages[decision_date] = float(target.abs().sum())
        else:
            charge = 0.0
        periods_since_trade += 1
        if current is None:  # pragma: no cover - the first iteration always trades
            raise RuntimeError("no weights were set on the first rebalance")

        period_returns = returns.iloc[i]
        gross_return = float((current * period_returns.reindex(current.index)).sum())
        gross.append(gross_return)
        net.append(gross_return - charge)
        dates.append(pd.Timestamp(realisation_date))
        current = _drift_weights(current, period_returns)

    index = pd.DatetimeIndex(dates)
    return BacktestResult(
        name=name,
        returns=pd.Series(net, index=index, name=name),
        gross_returns=pd.Series(gross, index=index, name=name),
        weights=pd.DataFrame(weight_rows).T,
        turnover=pd.Series(turnovers, name="turnover"),
        costs=pd.Series(charges, name="cost"),
        leverage=pd.Series(leverages, name="leverage"),
        periods_per_year=periods_per_year,
    )


def compare_strategies(
    returns: pd.DataFrame,
    builders: dict[str, PortfolioBuilder],
    lookback: int = 60,
    rebalance_every: int = 1,
    cost_bps: float = 10.0,
    periods_per_year: float = 12.0,
) -> dict[str, BacktestResult]:
    """Run the same walk-forward on several strategies over an identical sample.

    Every strategy sees exactly the same dates and the same estimation windows, so the
    comparison is paired and differences cannot come from one strategy getting a longer
    or luckier sample.
    """
    return {
        name: walk_forward(
            returns,
            builder,
            name,
            lookback=lookback,
            rebalance_every=rebalance_every,
            cost_bps=cost_bps,
            periods_per_year=periods_per_year,
        )
        for name, builder in builders.items()
    }
