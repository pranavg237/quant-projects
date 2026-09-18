r"""Performance metrics for a market-making run.

The headline number for a market maker is **not** PnL. A strategy that quotes wider makes
more per trade and trades less; one that carries inventory makes more on a trending day and
loses more on a reverting one. What separates strategies is the *risk* they take to earn
their PnL, so the comparisons here are built around:

* **Sharpe ratio** across independent simulation runs, not within a path. A market maker's
  within-path PnL increments are dominated by mark-to-market swings on inventory and are
  strongly autocorrelated, so an intraday Sharpe is close to meaningless. Across runs each
  observation is one independent session and the ratio is interpretable.
* **Inventory variance**, the quantity Avellaneda-Stoikov actually optimises.
* **PnL decomposition** into spread earned and inventory mark-to-market. These are the two
  sides of the trade-off, and separating them shows *why* a strategy wins rather than that
  it won.
* **Markout**, the standard desk measure of adverse selection: how the mid moves after your
  fill. If you buy and the price then falls, you were picked off. Averaged over fills and
  signed by direction, markout at increasing horizons is a direct read on how much of the
  captured spread you keep.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .engine import SimulationResult
from .types import FloatArray

__all__ = [
    "MakerMetrics",
    "markout_curve",
    "pnl_decomposition",
    "summarise_runs",
]


@dataclass(frozen=True)
class MakerMetrics:
    """Summary statistics over a set of independent simulation runs.

    Attributes:
        policy: Policy name.
        n_runs: Number of runs.
        mean_pnl: Mean terminal PnL.
        std_pnl: Standard deviation of terminal PnL across runs.
        sharpe: ``mean_pnl / std_pnl``. Per session, not annualised -- annualising would
            require assuming how many independent sessions a year holds, which is exactly
            the kind of unearned scaling factor that makes market-making Sharpes look
            absurd.
        median_pnl: Median terminal PnL. Reported alongside the mean because market-making
            PnL is left-skewed: many small wins, occasional large losses.
        pnl_5th_percentile: 5th percentile of terminal PnL, i.e. a simple session VaR.
        mean_abs_inventory: Mean absolute inventory over all steps and runs.
        std_final_inventory: Standard deviation of terminal inventory across runs.
        max_abs_inventory: Largest absolute inventory reached in any run.
        mean_trades: Mean number of fills per run.
        mean_spread_captured: Mean quoted spread, in price units.
        max_drawdown: Mean over runs of the worst peak-to-trough fall in mark-to-market.
    """

    policy: str
    n_runs: int
    mean_pnl: float
    std_pnl: float
    sharpe: float
    median_pnl: float
    pnl_5th_percentile: float
    mean_abs_inventory: float
    std_final_inventory: float
    max_abs_inventory: float
    mean_trades: float
    mean_spread_captured: float
    max_drawdown: float

    def as_dict(self) -> dict[str, float | str | int]:
        """Flat dict, for building a DataFrame."""
        return asdict(self)


def _max_drawdown(equity: FloatArray) -> float:
    """Worst peak-to-trough fall in an equity curve, as a positive number."""
    running_peak = np.maximum.accumulate(equity)
    return float(np.max(running_peak - equity))


def summarise_runs(results: list[SimulationResult]) -> MakerMetrics:
    """Aggregate a list of runs of the *same* policy into one summary.

    Raises:
        ValueError: if ``results`` is empty.
    """
    if not results:
        raise ValueError("cannot summarise an empty list of runs")

    pnl = np.array([r.final_pnl for r in results], dtype=np.float64)
    final_q = np.array([r.final_inventory for r in results], dtype=np.float64)
    abs_q = np.concatenate([np.abs(r.inventory) for r in results])
    trades = np.array([r.n_trades for r in results], dtype=np.float64)
    drawdowns = np.array([_max_drawdown(r.mark_to_market) for r in results], dtype=np.float64)

    spreads = []
    for r in results:
        width = r.ask_quotes - r.bid_quotes
        finite = width[np.isfinite(width)]
        if finite.size:
            spreads.append(float(finite.mean()))
    std_pnl = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0

    return MakerMetrics(
        policy=results[0].policy_name,
        n_runs=len(results),
        mean_pnl=float(pnl.mean()),
        std_pnl=std_pnl,
        sharpe=float(pnl.mean() / std_pnl) if std_pnl > 0 else float("nan"),
        median_pnl=float(np.median(pnl)),
        pnl_5th_percentile=float(np.percentile(pnl, 5)),
        mean_abs_inventory=float(abs_q.mean()),
        std_final_inventory=float(final_q.std(ddof=1)) if final_q.size > 1 else 0.0,
        max_abs_inventory=float(abs_q.max()),
        mean_trades=float(trades.mean()),
        mean_spread_captured=float(np.mean(spreads)) if spreads else float("nan"),
        max_drawdown=float(drawdowns.mean()),
    )


def pnl_decomposition(result: SimulationResult) -> dict[str, float]:
    r"""Split terminal PnL into spread capture and inventory mark-to-market.

    Writing terminal wealth as :math:`W_T = X_T + q_T S_T` and telescoping the cash process
    over fills at prices :math:`p_i` with signed sizes :math:`\Delta q_i`,

    .. math::
        W_T = \underbrace{\sum_i \Delta q_i (S_{t_i} - p_i)}_{\text{spread capture}}
            + \underbrace{\sum_i \Delta q_i (S_T - S_{t_i})}_{\text{inventory P\&L}},

    where :math:`S_{t_i}` is the mid at the time of fill :math:`i`. The first term is what
    the maker earned for providing liquidity -- always positive if it quotes outside the
    mid. The second is what happened to the position afterwards, and is where both
    inventory risk and adverse selection land.

    The identity is exact, and :func:`~mmsim.metrics.pnl_decomposition` checks it against
    the simulated wealth so a bookkeeping error cannot slip through.

    Returns:
        ``{"spread_capture", "inventory_pnl", "total", "reported_total",
        "reconciliation_error"}``.
    """
    fills = result.fill_prices
    terminal_mid = float(result.mid[-1])
    if fills.empty:
        return {
            "spread_capture": 0.0,
            "inventory_pnl": 0.0,
            "total": 0.0,
            "reported_total": result.final_pnl,
            "reconciliation_error": result.final_pnl,
        }

    sign = np.where(fills["side"].to_numpy() == "buy", 1.0, -1.0)
    size = fills["size"].to_numpy(dtype=np.float64) if "size" in fills else np.ones(len(fills))
    signed = sign * size
    price = fills["price"].to_numpy(dtype=np.float64)
    mid_at_fill = fills["mid"].to_numpy(dtype=np.float64)

    spread_capture = float(np.sum(signed * (mid_at_fill - price)))
    inventory_pnl = float(np.sum(signed * (terminal_mid - mid_at_fill)))
    total = spread_capture + inventory_pnl
    return {
        "spread_capture": spread_capture,
        "inventory_pnl": inventory_pnl,
        "total": total,
        "reported_total": result.final_pnl,
        "reconciliation_error": total - result.final_pnl,
    }


def markout_curve(
    result: SimulationResult, horizons: tuple[int, ...] = (1, 5, 10, 25, 50, 100, 250)
) -> pd.DataFrame:
    r"""Signed mid move after each fill, averaged over fills -- the adverse-selection curve.

    For a fill at step :math:`i` with signed size :math:`\Delta q`, the markout at horizon
    :math:`h` is

    .. math:: m_h = \operatorname{sign}(\Delta q)\,(S_{i+h} - S_i).

    A maker that buys just before the price falls has negative markout: it was picked off.
    A flat curve means the flow carried no information. The curve typically falls with
    :math:`h` and then levels off once the information in the trade is fully impounded,
    and the level it reaches is the per-fill adverse-selection cost the maker must earn
    back from the spread.

    Args:
        result: A run produced by :func:`~mmsim.engine.simulate_book`. Requires a ``step``
            column in ``fill_prices``, which the reference engine does not produce.
        horizons: Markout horizons in simulation steps.

    Returns:
        Frame with ``horizon``, ``mean_markout``, ``std_error`` and ``n_fills``.

    Raises:
        ValueError: if the result has no step-indexed fills.
    """
    fills = result.fill_prices
    if fills.empty:
        return pd.DataFrame(columns=["horizon", "mean_markout", "std_error", "n_fills"])
    if "step" not in fills.columns:
        raise ValueError(
            "markout needs step-indexed fills; use simulate_book, not simulate_reference"
        )

    steps = fills["step"].to_numpy(dtype=np.int64)
    sign = np.where(fills["side"].to_numpy() == "buy", 1.0, -1.0)
    mid = result.mid
    last = mid.size - 1

    rows: list[dict[str, float]] = []
    for horizon in horizons:
        future = np.minimum(steps + horizon, last)
        markout = sign * (mid[future] - mid[steps])
        n = markout.size
        rows.append(
            {
                "horizon": float(horizon),
                "mean_markout": float(markout.mean()),
                "std_error": float(markout.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "n_fills": float(n),
            }
        )
    return pd.DataFrame(rows)
