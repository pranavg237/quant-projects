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
    "fill_markouts",
    "informed_markout_theory",
    "markout_curve",
    "markout_decomposition",
    "pnl_decomposition",
    "session_standard_errors",
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


def session_standard_errors(
    pnl: FloatArray,
    final_inventory: FloatArray,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> dict[str, float]:
    r"""Standard errors, across independent sessions, of the headline session statistics.

    Each session (one seed) is one independent observation, so the sampling error of every
    cross-session statistic comes from resampling sessions:

    * ``mean_pnl_se`` is the textbook :math:`s/\sqrt{n}`;
    * ``std_pnl_se``, ``sharpe_se`` and ``std_final_inventory_se`` are **bootstrap**
      standard errors (sessions resampled with replacement, ``n_bootstrap`` times). The
      normal-theory formulas for these assume Gaussian PnL, and market-making PnL is
      skewed, so the bootstrap is the more honest choice. The Sharpe is per session and
      not annualised, like every Sharpe in this project.

    The bootstrap generator is seeded, so the reported errors are reproducible.

    Args:
        pnl: Terminal PnL of each session.
        final_inventory: Terminal inventory of each session, aligned with ``pnl``.
        n_bootstrap: Bootstrap resamples.
        seed: Seed for the resampling.

    Returns:
        ``{"mean_pnl_se", "std_pnl_se", "sharpe_se", "std_final_inventory_se"}``; all
        ``nan`` when there are fewer than two sessions.

    Raises:
        ValueError: if the two arrays are not the same length.
    """
    x = np.asarray(pnl, dtype=np.float64)
    q = np.asarray(final_inventory, dtype=np.float64)
    if x.shape != q.shape:
        raise ValueError("pnl and final_inventory must be aligned by session")
    n = x.size
    if n < 2:
        nan = float("nan")
        return {
            "mean_pnl_se": nan,
            "std_pnl_se": nan,
            "sharpe_se": nan,
            "std_final_inventory_se": nan,
        }
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_x = x[idx]
    boot_std = boot_x.std(axis=1, ddof=1)
    positive = boot_std > 0
    boot_sharpe = boot_x.mean(axis=1)[positive] / boot_std[positive]
    return {
        "mean_pnl_se": float(x.std(ddof=1) / np.sqrt(n)),
        "std_pnl_se": float(boot_std.std(ddof=1)),
        "sharpe_se": float(boot_sharpe.std(ddof=1)) if boot_sharpe.size > 1 else float("nan"),
        "std_final_inventory_se": float(q[idx].std(axis=1, ddof=1).std(ddof=1)),
    }


def fill_markouts(
    result: SimulationResult,
    horizons: tuple[int, ...] = (1, 5, 20, 100),
    tick_size: float = 0.01,
) -> pd.DataFrame:
    r"""One row per maker fill: the edge earned at the fill and the markout after it.

    For a fill of signed size :math:`\Delta q` at price :math:`p`, with :math:`S_i` the
    efficient mid the maker quoted against and :math:`S_{i+h}` the mid :math:`h` steps
    later, the per-unit quantities are

    .. math::
        \text{edge} = \operatorname{sign}(\Delta q)\,(S_i - p), \qquad
        m_h = \operatorname{sign}(\Delta q)\,(S_{i+h} - S_i), \qquad
        \text{realised}_h = \text{edge} + m_h.

    The edge is what the maker earned at the moment of the fill, measured against the mid;
    the markout is what the market then took back; the realised spread is what was left.
    The adverse-selection cost is :math:`-m_h`. This is the standard
    effective spread = realised spread + price impact decomposition, seen from the
    maker's side.

    A markout whose horizon runs past the end of the session is ``nan`` rather than being
    truncated at the last step: truncating would mix horizons and quietly shrink the
    long-horizon markouts of late fills.

    Args:
        result: A run from :func:`~mmsim.engine.simulate_book`.
        horizons: Markout horizons, in steps.
        tick_size: Used to express edge and markouts in ticks.

    Returns:
        Columns ``step``, ``side``, ``counterparty``, ``size``, ``edge_ticks`` and one
        ``markout_{h}_ticks`` per horizon. Empty if the run had no fills.

    Raises:
        ValueError: if the run's fills are not step-indexed (i.e. from the reference
            engine).
    """
    fills = result.fill_prices
    columns = ["step", "side", "counterparty", "size", "edge_ticks"] + [
        f"markout_{h}_ticks" for h in horizons
    ]
    if fills.empty:
        return pd.DataFrame(columns=columns)
    if "step" not in fills.columns:
        raise ValueError(
            "markout needs step-indexed fills; use simulate_book, not simulate_reference"
        )
    steps = fills["step"].to_numpy(dtype=np.int64)
    sign = np.where(fills["side"].to_numpy() == "buy", 1.0, -1.0)
    price = fills["price"].to_numpy(dtype=np.float64)
    mid = result.mid
    last = mid.size - 1

    out = pd.DataFrame(
        {
            "step": steps,
            "side": fills["side"].to_numpy(),
            "counterparty": (
                fills["counterparty"].to_numpy()
                if "counterparty" in fills.columns
                else np.full(steps.size, "unknown")
            ),
            "size": (
                fills["size"].to_numpy(dtype=np.float64)
                if "size" in fills.columns
                else np.ones(steps.size)
            ),
            "edge_ticks": sign * (mid[steps] - price) / tick_size,
        }
    )
    for h in horizons:
        future = steps + h
        valid = future <= last
        values = np.full(steps.size, np.nan)
        values[valid] = sign[valid] * (mid[future[valid]] - mid[steps[valid]]) / tick_size
        out[f"markout_{h}_ticks"] = values
    return out


def _ratio_with_se(numer: FloatArray, denom: FloatArray) -> tuple[float, float]:
    r"""Ratio of sums :math:`\sum_j x_j / \sum_j w_j` and its session-clustered standard error.

    Fills inside one session are not independent -- they share a price path and an
    inventory history -- so the standard error treats each *session* :math:`j` as the unit
    of observation. The delta-method (linearised) variance of a ratio estimator
    :math:`R` over :math:`n` sessions is
    :math:`\frac{1}{n(n-1)\bar w^2}\sum_j (x_j - R w_j)^2`.
    """
    n = numer.size
    total_w = float(denom.sum())
    if total_w <= 0.0:
        return float("nan"), float("nan")
    ratio = float(numer.sum()) / total_w
    if n < 2:
        return ratio, float("nan")
    resid = numer - ratio * denom
    se = float(np.sqrt(np.sum(resid**2) / (n * (n - 1))) / (total_w / n))
    return ratio, se


def markout_decomposition(
    runs: dict[str, list[SimulationResult]],
    horizons: tuple[int, ...] = (1, 5, 20, 100),
    tick_size: float = 0.01,
) -> pd.DataFrame:
    r"""Realised spread and adverse-selection cost, by policy, counterparty and horizon.

    For every policy, every counterparty type (``informed``, ``uninformed`` and ``all``)
    and every horizon :math:`h`, reports size-weighted **per-unit** averages over fills
    (see :func:`fill_markouts`):

    * ``edge_ticks``: spread earned at the fill, against the mid quoted against;
    * ``markout_ticks``: signed mid move :math:`h` steps later;
    * ``adverse_selection_ticks`` :math:`= -` ``markout_ticks``;
    * ``realised_spread_ticks`` :math:`=` ``edge_ticks`` :math:`+` ``markout_ticks``,

    each with a standard error clustered by session (see :func:`_ratio_with_se`). The same
    quantities are also reported as **PnL per session**, in price units
    (``*_pnl_per_session``): per-unit value times size, summed over a session's fills and
    averaged over sessions. Summed over the two counterparty types, ``edge_pnl_per_session``
    is the ``spread_capture`` leg of :func:`pnl_decomposition`, minus the few fills too
    close to the end of the session to have a markout at that horizon.

    Args:
        runs: Book-engine runs keyed by policy name, e.g. ``ComparisonResult.runs``.
        horizons: Markout horizons, in steps.
        tick_size: Price units per tick.

    Returns:
        One row per ``(policy, counterparty, horizon)``.
    """
    rows: list[dict[str, float | str]] = []
    for policy, results in runs.items():
        per_run = [fill_markouts(r, horizons, tick_size) for r in results]
        n_sessions = max(len(per_run), 1)
        for counterparty in ("informed", "uninformed", "all"):
            for h in horizons:
                col = f"markout_{h}_ticks"
                size_sum = np.zeros(len(per_run))
                edge_sum = np.zeros(len(per_run))
                mark_sum = np.zeros(len(per_run))
                n_fills = 0
                for j, frame in enumerate(per_run):
                    if frame.empty:
                        continue
                    keep = frame[col].notna().to_numpy()
                    if counterparty != "all":
                        keep = keep & (frame["counterparty"] == counterparty).to_numpy()
                    size = frame["size"].to_numpy(dtype=np.float64)[keep]
                    size_sum[j] = size.sum()
                    edge_sum[j] = float(np.sum(size * frame["edge_ticks"].to_numpy()[keep]))
                    mark_sum[j] = float(np.sum(size * frame[col].to_numpy(dtype=np.float64)[keep]))
                    n_fills += int(keep.sum())
                edge, edge_se = _ratio_with_se(edge_sum, size_sum)
                mark, mark_se = _ratio_with_se(mark_sum, size_sum)
                real, real_se = _ratio_with_se(edge_sum + mark_sum, size_sum)
                to_pnl = tick_size / n_sessions
                rows.append(
                    {
                        "policy": policy,
                        "counterparty": counterparty,
                        "horizon": float(h),
                        "n_sessions": float(len(per_run)),
                        "n_fills": float(n_fills),
                        "volume_per_session": float(size_sum.sum()) / n_sessions,
                        "edge_ticks": edge,
                        "edge_ticks_se": edge_se,
                        "markout_ticks": mark,
                        "markout_ticks_se": mark_se,
                        "adverse_selection_ticks": -mark,
                        "realised_spread_ticks": real,
                        "realised_spread_ticks_se": real_se,
                        "edge_pnl_per_session": float(edge_sum.sum()) * to_pnl,
                        "adverse_selection_pnl_per_session": -float(mark_sum.sum()) * to_pnl,
                        "realised_pnl_per_session": float((edge_sum + mark_sum).sum()) * to_pnl,
                    }
                )
    return pd.DataFrame(rows)


def informed_markout_theory(
    horizons: FloatArray | tuple[int, ...], impact_ticks: float, impact_speed: float
) -> FloatArray:
    r"""Expected markout, in ticks, of a fill against an informed order, from the model alone.

    An informed order adds :math:`J` ticks of pending impact, of which a fraction
    :math:`v` (``impact_speed``) is released into the efficient price in the same step and
    in every step after. After :math:`h` steps the released amount is
    :math:`J(1-(1-v)^h)`, and the maker, on the other side of the trade, is marked down by
    exactly that:

    .. math:: \mathbb{E}[m_h \mid \text{informed}] = -J\,\bigl(1 - (1-v)^h\bigr).

    This is a first-principles benchmark for the measured informed markout. It ignores
    everything *else* that moves the price after the fill, which averages to zero only if
    being hit is independent of the other informed impact still pending at the time.
    """
    h = np.asarray(horizons, dtype=np.float64)
    return np.asarray(-impact_ticks * (1.0 - (1.0 - impact_speed) ** h), dtype=np.float64)
