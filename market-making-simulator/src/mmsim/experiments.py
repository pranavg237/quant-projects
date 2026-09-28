"""Multi-run comparisons and parameter sensitivity analysis.

Two design decisions do most of the work here.

**Common random numbers.** Every policy in a comparison is run against the *same* sequence
of seeds, so run ``i`` of the Avellaneda-Stoikov maker faces the same price path and the
same order flow as run ``i`` of the symmetric maker. The difference between them is then
almost entirely the strategy rather than the draw, and the paired difference has far lower
variance than the difference of two independent means. With 200 runs this turns a
comparison that is barely significant into one that is not close. It costs nothing and it
is the single cheapest thing you can do to make a simulation study trustworthy.

**Paired significance tests.** Because the runs are paired, the right test is a paired
t-test on the per-run PnL difference, not a two-sample test. The paired statistic is
reported alongside every comparison, so "strategy A beat strategy B" always comes with how
sure we are.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .avellaneda_stoikov import AvellanedaStoikovParams
from .engine import FillModel, SimulationResult, simulate_book, simulate_reference
from .flow import FlowConfig
from .metrics import MakerMetrics, summarise_runs
from .strategies import (
    AvellanedaStoikovPolicy,
    InventoryLimitPolicy,
    QuotePolicy,
    SymmetricPolicy,
    average_optimal_spread,
)
from .types import MarketConfig

__all__ = [
    "ComparisonResult",
    "build_policy_set",
    "compare_policies_book",
    "compare_policies_reference",
    "paired_test",
    "sensitivity_sweep",
]


@dataclass
class ComparisonResult:
    """Outcome of comparing several policies over common random numbers.

    Attributes:
        metrics: One row per policy.
        pnl: ``(n_runs, n_policies)`` terminal PnL, aligned by run so columns are paired.
        runs: The raw runs, keyed by policy name.
        paired_tests: Paired t-tests of each policy against the first one.
    """

    metrics: pd.DataFrame
    pnl: pd.DataFrame
    runs: dict[str, list[SimulationResult]]
    paired_tests: pd.DataFrame


def paired_test(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str) -> dict[str, float | str]:
    """Paired t-test on ``a - b``, for runs that shared a seed.

    Returns:
        A dict with the mean difference, its standard error, the t statistic, the
        two-sided p-value and the number of pairs.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = diff.size
    mean = float(diff.mean())
    std_error = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    if n > 1 and diff.std(ddof=1) > 0:
        t_stat, p_value = stats.ttest_rel(a, b)
    else:  # pragma: no cover - identical policies
        t_stat, p_value = float("nan"), float("nan")
    return {
        "comparison": f"{label_a} - {label_b}",
        "mean_difference": mean,
        "std_error": std_error,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "n_pairs": float(n),
    }


def _assemble(
    runs: dict[str, list[SimulationResult]],
) -> ComparisonResult:
    metrics = pd.DataFrame([summarise_runs(v).as_dict() for v in runs.values()])
    pnl = pd.DataFrame({name: [r.final_pnl for r in v] for name, v in runs.items()})
    names = list(runs)
    tests = pd.DataFrame(
        [
            paired_test(pnl[names[0]].to_numpy(), pnl[other].to_numpy(), names[0], other)
            for other in names[1:]
        ]
    )
    return ComparisonResult(metrics=metrics, pnl=pnl, runs=runs, paired_tests=tests)


def build_policy_set(
    params: AvellanedaStoikovParams,
    inventory_limit: float | None = None,
) -> list[QuotePolicy]:
    """The three policies compared throughout, all at the same average spread.

    Matching the spread is what makes the comparison about inventory management rather than
    about how wide each strategy happens to quote. Without it, the "better" strategy is
    simply whichever one was configured to be wider.

    Args:
        params: Avellaneda-Stoikov parameters; also sets the matched spread.
        inventory_limit: Position limit for the third policy. Defaults to roughly the
            inventory the A-S maker typically carries, so the limit binds occasionally
            rather than never or always.
    """
    avg_spread = average_optimal_spread(params)
    limit = inventory_limit if inventory_limit is not None else 5.0
    return [
        AvellanedaStoikovPolicy(params),
        SymmetricPolicy(half_spread=avg_spread / 2.0),
        InventoryLimitPolicy(half_spread=avg_spread / 2.0, max_inventory=limit),
    ]


def compare_policies_reference(
    policies: Sequence[QuotePolicy],
    params: AvellanedaStoikovParams,
    n_runs: int = 500,
    n_steps: int = 200,
    initial_mid: float = 100.0,
    fill_model: FillModel | str = FillModel.EXACT,
    base_seed: int = 20260918,
) -> ComparisonResult:
    """Compare policies in the idealised Avellaneda-Stoikov world, over common seeds."""
    runs: dict[str, list[SimulationResult]] = {}
    for policy in policies:
        name = getattr(policy, "name", type(policy).__name__)
        runs[name] = [
            simulate_reference(
                policy,
                params,
                n_steps=n_steps,
                initial_mid=initial_mid,
                fill_model=fill_model,
                seed=base_seed + i,
            )
            for i in range(n_runs)
        ]
    return _assemble(runs)


def compare_policies_book(
    policies: Sequence[QuotePolicy],
    flow_config: FlowConfig,
    market: MarketConfig | None = None,
    n_runs: int = 100,
    n_steps: int = 3_000,
    horizon: float = 1.0,
    requote_every: int = 1,
    base_seed: int = 20260918,
) -> ComparisonResult:
    """Compare policies in the full order-book world, over common seeds."""
    market = market or MarketConfig()
    runs: dict[str, list[SimulationResult]] = {}
    for policy in policies:
        name = getattr(policy, "name", type(policy).__name__)
        runs[name] = [
            simulate_book(
                policy,
                flow_config,
                market,
                n_steps=n_steps,
                horizon=horizon,
                requote_every=requote_every,
                seed=base_seed + i,
            )
            for i in range(n_runs)
        ]
    return _assemble(runs)


def sensitivity_sweep(
    values: Sequence[float],
    build: Callable[[float], tuple[QuotePolicy, AvellanedaStoikovParams]],
    parameter: str,
    n_runs: int = 200,
    n_steps: int = 200,
    fill_model: FillModel | str = FillModel.EXACT,
    base_seed: int = 4242,
) -> pd.DataFrame:
    """Sweep one parameter in the reference engine and report metrics at each value.

    Args:
        values: Parameter values to sweep.
        build: Maps a value to ``(policy, params)``. Returning both lets the sweep change
            the *model* (which alters how the policy quotes) or the *market* (which alters
            what the policy faces), and the caller decides which.
        parameter: Name of the swept parameter, used as the output column.
        n_runs: Runs at each value.
        n_steps: Steps per run.
        fill_model: Fill discretisation.
        base_seed: Common random numbers are reused across values as well as across
            policies, so the curve is smooth in the parameter rather than jagged with
            sampling noise.

    Returns:
        One row per value, with the columns of :class:`~mmsim.metrics.MakerMetrics`.
    """
    rows: list[dict[str, float | str | int]] = []
    for value in values:
        policy, params = build(float(value))
        runs = [
            simulate_reference(
                policy, params, n_steps=n_steps, fill_model=fill_model, seed=base_seed + i
            )
            for i in range(n_runs)
        ]
        summary: MakerMetrics = summarise_runs(runs)
        row = summary.as_dict()
        row[parameter] = float(value)
        rows.append(row)
    frame = pd.DataFrame(rows)
    columns = [parameter] + [c for c in frame.columns if c != parameter]
    return frame[columns]
