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
from .metrics import MakerMetrics, session_standard_errors, summarise_runs
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
    "SessionRunner",
    "book_runner",
    "build_policy_set",
    "compare_policies_book",
    "compare_policies_reference",
    "paired_test",
    "policy_sensitivity_sweep",
    "reference_runner",
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


#: Runs one session of ``policy`` with the given seed.
SessionRunner = Callable[[QuotePolicy, int], SimulationResult]


def policy_sensitivity_sweep(
    values: Sequence[float],
    build: Callable[[float], tuple[Sequence[QuotePolicy], SessionRunner]],
    parameter: str,
    n_runs: int = 200,
    base_seed: int = 4242,
    n_bootstrap: int = 1_000,
) -> pd.DataFrame:
    """Sweep one parameter for several policies, with standard errors across sessions.

    At each value, ``build`` returns the policies to compare and a function that runs one
    session of a policy for a given seed. Taking a runner rather than a set of parameters
    lets the same sweep drive either engine: the idealised reference world (for the
    parameters of the Avellaneda-Stoikov model itself) or the full order book (for
    properties of the order flow, such as the informed fraction, that the reference world
    does not have).

    Session ``i`` uses seed ``base_seed + i`` for every policy *and* every parameter value,
    so policies are compared on common random numbers and the curves are smooth in the
    parameter rather than jagged with independent sampling noise. The standard errors
    reported are for each policy's statistic on its own; because of the common random
    numbers, *differences* between policies at the same value are estimated more precisely
    than those errors suggest.

    Args:
        values: Parameter values to sweep.
        build: Maps a value to ``(policies, runner)``.
        parameter: Name of the swept parameter.
        n_runs: Independent sessions per policy per value.
        base_seed: First seed.
        n_bootstrap: Bootstrap resamples for the standard errors; see
            :func:`~mmsim.metrics.session_standard_errors`.

    Returns:
        Long format, one row per ``(value, policy)``: ``parameter``, ``value``, the columns
        of :class:`~mmsim.metrics.MakerMetrics`, and ``mean_pnl_se``, ``std_pnl_se``,
        ``sharpe_se`` and ``std_final_inventory_se``. ``sharpe`` is per session and not
        annualised.
    """
    rows: list[dict[str, float | str | int]] = []
    for value in values:
        policies, runner = build(float(value))
        for policy in policies:
            runs = [runner(policy, base_seed + i) for i in range(n_runs)]
            summary: MakerMetrics = summarise_runs(runs)
            errors = session_standard_errors(
                np.array([r.final_pnl for r in runs]),
                np.array([r.final_inventory for r in runs]),
                n_bootstrap=n_bootstrap,
                seed=base_seed,
            )
            row: dict[str, float | str | int] = {"parameter": parameter, "value": float(value)}
            row.update(summary.as_dict())
            row.update(errors)
            rows.append(row)
    return pd.DataFrame(rows)


def reference_runner(
    params: AvellanedaStoikovParams,
    n_steps: int = 200,
    fill_model: FillModel | str = FillModel.EXACT,
) -> SessionRunner:
    """A :data:`SessionRunner` for the idealised Avellaneda-Stoikov world ``params``."""

    def run(policy: QuotePolicy, seed: int) -> SimulationResult:
        return simulate_reference(policy, params, n_steps=n_steps, fill_model=fill_model, seed=seed)

    return run


def book_runner(
    flow_config: FlowConfig,
    market: MarketConfig | None = None,
    n_steps: int = 3_000,
    horizon: float = 1.0,
) -> SessionRunner:
    """A :data:`SessionRunner` for the full order-book world."""

    def run(policy: QuotePolicy, seed: int) -> SimulationResult:
        return simulate_book(
            policy, flow_config, market, n_steps=n_steps, horizon=horizon, seed=seed
        )

    return run


def sensitivity_sweep(
    values: Sequence[float],
    build: Callable[[float], tuple[QuotePolicy, AvellanedaStoikovParams]],
    parameter: str,
    n_runs: int = 200,
    n_steps: int = 200,
    fill_model: FillModel | str = FillModel.EXACT,
    base_seed: int = 4242,
) -> pd.DataFrame:
    """Single-policy sweep in the reference engine; a wrapper over :func:`policy_sensitivity_sweep`.

    Args:
        values: Parameter values to sweep.
        build: Maps a value to ``(policy, params)``. Returning both lets the sweep change
            the *model* (which alters how the policy quotes) or the *market* (which alters
            what the policy faces), and the caller decides which.
        parameter: Name of the swept parameter, used as the output column.
        n_runs: Runs at each value.
        n_steps: Steps per run.
        fill_model: Fill discretisation.
        base_seed: Common random numbers are reused across values.

    Returns:
        One row per value: the swept value in a column named ``parameter``, then the
        columns of :class:`~mmsim.metrics.MakerMetrics` and their standard errors.
    """

    def build_one(value: float) -> tuple[list[QuotePolicy], SessionRunner]:
        policy, params = build(value)
        return [policy], reference_runner(params, n_steps=n_steps, fill_model=fill_model)

    frame = policy_sensitivity_sweep(
        values, build_one, parameter, n_runs=n_runs, base_seed=base_seed
    )
    return frame.drop(columns="parameter").rename(columns={"value": parameter})
