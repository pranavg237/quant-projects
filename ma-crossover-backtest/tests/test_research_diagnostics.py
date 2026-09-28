"""Tests for the parameter-sensitivity surface and the family-wide deflated Sharpe ratio.

Both are offline: prices are synthetic and grid files are written to a temp directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics
from quantbt.data import PriceData
from quantbt.research.multiple_testing import collect_trials, dsr_row, dsr_sensitivity
from quantbt.research.sensitivity import (
    ma_sharpe_grid,
    neighbourhood_sharpe,
    surface_summary,
)
from quantbt.validation import deflated_sharpe_ratio, probabilistic_sharpe_ratio
from quantbt.vectorized import backtest_ma_crossover

# -- sensitivity surface ------------------------------------------------------------------


def test_ma_sharpe_grid_matches_single_backtests(random_walk: PriceData) -> None:
    """Each cell is exactly the excess, net-of-cost Sharpe of that one backtest."""
    rf = 0.02
    start = random_walk.index[120]
    table = ma_sharpe_grid(random_walk, [5, 10, 40], [20, 40], start=start, rf=rf)
    # (40, 20) and (40, 40) are not crossovers and are skipped
    assert list(zip(table["fast"], table["slow"], strict=True)) == [
        (5, 20),
        (5, 40),
        (10, 20),
        (10, 40),
    ]
    one = backtest_ma_crossover(random_walk, 10, 40, start=start, rf=rf)
    cell = float(table.loc[(table["fast"] == 10) & (table["slow"] == 40), "sharpe"].iloc[0])
    assert cell == pytest.approx(metrics.sharpe(one.returns, rf=rf), rel=1e-12)
    # costs are on: a zero-cost run of the same cell has a higher Sharpe
    free = backtest_ma_crossover(
        random_walk, 10, 40, start=start, rf=rf, slippage_bps=0, commission_bps=0
    )
    assert metrics.sharpe(free.returns, rf=rf) > cell
    with pytest.raises(ValueError, match="no valid"):
        ma_sharpe_grid(random_walk, [50], [20], start=start)


def _toy_surface() -> pd.DataFrame:
    """A 3x3 surface with a spike at (2, 20) and a flat plateau elsewhere."""
    rows = []
    for f in (1, 2, 3):
        for s in (10, 20, 30):
            rows.append({"fast": f, "slow": s, "sharpe": 1.0 if (f, s) == (2, 20) else 0.2})
    return pd.DataFrame(rows)


def test_neighbourhood_separates_spike_from_plateau() -> None:
    table = _toy_surface()
    spike = neighbourhood_sharpe(table, 2, 20)
    assert spike["sharpe"] == 1.0
    assert spike["n_neighbours"] == 8
    assert spike["neighbour_mean"] == pytest.approx(0.2)
    corner = neighbourhood_sharpe(table, 1, 10)
    assert corner["n_neighbours"] == 3  # an edge cell has fewer neighbours
    assert corner["neighbour_mean"] == pytest.approx((0.2 + 0.2 + 1.0) / 3)
    with pytest.raises(ValueError, match="not on the grid"):
        neighbourhood_sharpe(table, 4, 20)
    summary = surface_summary(table, benchmark_sharpe=0.5, points=[(2, 20), (3, 30)])
    assert list(summary["beats_benchmark"]) == [True, False]
    assert summary["percentile_in_grid"].iloc[0] == pytest.approx(8 / 9)
    assert summary["percentile_in_grid"].iloc[1] == 0.0


# -- family-wide deflated Sharpe ------------------------------------------------------------


def _write_grid(root: Path, name: str, params: dict[str, list[float]], sharpe: list[float]) -> None:
    (root / name).mkdir(parents=True)
    frame = pd.DataFrame({**params, "objective": sharpe, "sharpe": sharpe, "trades": 1})
    frame.to_csv(root / name / "grid.csv", index=False)


def test_collect_trials_counts_every_configuration_of_every_strategy(tmp_path: Path) -> None:
    _write_grid(tmp_path, "a", {"short": [10, 20], "long": [50, 50]}, [0.5, 0.3])
    _write_grid(tmp_path, "b", {"window": [5, 10, 20]}, [0.1, -0.2, 0.0])
    trials = collect_trials(tmp_path, ["a", "b"])
    assert len(trials) == 5  # 2 + 3, not the size of either grid alone
    assert list(trials["strategy"]) == ["a", "a", "b", "b", "b"]
    assert trials["params"].iloc[0] == "short=10, long=50"
    assert trials["sharpe"].tolist() == [0.5, 0.3, 0.1, -0.2, 0.0]
    with pytest.raises(ValueError, match="no trials"):
        collect_trials(tmp_path, [])


def test_dsr_row_agrees_with_the_returns_based_functions() -> None:
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0004, 0.01, 2500), index=pd.bdate_range("2010-01-01", periods=2500))
    row = dsr_row(r, 63, 0.15)
    assert row["sharpe"] == pytest.approx(metrics.sharpe(r), rel=1e-12)
    assert row["psr"] == pytest.approx(probabilistic_sharpe_ratio(r), rel=1e-12)
    assert row["dsr"] == pytest.approx(
        deflated_sharpe_ratio(r, 63, var_trials_sharpe_annual=0.15), rel=1e-12
    )
    assert row["n_obs"] == 2500
    assert 0 < row["sr0_annual"] < 1.5

    sens = dsr_sensitivity(r, {"one": 1, "many": 63}, {"small": 0.01, "large": 0.15})
    assert len(sens) == 4
    by = sens.set_index(["n_assumption", "var_assumption"])["dsr"]
    # one trial deflates nothing, whatever the variance
    assert by[("one", "small")] == pytest.approx(row["psr"])
    assert by[("one", "large")] == pytest.approx(row["psr"])
    # more trials, or more dispersed trials, can only raise the hurdle
    assert by[("many", "large")] < by[("many", "small")] < by[("one", "small")]
