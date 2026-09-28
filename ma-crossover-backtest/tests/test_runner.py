"""Tests for the research pipeline that produces every number in RESULTS.md.

``run_spec`` is the single code path behind the results table, so it is worth pinning
down: that the benchmark stays out of the tradable universe, that the persisted files
match the returned object, and that the in-sample/out-of-sample comparison is computed
over the same span with the same risk-free rate.

Everything here is offline: the downloader and the French factor loader are fakes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbt.data import PriceData
from quantbt.factors.french import MODELS
from quantbt.research import runner
from quantbt.research.runner import StrategySpec, load_rf, results_table, run_spec
from quantbt.research.universes import (
    ALL_SYMBOLS,
    ASSET_CLASS_ETFS,
    LARGE_CAP_STOCKS,
    PAIR_CANDIDATES,
    PAIR_SYMBOLS,
    SECTOR_ETFS,
)
from quantbt.strategies import MACrossover
from quantbt.strategy import Context, Strategy

from .conftest import yahoo_like_frame

FACTOR_START = "2009-01-01"
FACTOR_BARS = 4000


def fake_factors(
    model: str = "ff3",
    frequency: str = "daily",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    **_: Any,
) -> pd.DataFrame:
    """A French-library-shaped daily factor frame: MODELS[model] + RF, in decimals."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range(FACTOR_START, periods=FACTOR_BARS)
    cols = {f: rng.normal(0.0002, 0.008, FACTOR_BARS) for f in MODELS[model]}
    cols["RF"] = np.full(FACTOR_BARS, 0.00008)  # ~2% a year
    return pd.DataFrame(cols, index=index).loc[start:end]


def fake_loader(symbols: list[str], start: Any = None, end: Any = None, **_: Any) -> PriceData:
    """Yahoo-shaped frames for any requested symbols, deterministic per symbol."""
    frames = {
        s: yahoo_like_frame(n=FACTOR_BARS, start=FACTOR_START, seed=abs(hash(s)) % 1000)
        for s in symbols
    }
    return PriceData.from_frames(frames).slice(start, end)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "load_yahoo", fake_loader)
    monkeypatch.setattr(runner, "load_factors", fake_factors)


@pytest.fixture
def spec() -> StrategySpec:
    return StrategySpec(
        name="ma_test",
        symbols=["AAA"],
        factory=lambda p: MACrossover("AAA", p["short"], p["long"]),
        grid={"short": [5, 10], "long": [20, 40]},
        constraint=lambda p: p["short"] < p["long"],
        start="2011-01-01",
        benchmark="BENCH",
        description="test spec",
        caveats=["not a real strategy"],
    )


def run(spec: StrategySpec, tmp_path: Path) -> Any:
    return run_spec(
        spec,
        end="2018-12-31",
        out_root=tmp_path,
        train_years=2.0,
        test_years=1.0,
        warmup_years=1,
        n_bootstrap=50,
        log=lambda _msg: None,
    )


# -- the pipeline ---------------------------------------------------------------------


def test_run_spec_produces_a_complete_row(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    res = run(spec, tmp_path)
    expected = {
        "strategy",
        "oos_start",
        "oos_end",
        "folds",
        "n_symbols",
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "max_drawdown",
        "max_dd_days",
        "turnover",
        "exposure",
        "is_sharpe",
        "is_params",
        "bench_cagr",
        "bench_sharpe",
        "bench_max_dd",
        "psr",
        "boot_ci_lo",
        "boot_ci_hi",
        "pbo",
        "dsr_is",
        "ff5_alpha",
        "ff5_alpha_t",
        "ff5_r2",
        "beta_mkt",
        "beta_mom",
        "mom_t",
        "ff6_alpha",
        "ff6_alpha_t",
    }
    assert expected <= set(res.row.index)
    assert res.row["strategy"] == "ma_test"
    assert res.row["n_symbols"] == 1  # the benchmark is not counted as tradable
    assert res.row["folds"] >= 5
    assert 0.0 <= res.row["pbo"] <= 1.0
    assert 0.0 <= res.row["psr"] <= 1.0
    assert res.row["boot_ci_lo"] <= res.row["boot_ci_hi"]
    assert json.loads(str(res.row["is_params"])).keys() == {"short", "long"}


def test_run_spec_keeps_the_benchmark_out_of_the_tradable_universe(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    """A strategy must not be able to buy the thing it is measured against."""
    res = run(spec, tmp_path)
    assert list(res.oos_weights.columns) == ["AAA", "BENCH"]  # BENCH is loaded, for marking
    assert (res.oos_weights["BENCH"] == 0.0).all()  # ... but never held
    assert res.benchmark_returns.abs().sum() > 0  # and its returns are still measured


def test_a_strategy_cannot_order_a_symbol_outside_the_universe(
    offline: None, tmp_path: Path
) -> None:
    class BuysTheBenchmark(Strategy):
        name = "cheater"

        def on_bar(self, ctx: Context) -> None:
            ctx.order_target_weight("BENCH", 1.0)

    cheating = StrategySpec(
        name="cheater",
        symbols=["AAA"],
        factory=lambda _p: BuysTheBenchmark(),
        grid={"unused": [1]},
        start="2011-01-01",
        benchmark="BENCH",
    )
    with pytest.raises(KeyError, match="survivorship bias"):
        run(cheating, tmp_path)


def test_run_spec_persists_every_artefact(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    res = run(spec, tmp_path)
    out = tmp_path / "ma_test"
    assert res.out_dir == out
    for name in (
        "oos_returns.csv",
        "oos_weights.csv",
        "benchmark_returns.csv",
        "rf.csv",
        "folds.csv",
        "grid.csv",
        "factors.csv",
        "overfit_is.csv",
        "overfit_oos.csv",
        "oos_vs_is.csv",
        "summary.json",
        "spec.json",
    ):
        assert (out / name).exists(), name

    # the persisted returns are the returns that were reported on
    saved = pd.read_csv(out / "oos_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
    pd.testing.assert_series_equal(saved, res.oos_returns, check_names=False, check_freq=False)

    written = json.loads((out / "summary.json").read_text())
    assert written["sharpe"] == pytest.approx(float(res.row["sharpe"]))

    saved_spec = json.loads((out / "spec.json").read_text())
    assert saved_spec["caveats"] == ["not a real strategy"]
    assert saved_spec["benchmark"] == "BENCH"
    assert saved_spec["train_years"] == 2.0


def test_in_sample_and_out_of_sample_cover_the_same_span(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    """``is_sharpe`` is the hindsight-best parameter set over the *same* dates as the
    stitched OOS series -- otherwise the overfitting gap would just be two different
    market regimes."""
    res = run(spec, tmp_path)
    comparison = pd.read_csv(res.out_dir / "oos_vs_is.csv", index_col=0)
    assert comparison.loc["start", "out_of_sample"] == comparison.loc["start", "in_sample_best"]
    assert comparison.loc["end", "out_of_sample"] == comparison.loc["end", "in_sample_best"]
    assert comparison.loc["periods", "out_of_sample"] == comparison.loc["periods", "in_sample_best"]


def test_folds_are_contiguous_and_never_overlap(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    res = run(spec, tmp_path)
    folds = res.folds
    assert (pd.to_datetime(folds["train_end"]) < pd.to_datetime(folds["test_start"])).all()
    starts = pd.to_datetime(folds["test_start"])
    ends = pd.to_datetime(folds["test_end"])
    assert (starts.iloc[1:].to_numpy() > ends.iloc[:-1].to_numpy()).all()
    assert not res.oos_returns.index.has_duplicates


def test_rf_is_the_same_series_everywhere(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    """Every Sharpe in the row must be an excess Sharpe against the same rate."""
    res = run(spec, tmp_path)
    assert res.rf.index.equals(res.oos_returns.index)
    assert res.rf.to_numpy() == pytest.approx(0.00008)


def test_load_rf_returns_a_daily_decimal_rate(offline: None) -> None:
    rf = load_rf("2012-01-01", "2012-12-31")
    assert rf.name == "RF"
    assert 0.0 <= float(rf.mean()) < 0.001
    assert rf.index.min() >= pd.Timestamp("2012-01-01")


def test_results_table_indexes_by_strategy(
    offline: None, spec: StrategySpec, tmp_path: Path
) -> None:
    res = run(spec, tmp_path)
    other = res.row.copy()
    other["strategy"] = "second"
    other["sharpe"] = np.nan
    table = results_table([res.row, other])
    assert list(table.index) == ["ma_test", "second"]
    assert table.loc["second", "sharpe"] is None  # NaN -> None for clean markdown


# -- the hand-written universes -------------------------------------------------------


def test_universes_are_unique_and_consistent() -> None:
    for name, symbols in (
        ("etfs", ASSET_CLASS_ETFS),
        ("sectors", SECTOR_ETFS),
        ("stocks", LARGE_CAP_STOCKS),
        ("pairs", PAIR_SYMBOLS),
    ):
        assert len(symbols) == len(set(symbols)), f"{name} has duplicates"
    assert all(a != b for a, b in PAIR_CANDIDATES)
    assert sorted({s for pair in PAIR_CANDIDATES for s in pair}) == PAIR_SYMBOLS
    assert set(ALL_SYMBOLS) == (
        set(ASSET_CLASS_ETFS) | set(SECTOR_ETFS) | set(LARGE_CAP_STOCKS) | set(PAIR_SYMBOLS)
    )
    assert "WBA" not in ALL_SYMBOLS  # delisted; see the note in universes.py
