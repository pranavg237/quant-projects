"""Walk-forward and bootstrap tests. The central property: parameters chosen in a fold
depend only on that fold's training window."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics
from quantbt.data import PriceData, synthetic_prices
from quantbt.execution import ExecutionSimulator, NoCommission, NoSlippage
from quantbt.strategies import MACrossover
from quantbt.strategy import Context, Strategy
from quantbt.validation import (
    bootstrap_metrics,
    deflated_sharpe_from_stats,
    deflated_sharpe_ratio,
    excess_returns,
    expand_grid,
    expected_max_sharpe,
    grid_search,
    min_track_record_length,
    overfit_report,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    psr_from_stats,
    sharpe_std_error,
    stationary_bootstrap,
    walk_forward,
)
from quantbt.validation.walkforward import OBJECTIVES, sharpe_objective

NO_COST = {"execution": ExecutionSimulator(NoCommission(), NoSlippage())}


class Constant(Strategy):
    """Holds a fixed weight; used to make the 'best parameter' depend on the window."""

    name = "constant"

    def __init__(self, weight: float) -> None:
        self.weight = weight
        self._sent = False

    def on_bar(self, ctx: Context) -> None:
        if not self._sent:
            ctx.order_target_weight("SYN", self.weight)
            self._sent = True


def _regime_data() -> PriceData:
    """Up-trend for 3 years, then down-trend for 3 years, then up again (no noise)."""
    n = 252 * 6
    index = pd.bdate_range("2010-01-01", periods=n)
    drift = np.where((np.arange(n) // (252 * 3)) % 2 == 0, 0.0005, -0.0005)
    price = 100 * np.exp(np.cumsum(drift))
    frame = pd.DataFrame(
        {"Open": price, "High": price * 1.001, "Low": price * 0.999, "Close": price, "Volume": 1.0},
        index=index,
    )
    return PriceData.from_frames({"SYN": frame})


def test_expand_grid_and_constraint() -> None:
    combos = expand_grid({"a": [1, 2], "b": [10, 20]}, constraint=lambda c: c["a"] < c["b"] / 10)
    assert combos == [{"a": 1, "b": 20}]
    with pytest.raises(ValueError):
        expand_grid({"a": [1]}, constraint=lambda c: False)


def test_grid_search_ranks_and_returns_matrix(random_walk: PriceData) -> None:
    gr = grid_search(
        random_walk,
        lambda p: MACrossover("SYN", p["short"], p["long"]),
        {"short": [5, 10], "long": [20, 40]},
        start=random_walk.index[100],
        end=random_walk.index[400],
        objective="total_return",
        run_kwargs=NO_COST,
    )
    assert len(gr.combos) == 4 and gr.returns.shape == (301, 4)
    assert gr.table["objective"].iloc[gr.best_index] == gr.table["objective"].max()
    assert set(gr.best_params) == {"short", "long"}
    with pytest.raises(ValueError, match="objective"):
        grid_search(random_walk, lambda p: MACrossover("SYN"), {"x": [1]}, objective="nope")
    for name, fn in OBJECTIVES.items():
        assert callable(fn), name


def test_walk_forward_chooses_on_train_only() -> None:
    """In an up/down/up regime series the in-sample best weight on a training window is
    the *wrong* one for the following test window. Walk-forward must pick the train-best
    (and therefore lose out of sample), never the test-best."""
    data = _regime_data()
    factory = lambda p: Constant(p["weight"])  # noqa: E731
    wf = walk_forward(
        data,
        factory,
        {"weight": [-1.0, 1.0]},
        start=data.index[0],
        train_years=3,
        test_years=3,
        objective="total_return",
        run_kwargs=NO_COST,
    )
    assert len(wf.folds) == 1
    fold = wf.folds[0]
    assert fold.params == {"weight": 1.0}  # up-trend in training
    assert fold.train_objective > 0 and fold.test_objective < 0  # ... and it loses in the test
    assert fold.test_start > fold.train_end
    assert fold.train_end - fold.train_start >= pd.Timedelta(days=1000)
    # the stitched OOS series covers exactly the test window
    assert wf.oos_returns.index[0] >= fold.test_start and wf.oos_returns.index[-1] <= fold.test_end
    assert not wf.oos_returns.index.has_duplicates
    summary = wf.summary()
    assert {"out_of_sample", "in_sample_best"} == set(summary.columns)
    total = summary.loc["total_return"].astype(float)
    assert total["out_of_sample"] < 0
    assert total["in_sample_best"] > 0  # the in-sample best knows the answer
    table = wf.fold_table()
    assert list(table["weight"]) == [1.0]
    assert wf.parameter_stability.shape == (1, 1)
    assert wf.objective_name == "total_return"


def _ohlcv(data: PriceData, symbol: str = "SYN") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": data.open[symbol],
            "High": data.high[symbol],
            "Low": data.low[symbol],
            "Close": data.close[symbol],
            "Volume": data.volume[symbol],
        }
    )


def test_walk_forward_params_are_invariant_to_future_data(random_walk: PriceData) -> None:
    """The direct no-look-ahead test for parameter selection.

    Run walk-forward, then replace every bar after fold ``k``'s training window with a
    completely different price path (a strong trend from another seed) and run it again.
    If selection only uses data up to ``train_end``, folds 0..k must choose exactly the
    same parameters with exactly the same training scores, and the out-of-sample returns
    before the splice must be bit-identical. Fold k's *test* score must change, which
    proves the splice reached the data the strategy is scored on.
    """
    factory = lambda p: MACrossover("SYN", p["short"], p["long"])  # noqa: E731
    grid: dict[str, list[Any]] = {"short": [5, 10, 20], "long": [30, 60]}
    common: dict[str, Any] = {
        "start": random_walk.index[80],
        "train_years": 1.5,
        "test_years": 0.5,
        "run_kwargs": NO_COST,
        "constraint": lambda p: p["short"] < p["long"],
    }
    base = walk_forward(random_walk, factory, grid, **common)
    k = 2
    assert len(base.folds) > k + 1
    cut = base.folds[k].train_end

    other = _ohlcv(synthetic_prices(1500, drift=0.002, vol=0.01, gap_vol=0.005, seed=7))
    original = _ohlcv(random_walk)
    after = original.index > cut
    scale = original["Close"][~after].iloc[-1] / other["Close"][~after].iloc[-1]
    spliced = original.copy()
    for column in ("Open", "High", "Low", "Close"):
        spliced.loc[after, column] = other.loc[after, column] * scale
    changed = walk_forward(PriceData.from_frames({"SYN": spliced}), factory, grid, **common)

    for i in range(k + 1):
        before, after_fold = base.folds[i], changed.folds[i]
        assert before.params == after_fold.params, f"fold {i} parameters used future data"
        assert before.train_objective == after_fold.train_objective
        pd.testing.assert_frame_equal(before.train_table, after_fold.train_table)
    for i in range(k):
        assert base.folds[i].test_objective == changed.folds[i].test_objective
    pd.testing.assert_series_equal(base.oos_returns[:cut], changed.oos_returns[:cut])
    # the splice did reach the scored data: fold k's test window now looks different
    assert base.folds[k].test_objective != changed.folds[k].test_objective


def test_walk_forward_rolling_vs_anchored(random_walk: PriceData) -> None:
    factory = lambda p: MACrossover("SYN", p["short"], p["long"])  # noqa: E731
    grid: dict[str, list[Any]] = {"short": [5, 10], "long": [30]}
    common = {
        "start": random_walk.index[60],
        "train_years": 1.5,
        "test_years": 1.0,
        "run_kwargs": NO_COST,
        "objective": sharpe_objective,
    }
    rolling = walk_forward(random_walk, factory, grid, anchored=False, **common)  # type: ignore[arg-type]
    anchored = walk_forward(random_walk, factory, grid, anchored=True, **common)  # type: ignore[arg-type]
    assert len(rolling.folds) == len(anchored.folds) >= 3
    assert all(f.train_start == anchored.folds[0].train_start for f in anchored.folds)
    starts = [f.train_start for f in rolling.folds]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)
    # consecutive OOS windows tile the period without gaps or overlap
    idx = rolling.oos_returns.index
    assert idx.is_monotonic_increasing and not idx.has_duplicates
    assert rolling.objective_name == "sharpe_objective"


def test_walk_forward_argument_errors(random_walk: PriceData) -> None:
    factory = lambda p: MACrossover("SYN")  # noqa: E731
    with pytest.raises(ValueError, match="before end"):
        walk_forward(
            random_walk, factory, {"x": [1]}, start=random_walk.index[-1], end=random_walk.index[0]
        )
    with pytest.raises(ValueError, match="training window"):
        walk_forward(random_walk, factory, {"x": [1]}, start=random_walk.index[-30], train_years=5)


# --- bootstrap / overfitting ------------------------------------------------------------------


def _normal_returns(n: int, mean: float, sd: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n), index=pd.bdate_range("2010-01-01", periods=n))


def test_sharpe_standard_error_matches_lo_for_normal_returns() -> None:
    r = _normal_returns(5000, 0.0004, 0.01, 1)
    sr = float(r.mean() / r.std(ddof=1))
    approx = np.sqrt((1 + sr**2 / 2) / (len(r) - 1))
    assert sharpe_std_error(r) == pytest.approx(approx, rel=0.05)
    assert sharpe_std_error(r.iloc[:3]) > 0


def test_psr_and_min_track_record() -> None:
    strong = _normal_returns(2000, 0.001, 0.01, 3)  # Sharpe ~1.6
    noise = _normal_returns(2000, 0.0, 0.01, 12)
    assert probabilistic_sharpe_ratio(strong) > 0.99
    assert probabilistic_sharpe_ratio(noise) < 0.9
    assert min_track_record_length(strong) < 2000
    assert min_track_record_length(noise - noise.mean() - 1e-3) == float("inf")
    assert np.isnan(probabilistic_sharpe_ratio(pd.Series([0.0, 0.0, 0.0, 0.0])))
    assert np.isnan(min_track_record_length(pd.Series([0.0, 0.0, 0.0, 0.0])))


def test_deflated_sharpe_penalises_many_trials() -> None:
    r = _normal_returns(1000, 0.0006, 0.01, 4)
    psr = probabilistic_sharpe_ratio(r)
    few = deflated_sharpe_ratio(r, n_trials=2, var_trials_sharpe_annual=0.25)
    many = deflated_sharpe_ratio(r, n_trials=1000, var_trials_sharpe_annual=0.25)
    assert psr > few > many
    trials = np.array([0.1, 0.5, -0.2, 0.8, 0.3, np.nan])
    from_trials = deflated_sharpe_ratio(r, n_trials=5, trials_sharpe_annual=trials)
    assert 0 <= from_trials <= 1
    assert deflated_sharpe_ratio(r, n_trials=1, var_trials_sharpe_annual=1.0) == pytest.approx(psr)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(r, n_trials=5)


def test_deflated_sharpe_matches_bailey_lopez_de_prado_worked_example() -> None:
    """The numerical example in Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".

    A strategy with an annualised Sharpe of 2.5 over five years of daily data (T = 1250),
    skew -3 and kurtosis 10, selected as the best of N = 100 trials whose annualised
    Sharpe ratios have variance 0.5. Every intermediate number below was computed by hand
    from the paper's equations (1) and (2), not by the code under test:

    * per-period variance of trial Sharpes: V = 0.5 / 250 = 0.002
    * Phi^-1(1 - 1/100) = 2.326348, Phi^-1(1 - 1/(100 e)) = 2.680210
    * SR0 = sqrt(0.002) * (0.422784 * 2.326348 + 0.577216 * 2.680210) = 0.113172
    * SR  = 2.5 / sqrt(250) = 0.158114
    * z   = (0.158114 - 0.113172) * sqrt(1249) / sqrt(1 + 3 * 0.158114 + 9/4 * 0.158114^2)
          = 0.044942 * 35.341194 / 1.237171 = 1.28382
    * DSR = Phi(1.28382) = 0.9004 (the paper reports 0.9004)
    """
    sr0 = expected_max_sharpe(100, 0.5 / 250)
    assert sr0 == pytest.approx(0.113172, abs=2e-6)
    dsr = deflated_sharpe_from_stats(
        sr=2.5 / np.sqrt(250),
        n_obs=1250,
        skew=-3.0,
        kurtosis=10.0,
        n_trials=100,
        var_trials_sharpe=0.5 / 250,
    )
    assert dsr == pytest.approx(0.9004, abs=5e-5)
    # the same inputs against zero instead of SR0 is the PSR, and must be higher
    psr = psr_from_stats(2.5 / np.sqrt(250), 0.0, 1250, -3.0, 10.0)
    assert psr > dsr
    # normal returns (skew 0, kurtosis 3), SR 0.05, T 1001, benchmark 0:
    # z = 0.05 * sqrt(1000) / sqrt(1 + (3 - 1)/4 * 0.05^2) = 1.581139 / 1.000625 = 1.580152
    # and Phi(1.580152) = 0.942964
    assert psr_from_stats(0.05, 0.0, 1001, 0.0, 3.0) == pytest.approx(0.942964, abs=1e-5)
    assert np.isnan(psr_from_stats(0.05, 0.0, 1, 0.0, 3.0))


def test_deflated_sharpe_from_returns_agrees_with_the_stats_formula() -> None:
    """The returns-based DSR uses per-period Sharpe, population skew and raw kurtosis."""
    rng = np.random.default_rng(9)
    raw = rng.standard_t(df=5, size=1500) * 0.01 + 0.0006
    r = pd.Series(raw, index=pd.bdate_range("2012-01-02", periods=1500))
    ppy = metrics.periods_per_year(r.index)
    mu, sd = raw.mean(), raw.std(ddof=1)
    dev = raw - mu
    m2 = (dev**2).mean()
    skew = (dev**3).mean() / m2**1.5
    kurt = (dev**4).mean() / m2**2
    expected = deflated_sharpe_from_stats(mu / sd, 1500, skew, kurt, 63, 0.09 / ppy)
    assert deflated_sharpe_ratio(r, 63, var_trials_sharpe_annual=0.09) == pytest.approx(
        expected, rel=1e-12
    )
    # one trial deflates nothing: the DSR collapses to the PSR
    assert deflated_sharpe_from_stats(mu / sd, 1500, skew, kurt, 1, 0.09) == pytest.approx(
        probabilistic_sharpe_ratio(r), rel=1e-12
    )


def test_stationary_bootstrap_ci_contains_truth() -> None:
    r = _normal_returns(1500, 0.0005, 0.01, 5)
    res = stationary_bootstrap(r, n_samples=400, seed=1)
    lo, hi = res.ci("sharpe")
    true_sharpe = 0.0005 / 0.01 * np.sqrt(252)
    assert lo < true_sharpe < hi
    assert lo < res.observed["sharpe"] < hi
    assert 0 <= res.p_value("sharpe") <= 1
    assert res.samples.shape == (400, 3)
    assert res.block_len == pytest.approx(1500 ** (1 / 3))
    table = bootstrap_metrics(r, n_samples=50, seed=2)
    assert list(table.index) == ["sharpe", "cagr", "max_drawdown"]
    assert "q50" in table.columns and "observed" in table.columns
    with pytest.raises(ValueError):
        stationary_bootstrap(r.iloc[:5])
    explicit = stationary_bootstrap(r, n_samples=10, block_len=20, seed=3)
    assert explicit.block_len == 20


def test_pbo_high_on_noise_low_on_signal() -> None:
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2010-01-01", periods=2000)
    noise = pd.DataFrame(rng.normal(0, 0.01, (2000, 30)), index=idx)
    out = pbo_cscv(noise, n_blocks=8)
    assert 0.3 <= out["pbo"] <= 0.8  # about a coin flip for pure noise
    assert out["n_splits"] == 70
    signal = noise.copy()
    signal[0] = signal[0] + 0.002  # one configuration has a real edge
    out2 = pbo_cscv(signal, n_blocks=8)
    assert out2["pbo"] < 0.1
    with pytest.raises(ValueError):
        pbo_cscv(noise, n_blocks=7)
    with pytest.raises(ValueError):
        pbo_cscv(noise.iloc[:, :1])
    with pytest.raises(ValueError):
        pbo_cscv(noise.iloc[:10], n_blocks=8)


def test_overfit_report_flags() -> None:
    rng = np.random.default_rng(8)
    idx = pd.bdate_range("2012-01-01", periods=1500)
    grid = pd.DataFrame(rng.normal(0, 0.01, (1500, 20)), index=idx)
    sharpes = grid.apply(lambda c: metrics.sharpe(c))
    best = grid[sharpes.idxmax()]
    rep = overfit_report(
        best, trials_sharpe_annual=sharpes.to_numpy(), grid_returns=grid, n_samples=100
    )
    assert rep.n_trials == 20
    assert rep.dsr is not None and rep.dsr < rep.psr
    assert (
        any("DSR" in f for f in rep.flags)
        or any("PSR" in f for f in rep.flags)
        or any("PBO" in f for f in rep.flags)
    )
    s = rep.to_series()
    assert "flags" in s.index and s["n_trials"] == 20
    good = _normal_returns(1500, 0.0015, 0.01, 9)
    rep2 = overfit_report(good, n_samples=100)
    assert rep2.dsr is None and rep2.pbo is None and rep2.flags == []
    assert rep2.to_series()["flags"] == "none"
    bad = -good
    rep3 = overfit_report(bad, n_trials=3, n_samples=50)
    assert "non-positive Sharpe" in rep3.flags


def test_overfit_report_uses_excess_returns() -> None:
    """A cash-like series has a high raw Sharpe and a negative excess one.

    The two must not be mixed: the summary metrics are excess, so the diagnostics are too.
    """
    n = 2000
    rng = np.random.default_rng(31)
    idx = pd.bdate_range("2010-01-01", periods=n)
    rf = pd.Series(2e-5, index=idx)  # about 0.5% a year
    cash_like = rf + rng.normal(-2e-6, 1e-5, n)  # tracks cash, a hair below it on average
    assert metrics.sharpe(cash_like) > 5  # raw: almost no volatility, positive mean
    raw = overfit_report(cash_like, n_samples=50)
    excess = overfit_report(cash_like, rf=rf, n_samples=50)
    assert raw.sharpe > 5 and excess.sharpe < 0
    assert "non-positive Sharpe" in excess.flags
    assert excess.psr < 0.5
    # a float annual rate works the same way
    flat = overfit_report(cash_like, rf=0.005, n_samples=50)
    assert flat.sharpe < raw.sharpe
    assert excess_returns(cash_like, None) is cash_like
    # a constant series has no Sharpe at all, and must not crash the bootstrap
    flat_returns = pd.Series(0.0, index=idx)
    quiet = overfit_report(flat_returns, n_samples=20)
    assert not np.isfinite(quiet.sharpe) and "non-positive Sharpe" in quiet.flags


def test_pbo_uses_excess_returns_for_the_grid() -> None:
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2010-01-01", periods=1200)
    grid = pd.DataFrame(rng.normal(0, 0.01, (1200, 8)), index=idx)
    rf = pd.Series(1e-4, index=idx)
    rep = overfit_report(grid[0], rf=rf, grid_returns=grid, n_samples=50, n_trials=8)
    assert rep.pbo is not None and 0.0 <= rep.pbo <= 1.0
