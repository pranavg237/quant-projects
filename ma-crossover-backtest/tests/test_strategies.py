"""Behavioural tests for the built-in strategies on constructed data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbt.data import PriceData, synthetic_prices
from quantbt.engine import run_backtest
from quantbt.execution import ExecutionSimulator, NoCommission, NoSlippage
from quantbt.strategies import (
    CrossSectionalMomentum,
    MeanReversion,
    PairsTrading,
    TimeSeriesMomentum,
    fit_pair,
)
from quantbt.strategies._common import is_rebalance_bar, realized_vol
from quantbt.strategy import Context, Strategy

NO_COST = ExecutionSimulator(NoCommission(), NoSlippage())


def _frame(price: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": price, "High": price * 1.001, "Low": price * 0.999, "Close": price, "Volume": 1e6},
        index=index,
    )


def _trend_universe(n: int = 700, seed: int = 0) -> PriceData:
    """UP drifts up, DOWN drifts down, FLAT is noise: momentum signs are unambiguous."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n)
    up = 100 * np.exp(np.cumsum(rng.normal(0.0015, 0.008, n)))
    down = 100 * np.exp(np.cumsum(rng.normal(-0.0015, 0.008, n)))
    flat = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, n)))
    return PriceData.from_frames(
        {"UP": _frame(up, index), "DOWN": _frame(down, index), "FLAT": _frame(flat, index)}
    )


def test_rebalance_helpers(random_walk: PriceData) -> None:
    seen: list[tuple[bool, bool, bool]] = []

    class Probe(Strategy):
        name = "probe"

        def on_bar(self, ctx: Context) -> None:
            seen.append(
                (
                    is_rebalance_bar(ctx, "daily"),
                    is_rebalance_bar(ctx, "monthly"),
                    is_rebalance_bar(ctx, 5),
                )
            )
            if ctx.position_index == 30:
                with pytest.raises(ValueError):
                    is_rebalance_bar(ctx, "weekly")
                raise StopIteration

    with pytest.raises(StopIteration):
        run_backtest(random_walk, Probe())
    assert all(d for d, _, _ in seen)
    months = sum(m for _, m, _ in seen)
    assert 1 <= months <= 3  # first bar plus one or two month boundaries in 31 bars
    assert sum(f for _, _, f in seen) == 7  # bars 0,5,...,30
    vol = realized_vol(random_walk.close, 60)
    assert vol["SYN"] > 0


def test_tsmom_long_up_short_down() -> None:
    data = _trend_universe()
    strat = TimeSeriesMomentum(lookback=126, skip=10, vol_lookback=40, vol_target=0.10)
    res = run_backtest(data, strat, start=data.index[200], execution=NO_COST)
    w = res.weights.iloc[-1]
    assert w["UP"] > 0 and w["DOWN"] < 0
    # weights are marked to market, so they drift between monthly rebalances
    assert res.weights.abs().sum(axis=1).max() <= strat.max_gross * 1.1
    assert res.records["n_eligible"].dropna().iloc[-1] == 3
    assert res.summary()["sharpe"] > 1.0  # trends are by construction persistent
    lo = TimeSeriesMomentum(lookback=126, skip=10, vol_lookback=40, long_only=True)
    res_lo = run_backtest(data, lo, start=data.index[200], execution=NO_COST)
    assert (res_lo.weights >= -1e-12).all().all()
    with pytest.raises(ValueError):
        TimeSeriesMomentum(lookback=10, skip=20)


def test_tsmom_caps_weights_and_gross() -> None:
    data = _trend_universe()
    strat = TimeSeriesMomentum(
        lookback=126, skip=0, vol_lookback=40, vol_target=1.0, max_weight=0.3, max_gross=0.5
    )
    res = run_backtest(data, strat, start=data.index[200], execution=NO_COST)
    assert res.weights.abs().max().max() <= 0.3 * 1.1
    assert res.weights.abs().sum(axis=1).max() <= 0.5 * 1.1


def test_xsmom_ranks_and_is_dollar_neutral() -> None:
    data = _trend_universe()
    strat = CrossSectionalMomentum(lookback=126, skip=10, top_frac=0.34, min_names=3)
    res = run_backtest(data, strat, start=data.index[200], execution=NO_COST)
    w = res.weights.iloc[-1]
    assert w["UP"] > 0 and w["DOWN"] < 0 and abs(w["FLAT"]) < 1e-9
    # dollar neutral on the bar after each rebalance (weights drift with prices after that)
    rebalance_bars = res.records["n_ranked"].dropna().index
    positions = res.weights.index.get_indexer(rebalance_bars[:-1]) + 1
    assert abs(res.weights.iloc[positions].sum(axis=1)).max() < 0.02
    long_only = CrossSectionalMomentum(
        lookback=126, skip=10, top_frac=0.34, long_only=True, min_names=3
    )
    res_lo = run_backtest(data, long_only, start=data.index[200], execution=NO_COST)
    assert (res_lo.weights >= -1e-12).all().all() and res_lo.weights.iloc[-1]["UP"] > 0.9
    assert res.records["n_ranked"].dropna().iloc[-1] == 3
    with pytest.raises(ValueError):
        CrossSectionalMomentum(top_frac=0.9)


def test_xsmom_ignores_names_without_history() -> None:
    a = _trend_universe(700)
    late = synthetic_prices(100, symbols=("LATE",), start=str(a.index[-100].date()))
    frames = a.to_frames()
    frames["LATE"] = late.to_frames()["LATE"]
    data = PriceData.from_frames(frames)
    strat = CrossSectionalMomentum(lookback=126, skip=10, top_frac=0.34, min_names=3)
    res = run_backtest(data, strat, start=data.index[200], execution=NO_COST)
    assert (res.positions["LATE"] == 0).all()
    assert res.records["n_ranked"].max() == 3


def test_mean_reversion_buys_dips_and_exits_on_recovery() -> None:
    n = 120
    index = pd.bdate_range("2020-01-01", periods=n)
    price = np.full(n, 100.0) + np.sin(np.arange(n) / 3.0) * 0.5  # small oscillation
    price[60] = 90.0  # a one-day crash: z << -2
    price[61:66] = 95.0  # partial recovery
    data = PriceData.from_frames({"A": _frame(price, index)})
    strat = MeanReversion(window=20, entry_z=2.0, exit_z=0.0, max_positions=1)
    res = run_backtest(data, strat, execution=NO_COST)
    pos = res.positions["A"]
    entry = pos[pos > 0].index[0]
    assert entry == index[61]  # signal at the close of the crash bar, filled next open
    assert float(res.records["min_z"].loc[index[60]]) < -2
    assert pos.loc[index[80]] == 0  # exited once the z-score recovered
    assert len(res.round_trips) >= 1 and res.round_trips[0].pnl > 0
    with pytest.raises(ValueError):
        MeanReversion(entry_z=1.0, exit_z=1.0)


def test_mean_reversion_respects_max_positions_and_max_hold() -> None:
    n = 100
    index = pd.bdate_range("2020-01-01", periods=n)
    frames = {}
    for k, s in enumerate(["A", "B", "C"]):
        price = np.full(n, 100.0) + np.cos(np.arange(n) / 2.0 + k) * 0.3
        price[50:] = 80.0 - k  # everyone crashes and stays down (no recovery)
        frames[s] = _frame(price, index)
    data = PriceData.from_frames(frames)
    strat = MeanReversion(window=20, entry_z=2.0, exit_z=0.0, max_positions=2, max_hold=5)
    res = run_backtest(data, strat, execution=NO_COST)
    assert (res.positions.gt(0).sum(axis=1) <= 2).all()
    assert res.records["n_positions"].max() == 2
    # with max_hold the positions are forced out after 5 bars even though z stays negative
    held = res.positions.gt(0).any(axis=1)
    assert held.loc[index[52]] and not held.loc[index[70]]
    short = MeanReversion(window=20, entry_z=2.0, exit_z=0.0, max_positions=2, long_only=False)
    frames_up = {
        s: f.assign(**{c: 200 - f[c] for c in ("Open", "High", "Low", "Close")})
        for s, f in frames.items()
    }
    for f in frames_up.values():
        f[["High", "Low"]] = f[["Low", "High"]].to_numpy()
    data_up = PriceData.from_frames(frames_up)
    res_s = run_backtest(data_up, short, execution=NO_COST)
    assert (res_s.weights < 0).any().any()


def _cointegrated_pair(n: int = 600, seed: int = 1) -> PriceData:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-01", periods=n)
    log_b = np.log(50.0) + np.cumsum(rng.normal(0, 0.01, n))
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 0.01)  # mean-reverting
    log_a = np.log(2.0) + 1.0 * log_b + spread
    unrelated = np.log(30.0) + np.cumsum(rng.normal(0, 0.01, n))
    return PriceData.from_frames(
        {
            "A": _frame(np.exp(log_a), index),
            "B": _frame(np.exp(log_b), index),
            "U": _frame(np.exp(unrelated), index),
        }
    )


def test_fit_pair_detects_cointegration() -> None:
    data = _cointegrated_pair()
    la, lb, lu = (np.log(data.close[s]) for s in ("A", "B", "U"))
    good = fit_pair(la, lb, "A", "B")
    assert good.pvalue < 0.05 and good.hedge == pytest.approx(1.0, abs=0.1)
    bad = fit_pair(la, lu, "A", "U")
    assert bad.pvalue > good.pvalue
    z = good.zscore(float(la.iloc[-1]), float(lb.iloc[-1]))
    assert np.isfinite(z)


def test_pairs_trades_the_spread_and_stays_dollar_neutral() -> None:
    data = _cointegrated_pair()
    strat = PairsTrading(
        [("A", "B"), ("A", "U")],
        formation=252,
        refit_every=63,
        pvalue=0.05,
        entry_z=1.5,
        exit_z=0.3,
    )
    res = run_backtest(data, strat, start=data.index[252], execution=NO_COST)
    assert strat.models and {(m.a, m.b) for m in strat.models} == {("A", "B")}
    assert (res.positions["U"] == 0).all()
    active = res.weights[res.weights.abs().sum(axis=1) > 0]
    assert len(active) > 10
    assert (
        abs(active.sum(axis=1)).max() < 0.05
    )  # long one leg, short the other (drifts intra-trade)
    assert len(res.round_trips) >= 2
    assert "z_A_B" in res.records.columns and res.records["n_pairs"].max() == 1
    with pytest.raises(ValueError):
        PairsTrading([("A", "B")], entry_z=1.0, exit_z=2.0)


def test_pairs_skips_missing_symbols_and_handles_stops() -> None:
    data = _cointegrated_pair()
    strat = PairsTrading(
        [("A", "B"), ("A", "NOPE")],
        formation=252,
        refit_every=10,
        entry_z=0.1,
        exit_z=0.05,
        stop_z=0.2,
    )
    uni_data = data  # NOPE is not in the data: the pair must be skipped, not crash
    res = run_backtest(uni_data, strat, start=data.index[252], execution=NO_COST)
    assert res.records["n_pairs"].max() == 1
    assert res.records["n_open"].max() <= 1
