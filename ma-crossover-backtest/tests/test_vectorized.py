"""Tests for the corrected vectorised backtest, including the bias detectors.

Each ``test_bias_*`` test constructs data on which a specific bug would produce an
impossible result and asserts the fixed code does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics as m
from quantbt.data import PriceData, synthetic_prices
from quantbt.vectorized import backtest_long_flat, backtest_ma_crossover, ma_crossover_signal


def _jump_data(jump_day: int = 50, n: int = 100, size: float = 0.10) -> PriceData:
    """Flat at 100 except a gap up of ``size`` at the *open* of ``jump_day`` that persists."""
    index = pd.bdate_range("2021-01-01", periods=n)
    price = np.full(n, 100.0)
    price[jump_day:] = 100.0 * (1 + size)
    frame = pd.DataFrame(
        {"Open": price, "High": price, "Low": price, "Close": price, "Volume": 1.0}, index=index
    )
    return PriceData.from_frames({"J": frame})


def test_signal_basic() -> None:
    s = pd.Series(np.arange(1, 21, dtype=float), index=pd.bdate_range("2020-01-01", periods=20))
    sig = ma_crossover_signal(s, 2, 5)
    assert sig.iloc[:4].tolist() == [0, 0, 0, 0]  # warm-up is flat
    assert sig.iloc[4:].tolist() == [1] * 16  # rising series: fast > slow
    with pytest.raises(ValueError):
        ma_crossover_signal(s, 5, 2)


def test_bias_same_bar_fill_cannot_capture_signal_bar_gap() -> None:
    """A signal fired at the close of day t-1 must not earn the gap at the open of day t.

    The original script multiplied return[t] by signal[t-1], i.e. it entered at close[t-1].
    With a jump at the open of day t, that would book +10%. A next-open fill books nothing.
    """
    data = _jump_data(jump_day=50)
    signal = pd.Series(0, index=data.index)
    signal.iloc[49:] = 1  # "buy" decided at the close of day 49 (the bar before the jump)
    res = backtest_long_flat(data, signal, slippage_bps=0, commission_bps=0)
    assert res.frame["position"].iloc[50] == 1  # in the market from the open of day 50
    assert res.returns.iloc[50] == pytest.approx(0.0)  # ... but filled at the post-gap open
    assert res.equity.iloc[-1] == pytest.approx(1.0)
    # The buggy alignment would have booked the whole gap:
    buggy = (data.single("close").pct_change() * signal.shift(1)).fillna(0)
    assert (1 + buggy).prod() == pytest.approx(1.10)


def test_bias_perfect_foresight_signal_earns_nothing_with_correct_fills() -> None:
    """A signal that knows the overnight gap (close[t] -> open[t+1]) is pure lookahead.

    Under the original alignment (return[t+1] * signal[t], i.e. filled at close[t]) it
    books every gap and produces an absurd Sharpe. With next-open fills the gap has
    already happened by the time the order fills, and the strategy is left holding a
    driftless intraday move, so the Sharpe collapses toward zero.
    """
    data = synthetic_prices(2000, drift=0.0, vol=0.01, gap_vol=0.01, seed=7)
    close = data.single("close")
    open_ = data.single("open")
    gap = open_.shift(-1) / close - 1.0  # known only in the future
    signal = (gap > 0).astype(int)
    buggy = (close.pct_change() * signal.shift(1)).dropna()
    assert m.sharpe(buggy) > 4  # impossible in reality
    res = backtest_long_flat(data, signal, slippage_bps=0, commission_bps=0)
    assert abs(m.sharpe(res.returns)) < 1.0


def test_bias_open_equals_close_makes_same_bar_fill_look_harmless() -> None:
    """Why the detector above needs overnight gaps: with open[t+1] == close[t] a next-open
    fill *is* a same-bar-close fill, so synthetic data without gaps cannot expose the bug."""
    data = synthetic_prices(500, drift=0.0, vol=0.01, gap_vol=0.0, seed=3)
    assert np.allclose(data.single("open").iloc[1:], data.single("close").iloc[:-1])


def test_bias_random_walk_has_no_edge_after_costs() -> None:
    """On a driftless random walk no MA crossover should show a large positive Sharpe."""
    data = synthetic_prices(3000, drift=0.0, vol=0.01, gap_vol=0.003, seed=11)
    res = backtest_ma_crossover(data, 20, 100, slippage_bps=5, commission_bps=1)
    assert m.sharpe(res.returns) < 1.0


def test_costs_reduce_equity_by_expected_amount() -> None:
    data = _jump_data(jump_day=1000, n=200)  # flat price forever (jump never happens)
    signal = pd.Series(0, index=data.index)
    signal.iloc[10:20] = 1  # one round trip
    res = backtest_long_flat(data, signal, slippage_bps=10, commission_bps=5)
    # buy at 100 * 1.001 plus 5 bps commission on the notional, sell at 100 * 0.999 less 5 bps
    expected = (100 / (100.1 * (1 + 5e-4))) * (0.999 * (1 - 5e-4))
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert res.trade_pnls.iloc[0] == pytest.approx(expected - 1, rel=1e-9)
    assert len(res.trade_pnls) == 1
    zero = backtest_long_flat(data, signal, slippage_bps=0, commission_bps=0)
    assert zero.equity.iloc[-1] == pytest.approx(1.0)


def test_alignment_fill_is_exactly_one_bar_after_signal(trending: PriceData) -> None:
    res = backtest_ma_crossover(trending, 10, 50, slippage_bps=0, commission_bps=0)
    f = res.frame
    first_signal = f.index[f["signal"] == 1][0]
    first_position = f.index[f["position"] == 1][0]
    assert int(f.index.get_loc(first_position)) == int(f.index.get_loc(first_signal)) + 1  # type: ignore[arg-type]
    # the fill price on the entry bar is that bar's open, not the previous close
    entry_open = trending.single("open").loc[first_position]
    assert f.loc[first_position, "fill_price"] == pytest.approx(entry_open)
    assert f.loc[first_position, "trade"] == 1


def test_warmup_window_does_not_touch_evaluation(trending: PriceData) -> None:
    """Evaluation starting mid-series has a defined signal from the first bar."""
    res = backtest_ma_crossover(
        trending, 10, 50, start="2020-08-03", slippage_bps=0, commission_bps=0
    )
    assert res.frame.index[0] == pd.Timestamp("2020-08-03")
    assert res.frame["benchmark_return"].iloc[0] == 0.0
    assert res.frame["position"].iloc[0] in (0, 1)
    assert res.frame["equity"].iloc[0] > 0
    with pytest.raises(ValueError):
        backtest_ma_crossover(trending, 10, 50, start="2030-01-01")


def test_rf_earned_only_when_flat() -> None:
    data = _jump_data(jump_day=1000, n=252)
    signal = pd.Series(0, index=data.index)
    res = backtest_long_flat(data, signal, rf=0.05, slippage_bps=0, commission_bps=0)
    assert res.equity.iloc[-1] == pytest.approx(1.05, rel=1e-9)  # 252 flat bars = one year
    rf_series = pd.Series(0.0001, index=data.index)
    res2 = backtest_long_flat(data, signal, rf=rf_series, slippage_bps=0, commission_bps=0)
    assert res2.equity.iloc[-1] == pytest.approx(1.0001**252, rel=1e-9)
    always_in = backtest_long_flat(data, 1 - signal, rf=0.05, slippage_bps=0, commission_bps=0)
    # bar 0 is flat (position = signal.shift(1)), then invested in a flat price: no interest
    # ... and cash held overnight into the entry bar earns that bar's rate as well
    assert always_in.equity.iloc[-1] == pytest.approx(1.05 ** (2 / 252), rel=1e-9)


def test_input_validation(random_walk: PriceData) -> None:
    sig = pd.Series(0, index=random_walk.index)
    with pytest.raises(ValueError, match="0/1"):
        backtest_long_flat(random_walk, sig + 2)
    with pytest.raises(ValueError, match="index"):
        backtest_long_flat(random_walk, sig.iloc[:-1])
    with pytest.raises(ValueError, match="non-negative"):
        backtest_long_flat(random_walk, sig, slippage_bps=-1)


def test_summary_from_result(random_walk: PriceData) -> None:
    res = backtest_ma_crossover(random_walk, 20, 60)
    s = res.summary()
    assert 0 <= s["time_in_market"] <= 1
    assert s["trades"] >= 1
    assert "benchmark_sharpe" in s
    assert res.benchmark_returns.iloc[0] == 0.0
