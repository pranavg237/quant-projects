"""Event-driven engine tests: equivalence with the vectorised model, lookahead guards,
point-in-time universes, execution models and bookkeeping."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from quantbt.data import PriceData, synthetic_prices
from quantbt.engine import run_backtest
from quantbt.execution import (
    ExecutionSimulator,
    Fill,
    FixedBpsSlippage,
    NoCommission,
    NoSlippage,
    Order,
    OrderKind,
    PercentageCommission,
    PerShareCommission,
    VolumeShareSlippage,
)
from quantbt.portfolio import Portfolio
from quantbt.strategies import BuyAndHold, MACrossover
from quantbt.strategy import Context, Strategy
from quantbt.universe import PointInTimeUniverse, StaticUniverse, from_data
from quantbt.vectorized import backtest_ma_crossover
from tests.conftest import yahoo_like_frame


def _exec(slip: float = 5.0, comm: float = 1e-4, **kw: object) -> ExecutionSimulator:
    return ExecutionSimulator(
        commission=PercentageCommission(comm),
        slippage=FixedBpsSlippage(slip),
        **kw,  # type: ignore[arg-type]
    )


# --- equivalence with the Phase 2 vectorised backtest -----------------------------------


@pytest.mark.parametrize("rf", [0.0, 0.03])
@pytest.mark.parametrize("short,long", [(10, 40), (20, 100)])
def test_engine_matches_vectorized(
    random_walk: PriceData, short: int, long: int, rf: float
) -> None:
    start = random_walk.index[300]
    vec = backtest_ma_crossover(
        random_walk, short, long, start=start, slippage_bps=5.0, commission_bps=1.0, rf=rf
    )
    res = run_backtest(
        random_walk,
        MACrossover("SYN", short, long),
        start=start,
        execution=_exec(5.0, 1e-4),
        rf=rf,
        benchmark="SYN",
        initial_capital=1.0,
    )
    assert res.equity.index.equals(vec.equity.index)
    np.testing.assert_allclose(res.equity.to_numpy(), vec.equity.to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(res.returns.to_numpy(), vec.returns.to_numpy(), atol=1e-12)
    assert res.benchmark_returns is not None
    np.testing.assert_allclose(res.benchmark_returns, vec.benchmark_returns)
    # same number of round trips and the same trade P&Ls (as fractions of entry equity)
    assert len(res.round_trips) == len(vec.trade_pnls)
    assert res.summary()["sharpe"] == pytest.approx(vec.summary(rf=rf)["sharpe"], rel=1e-8)


def test_engine_matches_vectorized_on_real_shape_data() -> None:
    raw = yahoo_like_frame(n=1200, seed=5, dividends={300: 0.4, 700: 0.5}, splits={500: 3.0})
    data = PriceData.from_frames({"X": raw})
    start = data.index[250]
    vec = backtest_ma_crossover(data, 20, 60, start=start, slippage_bps=3.0, commission_bps=2.0)
    res = run_backtest(
        data, MACrossover("X", 20, 60), start=start, execution=_exec(3.0, 2e-4), initial_capital=1.0
    )
    np.testing.assert_allclose(res.equity.to_numpy(), vec.equity.to_numpy(), rtol=1e-10)


# --- lookahead guards ---------------------------------------------------------------------


class _Spy(Strategy):
    """Asserts that nothing past ``now`` is visible, and records what it saw."""

    name = "spy"

    def __init__(self) -> None:
        self.seen: list[pd.Timestamp] = []

    def on_bar(self, ctx: Context) -> None:
        h = ctx.history("close", 10)
        assert h.index[-1] == ctx.now
        assert (h.index <= ctx.now).all()
        full = ctx.history("close")
        assert full.index[-1] == ctx.now
        assert ctx.price("SYN") == pytest.approx(float(full["SYN"].iloc[-1]))
        self.seen.append(ctx.now)


def test_context_history_never_sees_the_future(random_walk: PriceData) -> None:
    spy = _Spy()
    res = run_backtest(random_walk, spy, start=random_walk.index[100], end=random_walk.index[200])
    assert spy.seen[0] == random_walk.index[100]
    assert spy.seen[-1] == random_walk.index[200]
    assert len(res.equity) == 101


class _GapChaser(Strategy):
    """Knows tomorrow's open (cheating) and buys when it gaps up."""

    name = "gap_chaser"

    def __init__(self, data: PriceData) -> None:
        self._open = data.single("open")
        self._close = data.single("close")

    def on_bar(self, ctx: Context) -> None:
        pos = ctx.position_index
        if pos + 1 >= len(self._open):
            return
        gap_up = self._open.iloc[pos + 1] > self._close.iloc[pos]
        ctx.order_target_weight("SYN", 1.0 if gap_up else 0.0)


def test_bias_engine_fills_after_the_gap(random_walk: PriceData) -> None:
    """Peeking at tomorrow's open still cannot capture the gap: the fill *is* that open."""
    strat = _GapChaser(random_walk)
    res = run_backtest(
        random_walk,
        strat,
        execution=ExecutionSimulator(NoCommission(), NoSlippage()),
        initial_capital=1.0,
    )
    assert abs(float(res.summary()["sharpe"])) < 1.0
    # ... whereas booking the gap would have been a money machine
    open_, close = random_walk.single("open"), random_walk.single("close")
    gaps = (open_.shift(-1) / close - 1.0).clip(lower=0).dropna()
    assert (1 + gaps).prod() > 5  # type: ignore[operator]


def test_orders_on_last_bar_are_rejected_not_filled(random_walk: PriceData) -> None:
    class Always(Strategy):
        name = "always"

        def on_bar(self, ctx: Context) -> None:
            ctx.order("SYN", 1.0)

    res = run_backtest(random_walk, Always(), end=random_walk.index[20])
    assert len(res.fills) == 20
    assert len(res.rejections) == 1
    assert res.rejections[0].reason == "submitted on the final bar"
    assert res.summary()["rejected_orders"] == 1


def test_fill_at_close_is_one_bar_after_signal() -> None:
    n, jump = 60, 30
    index = pd.bdate_range("2021-01-01", periods=n)
    close = np.full(n, 100.0)
    close[jump:] = 110.0
    open_ = np.full(n, 100.0)
    open_[jump + 1 :] = 110.0  # the open catches up a bar later, so open != close on the jump bar
    frame = pd.DataFrame(
        {"Open": open_, "High": 120, "Low": 90, "Close": close, "Volume": 1.0}, index=index
    )
    data = PriceData.from_frames({"J": frame})

    class BuyOnJumpBar(Strategy):
        name = "b"

        def on_bar(self, ctx: Context) -> None:
            if ctx.now == index[jump - 1]:
                ctx.order_target_weight("J", 1.0)

    res_open = run_backtest(
        data,
        BuyOnJumpBar(),
        execution=ExecutionSimulator(NoCommission(), NoSlippage(), fill_at="open"),
    )
    res_close = run_backtest(
        data,
        BuyOnJumpBar(),
        execution=ExecutionSimulator(NoCommission(), NoSlippage(), fill_at="close"),
    )
    # filled at the open of the jump bar (100) -> books the jump; at the close (110) -> does not
    assert res_open.fills[0].price == 100.0 and res_open.fills[0].timestamp == index[jump]
    assert res_close.fills[0].price == 110.0
    assert res_open.growth.iloc[-1] == pytest.approx(1.10)
    assert res_close.growth.iloc[-1] == pytest.approx(1.0)


# --- universes and survivorship -----------------------------------------------------------


def test_bias_cannot_trade_symbol_before_listing_or_after_delisting() -> None:
    a = yahoo_like_frame(n=200, start="2020-01-01")
    b = yahoo_like_frame(n=60, start="2020-03-02", seed=2)  # lists late, delists early
    data = PriceData.from_frames({"A": a, "B": b})
    uni = from_data(data)
    assert uni.members(pd.Timestamp("2020-01-15")) == ["A"]
    assert "B" in uni.members(pd.Timestamp("2020-04-01"))
    assert uni.members(data.index[-1]) == ["A"]

    class WantsB(Strategy):
        name = "wants_b"

        def __init__(self) -> None:
            self.errors = 0

        def on_bar(self, ctx: Context) -> None:
            try:
                ctx.order_target_weight("B", 0.5)
            except KeyError:
                self.errors += 1

    strat = WantsB()
    res = run_backtest(data, strat, execution=ExecutionSimulator(NoCommission(), NoSlippage()))
    assert strat.errors > 0  # every bar where B was not a member raised
    # B was force-liquidated when it left the universe; no position remains at the end
    assert res.positions["B"].iloc[-1] == 0.0
    assert res.positions["B"].abs().max() > 0
    last_b_bar = data.close["B"].dropna().index[-1]
    # position is closed on the first bar after the last B print
    after = res.positions["B"].loc[last_b_bar:]
    assert after.iloc[0] != 0 and (after.iloc[1:] == 0).all()


def test_point_in_time_universe_and_static() -> None:
    pit = PointInTimeUniverse.from_table(
        [("A", "2020-01-01", "2020-06-30"), ("B", None, None), ("A", "2021-01-01", None)]
    )
    assert pit.all_symbols == ["A", "B"]
    assert pit.members(pd.Timestamp("2020-03-01")) == ["A", "B"]
    assert pit.members(pd.Timestamp("2020-09-01")) == ["B"]
    assert pit.members(pd.Timestamp("2021-02-01")) == ["A", "B"]
    with pytest.raises(ValueError):
        PointInTimeUniverse({"A": [(pd.Timestamp("2021-01-01"), pd.Timestamp("2020-01-01"))]})
    st = StaticUniverse(["X", "Y"])
    assert st.members(pd.Timestamp("1999-01-01")) == ["X", "Y"] and st.all_symbols == ["X", "Y"]
    data = synthetic_prices(10)
    with pytest.raises(ValueError, match="without data"):
        run_backtest(data, BuyAndHold(), universe=StaticUniverse(["SYN", "NOPE"]))
    assert from_data(data, min_history=20).members(data.index[-1]) == []


# --- execution models -----------------------------------------------------------------------


def test_commission_models() -> None:
    pct = PercentageCommission(0.001)
    assert pct.calculate(-10, 50.0) == pytest.approx(0.5)
    shares = pct.shares_for_notional(1000.0, 10.0)
    assert shares * 10.0 + pct.calculate(shares, 10.0) == pytest.approx(1000.0)
    with pytest.raises(ValueError):
        PercentageCommission(-1)
    ps = PerShareCommission(0.005, minimum=1.0)
    assert ps.calculate(100, 10.0) == 1.0  # minimum binds
    assert ps.calculate(1000, 10.0) == 5.0
    assert ps.calculate(0, 10.0) == 0.0
    s = ps.shares_for_notional(100_000.0, 10.0)
    assert s * 10.0 + ps.calculate(s, 10.0) == pytest.approx(100_000.0)
    s2 = ps.shares_for_notional(100.0, 10.0)  # minimum binds: (100 - 1) / 10
    assert s2 == pytest.approx(9.9)
    assert ps.shares_for_notional(0.5, 10.0) == 0.0
    assert NoCommission().shares_for_notional(100.0, 4.0) == 25.0


def test_slippage_models() -> None:
    assert FixedBpsSlippage(10).fill_price(100.0, 1, 1e6) == pytest.approx(100.1)
    assert FixedBpsSlippage(10).fill_price(100.0, -1, 1e6) == pytest.approx(99.9)
    with pytest.raises(ValueError):
        FixedBpsSlippage(-1)
    vs = VolumeShareSlippage(price_impact=0.1, fixed_bps=0.0)
    assert vs.fill_price(100.0, 1000, 10_000) == pytest.approx(100.0 * (1 + 0.1 * 0.01))
    assert vs.fill_price(100.0, -1000, 10_000) == pytest.approx(100.0 * (1 - 0.1 * 0.01))
    assert vs.fill_price(100.0, 1000, 0.0) == 100.0
    assert NoSlippage().fill_price(3.0, 5, 1) == 3.0


def test_simulator_options_and_rejections() -> None:
    with pytest.raises(ValueError):
        ExecutionSimulator(fill_at="midpoint")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ExecutionSimulator(max_volume_share=2.0)
    with pytest.raises(ValueError):
        Order("A", OrderKind.QUANTITY, float("nan"), pd.Timestamp("2020-01-01"))
    sim = ExecutionSimulator(NoCommission(), NoSlippage(), lot_size=100, max_volume_share=0.1)
    ts = pd.Timestamp("2020-01-02")
    o = Order("A", OrderKind.TARGET_WEIGHT, 1.0, ts)
    f = sim.fill(o, ts, price=10.0, volume=100_000, equity=12_345.0, current_quantity=0.0)
    assert isinstance(f, Fill) and f.quantity == 1200  # 1234.5 truncated to lots
    f2 = sim.fill(o, ts, price=10.0, volume=1_000, equity=12_345.0, current_quantity=0.0)
    assert isinstance(f2, Fill) and f2.quantity == 100  # capped at 10% of volume
    r = sim.fill(o, ts, price=float("nan"), volume=1.0, equity=1.0, current_quantity=0.0)
    assert r.reason == "no price on fill bar"  # type: ignore[union-attr]
    r2 = sim.fill(o, ts, price=10.0, volume=1.0, equity=100.0, current_quantity=0.0)
    assert r2.reason == "zero quantity after sizing"  # type: ignore[union-attr]
    # selling to a lower target
    sell = sim.fill(
        Order("A", OrderKind.TARGET_WEIGHT, 0.5, ts),
        ts,
        10.0,
        1e9,
        equity=10_000.0,
        current_quantity=1000.0,
    )
    assert sell.quantity == -500  # type: ignore[union-attr]


# --- portfolio bookkeeping ----------------------------------------------------------------


def _fill(symbol: str, qty: float, price: float, day: int, commission: float = 0.0) -> Fill:

    return Fill(
        0,
        symbol,
        pd.Timestamp("2020-01-01") + pd.Timedelta(days=day),
        qty,
        price,
        commission,
        price,
    )


def test_portfolio_round_trips_long_short_partial_and_flip() -> None:
    p = Portfolio(cash=10_000.0)
    p.apply_fill(_fill("A", 100, 10.0, 0, commission=1.0))
    assert p.cash == pytest.approx(10_000 - 1000 - 1)
    p.mark({"A": 12.0})
    assert p.equity == pytest.approx(8999 + 1200)
    assert p.weights()["A"] == pytest.approx(1200 / 10199)
    p.apply_fill(_fill("A", 50, 14.0, 1))  # add: average cost 11.33
    assert p.positions["A"].cost_basis == pytest.approx((1000 + 700) / 150)
    p.apply_fill(_fill("A", -50, 15.0, 2, commission=0.5))  # partial exit
    assert len(p.round_trips) == 1
    rt = p.round_trips[0]
    assert rt.quantity == 50 and rt.direction == 1
    assert rt.pnl == pytest.approx(50 * (15.0 - 1700 / 150) - 0.5 - 1.0 * (50 / 150))
    p.apply_fill(_fill("A", -200, 16.0, 3))  # close the remaining 100 and flip short 100
    assert len(p.round_trips) == 2
    assert p.positions["A"].quantity == -100
    assert p.positions["A"].cost_basis == 16.0
    p.apply_fill(_fill("A", 100, 15.0, 4))  # cover: short pnl = 100 * (16 - 15)
    assert len(p.round_trips) == 3
    assert p.round_trips[2].direction == -1 and p.round_trips[2].pnl == pytest.approx(100.0)
    assert p.positions["A"].quantity == 0
    assert p.quantity("Z") == 0.0
    assert p.total_commission == pytest.approx(1.5)
    empty = Portfolio(cash=0.0)
    empty.positions["A"] = __import__("quantbt.portfolio", fromlist=["Position"]).Position("A")
    assert empty.weights() == {"A": 0.0}


def test_interest_accrues_on_cash_only() -> None:
    data = synthetic_prices(30, vol=0.0, gap_vol=0.0)  # flat price

    class Nothing(Strategy):
        name = "nothing"

        def on_bar(self, ctx: Context) -> None:
            pass

    res = run_backtest(data, Nothing(), rf=0.05, initial_capital=100.0)
    assert res.equity.iloc[-1] == pytest.approx(100.0 * 1.05 ** (30 / 252))
    rf_series = pd.Series(0.001, index=data.index)
    res2 = run_backtest(data, Nothing(), rf=rf_series, initial_capital=100.0)
    assert res2.equity.iloc[-1] == pytest.approx(100.0 * 1.001**30)


# --- results layer and misc ---------------------------------------------------------------


def test_buy_and_hold_and_result_frames(random_walk: PriceData) -> None:
    res = run_backtest(random_walk, BuyAndHold(), start=random_walk.index[5], benchmark="SYN")
    assert len(res.fills) == 1
    assert res.weights["SYN"].iloc[-1] == pytest.approx(1.0, abs=1e-3)
    frame = res.to_frame()
    assert {"equity", "cash", "return", "drawdown", "gross_exposure", "benchmark_return"} <= set(
        frame.columns
    )
    assert res.fills_frame().shape[0] == 1
    assert res.trades_frame().empty  # never closed
    s = res.summary()
    assert s["trades"] == 0 and "cost_drag_pct" in s
    assert res.params == {"weight": 1.0}
    empty = run_backtest(random_walk, BuyAndHold(), end=random_walk.index[0])
    assert empty.fills_frame().empty and empty.trades_frame().empty


def test_engine_argument_validation(random_walk: PriceData) -> None:
    with pytest.raises(ValueError, match="no bars"):
        run_backtest(random_walk, BuyAndHold(), start="2050-01-01")
    with pytest.raises(ValueError, match="benchmark"):
        run_backtest(random_walk, BuyAndHold(), benchmark="NOPE")
    with pytest.raises(ValueError):
        MACrossover("SYN", 50, 20)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run_backtest(random_walk, MACrossover("SYN", 10, 200), start=random_walk.index[50])
    assert any("warm-up" in str(x.message) for x in w)
    with pytest.raises(ValueError, match="empty"):
        run_backtest(random_walk.slice("2050-01-01", "2050-01-02"), BuyAndHold())


def test_context_helpers(random_walk: PriceData) -> None:
    class Probe(Strategy):
        name = "probe"

        def on_bar(self, ctx: Context) -> None:
            assert ctx.has_price("SYN")
            assert ctx.cash == ctx.portfolio.cash
            assert ctx.quantity("SYN") == 0.0 and ctx.weight("SYN") == 0.0 and ctx.weights == {}
            orders = ctx.order_target_weights({"SYN": 0.5})
            assert len(orders) == 1
            assert ctx.params == {"p": 1}
            raise StopIteration  # one bar is enough

    with pytest.raises(StopIteration):
        run_backtest(random_walk, Probe(), params={"p": 1})


def test_order_target_weights_closes_others() -> None:
    data = synthetic_prices(50, symbols=("A", "B"))

    class Rotate(Strategy):
        name = "rotate"

        def on_bar(self, ctx: Context) -> None:
            if ctx.position_index == 0:
                ctx.order_target_weights({"A": 0.5, "B": 0.5})
            elif ctx.position_index == 5:
                ctx.order_target_weights({"A": 1.0})  # B goes to zero

    res = run_backtest(data, Rotate(), execution=ExecutionSimulator(NoCommission(), NoSlippage()))
    assert res.positions["B"].iloc[3] > 0
    assert res.positions["B"].iloc[10] == 0
    assert len(res.round_trips) == 1 and res.round_trips[0].symbol == "B"


def test_portfolio_zeroes_floating_point_dust_on_full_exit() -> None:
    """A target-weight-0 order sizes as -value/price, which is not exactly -quantity.

    With an absolute epsilon the leftover dust persisted as a phantom position and a
    phantom open round trip, and eventually divided by a zero cost basis.
    """
    p = Portfolio(cash=1_000_000.0)
    qty = 1_000_000.0 / 137.77  # a price that does not divide evenly
    p.apply_fill(_fill("A", qty, 137.77, 0))
    p.mark({"A": 140.0})
    # sell "everything" the way the simulator sizes it: -(quantity * price) / price
    sell = -(p.quantity("A") * 140.0) / 140.0
    p.apply_fill(_fill("A", sell, 140.0, 1))
    assert p.quantity("A") == 0.0
    assert len(p.round_trips) == 1
    assert p.round_trips[0].return_pct == pytest.approx(140.0 / 137.77 - 1.0, rel=1e-9)
    assert not p._open  # no phantom trip left behind


def test_round_trip_with_zero_cost_basis_reports_zero_return() -> None:
    p = Portfolio(cash=100.0)
    p.apply_fill(_fill("A", 10, 0.0, 0))  # a zero-priced fill (bad data, not a crash)
    p.apply_fill(_fill("A", -10, 0.0, 1))
    assert len(p.round_trips) == 1
    assert p.round_trips[0].return_pct == 0.0


# -- short borrow cost -------------------------------------------------------------------


class _AlwaysShort(Strategy):
    name = "always_short"

    def __init__(self, symbol: str, weight: float = -1.0) -> None:
        self.symbol = symbol
        self.weight = weight

    def on_bar(self, ctx: Context) -> None:
        if ctx.quantity(self.symbol) == 0:
            ctx.order_target_weight(self.symbol, self.weight)


def test_borrow_cost_is_charged_only_on_shorts() -> None:
    """Carrying a short must cost the stock-loan fee; a long must not be charged."""
    data = synthetic_prices(260, symbols=("A",), drift=0.0, vol=0.005, seed=5)
    no_cost = ExecutionSimulator(NoCommission(), NoSlippage())
    short_free = run_backtest(data, _AlwaysShort("A"), execution=no_cost)
    short_paid = run_backtest(data, _AlwaysShort("A"), execution=no_cost, borrow_rate=0.03)
    long_paid = run_backtest(data, _AlwaysShort("A", 1.0), execution=no_cost, borrow_rate=0.03)

    assert short_free.total_borrow_cost == 0.0
    assert long_paid.total_borrow_cost == 0.0  # a long position is not borrowed
    assert short_paid.total_borrow_cost > 0.0
    assert short_paid.equity.iloc[-1] < short_free.equity.iloc[-1]

    # ~3% a year on ~100% short exposure over ~one year of bars
    drag = 1.0 - short_paid.equity.iloc[-1] / short_free.equity.iloc[-1]
    assert 0.02 < drag < 0.04
    assert short_paid.summary()["total_borrow_cost"] == pytest.approx(short_paid.total_borrow_cost)


def test_borrow_rate_must_be_non_negative() -> None:
    data = synthetic_prices(60, symbols=("A",), seed=1)
    with pytest.raises(ValueError, match="borrow_rate"):
        run_backtest(data, _AlwaysShort("A"), borrow_rate=-0.01)


def test_summary_surfaces_leverage() -> None:
    """A backtest that borrows cash must not be able to pass as an unlevered one."""
    data = synthetic_prices(120, symbols=("A",), drift=0.0, vol=0.005, seed=2)

    class Levered(Strategy):
        name = "levered"

        def on_bar(self, ctx: Context) -> None:
            if ctx.quantity("A") == 0:
                ctx.order_target_weight("A", 1.5)

    s = run_backtest(data, Levered(), execution=ExecutionSimulator(NoCommission(), NoSlippage()))
    summary = s.summary()
    assert summary["max_gross_exposure"] > 1.4
    assert summary["min_cash_weight"] < -0.4  # the cash leg is a margin loan
