"""Markout decomposition by counterparty, and standard errors across sessions.

The hand-built cases pin the arithmetic exactly. The simulation cases check the one thing a
markout analysis must get right to be believed: fills against informed takers lose what the
model says they should, and fills against uninformed takers lose nothing at short horizons.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmsim.avellaneda_stoikov import AvellanedaStoikovParams
from mmsim.engine import SimulationResult, simulate_book, simulate_reference
from mmsim.flow import FlowConfig
from mmsim.metrics import (
    fill_markouts,
    informed_markout_theory,
    markout_decomposition,
    pnl_decomposition,
    session_standard_errors,
)
from mmsim.strategies import AvellanedaStoikovPolicy, SymmetricPolicy
from mmsim.types import MarketConfig


def _hand_built_run() -> SimulationResult:
    """Five mids, two fills.

    * step 1: bought 1.0 at 99.99 from an **informed** seller, mid 100.01
      -> edge +2 ticks; markout h=1: 99.99 - 100.01 = -2 ticks; h=2: 99.98 - 100.01 = -3
    * step 2: sold 0.5 at 100.02 to an **uninformed** buyer, mid 99.99
      -> edge +3 ticks; markout h=1: -(99.98 - 99.99) = +1 tick; h=2: -(100.02 - 99.99) = -3
    """
    mid = np.array([100.00, 100.01, 99.99, 99.98, 100.02])
    fills = pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "step": [1, 2],
            "side": ["buy", "sell"],
            "price": [99.99, 100.02],
            "mid": [100.01, 99.99],
            "size": [1.0, 0.5],
            "counterparty": ["informed", "uninformed"],
        }
    )
    n = mid.size
    return SimulationResult(
        times=np.linspace(0.0, 0.4, n),
        mid=mid,
        inventory=np.array([0.0, 0.0, 1.0, 0.5, 0.5]),
        cash=np.zeros(n),
        bid_quotes=np.full(n, np.nan),
        ask_quotes=np.full(n, np.nan),
        buy_fills=np.zeros(n),
        sell_fills=np.zeros(n),
        fill_prices=fills,
        engine="book",
        policy_name="hand",
    )


def test_fill_markouts_by_hand() -> None:
    frame = fill_markouts(_hand_built_run(), horizons=(1, 2, 3), tick_size=0.01)
    assert list(frame["counterparty"]) == ["informed", "uninformed"]
    np.testing.assert_allclose(frame["edge_ticks"], [2.0, 3.0], atol=1e-9)
    np.testing.assert_allclose(frame["markout_1_ticks"], [-2.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(frame["markout_2_ticks"], [-3.0, -3.0], atol=1e-9)
    # step 2 + 3 = 5 is past the last index (4): missing, not truncated to the last mid.
    assert frame["markout_3_ticks"].iloc[0] == pytest.approx(1.0, abs=1e-9)
    assert np.isnan(frame["markout_3_ticks"].iloc[1])


def test_markout_decomposition_by_hand() -> None:
    run = _hand_built_run()
    table = markout_decomposition({"hand": [run, run]}, horizons=(1,), tick_size=0.01)
    row = table.set_index("counterparty")

    assert row.loc["informed", "edge_ticks"] == pytest.approx(2.0)
    assert row.loc["informed", "markout_ticks"] == pytest.approx(-2.0)
    assert row.loc["informed", "adverse_selection_ticks"] == pytest.approx(2.0)
    assert row.loc["informed", "realised_spread_ticks"] == pytest.approx(0.0, abs=1e-9)
    assert row.loc["uninformed", "realised_spread_ticks"] == pytest.approx(4.0)

    # "all" is size-weighted: (2 * 1 + 3 * 0.5) / 1.5 and (-2 * 1 + 1 * 0.5) / 1.5.
    assert row.loc["all", "edge_ticks"] == pytest.approx(3.5 / 1.5)
    assert row.loc["all", "markout_ticks"] == pytest.approx(-1.0)
    assert row.loc["all", "volume_per_session"] == pytest.approx(1.5)
    # Per session, in price units: 3.5 ticks of edge * 0.01.
    assert row.loc["all", "edge_pnl_per_session"] == pytest.approx(0.035)
    assert row.loc["all", "realised_pnl_per_session"] == pytest.approx(
        row.loc["all", "edge_pnl_per_session"] - row.loc["all", "adverse_selection_pnl_per_session"]
    )
    # Two identical sessions: no between-session variation, so zero clustered error.
    assert row.loc["all", "markout_ticks_se"] == pytest.approx(0.0, abs=1e-12)


def test_split_by_inventory_effect_by_hand() -> None:
    """The buy from flat adds to the position; the sell while long 1.0 reduces it."""
    run = _hand_built_run()
    frame = fill_markouts(run, horizons=(1,), tick_size=0.01)
    assert list(frame["inventory_effect"]) == ["adds", "reduces"]

    table = markout_decomposition(
        {"hand": [run, run]}, horizons=(1,), tick_size=0.01, by="inventory_effect"
    ).set_index("inventory_effect")
    assert list(table.index) == ["adds", "reduces", "all"]
    assert table.loc["reduces", "markout_ticks"] == pytest.approx(1.0)
    assert table.loc["adds", "markout_ticks"] == pytest.approx(-2.0)

    only_uninformed = markout_decomposition(
        {"hand": [run]},
        horizons=(1,),
        tick_size=0.01,
        by="inventory_effect",
        where={"counterparty": "uninformed"},
    ).set_index("inventory_effect")
    assert list(only_uninformed.index) == ["reduces", "all"]
    assert only_uninformed.loc["all", "volume_per_session"] == pytest.approx(0.5)


def test_counterparty_is_recorded_on_every_book_fill(
    quiet_flow: FlowConfig, informed_flow: FlowConfig, market: MarketConfig
) -> None:
    policy = SymmetricPolicy(half_spread=0.04)
    quiet = simulate_book(policy, quiet_flow, market, n_steps=600, seed=3).fill_prices
    assert set(quiet["counterparty"]) == {"uninformed"}
    toxic = simulate_book(policy, informed_flow, market, n_steps=600, seed=3).fill_prices
    assert set(toxic["counterparty"]) == {"informed", "uninformed"}


def test_edge_at_horizon_zero_is_the_spread_capture_leg(
    informed_flow: FlowConfig, market: MarketConfig
) -> None:
    """The markout table and the PnL decomposition must describe the same money."""
    runs = [
        simulate_book(SymmetricPolicy(half_spread=0.04), informed_flow, market, n_steps=500, seed=s)
        for s in range(6)
    ]
    table = markout_decomposition({"s": runs}, horizons=(0,), tick_size=market.tick_size)
    edge = table.set_index("counterparty")["edge_pnl_per_session"]
    capture = float(np.mean([pnl_decomposition(r)["spread_capture"] for r in runs]))
    assert edge["all"] == pytest.approx(capture, rel=1e-9)
    assert edge["informed"] + edge["uninformed"] == pytest.approx(capture, rel=1e-9)


def test_informed_markout_matches_the_model(market: MarketConfig) -> None:
    """Informed fills lose -J(1-(1-v)^h); uninformed fills lose ~nothing at short horizons.

    This is the validation of the whole analysis: the informed markout is a first-principles
    number (impact J released at rate v per step), and the measurement recovers it.
    """
    flow = FlowConfig(informed_fraction=0.2, info_impact_ticks=2.0, info_impact_speed=0.05)
    runs = [
        simulate_book(SymmetricPolicy(half_spread=0.04), flow, market, n_steps=1500, seed=s)
        for s in range(25)
    ]
    horizons = (1, 5, 20)
    table = markout_decomposition({"s": runs}, horizons=horizons, tick_size=market.tick_size)
    theory = informed_markout_theory(horizons, 2.0, 0.05)
    informed = table[table["counterparty"] == "informed"].sort_values("horizon")
    for (_, row), expected in zip(informed.iterrows(), theory, strict=True):
        assert abs(row["markout_ticks"] - expected) < 4.0 * row["markout_ticks_se"] + 0.02
    uninformed = table[(table["counterparty"] == "uninformed") & (table["horizon"] == 1.0)]
    assert abs(float(uninformed["markout_ticks"].iloc[0])) < 0.05


def test_informed_markout_theory_values() -> None:
    values = informed_markout_theory((0, 1, 2, 10_000), impact_ticks=2.0, impact_speed=0.05)
    np.testing.assert_allclose(values[:3], [0.0, -0.1, -2.0 * (1 - 0.95**2)])
    assert values[-1] == pytest.approx(-2.0)


def test_fill_markouts_needs_book_fills(paper_params: AvellanedaStoikovParams) -> None:
    result = simulate_reference(AvellanedaStoikovPolicy(paper_params), paper_params, seed=1)
    with pytest.raises(ValueError, match="simulate_book"):
        fill_markouts(result)


def test_empty_run_has_no_markouts() -> None:
    run = _hand_built_run()
    run.fill_prices = pd.DataFrame()
    assert fill_markouts(run).empty
    table = markout_decomposition({"empty": [run]}, horizons=(1,))
    assert table["n_fills"].eq(0).all()
    assert table["edge_ticks"].isna().all()


def test_session_standard_errors_against_normal_theory() -> None:
    """For Gaussian PnL the bootstrap must agree with the textbook formulas."""
    rng = np.random.default_rng(7)
    n = 2_000
    pnl = rng.normal(5.0, 2.0, n)  # Sharpe per session 2.5
    q = rng.normal(0.0, 3.0, n)
    se = session_standard_errors(pnl, q, n_bootstrap=800, seed=1)
    assert se["mean_pnl_se"] == pytest.approx(pnl.std(ddof=1) / np.sqrt(n))
    assert se["std_pnl_se"] == pytest.approx(2.0 / np.sqrt(2 * n), rel=0.2)
    assert se["std_final_inventory_se"] == pytest.approx(3.0 / np.sqrt(2 * n), rel=0.2)
    # Lo (2002), iid returns: se(SR) = sqrt((1 + SR^2 / 2) / n).
    assert se["sharpe_se"] == pytest.approx(np.sqrt((1 + 2.5**2 / 2) / n), rel=0.2)
    # Seeded: reproducible.
    assert session_standard_errors(pnl, q, n_bootstrap=800, seed=1) == se


def test_session_standard_errors_edge_cases() -> None:
    one = session_standard_errors(np.array([1.0]), np.array([0.0]))
    assert all(np.isnan(v) for v in one.values())
    with pytest.raises(ValueError, match="aligned"):
        session_standard_errors(np.zeros(3), np.zeros(4))
