"""Quoting policies."""

from __future__ import annotations

import numpy as np
import pytest

from mmsim.avellaneda_stoikov import AvellanedaStoikovParams, HorizonMode
from mmsim.strategies import (
    AvellanedaStoikovPolicy,
    InventoryLimitPolicy,
    MakerState,
    QuotePolicy,
    SymmetricPolicy,
    average_optimal_spread,
)


def _state(inventory: float = 0.0, time: float = 0.0) -> MakerState:
    return MakerState(mid=100.0, inventory=inventory, cash=0.0, time=time, horizon=1.0)


def test_all_policies_satisfy_the_protocol(paper_params: AvellanedaStoikovParams) -> None:
    policies = [
        AvellanedaStoikovPolicy(paper_params),
        SymmetricPolicy(half_spread=0.5),
        InventoryLimitPolicy(half_spread=0.5, max_inventory=3.0),
    ]
    for policy in policies:
        assert isinstance(policy, QuotePolicy)
        assert isinstance(policy.name, str) and policy.name


def test_symmetric_ignores_inventory() -> None:
    policy = SymmetricPolicy(half_spread=0.75)
    for q in (-10.0, 0.0, 10.0):
        assert policy.quote(_state(q)) == (99.25, 100.75)


def test_avellaneda_stoikov_does_not_ignore_inventory(
    paper_params: AvellanedaStoikovParams,
) -> None:
    policy = AvellanedaStoikovPolicy(paper_params)
    flat_bid, flat_ask = policy.quote(_state(0.0))
    long_bid, long_ask = policy.quote(_state(4.0))
    assert flat_bid is not None and long_bid is not None
    assert flat_ask is not None and long_ask is not None
    assert long_bid < flat_bid
    assert long_ask < flat_ask
    assert (long_ask - long_bid) == pytest.approx(flat_ask - flat_bid)


def test_inventory_limit_withdraws_one_side() -> None:
    policy = InventoryLimitPolicy(half_spread=0.5, max_inventory=3.0)
    assert policy.quote(_state(3.0))[0] is None
    assert policy.quote(_state(3.0))[1] == 100.5
    assert policy.quote(_state(-3.0))[1] is None
    assert policy.quote(_state(-3.0))[0] == 99.5
    assert policy.quote(_state(0.0)) == (99.5, 100.5)


def test_average_spread_finite_versus_stationary(
    paper_params: AvellanedaStoikovParams, stationary_params: AvellanedaStoikovParams
) -> None:
    from mmsim.avellaneda_stoikov import optimal_spread

    # Finite: the average of a spread that shrinks linearly to the markup.
    markup = float(optimal_spread(0.0, paper_params))
    expected = markup + 0.5 * paper_params.gamma * paper_params.sigma**2 * paper_params.horizon
    assert average_optimal_spread(paper_params) == pytest.approx(expected)
    # Stationary: constant, so the average is the spread itself.
    assert average_optimal_spread(stationary_params) == pytest.approx(
        float(optimal_spread(stationary_params.horizon, stationary_params))
    )


def test_matched_spread_really_matches(paper_params: AvellanedaStoikovParams) -> None:
    """The benchmark quotes the same average width, so the comparison is about inventory."""
    avg = average_optimal_spread(paper_params)
    as_policy = AvellanedaStoikovPolicy(paper_params)
    sym_policy = SymmetricPolicy(half_spread=avg / 2)

    times = np.linspace(0.0, paper_params.horizon, 201)
    as_widths = []
    for t in times:
        bid, ask = as_policy.quote(_state(0.0, float(t)))
        assert bid is not None and ask is not None
        as_widths.append(ask - bid)
    sym_bid, sym_ask = sym_policy.quote(_state(0.0))
    assert sym_bid is not None and sym_ask is not None
    assert float(np.mean(as_widths)) == pytest.approx(sym_ask - sym_bid, rel=1e-3)


def test_policy_validation() -> None:
    with pytest.raises(ValueError, match="half_spread"):
        SymmetricPolicy(half_spread=0.0)
    with pytest.raises(ValueError, match="half_spread"):
        InventoryLimitPolicy(half_spread=-1.0, max_inventory=1.0)
    with pytest.raises(ValueError, match="max_inventory"):
        InventoryLimitPolicy(half_spread=1.0, max_inventory=0.0)


def test_stationary_policy_is_time_invariant(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    policy = AvellanedaStoikovPolicy(stationary_params)
    assert policy.quote(_state(2.0, 0.0)) == policy.quote(_state(2.0, 0.99))
    assert stationary_params.horizon_mode is HorizonMode.STATIONARY


def test_quote_type_rejects_a_crossed_quote() -> None:
    from mmsim.types import Quote

    Quote(bid_ticks=99, ask_ticks=101)
    Quote(bid_ticks=None, ask_ticks=101)
    with pytest.raises(ValueError, match="crossed"):
        Quote(bid_ticks=101, ask_ticks=99)
    with pytest.raises(ValueError, match="crossed"):
        Quote(bid_ticks=100, ask_ticks=100)
