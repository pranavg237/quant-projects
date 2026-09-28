"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from mmsim.avellaneda_stoikov import AvellanedaStoikovParams, HorizonMode
from mmsim.book import LimitOrderBook
from mmsim.flow import FlowConfig
from mmsim.types import MarketConfig, Side

#: Avellaneda & Stoikov (2008) Table 1 parameters, used wherever the published numbers
#: are the reference.
PAPER_PARAMS = AvellanedaStoikovParams(
    gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0, horizon=1.0
)


@pytest.fixture
def book() -> LimitOrderBook:
    """An empty book."""
    return LimitOrderBook()


@pytest.fixture
def populated_book() -> LimitOrderBook:
    """A book with three bid and three ask levels and two orders at the touch.

    ::

        asks   102 x 4  (one order)
               101 x 7  (two orders: 'alice' 3 then 'bob' 4)
        ------------------
        bids    99 x 8   (two orders: 'alice' 5 then 'bob' 3)
                98 x 6   (one order)
    """
    b = LimitOrderBook()
    b.submit_limit(Side.BID, 99, 5.0, 0.0, "alice")
    b.submit_limit(Side.BID, 99, 3.0, 0.1, "bob")
    b.submit_limit(Side.BID, 98, 6.0, 0.2, "carol")
    b.submit_limit(Side.ASK, 101, 3.0, 0.3, "alice")
    b.submit_limit(Side.ASK, 101, 4.0, 0.4, "bob")
    b.submit_limit(Side.ASK, 102, 4.0, 0.5, "carol")
    return b


@pytest.fixture
def paper_params() -> AvellanedaStoikovParams:
    """The published Table 1 parameters."""
    return PAPER_PARAMS


@pytest.fixture
def stationary_params() -> AvellanedaStoikovParams:
    """Time-homogeneous variant of the paper parameters."""
    return AvellanedaStoikovParams(
        gamma=0.1,
        kappa=1.5,
        arrival_rate=140.0,
        sigma=2.0,
        horizon=0.5,
        horizon_mode=HorizonMode.STATIONARY,
    )


@pytest.fixture
def market() -> MarketConfig:
    """Penny ticks at a price of 100."""
    return MarketConfig(tick_size=0.01, initial_mid_ticks=10_000)


@pytest.fixture
def quiet_flow() -> FlowConfig:
    """Order flow with no informed traders."""
    return FlowConfig(informed_fraction=0.0)


@pytest.fixture
def informed_flow() -> FlowConfig:
    """Order flow where 20% of market orders move the price 2 ticks in their direction."""
    return FlowConfig(informed_fraction=0.2, info_impact_ticks=2.0)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator."""
    return np.random.default_rng(20260918)
