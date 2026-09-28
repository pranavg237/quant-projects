from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantbt.data import PriceData, synthetic_prices


@pytest.fixture
def random_walk() -> PriceData:
    """1500 bars of a driftless random walk with overnight gaps (no exploitable structure)."""
    return synthetic_prices(1500, drift=0.0, vol=0.01, gap_vol=0.005, seed=42)


@pytest.fixture
def trending() -> PriceData:
    """A deterministic series: flat, then a linear ramp, then flat, then a decline."""
    n = 600
    index = pd.bdate_range("2020-01-01", periods=n)
    price = np.concatenate(
        [
            np.full(150, 100.0),
            np.linspace(100.0, 200.0, 150),
            np.full(150, 200.0),
            np.linspace(200.0, 120.0, 150),
        ]
    )
    frame = pd.DataFrame(
        {"Open": price, "High": price * 1.001, "Low": price * 0.999, "Close": price, "Volume": 1e6},
        index=index,
    )
    return PriceData.from_frames({"TREND": frame})


def yahoo_like_frame(
    n: int = 300,
    start: str = "2020-01-01",
    seed: int = 1,
    dividends: dict[int, float] | None = None,
    splits: dict[int, float] | None = None,
) -> pd.DataFrame:
    """A raw Yahoo-style frame (split-adjusted Close, Adj Close from dividends and splits)."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    div = np.zeros(n)
    spl = np.zeros(n)
    for i, d in (dividends or {}).items():
        div[i] = d
    for i, s in (splits or {}).items():
        spl[i] = s
    # CRSP-style adjustment factor from dividends only (Close is already split-adjusted)
    f = np.ones(n)
    for i in range(1, n):
        if div[i] > 0:
            f[i] = 1 - div[i] / close[i - 1]
    factor = np.ones(n)
    for i in range(n - 2, -1, -1):
        factor[i] = factor[i + 1] * f[i + 1]
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.002, n)),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close * factor,
            "Volume": rng.integers(100_000, 1_000_000, n).astype(float),
            "Dividends": div,
            "Stock Splits": spl,
        },
        index=index,
    )
