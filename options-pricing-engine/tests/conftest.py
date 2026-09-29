"""Shared fixtures.

The most valuable fixture here is :func:`synthetic_snapshot`: an option chain generated
from a *known* Black-Scholes smile with a known rate and dividend yield. Running it
through the full cleaning -> forward-extraction -> surface pipeline and checking that the
original volatilities come back out is an end-to-end correctness test that no amount of
unit testing on individual functions replaces.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from optpricing import blackscholes as bs
from optpricing import data as data_mod
from optpricing.heston import HestonParams
from optpricing.types import OptionType

NY = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Reference smile used to build the synthetic chain: a downward-sloping equity skew that
#: flattens with maturity, plus a mild call-wing upturn.
TRUE_SPOT = 500.0
TRUE_RATE = 0.042
TRUE_DIVIDEND = 0.013
SYNTHETIC_EXPIRY_DAYS = (14, 35, 91, 182, 365)


def reference_vol(log_moneyness: np.ndarray, tau: float) -> np.ndarray:
    r"""A plausible equity smile: ``0.18 - 0.35 k / sqrt(tau + 0.25) + 0.6 k^2``.

    Not a model, just a smooth arbitrage-plausible function that is steep and short-dated
    and flattens out, so the tests exercise the same regime as the real data.
    """
    k = np.asarray(log_moneyness, dtype=np.float64)
    return 0.18 - 0.35 * k / np.sqrt(tau + 0.25) + 0.6 * k * k


@pytest.fixture(scope="session")
def synthetic_snapshot() -> data_mod.ChainSnapshot:
    """A clean, arbitrage-free chain built from :func:`reference_vol`.

    Quotes are exact Black-Scholes mids with a symmetric 1% bid-ask around them, all
    strikes carry open interest, and the rate and dividend yield are constants known to
    the tests. Any pipeline error therefore shows up as a discrepancy, not as noise.
    """
    asof = dt.datetime(2026, 9, 18, 11, 0, tzinfo=NY)
    rows: list[dict[str, object]] = []
    for days in SYNTHETIC_EXPIRY_DAYS:
        expiry = (asof + dt.timedelta(days=days)).date()
        tau = data_mod.year_fraction(asof, expiry)
        forward = TRUE_SPOT * np.exp((TRUE_RATE - TRUE_DIVIDEND) * tau)
        ks = np.linspace(-0.25, 0.20, 24)
        strikes = np.round(forward * np.exp(ks), 0)
        vols = reference_vol(np.log(strikes / forward), tau)
        for side in (OptionType.CALL, OptionType.PUT):
            mids = np.asarray(
                bs.price(TRUE_SPOT, strikes, tau, TRUE_RATE, vols, side, TRUE_DIVIDEND)
            )
            for strike, mid in zip(strikes, mids, strict=True):
                half = max(0.005 * mid, 0.01)
                rows.append(
                    {
                        "contract": f"SYN{expiry:%y%m%d}{side.value[0].upper()}{strike:08.0f}",
                        "expiry": expiry,
                        "option_type": side.value,
                        "strike": float(strike),
                        "bid": float(mid - half),
                        "ask": float(mid + half),
                        "last": float(mid),
                        "volume": 100.0,
                        "open_interest": 1000.0,
                        "yf_implied_vol": np.nan,
                        "tau": tau,
                        "mid": float(mid),
                    }
                )
    return data_mod.ChainSnapshot(
        ticker="SYN", asof=asof, spot=TRUE_SPOT, quotes=pd.DataFrame(rows)
    )


@pytest.fixture(scope="session")
def flat_rate_curve() -> data_mod.RateCurve:
    """A flat curve at exactly ``TRUE_RATE`` so parity recovers the forward exactly."""
    return data_mod.RateCurve([0.01, 30.0], [TRUE_RATE, TRUE_RATE])


@pytest.fixture(scope="session")
def synthetic_surface(
    synthetic_snapshot: data_mod.ChainSnapshot, flat_rate_curve: data_mod.RateCurve
) -> pd.DataFrame:
    """The synthetic chain run through the real cleaning and surface pipeline.

    The chain is European by construction, so it is built with ``exercise="european"``;
    what the American correction does to it is tested in ``test_american.py``.
    """
    from optpricing import surface as surface_mod

    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    return surface_mod.build_surface(
        synthetic_snapshot, clean, flat_rate_curve, exercise="european"
    )


@pytest.fixture(scope="session")
def true_heston_params() -> HestonParams:
    """Parameters used to generate the Heston test surface. Feller is satisfied."""
    return HestonParams(v0=0.035, kappa=2.4, theta=0.045, xi=0.35, rho=-0.65)


@pytest.fixture(scope="session")
def cached_spy_snapshot() -> data_mod.ChainSnapshot:
    """The committed real SPY snapshot, skipped if it is not in the repo."""
    candidates = sorted((REPO_ROOT / "data" / "snapshots").glob("SPY_*.csv"))
    if not candidates:
        pytest.skip("no committed SPY snapshot")
    return data_mod.ChainSnapshot.from_csv(candidates[-1])


#: The rate curve the SPY regression tests price with (close to the committed one).
SPY_TEST_CURVE = data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])


@pytest.fixture(scope="session")
def spy_build(cached_spy_snapshot: data_mod.ChainSnapshot) -> Any:
    """The committed SPY chain through the default (American-corrected) surface.

    Built once per session: the de-Americanisation fixed point costs ~15 s.
    """
    from optpricing import surface as surface_mod

    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    return surface_mod.build_surface_detailed(cached_spy_snapshot, clean, SPY_TEST_CURVE)


@pytest.fixture(scope="session")
def spy_surface(spy_build: Any) -> pd.DataFrame:
    """The surface of :func:`spy_build`."""
    surface: pd.DataFrame = spy_build.surface
    return surface
