"""Implied-volatility inversion."""

from __future__ import annotations

import numpy as np
import pytest

from optpricing import blackscholes as bs
from optpricing import implied_vol as iv
from optpricing.types import OptionType


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("true_vol", [0.05, 0.12, 0.25, 0.60, 1.50])
@pytest.mark.parametrize("strike", [70.0, 95.0, 100.0, 110.0, 140.0])
@pytest.mark.parametrize("tau", [0.02, 0.25, 2.0])
def test_round_trip(option_type: OptionType, true_vol: float, strike: float, tau: float) -> None:
    """Price at a known vol, invert, get the vol back (where the quote is invertible)."""
    spot, rate, q = 100.0, 0.03, 0.01
    price = float(bs.price(spot, strike, tau, rate, true_vol, option_type, q))
    recovered = iv.implied_vol_scalar(price, spot, strike, tau, rate, option_type, q)
    if np.isnan(recovered):
        # Only legitimate when the option has no extractable time value at float precision.
        lower, upper = iv.no_arbitrage_bounds(spot, strike, tau, rate, option_type, q)
        assert price <= float(lower) + 1e-12 or price >= float(upper) - 1e-12
    else:
        assert recovered == pytest.approx(true_vol, abs=1e-5)


def test_vectorised_round_trip_over_a_whole_smile() -> None:
    spot, rate, q = 100.0, 0.03, 0.01
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 125.0])
    taus = np.array([[0.05], [0.5], [2.0]])
    true = 0.18 - 0.4 * np.log(strikes / spot) / np.sqrt(taus + 0.3)
    prices = bs.price(spot, strikes, taus, rate, true, OptionType.CALL, q)

    result = iv.implied_vol(
        prices, spot, strikes, taus, rate, OptionType.CALL, q, return_diagnostics=True
    )
    assert isinstance(result, iv.ImpliedVolResult)
    assert result.vol.shape == (3, 5)
    valid = ~np.isnan(result.vol)
    assert valid.sum() > 10
    assert np.max(np.abs(result.vol[valid] - np.broadcast_to(true, result.vol.shape)[valid])) < 1e-4
    assert bool(result.converged[valid].all())
    # Newton should be doing most of the work on a well-behaved synthetic smile.
    assert result.newton_fraction > 0.7
    assert result.iterations[valid].max() < 40


def test_quotes_outside_no_arbitrage_bounds_return_nan() -> None:
    spot, strike, tau, rate = 100.0, 90.0, 1.0, 0.03
    lower, upper = iv.no_arbitrage_bounds(spot, strike, tau, rate, OptionType.CALL)
    assert np.isnan(iv.implied_vol_scalar(float(lower) * 0.5, spot, strike, tau, rate))
    assert np.isnan(iv.implied_vol_scalar(float(upper) * 1.5, spot, strike, tau, rate))
    assert np.isnan(iv.implied_vol_scalar(float(lower), spot, strike, tau, rate))
    assert np.isnan(iv.implied_vol_scalar(10.0, spot, strike, 0.0, rate))
    assert np.isnan(iv.implied_vol_scalar(10.0, 0.0, strike, tau, rate))


def test_no_arbitrage_bounds_values() -> None:
    spot, strike, tau, rate, q = 100.0, 90.0, 1.0, 0.03, 0.01
    lower, upper = iv.no_arbitrage_bounds(spot, strike, tau, rate, OptionType.CALL, q)
    assert float(lower) == pytest.approx(100 * np.exp(-0.01) - 90 * np.exp(-0.03))
    assert float(upper) == pytest.approx(100 * np.exp(-0.01))
    lower_p, upper_p = iv.no_arbitrage_bounds(spot, strike, tau, rate, OptionType.PUT, q)
    assert float(lower_p) == pytest.approx(0.0)
    assert float(upper_p) == pytest.approx(90 * np.exp(-0.03))


def test_low_vega_quotes_still_converge_via_bisection() -> None:
    """Deep OTM with days to go: vega is ~0, so bare Newton would diverge."""
    spot, strike, tau, rate = 100.0, 160.0, 5 / 365, 0.03
    true_vol = 0.9
    price = float(bs.price(spot, strike, tau, rate, true_vol, OptionType.CALL))
    result = iv.implied_vol(
        price, spot, strike, tau, rate, OptionType.CALL, return_diagnostics=True
    )
    assert isinstance(result, iv.ImpliedVolResult)
    assert float(result.vol) == pytest.approx(true_vol, abs=1e-3)
    assert bool(result.converged)


def test_solver_brackets_an_extreme_volatility() -> None:
    """A 500% vol still inverts: the bracket runs to 1000%."""
    price = float(bs.price(100.0, 100.0, 1.0, 0.0, 5.0, OptionType.CALL))
    assert iv.implied_vol_scalar(price, 100.0, 100.0, 1.0, 0.0) == pytest.approx(5.0, abs=1e-5)


def test_calls_and_puts_at_the_same_strike_imply_the_same_vol() -> None:
    """Parity-consistent prices must invert to identical vols -- no kink at the forward."""
    spot, rate, q, tau = 100.0, 0.04, 0.015, 0.6
    strikes = np.array([85.0, 95.0, 105.0, 115.0])
    true = np.array([0.30, 0.25, 0.21, 0.20])
    calls = bs.price(spot, strikes, tau, rate, true, OptionType.CALL, q)
    puts = np.asarray(calls) - spot * np.exp(-q * tau) + strikes * np.exp(-rate * tau)
    from_calls = np.asarray(iv.implied_vol(calls, spot, strikes, tau, rate, OptionType.CALL, q))
    from_puts = np.asarray(iv.implied_vol(puts, spot, strikes, tau, rate, OptionType.PUT, q))
    assert np.max(np.abs(from_calls - from_puts)) < 1e-8


def test_manaster_koehler_seed_is_finite_everywhere() -> None:
    spot = np.array([100.0, 100.0, 100.0])
    strike = np.array([100.0, 1e-6, 1e6])
    tau = np.array([1.0, 1e-8, 10.0])
    seed = iv._initial_guess(spot, strike, tau, np.zeros(3), np.zeros(3))
    assert np.all(np.isfinite(seed))
    assert np.all((seed >= 0.05) & (seed <= 3.0))


def test_return_type_without_diagnostics_is_a_bare_array() -> None:
    out = iv.implied_vol(10.45, 100.0, 100.0, 1.0, 0.05)
    assert isinstance(out, np.ndarray)
    assert not isinstance(out, iv.ImpliedVolResult)
