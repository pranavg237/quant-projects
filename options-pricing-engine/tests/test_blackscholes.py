"""Black-Scholes price and Greeks.

The Greeks are checked against **central finite differences of the price function**, which
is an independent derivation: the analytic formulas and the price formula share only the
normal CDF, so an algebra slip in a Greek cannot hide.
"""

from __future__ import annotations

import numpy as np
import pytest

from optpricing import blackscholes as bs
from optpricing.types import OptionType

# Hull, *Options, Futures and Other Derivatives*, worked example:
# S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81.
HULL = (42.0, 40.0, 0.5, 0.10, 0.20)

# Standard reference case used throughout: ATM, 1y, r=5%, sigma=20%.
ATM = (100.0, 100.0, 1.0, 0.05, 0.20)


def test_hull_textbook_call_and_put() -> None:
    call = float(bs.price(*HULL, OptionType.CALL))
    put = float(bs.price(*HULL, OptionType.PUT))
    assert call == pytest.approx(4.7594, abs=5e-4)
    assert put == pytest.approx(0.8086, abs=5e-4)


def test_atm_reference_values() -> None:
    """Values that any Black-Scholes implementation must reproduce to 6 decimals."""
    g = bs.greeks(*ATM, OptionType.CALL)
    assert float(g.price) == pytest.approx(10.450584, abs=1e-6)
    assert float(g.delta) == pytest.approx(0.636831, abs=1e-6)
    assert float(g.gamma) == pytest.approx(0.018762, abs=1e-6)
    assert float(g.vega) == pytest.approx(37.524035, abs=1e-5)
    assert float(g.theta) == pytest.approx(-6.414028, abs=1e-5)
    assert float(g.rho) == pytest.approx(53.232482, abs=1e-5)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("spot", [70.0, 100.0, 130.0])
@pytest.mark.parametrize("tau", [0.05, 0.5, 2.0])
def test_put_call_parity(option_type: OptionType, spot: float, tau: float) -> None:
    """C - P = S e^{-q tau} - K e^{-r tau}, exactly, for every parameter set."""
    k, r, sigma, q = 100.0, 0.04, 0.25, 0.015
    call = bs.price(spot, k, tau, r, sigma, OptionType.CALL, q)
    put = bs.price(spot, k, tau, r, sigma, OptionType.PUT, q)
    gap = bs.put_call_parity_gap(call, put, spot, k, tau, r, q)
    assert float(np.abs(gap)) < 1e-10


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_greeks_against_finite_differences(option_type: OptionType) -> None:
    s, k, tau, r, sigma, q = 103.0, 100.0, 0.7, 0.035, 0.27, 0.012

    def price(
        spot: float = s,
        strike: float = k,
        t: float = tau,
        rate: float = r,
        vol: float = sigma,
        div: float = q,
    ) -> float:
        return float(bs.price(spot, strike, t, rate, vol, option_type, div))

    h_s, h_v, h_t, h_r, h_k = 1e-4 * s, 1e-5, 1e-6, 1e-6, 1e-4 * k

    fd_delta = (price(spot=s + h_s) - price(spot=s - h_s)) / (2 * h_s)
    fd_gamma = (price(spot=s + h_s) - 2 * price() + price(spot=s - h_s)) / h_s**2
    fd_vega = (price(vol=sigma + h_v) - price(vol=sigma - h_v)) / (2 * h_v)
    # theta is d/dt, i.e. minus d/dtau.
    fd_theta = -(price(t=tau + h_t) - price(t=tau - h_t)) / (2 * h_t)
    fd_rho = (price(rate=r + h_r) - price(rate=r - h_r)) / (2 * h_r)
    fd_dual = (price(strike=k + h_k) - price(strike=k - h_k)) / (2 * h_k)

    assert float(bs.delta(s, k, tau, r, sigma, option_type, q)) == pytest.approx(fd_delta, rel=1e-5)
    assert float(bs.gamma(s, k, tau, r, sigma, q)) == pytest.approx(fd_gamma, rel=1e-4)
    assert float(bs.vega(s, k, tau, r, sigma, q)) == pytest.approx(fd_vega, rel=1e-6)
    assert float(bs.theta(s, k, tau, r, sigma, option_type, q)) == pytest.approx(fd_theta, rel=1e-4)
    assert float(bs.rho(s, k, tau, r, sigma, option_type, q)) == pytest.approx(fd_rho, rel=1e-4)
    assert float(bs.dual_delta(s, k, tau, r, sigma, option_type, q)) == pytest.approx(
        fd_dual, rel=1e-5
    )


def test_second_order_greeks_against_finite_differences() -> None:
    """Vanna, volga and charm are cross-derivatives, so they get their own check."""
    s, k, tau, r, sigma, q = 96.0, 100.0, 1.3, 0.03, 0.22, 0.01
    h_s, h_v, h_t = 1e-3 * s, 1e-5, 1e-6

    def delta_at(vol: float) -> float:
        return float(bs.delta(s, k, tau, r, vol, OptionType.CALL, q))

    def vega_at(vol: float) -> float:
        return float(bs.vega(s, k, tau, r, vol, q))

    def delta_at_tau(t: float) -> float:
        return float(bs.delta(s, k, t, r, sigma, OptionType.CALL, q))

    fd_vanna = (delta_at(sigma + h_v) - delta_at(sigma - h_v)) / (2 * h_v)
    fd_volga = (vega_at(sigma + h_v) - vega_at(sigma - h_v)) / (2 * h_v)
    fd_charm = -(delta_at_tau(tau + h_t) - delta_at_tau(tau - h_t)) / (2 * h_t)

    assert float(bs.vanna(s, k, tau, r, sigma, q)) == pytest.approx(fd_vanna, rel=1e-4)
    assert float(bs.volga(s, k, tau, r, sigma, q)) == pytest.approx(fd_volga, rel=1e-4)
    assert float(bs.charm(s, k, tau, r, sigma, OptionType.CALL, q)) == pytest.approx(
        fd_charm, rel=1e-4
    )
    # Vanna is identical for calls and puts (parity is linear in S and independent of vol).
    assert float(bs.vanna(s, k, tau, r, sigma, q)) == pytest.approx(
        (
            float(bs.delta(s, k, tau, r, sigma + h_v, OptionType.PUT, q))
            - float(bs.delta(s, k, tau, r, sigma - h_v, OptionType.PUT, q))
        )
        / (2 * h_v),
        rel=1e-4,
    )
    assert abs(h_s) > 0  # keeps the unused-variable linter honest about intent


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_no_arbitrage_bounds_hold(option_type: OptionType) -> None:
    s, k, tau, r, q = 100.0, 110.0, 0.8, 0.03, 0.01
    vols = np.array([0.01, 0.1, 0.5, 2.0, 5.0])
    prices = np.asarray(bs.price(s, k, tau, r, vols, option_type, q))
    disc_s, disc_k = s * np.exp(-q * tau), k * np.exp(-r * tau)
    if option_type is OptionType.CALL:
        assert np.all(prices >= max(disc_s - disc_k, 0.0) - 1e-12)
        assert np.all(prices <= disc_s + 1e-12)
    else:
        assert np.all(prices >= max(disc_k - disc_s, 0.0) - 1e-12)
        assert np.all(prices <= disc_k + 1e-12)
    # Strictly increasing in volatility -- this is what makes implied vol well posed.
    assert np.all(np.diff(prices) > 0)


def test_degenerate_inputs_give_intrinsic_value() -> None:
    assert float(bs.price(110.0, 100.0, 0.0, 0.05, 0.2, OptionType.CALL)) == pytest.approx(10.0)
    assert float(bs.price(90.0, 100.0, 0.0, 0.05, 0.2, OptionType.CALL)) == pytest.approx(0.0)
    assert float(bs.price(90.0, 100.0, 0.0, 0.05, 0.2, OptionType.PUT)) == pytest.approx(10.0)
    # Zero vol with time left: the deterministic forward intrinsic, discounted.
    expected = 110.0 - 100.0 * np.exp(-0.05)
    assert float(bs.price(110.0, 100.0, 1.0, 0.05, 0.0, OptionType.CALL)) == pytest.approx(expected)
    assert float(bs.price(110.0, 100.0, -1.0, 0.05, 0.2, OptionType.CALL)) == pytest.approx(10.0)
    # Greeks are all zero where there is no optionality left.
    assert float(bs.gamma(110.0, 100.0, 0.0, 0.05, 0.2)) == 0.0
    assert float(bs.vega(110.0, 100.0, 0.0, 0.05, 0.2)) == 0.0
    assert float(bs.theta(110.0, 100.0, 0.0, 0.05, 0.2, OptionType.CALL)) == 0.0
    assert float(bs.vanna(110.0, 100.0, 0.0, 0.05, 0.2)) == 0.0
    assert float(bs.volga(110.0, 100.0, 0.0, 0.05, 0.2)) == 0.0
    assert float(bs.charm(110.0, 100.0, 0.0, 0.05, 0.2, OptionType.CALL)) == 0.0


def test_broadcasting_over_arrays() -> None:
    spots = np.array([90.0, 100.0, 110.0])
    taus = np.array([[0.25], [1.0]])
    prices = np.asarray(bs.price(spots, 100.0, taus, 0.05, 0.2, OptionType.CALL))
    assert prices.shape == (2, 3)
    # Longer expiry is worth more for every spot (no dividends).
    assert np.all(prices[1] > prices[0])
    # And every Greek broadcasts the same way.
    assert np.asarray(bs.greeks(spots, 100.0, taus, 0.05, 0.2).delta).shape == (2, 3)


def test_forward_price_and_gamma_symmetry() -> None:
    assert float(bs.forward_price(100.0, 1.0, 0.05, 0.02)) == pytest.approx(100.0 * np.exp(0.03))
    call_gamma = float(bs.gamma(100.0, 105.0, 0.5, 0.03, 0.2, 0.01))
    # Gamma has no option_type argument precisely because it is the same for both.
    h = 1e-3
    fd_put = (
        float(bs.delta(100.0 + h, 105.0, 0.5, 0.03, 0.2, OptionType.PUT, 0.01))
        - float(bs.delta(100.0 - h, 105.0, 0.5, 0.03, 0.2, OptionType.PUT, 0.01))
    ) / (2 * h)
    assert call_gamma == pytest.approx(fd_put, rel=1e-5)


def test_option_type_accepts_strings_and_rejects_nonsense() -> None:
    assert float(bs.price(*ATM, "call")) == float(bs.price(*ATM, OptionType.CALL))
    assert float(bs.price(*ATM, "PUT")) == float(bs.price(*ATM, OptionType.PUT))
    with pytest.raises(ValueError, match="unknown option type"):
        bs.price(*ATM, "straddle")


def test_greeks_as_dict_round_trip() -> None:
    d = bs.greeks(*ATM, OptionType.CALL).as_dict()
    assert set(d) == {
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "vanna",
        "volga",
        "charm",
        "dual_delta",
    }
