"""Heston characteristic-function pricing.

Validated three independent ways, because a Fourier pricer checked only against itself is
not checked at all:

1. the **Black-Scholes limit** (``xi -> 0`` with ``v0 = theta = sigma^2``) to 1e-9;
2. **two algebraically distinct quadratures** (Gil-Pelaez and Lewis) agreeing to 1e-6;
3. a **full-truncation Euler Monte Carlo**, agreeing to within its own standard error.
"""

from __future__ import annotations

import numpy as np
import pytest

from optpricing import blackscholes as bs
from optpricing import heston as hs
from optpricing.heston import HestonParams
from optpricing.types import OptionType

EQUITY = HestonParams(v0=0.04, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
FELLER_OK = HestonParams(v0=0.04, kappa=3.0, theta=0.06, xi=0.2, rho=-0.7)
STRIKES = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0])


@pytest.mark.parametrize("tau", [0.05, 0.25, 1.0, 3.0])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_black_scholes_limit(tau: float, option_type: OptionType) -> None:
    """With zero vol-of-vol and v0 = theta, Heston *is* Black-Scholes."""
    sigma = 0.25
    params = HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, xi=0.0, rho=-0.5)
    heston_px = np.asarray(hs.price(100.0, STRIKES, tau, 0.03, params, option_type, 0.01))
    bs_px = np.asarray(bs.price(100.0, STRIKES, tau, 0.03, sigma, option_type, 0.01))
    # ~1e-11 relative on a 100-notional option; the residual is Gauss-Legendre truncation.
    assert np.max(np.abs(heston_px - bs_px)) < 5e-9


def test_black_scholes_limit_with_mean_reverting_variance() -> None:
    """xi = 0 but v0 != theta: variance is deterministic, so integrated variance is exact."""
    params = HestonParams(v0=0.09, kappa=2.0, theta=0.04, xi=0.0, rho=0.0)
    tau = 1.5
    integrated = 0.04 * tau + (0.09 - 0.04) * (1 - np.exp(-2.0 * tau)) / 2.0
    effective_vol = np.sqrt(integrated / tau)
    heston_px = np.asarray(hs.price(100.0, STRIKES, tau, 0.02, params, OptionType.CALL))
    bs_px = np.asarray(bs.price(100.0, STRIKES, tau, 0.02, effective_vol, OptionType.CALL))
    assert np.max(np.abs(heston_px - bs_px)) < 5e-9


@pytest.mark.parametrize("tau", [0.02, 0.1, 0.5, 2.0, 5.0])
def test_gil_pelaez_and_lewis_agree(tau: float) -> None:
    """Two algebraically equivalent formulas -- disagreement means a quadrature problem."""
    a = np.asarray(hs.price(100.0, STRIKES, tau, 0.025, EQUITY, OptionType.CALL))
    b = np.asarray(hs.price(100.0, STRIKES, tau, 0.025, EQUITY, OptionType.CALL, method="lewis"))
    assert np.max(np.abs(a - b)) < 1e-6


@pytest.mark.parametrize("params", [EQUITY, FELLER_OK])
def test_monte_carlo_agrees_with_fourier(params: HestonParams) -> None:
    """A completely separate derivation: full-truncation Euler simulation."""
    tau = 1.0
    strikes = np.array([85.0, 100.0, 115.0])
    fourier = np.asarray(hs.price(100.0, strikes, tau, 0.025, params, OptionType.CALL))
    simulated, se = hs.mc_price(
        100.0, strikes, tau, 0.025, params, OptionType.CALL, n_paths=200_000, n_steps=400, seed=3
    )
    z = (np.asarray(simulated) - fourier) / np.asarray(se)
    # Euler discretisation leaves a small bias on top of the sampling error, so allow 4 SE.
    assert np.max(np.abs(z)) < 4.0


@pytest.mark.parametrize("tau", [0.1, 1.0, 4.0])
def test_put_call_parity_holds_under_heston(tau: float) -> None:
    calls = np.asarray(hs.price(100.0, STRIKES, tau, 0.03, EQUITY, OptionType.CALL, 0.012))
    puts = np.asarray(hs.price(100.0, STRIKES, tau, 0.03, EQUITY, OptionType.PUT, 0.012))
    gap = np.asarray(bs.put_call_parity_gap(calls, puts, 100.0, STRIKES, tau, 0.03, 0.012))
    assert np.max(np.abs(gap)) < 1e-8


def test_characteristic_function_properties() -> None:
    """psi(0) = 1 and psi(-i) = E[S_T/F] = 1, since the CF is written drift-free."""
    for tau in (0.1, 1.0, 5.0):
        assert complex(hs.char_func(0.0, tau, EQUITY)) == pytest.approx(1.0 + 0j, abs=1e-12)
        assert complex(hs.char_func(-1j, tau, EQUITY)) == pytest.approx(1.0 + 0j, abs=1e-10)
    # Hermitian symmetry: psi(-u) = conj(psi(u)) for real u.
    u = np.array([0.5, 2.0, 10.0])
    assert np.allclose(hs.char_func(-u, 1.0, EQUITY), np.conj(hs.char_func(u, 1.0, EQUITY)))
    # And |psi| <= 1 for real arguments.
    assert np.all(np.abs(hs.char_func(np.linspace(0, 200, 500), 1.0, EQUITY)) <= 1.0 + 1e-12)


def test_no_branch_cut_blowup_at_long_maturities() -> None:
    """The classic Heston bug: the original `g` grouping goes wrong for large tau."""
    for tau in (5.0, 10.0, 20.0, 30.0):
        px = np.asarray(hs.price(100.0, STRIKES, tau, 0.02, EQUITY, OptionType.CALL))
        assert np.all(np.isfinite(px))
        assert np.all(px >= 0.0)
        assert np.all(px <= 100.0 + 1e-8)  # a call is never worth more than the spot
        assert np.all(np.diff(px) < 0)  # monotonically decreasing in strike


def test_prices_are_monotone_and_convex_in_strike() -> None:
    strikes = np.linspace(60.0, 160.0, 101)
    px = np.asarray(hs.price(100.0, strikes, 1.0, 0.02, EQUITY, OptionType.CALL))
    assert np.all(np.diff(px) < 0)
    assert np.all(np.diff(px, 2) > -1e-10)  # convex => non-negative implied density


def test_heston_produces_a_downward_sloping_skew() -> None:
    """Negative rho must generate a put skew; positive rho must reverse it."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    down = hs.implied_vol_surface(100.0, strikes, [1.0], 0.02, EQUITY)[0]
    up = hs.implied_vol_surface(100.0, strikes, [1.0], 0.02, EQUITY.replace(rho=+0.5711))[0]
    assert down[0] > down[-1]
    assert up[0] < up[-1]


def test_vol_of_vol_adds_curvature() -> None:
    strikes = np.array([80.0, 100.0, 125.0])
    flat = HestonParams(v0=0.04, kappa=2.0, theta=0.04, xi=0.05, rho=0.0)
    curved = flat.replace(xi=1.2)
    flat_smile = hs.implied_vol_surface(100.0, strikes, [1.0], 0.0, flat)[0]
    curved_smile = hs.implied_vol_surface(100.0, strikes, [1.0], 0.0, curved)[0]

    def curvature(v: np.ndarray) -> float:
        return float(v[0] - 2 * v[1] + v[2])

    assert curvature(curved_smile) > curvature(flat_smile)


def test_zero_tau_returns_intrinsic() -> None:
    px = np.asarray(hs.price(110.0, np.array([100.0, 120.0]), 0.0, 0.03, EQUITY))
    assert px[0] == pytest.approx(10.0)
    assert px[1] == pytest.approx(0.0)


def test_parameter_validation_and_helpers() -> None:
    with pytest.raises(ValueError, match="v0 must be"):
        HestonParams(-0.1, 1.0, 0.04, 0.5, -0.5)
    with pytest.raises(ValueError, match="theta must be"):
        HestonParams(0.04, 1.0, -0.04, 0.5, -0.5)
    with pytest.raises(ValueError, match="kappa must be"):
        HestonParams(0.04, 0.0, 0.04, 0.5, -0.5)
    with pytest.raises(ValueError, match="xi must be"):
        HestonParams(0.04, 1.0, 0.04, -0.5, -0.5)
    with pytest.raises(ValueError, match="rho must be"):
        HestonParams(0.04, 1.0, 0.04, 0.5, -1.5)
    with pytest.raises(ValueError, match="expected 5 parameters"):
        HestonParams.from_array([0.1, 0.2])

    assert EQUITY.feller_ratio == pytest.approx(2 * EQUITY.kappa * EQUITY.theta / EQUITY.xi**2)
    assert not EQUITY.satisfies_feller
    assert FELLER_OK.satisfies_feller
    assert HestonParams(0.04, 1.0, 0.04, 0.0, 0.0).feller_ratio == float("inf")
    assert HestonParams.from_array(EQUITY.to_array()) == EQUITY
    assert EQUITY.replace(rho=0.1).rho == 0.1


def test_unknown_pricing_method_rejected() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        hs.price(100.0, 100.0, 1.0, 0.02, EQUITY, method="carr-madan")


def test_quadrature_is_converged_at_the_default_settings() -> None:
    """Doubling the nodes and widening the domain must not move the price."""
    for tau in (0.02, 0.3, 3.0):
        base = np.asarray(hs.price(100.0, STRIKES, tau, 0.025, EQUITY))
        fine = np.asarray(hs.price(100.0, STRIKES, tau, 0.025, EQUITY, n_quad=2048))
        assert np.max(np.abs(base - fine)) < 1e-7
