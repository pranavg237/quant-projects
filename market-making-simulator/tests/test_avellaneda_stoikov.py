"""The Avellaneda-Stoikov closed-form solution.

The most valuable check here is the **analytic average spread**: the model's time-averaged
spread at the paper's parameters is 1.4908, and the paper reports 1.49. That single number
pins the whole formula -- both the inventory-risk term and the monopolistic markup, and the
relative weight between them -- against an independent source.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mmsim.avellaneda_stoikov import (
    AvellanedaStoikovParams,
    HorizonMode,
    inventory_skew,
    optimal_quotes,
    optimal_spread,
    reservation_price,
)
from mmsim.strategies import average_optimal_spread


def test_average_spread_matches_the_published_value(paper_params: AvellanedaStoikovParams) -> None:
    """Avellaneda & Stoikov (2008) Table 1 reports an average spread of 1.49."""
    assert average_optimal_spread(paper_params) == pytest.approx(1.4908, abs=1e-4)


def test_optimal_spread_formula(paper_params: AvellanedaStoikovParams) -> None:
    r"""gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/kappa), evaluated by hand."""
    tau = 1.0
    expected = 0.1 * 4.0 * tau + (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5)
    assert float(optimal_spread(tau, paper_params)) == pytest.approx(expected, rel=1e-12)
    # At tau = 0 only the markup survives.
    assert float(optimal_spread(0.0, paper_params)) == pytest.approx(
        (2.0 / 0.1) * np.log(1.0 + 0.1 / 1.5), rel=1e-12
    )


def test_spread_is_linear_in_time_remaining(paper_params: AvellanedaStoikovParams) -> None:
    taus = np.linspace(0.0, 2.0, 21)
    spreads = np.asarray(optimal_spread(taus, paper_params))
    slope = np.diff(spreads) / np.diff(taus)
    assert np.allclose(slope, paper_params.gamma * paper_params.sigma**2)


def test_risk_neutral_limit_is_two_over_kappa() -> None:
    r"""As gamma -> 0 the markup tends to 2/kappa, not to infinity."""
    kappa = 1.5
    for gamma in (1e-12, 1e-10, 1e-9):
        params = AvellanedaStoikovParams(gamma=gamma, kappa=kappa, sigma=2.0)
        assert float(optimal_spread(0.0, params)) == pytest.approx(2.0 / kappa, rel=1e-6)
    # The markup approaches 2/kappa from *below*: it is decreasing in gamma, not
    # increasing. A more risk-averse maker charges a smaller monopolistic markup, because
    # it prefers a likelier small gain to a rarer large one. The total spread still widens
    # with gamma -- see the next test -- because the inventory-risk term grows faster.
    markups = [
        float(optimal_spread(0.0, AvellanedaStoikovParams(gamma=g, kappa=kappa, sigma=2.0)))
        for g in (1.0, 0.1, 0.01, 0.001)
    ]
    assert all(a < b for a, b in itertools.pairwise(markups))
    assert markups[-1] == pytest.approx(2.0 / kappa, rel=1e-3)
    assert markups[-1] < 2.0 / kappa


def test_total_spread_still_widens_with_risk_aversion() -> None:
    """Despite the markup shrinking, the inventory term dominates and the spread widens."""
    totals = [
        float(optimal_spread(1.0, AvellanedaStoikovParams(gamma=g, kappa=1.5, sigma=2.0)))
        for g in (0.001, 0.01, 0.1, 1.0)
    ]
    assert all(a < b for a, b in itertools.pairwise(totals))


def test_reservation_price_moves_against_inventory(paper_params: AvellanedaStoikovParams) -> None:
    """A long maker values the asset below the mid; a short maker above it."""
    mid = 100.0
    long_r = float(reservation_price(mid, 5.0, 1.0, paper_params))
    flat_r = float(reservation_price(mid, 0.0, 1.0, paper_params))
    short_r = float(reservation_price(mid, -5.0, 1.0, paper_params))
    assert long_r < flat_r == mid < short_r
    assert mid - long_r == pytest.approx(short_r - mid)  # symmetric in q
    assert flat_r - long_r == pytest.approx(5.0 * 0.1 * 4.0 * 1.0)


def test_quotes_skew_but_keep_a_constant_width(paper_params: AvellanedaStoikovParams) -> None:
    """The whole mechanism: inventory moves both quotes together, never the width."""
    widths = []
    previous_bid = np.inf
    for q in (-4.0, -2.0, 0.0, 2.0, 4.0):
        bid, ask = optimal_quotes(100.0, q, 0.0, paper_params)
        assert bid is not None and ask is not None
        widths.append(ask - bid)
        assert bid < previous_bid  # both quotes fall as inventory rises
        previous_bid = bid
    assert np.allclose(widths, widths[0])
    assert widths[0] == pytest.approx(float(optimal_spread(1.0, paper_params)))


def test_skew_scales_with_gamma_sigma_squared_and_time() -> None:
    base = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, sigma=2.0, horizon=1.0)
    assert float(inventory_skew(1.0, 1.0, base)) == pytest.approx(0.1 * 4.0)
    doubled_gamma = base.__class__(gamma=0.2, kappa=1.5, sigma=2.0, horizon=1.0)
    assert float(inventory_skew(1.0, 1.0, doubled_gamma)) == pytest.approx(
        2.0 * float(inventory_skew(1.0, 1.0, base))
    )
    doubled_sigma = base.__class__(gamma=0.1, kappa=1.5, sigma=4.0, horizon=1.0)
    assert float(inventory_skew(1.0, 1.0, doubled_sigma)) == pytest.approx(
        4.0 * float(inventory_skew(1.0, 1.0, base))
    )
    assert float(inventory_skew(1.0, 2.0, base)) == pytest.approx(
        2.0 * float(inventory_skew(1.0, 1.0, base))
    )


def test_finite_horizon_control_vanishes_at_the_bell(
    paper_params: AvellanedaStoikovParams,
) -> None:
    """The model's defect: as t -> T both the skew and the risk premium go to zero."""
    assert float(inventory_skew(5.0, paper_params.time_remaining(0.0), paper_params)) > 0
    assert float(inventory_skew(5.0, paper_params.time_remaining(1.0), paper_params)) == 0.0
    early = optimal_quotes(100.0, 5.0, 0.0, paper_params)
    late = optimal_quotes(100.0, 5.0, 1.0, paper_params)
    assert early[1] is not None and late[1] is not None
    assert early[1] - early[0] > late[1] - late[0]  # type: ignore[operator]


def test_stationary_mode_is_time_homogeneous(
    stationary_params: AvellanedaStoikovParams,
) -> None:
    quotes = [optimal_quotes(100.0, 3.0, t, stationary_params) for t in (0.0, 0.5, 0.99, 5.0)]
    assert all(q == quotes[0] for q in quotes)
    assert stationary_params.time_remaining(99.0) == stationary_params.horizon
    assert average_optimal_spread(stationary_params) == pytest.approx(
        float(optimal_spread(stationary_params.horizon, stationary_params))
    )


def test_inventory_limit_withdraws_a_side() -> None:
    params = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, sigma=2.0, max_inventory=3.0)
    assert optimal_quotes(100.0, 3.0, 0.0, params)[0] is None  # no bid when max long
    assert optimal_quotes(100.0, 3.0, 0.0, params)[1] is not None
    assert optimal_quotes(100.0, -3.0, 0.0, params)[1] is None  # no ask when max short
    assert optimal_quotes(100.0, 3.5, 0.0, params)[0] is None
    bid, ask = optimal_quotes(100.0, 0.0, 0.0, params)
    assert bid is not None and ask is not None


def test_parameter_validation() -> None:
    with pytest.raises(ValueError, match="gamma"):
        AvellanedaStoikovParams(gamma=-0.1)
    with pytest.raises(ValueError, match="kappa"):
        AvellanedaStoikovParams(kappa=0.0)
    with pytest.raises(ValueError, match="arrival_rate"):
        AvellanedaStoikovParams(arrival_rate=0.0)
    with pytest.raises(ValueError, match="sigma"):
        AvellanedaStoikovParams(sigma=-1.0)
    with pytest.raises(ValueError, match="horizon"):
        AvellanedaStoikovParams(horizon=0.0)
    with pytest.raises(ValueError, match="max_inventory"):
        AvellanedaStoikovParams(max_inventory=0.0)


def test_time_remaining_is_floored_at_zero(paper_params: AvellanedaStoikovParams) -> None:
    assert paper_params.time_remaining(2.0) == 0.0
    assert paper_params.horizon_mode is HorizonMode.FINITE
