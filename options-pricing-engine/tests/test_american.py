"""De-Americanisation: the lattice premium, the per-quote fixed point, the forward loop.

Three kinds of chain, each with a known right answer:

* an **American** chain (European Black-Scholes price plus the lattice premium at a known
  smile): the corrected surface must recover the smile and the forward exactly, and the
  uncorrected one must show the SPY symptoms (forward too low, call/put gap);
* a **European chain with nothing to correct** (r = q = 0, where early exercise is never
  optimal): the correction must be zero to round-off -- the control that catches a
  premium computed against the wrong European price;
* a **European chain with carry**: the correction must make recovery *worse*, so that
  it improving SPY is evidence rather than a free fit.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import NY, reference_vol
from optpricing import american, binomial, parity
from optpricing import blackscholes as bs
from optpricing import data as data_mod
from optpricing import surface as surface_mod
from optpricing.types import ExerciseStyle, OptionType

SPOT = 500.0
STEPS = 200
EXPIRY_DAYS = (21, 91, 365, 640)


def _chain(rate: float, dividend: float, exercise: ExerciseStyle) -> data_mod.ChainSnapshot:
    """A chain from :func:`reference_vol`, priced European or American (same lattice)."""
    asof = dt.datetime(2026, 9, 18, 11, 0, tzinfo=NY)
    rows: list[dict[str, object]] = []
    for days in EXPIRY_DAYS:
        expiry = (asof + dt.timedelta(days=days)).date()
        tau = data_mod.year_fraction(asof, expiry)
        forward = SPOT * np.exp((rate - dividend) * tau)
        strikes = np.unique(np.round(forward * np.exp(np.linspace(-0.3, 0.2, 31)), 0))
        vols = reference_vol(np.log(strikes / forward), tau)
        for side in (OptionType.CALL, OptionType.PUT):
            mids = np.asarray(bs.price(SPOT, strikes, tau, rate, vols, side, dividend))
            if exercise is ExerciseStyle.AMERICAN:
                mids = mids + american.early_exercise_premium(
                    SPOT, strikes, tau, rate, dividend, vols, side, STEPS
                )
            for strike, mid in zip(strikes, mids, strict=True):
                rows.append(
                    {
                        "expiry": expiry,
                        "option_type": side.value,
                        "strike": float(strike),
                        "bid": float(mid) - 0.01,
                        "ask": float(mid) + 0.01,
                        "volume": 10.0,
                        "open_interest": 10.0,
                        "tau": tau,
                    }
                )
    return data_mod.ChainSnapshot("SYN", asof, SPOT, pd.DataFrame(rows))


def _build(
    snap: data_mod.ChainSnapshot, rate: float, exercise: str, **kwargs: object
) -> surface_mod.SurfaceBuild:
    clean, _ = data_mod.clean_chain(snap)
    curve = data_mod.RateCurve([0.01, 30.0], [rate, rate])
    return surface_mod.build_surface_detailed(snap, clean, curve, exercise=exercise, **kwargs)  # type: ignore[arg-type]


def _max_vol_error(surf: pd.DataFrame, rate: float, dividend: float) -> float:
    """Worst |implied vol - reference smile|, with k measured at the *true* forward."""
    true_fwd = SPOT * np.exp((rate - dividend) * surf["tau"].to_numpy())
    k = np.log(surf["strike"].to_numpy() / true_fwd)
    truth = reference_vol(k, surf["tau"].to_numpy())  # vectorises over tau too
    return float(np.max(np.abs(surf["implied_vol"].to_numpy() - truth)))


RATE, DIV = 0.042, 0.004


@pytest.fixture(scope="module")
def american_chain() -> data_mod.ChainSnapshot:
    return _chain(RATE, DIV, ExerciseStyle.AMERICAN)


@pytest.fixture(scope="module")
def american_builds(american_chain):
    return {ex: _build(american_chain, RATE, ex) for ex in ("american", "european")}


# --- the lattice premium ---------------------------------------------------------------


@pytest.mark.parametrize("side", [OptionType.CALL, OptionType.PUT])
def test_batched_premium_matches_the_scalar_tree(side) -> None:
    strikes = np.array([380.0, 450.0, 500.0, 540.0, 620.0])
    vols = np.array([0.32, 0.24, 0.2, 0.18, 0.21])
    tau, r, q = 1.3, 0.04, 0.02
    got = american.early_exercise_premium(SPOT, strikes, tau, r, q, vols, side, 150)
    ref = [
        binomial.price(SPOT, k, tau, r, v, side, ExerciseStyle.AMERICAN, 150, "crr", q)
        - binomial.price(SPOT, k, tau, r, v, side, ExerciseStyle.EUROPEAN, 150, "crr", q)
        for k, v in zip(strikes, vols, strict=True)
    ]
    assert np.allclose(got, np.maximum(ref, 0.0), atol=1e-10, rtol=0)
    assert np.all(got >= 0.0)


def test_premium_vanishes_where_early_exercise_is_never_optimal() -> None:
    strikes = np.linspace(350.0, 650.0, 13)
    vols = np.full(strikes.size, 0.25)
    # No interest on the strike: an American put is European. No dividend: same for calls.
    puts = american.early_exercise_premium(SPOT, strikes, 1.5, 0.0, 0.02, vols, "put")
    calls = american.early_exercise_premium(SPOT, strikes, 1.5, 0.04, 0.0, vols, "call")
    assert np.max(puts) < 1e-10
    assert np.max(calls) < 1e-10


def test_premium_edge_cases() -> None:
    k = np.array([450.0, 500.0])
    assert np.all(np.isnan(american.early_exercise_premium(SPOT, k, 1.0, 0.04, 0.0, np.nan, "put")))
    expired = american.early_exercise_premium(SPOT, k, 0.0, 0.04, 0.0, np.array([0.2, 0.2]), "put")
    assert np.all(expired == 0.0)
    # A carry too large for the lattice spacing makes p leave [0, 1]: flagged, not priced.
    bad = american.early_exercise_premium(SPOT, k, 1.0, 5.0, 0.0, np.array([0.01, 0.01]), "put", 4)
    assert np.all(np.isnan(bad))


# --- the per-quote fixed point ---------------------------------------------------------


@pytest.mark.parametrize("tau", [0.05, 0.5, 1.75])
@pytest.mark.parametrize("side", [OptionType.CALL, OptionType.PUT])
def test_deamericanise_round_trip(tau, side) -> None:
    """American price built at a known vol in, that vol and the European price out."""
    r, q = 0.045, 0.012
    fwd = SPOT * np.exp((r - q) * tau)
    span = min(0.3, 0.5 * np.sqrt(tau))  # about 2.5 standard deviations: quotes with value
    ks = np.linspace(-span, -0.01, 12) if side is OptionType.PUT else np.linspace(0.0, span, 12)
    k = fwd * np.exp(ks)
    true_vol = reference_vol(np.log(k / fwd), tau)
    european = np.asarray(bs.price(SPOT, k, tau, r, true_vol, side, q))
    prices = european + american.early_exercise_premium(SPOT, k, tau, r, q, true_vol, side)
    res = american.deamericanise(prices, SPOT, k, tau, r, q, side)
    assert res.converged.all()
    assert np.max(np.abs(res.vol - true_vol)) < 1e-8
    assert np.max(np.abs(res.european_price - european)) < 1e-6


def test_fixed_point_contracts_geometrically() -> None:
    """The map's slope is premium-vega / option-vega: small for OTM quotes."""
    tau, r, q = 1.75, 0.045, 0.0
    fwd = SPOT * np.exp((r - q) * tau)
    k = fwd * np.exp(np.linspace(-0.4, -0.01, 20))
    vol = reference_vol(np.log(k / fwd), tau)
    prices = np.asarray(bs.price(SPOT, k, tau, r, vol, "put", q))
    prices += american.early_exercise_premium(SPOT, k, tau, r, q, vol, "put")
    res = american.deamericanise(prices, SPOT, k, tau, r, q, "put", tol=1e-12)
    steps = np.array(res.step_history)
    assert res.converged.all()
    assert 0.0 < res.contraction < 0.3
    # Each iteration shrinks the largest step by at least the measured contraction.
    shrinking = steps[:-1] > 1e-11
    assert np.all(steps[1:][shrinking] <= res.contraction * steps[:-1][shrinking] * 1.0001)
    # The first correction is the whole premium's worth of vol; it is not small.
    assert steps[0] > 1e-3
    assert res.iterations.max() <= 20


def test_fixed_point_reports_non_convergence_honestly() -> None:
    tau, r = 1.75, 0.045
    k = np.array([450.0, 480.0])
    vol = np.array([0.22, 0.2])
    prices = np.asarray(bs.price(SPOT, k, tau, r, vol, "put")) + american.early_exercise_premium(
        SPOT, k, tau, r, 0.0, vol, "put"
    )
    res = american.deamericanise(prices, SPOT, k, tau, r, 0.0, "put", max_iter=2)
    assert not res.converged.any()
    assert np.all(res.iterations == 2)
    # A quote below the European floor even before any correction stays nan.
    floor = american.deamericanise(np.array([1.0]), SPOT, np.array([300.0]), tau, r, 0.0, "call")
    assert np.isnan(floor.vol[0]) and not floor.converged[0]


def test_independent_fine_lattice_is_recovered_to_its_discretisation_error() -> None:
    """American prices from a 2,000-step tree, not the 200-step construction."""
    tau, r, q = 1.0, 0.045, 0.005
    fwd = SPOT * np.exp((r - q) * tau)
    k = np.round(fwd * np.exp(np.linspace(-0.3, -0.02, 6)))
    vol = reference_vol(np.log(k / fwd), tau)
    prices = np.array(
        [
            binomial.price(SPOT, ki, tau, r, v, "put", ExerciseStyle.AMERICAN, 2000, "crr", q)
            - binomial.price(SPOT, ki, tau, r, v, "put", ExerciseStyle.EUROPEAN, 2000, "crr", q)
            + float(bs.price(SPOT, ki, tau, r, v, "put", q))
            for ki, v in zip(k, vol, strict=True)
        ]
    )
    corrected = american.deamericanise(prices, SPOT, k, tau, r, q, "put").vol
    # max_iter=0 skips the correction: the plain Black-Scholes inversion of the quote.
    uncorrected = american.deamericanise(prices, SPOT, k, tau, r, q, "put", max_iter=0).vol
    # 200 vs 2,000 steps: below 0.02 vol points, and at least ten times the correction.
    assert np.max(np.abs(corrected - vol)) < 2e-4
    assert np.min(np.abs(uncorrected - vol)) > 10 * np.max(np.abs(corrected - vol))


# --- the forward loop and the surface ---------------------------------------------------


def test_corrected_surface_recovers_an_american_chain(american_builds) -> None:
    am, eu = american_builds["american"], american_builds["european"]
    assert am.converged and am.outer_iterations <= 6
    assert am.unconverged_quotes == 0 and am.dropped_by_correction == 0
    true_fwd = SPOT * np.exp((RATE - DIV) * am.forwards["tau"].to_numpy())
    # Corrected: the forward to a hundredth of a basis point...
    assert np.max(np.abs(am.forwards["forward"] / true_fwd - 1.0)) < 1e-6
    # ...uncorrected: biased low, by more the longer the expiry.
    shift = am.forwards["forward_shift_bp"].to_numpy()
    assert np.all(shift > 0) and np.all(np.diff(shift) > 0)
    assert shift[-1] > 20
    # And the smile: exact after correction, visibly wrong before.
    assert _max_vol_error(am.surface, RATE, DIV) < 1e-6
    assert _max_vol_error(eu.surface, RATE, DIV) > 5e-3


def test_european_prices_carry_the_premium_off(american_builds) -> None:
    surf = american_builds["american"].surface
    assert np.allclose(surf["mid_market"] - surf["ee_premium"], surf["mid"])
    assert np.allclose(surf["ask"] - surf["bid"], 0.02)  # the spread is untouched
    puts = surf.loc[surf["option_type"] == "put"]
    assert (puts["ee_premium"] > 0).all()
    # With a positive dividend yield the OTM calls carry a (tiny) premium too.
    calls = surf.loc[surf["option_type"] == "call"]
    assert calls["ee_premium"].max() < 0.05 * puts["ee_premium"].max()


def test_call_put_gap_closes_only_with_the_correction(american_chain) -> None:
    gaps = {}
    for ex in ("american", "european"):
        fitted = _build(american_chain, RATE, ex).forwards
        both = _build(
            american_chain, RATE, ex, otm_only=False, max_abs_log_moneyness=0.05, forwards=fitted
        )
        assert np.allclose(both.forwards["forward"], fitted["forward"], rtol=0, atol=0)
        gaps[ex] = surface_mod.call_put_gap(both.surface)
    assert np.max(np.abs(gaps["american"]["gap_vol_points"])) < 1e-4
    assert gaps["european"]["gap_vol_points"].iloc[-1] < -0.1  # the SPY sign


def test_parity_module_reproduces_the_surface_forward(american_chain, american_builds) -> None:
    """Self-consistency: the surface's forward is the parity module's American forward."""
    am = american_builds["american"]
    clean, _ = data_mod.clean_chain(american_chain)
    curve = data_mod.RateCurve([0.01, 30.0], [RATE, RATE])
    fwd = data_mod.implied_forward_curve(clean, SPOT, curve)
    pairs = parity.matched_pairs(american_chain, clean)
    adj = parity.american_adjusted_forward(pairs, SPOT, curve, fwd, am.surface)
    merged = adj.merge(am.forwards, on="expiry")
    assert np.max(np.abs(merged["forward_adjusted"] / merged["forward"] - 1.0)) < 1e-6


def test_control_nothing_to_correct_means_no_correction() -> None:
    """r = q = 0: American and European coincide, so the two surfaces must too."""
    snap = _chain(0.0, 0.0, ExerciseStyle.EUROPEAN)
    am, eu = _build(snap, 0.0, "american"), _build(snap, 0.0, "european")
    assert am.surface["ee_premium"].abs().max() < 1e-10
    assert np.allclose(am.forwards["forward"], eu.forwards["forward"], rtol=1e-12, atol=0)
    merged = am.surface.merge(eu.surface, on=["expiry", "strike", "option_type"])
    assert len(merged) == len(eu.surface)
    assert np.max(np.abs(merged["implied_vol_x"] - merged["implied_vol_y"])) < 1e-9


def test_control_the_correction_is_falsifiable_on_a_european_chain() -> None:
    """With carry but European prices, the correction must make recovery worse."""
    snap = _chain(RATE, DIV, ExerciseStyle.EUROPEAN)
    am, eu = _build(snap, RATE, "american"), _build(snap, RATE, "european")
    assert _max_vol_error(eu.surface, RATE, DIV) < 1e-6
    assert _max_vol_error(am.surface, RATE, DIV) > 5e-3
    # It pushes the forward the wrong way: above the true one.
    true_fwd = SPOT * np.exp((RATE - DIV) * am.forwards["tau"].to_numpy())
    assert np.all(am.forwards["forward"].to_numpy() > true_fwd)


def test_european_build_is_the_old_pipeline(synthetic_snapshot, flat_rate_curve) -> None:
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    b = surface_mod.build_surface_detailed(
        synthetic_snapshot, clean, flat_rate_curve, exercise="european"
    )
    assert b.outer_iterations == 1 and b.converged and b.forward_history == []
    assert (b.surface["ee_premium"] == 0).all()
    assert (b.surface["mid"] == b.surface["mid_market"]).all()
    assert (b.forwards["forward_shift_bp"] == 0).all()


def test_forward_loop_cap_is_reported(american_chain) -> None:
    b = _build(american_chain, RATE, "american", max_forward_iter=1)
    assert not b.converged
    assert len(b.forward_history) == 1


# --- the committed SPY snapshot ---------------------------------------------------------


def test_spy_correction_converges_and_moves_the_long_forward(spy_build) -> None:
    b = spy_build
    assert b.converged and b.forward_history[-1] < 1e-7
    assert b.unconverged_quotes == 0
    assert b.max_contraction < 0.5
    f = b.forwards
    long_dated = f.loc[f["tau"] > 1.5, "forward_shift_bp"]
    assert ((long_dated > 50) & (long_dated < 150)).all()
    surf = b.surface
    long_puts = surf.loc[(surf["tau"] > 1.5) & (surf["option_type"] == "put"), "ee_premium"]
    assert long_puts.max() > 1.0  # dollars: not negligible at 21 months
