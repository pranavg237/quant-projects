"""Volatility-surface construction and arbitrage checks.

The central test is a full round trip: build a chain from a known smile, push it through
cleaning, forward extraction and inversion, and check the original volatilities come back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import TRUE_SPOT, reference_vol
from optpricing import data as data_mod
from optpricing import surface as surface_mod


def test_surface_round_trip_recovers_the_input_smile(synthetic_surface: pd.DataFrame) -> None:
    """End-to-end: quotes in, the same volatilities out, to better than 0.01 vol points."""
    expected = reference_vol(
        synthetic_surface["log_moneyness"].to_numpy(dtype=np.float64), 0.0
    )  # placeholder, replaced per-expiry below
    assert expected.shape == (len(synthetic_surface),)

    errors = []
    for tau, group in synthetic_surface.groupby("tau"):
        truth = reference_vol(group["log_moneyness"].to_numpy(dtype=np.float64), float(tau))
        errors.append(group["implied_vol"].to_numpy(dtype=np.float64) - truth)
    worst = float(np.max(np.abs(np.concatenate(errors))))
    assert worst < 1e-4


def test_surface_keeps_only_out_of_the_money_quotes(synthetic_surface: pd.DataFrame) -> None:
    calls = synthetic_surface["option_type"] == "call"
    assert (synthetic_surface.loc[calls, "log_moneyness"] >= -1e-12).all()
    assert (synthetic_surface.loc[~calls, "log_moneyness"] < 1e-12).all()
    # Exactly one contract survives per (expiry, strike).
    assert not synthetic_surface.duplicated(subset=["expiry", "strike"]).any()


def test_surface_columns_and_derived_quantities(synthetic_surface: pd.DataFrame) -> None:
    required = {
        "expiry",
        "tau",
        "strike",
        "option_type",
        "mid",
        "forward",
        "discount",
        "rate",
        "dividend_yield",
        "log_moneyness",
        "implied_vol",
        "total_variance",
        "vega",
        "moneyness_bucket",
    }
    assert required.issubset(synthetic_surface.columns)
    assert np.allclose(
        synthetic_surface["total_variance"],
        synthetic_surface["implied_vol"] ** 2 * synthetic_surface["tau"],
    )
    assert (synthetic_surface["vega"] > 0).all()
    assert synthetic_surface["tau"].is_monotonic_increasing


def test_bid_ask_vols_bracket_the_mid_vol(synthetic_surface: pd.DataFrame) -> None:
    """Price is increasing in vol, so the bid and ask vols must bracket the mid vol."""
    s = synthetic_surface.dropna(subset=["iv_bid", "iv_ask"])
    assert len(s) > 0.9 * len(synthetic_surface)
    assert (s["iv_bid"] < s["implied_vol"]).all()
    assert (s["implied_vol"] < s["iv_ask"]).all()


def test_synthetic_surface_is_arbitrage_free(synthetic_surface: pd.DataFrame) -> None:
    report = surface_mod.arbitrage_report(synthetic_surface)
    assert report.calendar_checks > 50
    assert report.calendar_violations == 0
    assert report.butterfly_checks > 50
    assert report.butterfly_violations == 0
    assert "Calendar-spread violations" in str(report)


def test_arbitrage_report_detects_a_planted_calendar_violation(
    synthetic_surface: pd.DataFrame,
) -> None:
    """Halve the total variance of the longest expiry: that is a free calendar spread."""
    broken = synthetic_surface.copy()
    longest = broken["tau"].max()
    mask = broken["tau"] == longest
    broken.loc[mask, "implied_vol"] *= 0.4
    broken.loc[mask, "total_variance"] = broken.loc[mask, "implied_vol"] ** 2 * longest
    report = surface_mod.arbitrage_report(broken)
    assert report.calendar_violations > 0
    assert report.calendar_rate > 0.05
    assert report.worst_calendar_drop < 0


def test_arbitrage_report_detects_a_planted_butterfly_violation(
    synthetic_surface: pd.DataFrame,
) -> None:
    """Spike one strike's implied vol: the call price becomes locally concave."""
    broken = synthetic_surface.copy()
    tau = broken["tau"].unique()[2]
    candidates = broken.index[broken["tau"] == tau]
    victim = candidates[len(candidates) // 2]
    broken.loc[victim, "implied_vol"] *= 1.6
    clean_report = surface_mod.arbitrage_report(synthetic_surface)
    broken_report = surface_mod.arbitrage_report(broken)
    assert broken_report.butterfly_violations > clean_report.butterfly_violations
    assert broken_report.worst_butterfly < 0


def test_spread_aware_tolerance_matters_on_real_quotes(
    cached_spy_snapshot: data_mod.ChainSnapshot,
) -> None:
    """The finding: 26% of SPY strike triples look concave; 3% actually are.

    SPY lists $1 strikes near the money, where a one-lot butterfly is worth about as much
    as the $0.01 quote tick. A zero-tolerance convexity test is therefore measuring
    rounding. Netting off the bid-ask cost of the three legs leaves only the triples that
    could really be bought for less than nothing.
    """
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    curve = data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    surf = surface_mod.build_surface(cached_spy_snapshot, clean, curve)
    report = surface_mod.arbitrage_report(surf)
    assert report.butterfly_rate_zero_tol > 0.15
    assert report.butterfly_rate < report.butterfly_rate_zero_tol / 4.0
    # Calendar arbitrage genuinely is rare in a real chain.
    assert report.calendar_rate < 0.02


def test_real_spy_term_structure_is_upward_sloping(
    cached_spy_snapshot: data_mod.ChainSnapshot,
) -> None:
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    curve = data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    surf = surface_mod.build_surface(cached_spy_snapshot, clean, curve)
    term = surface_mod.atm_term_structure(surf)
    assert len(term) >= 8
    assert (term["atm_vol"] > 0.05).all()
    assert (term["atm_vol"] < 0.60).all()
    # Total variance must be monotone even where the vol level is not.
    assert term["atm_total_variance"].is_monotonic_increasing
    # A calm index in 2026: front vol well below one-year vol.
    assert term["atm_vol"].iloc[-1] > term["atm_vol"].iloc[0]


def test_real_spy_smile_has_a_put_skew(cached_spy_snapshot: data_mod.ChainSnapshot) -> None:
    """A 5%-moneyness risk reversal is positive at every expiry with coverage.

    Skew is measured by *interpolating* the smile to k = -0.05 and k = +0.05 rather than
    by averaging a window. Windows are the wrong tool here for two reasons: the SPY smile
    turns back up on the call side, so a wide window straddles the minimum; and the far
    call quotes are nickel options where the $0.01 tick is a fifth of the price, so their
    implied vols dominate any average they appear in.
    """
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    curve = data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    surf = surface_mod.build_surface(cached_spy_snapshot, clean, curve)

    checked = 0
    for expiry in sorted(surf["expiry"].unique()):
        smile = surface_mod.smile_slice(surf, expiry)
        assert smile["log_moneyness"].is_monotonic_increasing
        k = smile["log_moneyness"].to_numpy(dtype=np.float64)
        vol = smile["implied_vol"].to_numpy(dtype=np.float64)
        if k.min() > -0.05 or k.max() < 0.05:
            continue  # this expiry does not quote both 5% wings
        checked += 1
        put_5, atm_0, call_5 = np.interp([-0.05, 0.0, 0.05], k, vol)
        assert put_5 > atm_0 > call_5
        assert put_5 - call_5 > 0.02  # a meaningful risk reversal, not a marginal one
    assert checked >= 6


def test_real_spy_call_wing_turns_up(cached_spy_snapshot: data_mod.ChainSnapshot) -> None:
    """The smile bottoms out on the call side, not at the forward -- a real feature.

    This is why the skew test above interpolates at a fixed moneyness instead of averaging
    "everything beyond k = +0.1": past the trough the call wing rises back above the
    at-the-money level, and a wide window would report the wrong sign.
    """
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    curve = data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    surf = surface_mod.build_surface(cached_spy_snapshot, clean, curve)
    smile = surface_mod.smile_slice(surf, sorted(surf["expiry"].unique())[-1])
    wide = smile.loc[smile["log_moneyness"].between(-0.05, 0.5)]
    trough_k = float(wide.loc[wide["implied_vol"].idxmin(), "log_moneyness"])
    assert trough_k > 0.0
    far_call = wide.loc[wide["log_moneyness"] > trough_k + 0.15, "implied_vol"]
    assert len(far_call) > 3
    assert far_call.mean() > float(wide["implied_vol"].min())


def test_smile_slice_rejects_a_missing_expiry(synthetic_surface: pd.DataFrame) -> None:
    import datetime as dt

    with pytest.raises(KeyError, match="not present"):
        surface_mod.smile_slice(synthetic_surface, dt.date(1999, 1, 1))


def test_atm_term_structure(synthetic_surface: pd.DataFrame) -> None:
    term = surface_mod.atm_term_structure(synthetic_surface)
    assert len(term) == synthetic_surface["tau"].nunique()
    assert term["tau"].is_monotonic_increasing
    assert (term["n_quotes"] > 0).all()
    # The reference smile is 0.18 at k = 0 for every expiry.
    assert np.allclose(term["atm_vol"], 0.18, atol=0.01)
    assert np.allclose(term["atm_total_variance"], term["atm_vol"] ** 2 * term["tau"])


def test_surface_grid_does_not_extrapolate(synthetic_surface: pd.DataFrame) -> None:
    taus, ks, grid = surface_mod.surface_grid(
        synthetic_surface, n_moneyness=41, moneyness_range=(-0.6, 0.6)
    )
    assert grid.shape == (taus.size, ks.size)
    # The synthetic chain only quotes |k| <= 0.25, so the wings must be nan.
    assert np.isnan(grid[:, 0]).all()
    assert np.isnan(grid[:, -1]).all()
    inside = np.isfinite(grid)
    assert inside.sum() > 50
    assert np.all(grid[inside] > 0)


def test_surface_grid_handles_a_single_quote_expiry() -> None:
    frame = pd.DataFrame(
        {
            "tau": [0.5],
            "log_moneyness": [0.0],
            "total_variance": [0.02],
            "implied_vol": [0.2],
            "strike": [100.0],
            "forward": [100.0],
        }
    )
    _, _, grid = surface_mod.surface_grid(frame, n_moneyness=5)
    assert np.isnan(grid).all()


def test_thin_surface(synthetic_surface: pd.DataFrame) -> None:
    thin = surface_mod.thin_surface(synthetic_surface, max_per_expiry=6)
    counts = thin.groupby("tau").size()
    assert (counts <= 6).all()
    assert thin["tau"].nunique() == synthetic_surface["tau"].nunique()
    # Thinning is a subset, not a resample.
    assert set(thin["strike"]).issubset(set(synthetic_surface["strike"]))
    # Asking for more than exist returns everything in the window.
    wide = surface_mod.thin_surface(synthetic_surface, max_per_expiry=10_000)
    assert len(wide) == len(synthetic_surface)
    # An empty window returns an empty frame with the same columns.
    empty = surface_mod.thin_surface(synthetic_surface, 10, moneyness_range=(5.0, 6.0))
    assert empty.empty
    assert list(empty.columns) == list(synthetic_surface.columns)


def test_build_surface_rejects_an_unusable_chain(
    synthetic_snapshot: data_mod.ChainSnapshot, flat_rate_curve: data_mod.RateCurve
) -> None:
    calls_only = synthetic_snapshot.quotes.loc[synthetic_snapshot.quotes["option_type"] == "call"]
    with pytest.raises(ValueError, match="could not fit a forward"):
        surface_mod.build_surface(synthetic_snapshot, calls_only, flat_rate_curve)


def test_build_surface_rejects_an_empty_moneyness_window(
    synthetic_snapshot: data_mod.ChainSnapshot, flat_rate_curve: data_mod.RateCurve
) -> None:
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    with pytest.raises(ValueError, match="no quotes survive"):
        surface_mod.build_surface(
            synthetic_snapshot, clean, flat_rate_curve, max_abs_log_moneyness=1e-9
        )


def test_keeping_both_legs_gives_matching_vols(
    synthetic_snapshot: data_mod.ChainSnapshot, flat_rate_curve: data_mod.RateCurve
) -> None:
    """With otm_only=False, the call and put at a strike must imply the same vol.

    This is the payoff from carrying a parity-consistent forward through the inversion:
    without it the surface has a visible kink where the two legs meet.
    """
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    both = surface_mod.build_surface(synthetic_snapshot, clean, flat_rate_curve, otm_only=False)
    pivot = both.pivot_table(
        index=["tau", "strike"], columns="option_type", values="implied_vol"
    ).dropna()
    assert len(pivot) > 40
    assert float(np.max(np.abs(pivot["call"] - pivot["put"]))) < 1e-6
    assert TRUE_SPOT > 0
