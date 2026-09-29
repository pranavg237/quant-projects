"""Put-call parity analysis: European residuals, the American adjustment, model-free checks.

The synthetic chains here are built so the right answer is known: a European chain must
show no parity violations at all, and a chain whose puts are priced as *American* must
show exactly the high-strike pattern seen on SPY -- and the American adjustment must
remove it and recover the true forward.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from conftest import NY
from optpricing import binomial, parity
from optpricing import data as data_mod
from optpricing import surface as surface_mod
from optpricing.types import ExerciseStyle, OptionType


@pytest.fixture(scope="module")
def european_result(synthetic_snapshot, flat_rate_curve, synthetic_surface):
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    fwd = data_mod.implied_forward_curve(clean, synthetic_snapshot.spot, flat_rate_curve)
    return parity.run_parity_analysis(
        synthetic_snapshot, clean, flat_rate_curve, fwd, synthetic_surface, steps=60
    )


def test_matched_pairs_needs_both_legs_two_sided(synthetic_snapshot) -> None:
    quotes = synthetic_snapshot.quotes.copy()
    # Knock out the bid of the most expensive put (certainly paired): its strike must
    # drop out of the pairs.
    dearest_put = quotes.loc[quotes["option_type"] == "put", "mid"].idxmax()
    quotes.loc[dearest_put, "bid"] = 0.0
    snap = data_mod.ChainSnapshot("SYN", synthetic_snapshot.asof, synthetic_snapshot.spot, quotes)
    full = parity.matched_pairs(synthetic_snapshot)
    fewer = parity.matched_pairs(snap)
    assert len(fewer) == len(full) - 1
    assert not full["in_clean"].any()  # no cleaned chain passed
    for col in ("call_bid", "call_ask", "put_bid", "put_ask", "call_mid", "put_mid"):
        assert full[col].notna().all()


def test_a_european_chain_satisfies_parity_everywhere(european_result) -> None:
    res = european_result.residuals
    assert len(res) > 100
    assert res["in_clean"].mean() > 0.8  # the rest have a mid below the 5c cleaning floor
    assert res["residual"].abs().max() < 1e-6
    assert not res["violation"].any()
    dist = european_result.distribution
    assert len(dist) == 4
    assert (dist.loc[dist["theory"].str.startswith("European"), "violations"] == 0).all()


def test_the_american_adjustment_is_falsifiable(european_result) -> None:
    """Applied to a chain that really is European, the adjustment must make things worse.

    If it flattened the residuals whatever the data, the SPY result would mean nothing.
    """
    f = european_result.forwards
    long_dated = f.loc[f["tau"] > 0.4]
    assert (long_dated["dispersion_adjusted"] > long_dated["dispersion_raw"] + 0.1).all()


def test_european_chain_has_no_call_put_vol_gap(european_result) -> None:
    gap = european_result.vol_gap
    assert gap["gap_pipeline"].abs().max() < 1e-4


def test_monotonicity_and_upper_bound_hold_on_clean_quotes(european_result) -> None:
    assert european_result.stale["strike_monotonicity_violations"].sum() == 0
    assert european_result.upper_bound["upper_violations"].sum() == 0
    # A European chain with r > q legitimately has deep ITM puts *below* intrinsic, which
    # is exactly why that check is only meaningful for American options (next tests).
    assert european_result.stale["ask_below_intrinsic"].sum() > 0


def test_by_expiry_splits_in_and_out_of_window(european_result) -> None:
    by = european_result.by_expiry
    assert (by["in_window"] <= by["pairs"]).all()
    assert by["in_window"].sum() == int(european_result.residuals["in_window"].sum())


# --- American puts -----------------------------------------------------------------

SPOT, RATE, VOL, STEPS = 100.0, 0.05, 0.20, 150


@pytest.fixture(scope="module")
def american_chain():
    """Calls European (no dividend, so identical to American), puts American."""
    asof = dt.datetime(2026, 9, 18, 11, 0, tzinfo=NY)
    rows: list[dict[str, object]] = []
    for days in (182, 365):
        expiry = (asof + dt.timedelta(days=days)).date()
        tau = data_mod.year_fraction(asof, expiry)
        for strike in np.arange(85.0, 116.0, 2.5):
            for side in (OptionType.CALL, OptionType.PUT):
                px = binomial.price(
                    SPOT, strike, tau, RATE, VOL, side, ExerciseStyle.AMERICAN, STEPS, "crr"
                )
                # European call on the same lattice, so the only difference is exercise.
                rows.append(
                    {
                        "expiry": expiry,
                        "option_type": side.value,
                        "strike": float(strike),
                        "bid": px - 0.01,
                        "ask": px + 0.01,
                        "volume": 10.0,
                        "open_interest": 10.0,
                        "tau": tau,
                    }
                )
    snap = data_mod.ChainSnapshot("AMR", asof, SPOT, pd.DataFrame(rows))
    curve = data_mod.RateCurve([0.01, 30.0], [RATE, RATE])
    clean, _ = data_mod.clean_chain(snap)
    fwd = data_mod.implied_forward_curve(clean, SPOT, curve)
    flat = pd.DataFrame(
        {
            "expiry": np.repeat(fwd["expiry"].to_numpy(), 2),
            "log_moneyness": np.tile([-1.0, 1.0], len(fwd)),
            "implied_vol": VOL,
        }
    )
    result = parity.run_parity_analysis(snap, clean, curve, fwd, flat, steps=STEPS)
    return snap, fwd, result


def test_american_puts_bias_the_european_forward_low(american_chain) -> None:
    _, fwd, result = american_chain
    true_forward = SPOT * np.exp(RATE * fwd["tau"].to_numpy())
    # Early exercise makes in-the-money puts dearer, so K + (C - P)/D is too low.
    assert np.all(fwd["forward"].to_numpy() < true_forward - 0.1)
    adjusted = result.forwards["forward_adjusted"].to_numpy()
    assert np.allclose(adjusted, true_forward, atol=0.02)
    assert np.all(result.forwards["forward_shift_bp"] > 10)


def test_american_adjustment_flattens_the_residuals(american_chain) -> None:
    _, _, result = american_chain
    f = result.forwards
    assert np.all(f["dispersion_adjusted"] < f["dispersion_raw"] / 10)
    res = result.residuals
    inside = res.loc[res["in_window"]]
    assert inside["violation"].sum() > len(inside) / 3
    assert inside["violation_american"].sum() == 0
    # The European residual falls with strike -- the SPY signature.
    for _, g in inside.groupby("expiry"):
        slope = np.polyfit(g["strike"], g["residual"], 1)[0]
        assert slope < 0


def test_de_americanised_vols_meet_at_the_forward(american_chain) -> None:
    _, _, result = american_chain
    gap = result.vol_gap
    assert (gap["gap_american"].abs() < 0.05).all()
    assert (gap["gap_pipeline"].abs() > gap["gap_american"].abs()).all()


def test_early_exercise_premia_are_put_only_without_dividends() -> None:
    strikes = np.array([90.0, 100.0, 110.0])
    call, put = parity.early_exercise_premia(SPOT, strikes, 1.0, RATE, 0.0, np.full(3, VOL), 100)
    assert np.allclose(call, 0.0, atol=1e-10)
    assert np.all(put > 0) and np.all(np.diff(put) > 0)  # deeper ITM, more premium


# --- model-free checks on planted errors ----------------------------------------------


def test_a_clean_american_chain_passes_the_stale_quote_checks(american_chain) -> None:
    _, _, result = american_chain
    assert result.stale["ask_below_intrinsic"].sum() == 0
    assert result.stale["strike_monotonicity_violations"].sum() == 0


def test_stale_quote_checks_find_planted_errors(american_chain) -> None:
    snap, _, _ = american_chain
    q = snap.quotes.copy()
    first = q["expiry"].min()
    calls = q.loc[(q["option_type"] == "call") & (q["expiry"] == first)].sort_values("strike")
    deep, nxt = calls.index[0], calls.index[1]  # K = 85 (deepest ITM), then 87.5
    intrinsic = SPOT - q.loc[deep, "strike"]
    q.loc[deep, ["bid", "ask"]] = [intrinsic - 3.0, intrinsic - 2.0]  # offered below intrinsic
    q.loc[nxt, ["bid", "ask"]] = [intrinsic - 1.5, intrinsic - 1.0]  # bid above K=85's ask
    out = parity.stale_quote_checks(data_mod.ChainSnapshot("AMR", snap.asof, SPOT, q))
    row = out.loc[out["expiry"] == first].iloc[0]
    assert row["ask_below_intrinsic"] == 1
    assert row["strike_monotonicity_violations"] == 1
    assert row["implied_spot_min"] == pytest.approx(SPOT - 2.0)
    assert out.loc[out["expiry"] != first, "ask_below_intrinsic"].sum() == 0


def test_american_upper_bound_flags_an_overpriced_call(synthetic_snapshot) -> None:
    pairs = parity.matched_pairs(synthetic_snapshot)
    curve = data_mod.RateCurve([0.01, 30.0], [0.042, 0.042])
    assert parity.american_upper_bound(pairs, 500.0, curve)["upper_violations"].sum() == 0
    pairs.loc[0, "call_bid"] += 50.0
    bumped = parity.american_upper_bound(pairs, 500.0, curve)
    assert bumped["upper_violations"].sum() == 1
    assert bumped["worst_upper_excess"].max() > 0


def test_vol_at_rejects_a_missing_expiry(synthetic_surface) -> None:
    with pytest.raises(KeyError, match="not in the surface"):
        parity._vol_at(synthetic_surface, dt.date(1999, 1, 1), np.array([0.0]))


def test_parity_residuals_without_the_american_leg(synthetic_snapshot, flat_rate_curve) -> None:
    clean, _ = data_mod.clean_chain(synthetic_snapshot)
    fwd = data_mod.implied_forward_curve(clean, synthetic_snapshot.spot, flat_rate_curve)
    pairs = parity.matched_pairs(synthetic_snapshot, clean)
    res = parity.parity_residuals(pairs, synthetic_snapshot.spot, flat_rate_curve, fwd)
    assert "theo_american" not in res.columns
    assert not res["violation"].any()


# --- the finding on the committed SPY snapshot ---------------------------------------


def test_spy_long_dated_parity_needs_the_american_adjustment(cached_spy_snapshot) -> None:
    """On real SPY quotes the early-exercise premium, not noise, drives the dispersion."""
    snap = cached_spy_snapshot
    curve = data_mod.load_rate_curve(asof=snap.asof.date())
    clean, _ = data_mod.clean_chain(snap)
    fwd = data_mod.implied_forward_curve(clean, snap.spot, curve)
    surf = surface_mod.build_surface(snap, clean, curve)
    long_dated = fwd.loc[fwd["tau"] > 0.4].head(2)
    pairs = parity.matched_pairs(snap, clean)
    adj = parity.american_adjusted_forward(
        pairs, snap.spot, curve, long_dated, surf, iterations=2, steps=100
    )
    assert len(adj) == 2
    assert (adj["dispersion_adjusted"] < adj["dispersion_raw"] / 2).all()
    assert (adj["forward_adjusted"] > adj["forward_raw"]).all()
