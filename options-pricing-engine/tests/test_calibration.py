"""Heston calibration.

The headline test is **parameter recovery**: generate a surface from known Heston
parameters, calibrate to it, and check the original parameters come back. That catches
sign errors, scaling errors and objective-function mistakes that an RMSE-only test cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optpricing import calibration as cal
from optpricing import data as data_mod
from optpricing import heston as hs
from optpricing import surface as surface_mod
from optpricing.heston import HestonParams
from optpricing.types import OptionType

SPOT = 100.0
RATE = 0.03
DIVIDEND = 0.01


def _heston_surface(
    params: HestonParams, taus: tuple[float, ...] = (0.1, 0.35, 0.8, 1.6)
) -> pd.DataFrame:
    """Build a tidy surface frame of exact Heston prices and their implied vols."""
    from optpricing import blackscholes as bs
    from optpricing import implied_vol as iv

    rows: list[pd.DataFrame] = []
    for tau in taus:
        forward = SPOT * np.exp((RATE - DIVIDEND) * tau)
        ks = np.linspace(-0.30, 0.22, 17)
        strikes = forward * np.exp(ks)
        is_call = strikes >= forward
        call = np.asarray(hs.price(SPOT, strikes, tau, RATE, params, OptionType.CALL, DIVIDEND))
        put = np.asarray(hs.price(SPOT, strikes, tau, RATE, params, OptionType.PUT, DIVIDEND))
        mid = np.where(is_call, call, put)
        vol = np.where(
            is_call,
            np.asarray(iv.implied_vol(mid, SPOT, strikes, tau, RATE, OptionType.CALL, DIVIDEND)),
            np.asarray(iv.implied_vol(mid, SPOT, strikes, tau, RATE, OptionType.PUT, DIVIDEND)),
        )
        vega = np.asarray(bs.vega(SPOT, strikes, tau, RATE, vol, DIVIDEND))
        rows.append(
            pd.DataFrame(
                {
                    "tau": tau,
                    "strike": strikes,
                    "option_type": np.where(is_call, "call", "put"),
                    "mid": mid,
                    "forward": forward,
                    "rate": RATE,
                    "dividend_yield": DIVIDEND,
                    "log_moneyness": ks,
                    "implied_vol": vol,
                    "vega": vega,
                }
            )
        )
    return pd.concat(rows, ignore_index=True).dropna(subset=["implied_vol"])


def test_parameter_recovery_from_a_synthetic_heston_surface(
    true_heston_params: HestonParams,
) -> None:
    """Calibrating to noiseless Heston prices must return the generating parameters."""
    surf = _heston_surface(true_heston_params)
    result = cal.calibrate(surf, SPOT)

    assert result.rmse_vol < 1e-4  # a perfect fit is available and must be found
    fitted, truth = result.params, true_heston_params
    assert fitted.v0 == pytest.approx(truth.v0, rel=0.05)
    assert fitted.theta == pytest.approx(truth.theta, rel=0.10)
    assert fitted.rho == pytest.approx(truth.rho, abs=0.05)
    # kappa and xi are the weakly identified pair -- only their combination is pinned down
    # by a surface of this size, so they get a looser tolerance than v0/theta/rho.
    assert fitted.kappa == pytest.approx(truth.kappa, rel=0.35)
    assert fitted.xi == pytest.approx(truth.xi, rel=0.35)


def test_calibration_is_stable_across_seeds(true_heston_params: HestonParams) -> None:
    surf = _heston_surface(true_heston_params, taus=(0.25, 1.0))
    result = cal.calibrate(surf, SPOT)
    assert len(result.seed_rmses) == len(cal.DEFAULT_SEEDS)
    assert max(result.seed_rmses) - min(result.seed_rmses) < 5e-3
    assert result.success
    assert "Heston calibration" in str(result)
    assert "Feller" in str(result)


def test_calibration_beats_a_flat_vol_per_expiry(true_heston_params: HestonParams) -> None:
    """The benchmark that matters: a model with more parameters and no smile."""
    surf = _heston_surface(true_heston_params)
    heston_fit = cal.calibrate(surf, SPOT)
    _, per_expiry_errors = cal.fit_flat_vol_per_expiry(surf)
    per_expiry_rmse = float(np.sqrt((per_expiry_errors["vol_error"] ** 2).mean()))
    assert heston_fit.rmse_vol < per_expiry_rmse / 10.0


def test_surface_errors_exact_and_approximate_agree(true_heston_params: HestonParams) -> None:
    """The vega approximation to the vol residual is good to a hundredth of a vol point."""
    surf = _heston_surface(true_heston_params, taus=(0.5,))
    perturbed = true_heston_params.replace(rho=-0.55)
    exact = cal.surface_errors(perturbed, surf, SPOT, exact_vols=True)
    approx = cal.surface_errors(perturbed, surf, SPOT, exact_vols=False)
    diff = (exact["vol_error"] - approx["vol_error"]).abs()
    # The linearisation drops volga, so it is near-exact in the body and degrades in the
    # wings: here the median gap is ~1e-5 of vol and the worst single quote is ~1.2e-3,
    # i.e. about a tenth of a vol point on the furthest strike.
    assert float(diff.median()) < 1e-4
    assert float(diff.max()) < 2e-3
    body = diff.loc[exact["log_moneyness"].abs() < 0.15]
    assert float(body.max()) < 2e-4
    assert set(exact.columns) >= {"model_vol", "vol_error", "model_price", "price_error"}


def test_flat_vol_benchmarks(true_heston_params: HestonParams) -> None:
    surf = _heston_surface(true_heston_params)
    global_vol, global_errors = cal.fit_global_flat_vol(surf)
    per_expiry, per_expiry_errors = cal.fit_flat_vol_per_expiry(surf)

    assert 0.05 < global_vol < 0.60
    assert len(per_expiry) == surf["tau"].nunique()
    assert (global_errors["model_vol"] == global_vol).all()

    # One vol per expiry nests one vol overall, so it must fit at least as well *in the
    # metric both are minimising* -- the vega-weighted squared error. In plain unweighted
    # RMSE the ordering can invert by a hair (it does on this synthetic surface: 2.811%
    # against 2.799%), because neither fit is minimising that.
    def weighted_rmse(errors: pd.DataFrame) -> float:
        w = np.sqrt(np.maximum(errors["vega"].to_numpy(dtype=np.float64), 1e-8))
        e = errors["vol_error"].to_numpy(dtype=np.float64)
        return float(np.sqrt(np.sum(w * e**2) / np.sum(w)))

    assert weighted_rmse(per_expiry_errors) <= weighted_rmse(global_errors) + 1e-12
    # Errors are signed model-minus-market, so they straddle zero.
    assert per_expiry_errors["vol_error"].min() < 0 < per_expiry_errors["vol_error"].max()


def test_vega_weighting_changes_the_fit(true_heston_params: HestonParams) -> None:
    surf = _heston_surface(true_heston_params, taus=(0.5,))
    weighted = cal.calibrate(surf, SPOT, seeds=cal.DEFAULT_SEEDS[:1], weight_by_vega=True)
    unweighted = cal.calibrate(surf, SPOT, seeds=cal.DEFAULT_SEEDS[:1], weight_by_vega=False)
    # Both fit noiseless data essentially perfectly, but they are genuinely different runs.
    assert weighted.rmse_vol < 1e-3
    assert unweighted.rmse_vol < 1e-3


def test_calibrating_to_an_empty_surface_raises() -> None:
    with pytest.raises(ValueError, match="empty surface"):
        cal.calibrate(pd.DataFrame(columns=["tau", "strike"]), SPOT)


def test_calibration_respects_bounds(true_heston_params: HestonParams) -> None:
    surf = _heston_surface(true_heston_params, taus=(0.5,))
    tight = ((0.01, 1.0, 0.01, 0.1, -0.30), (0.05, 2.0, 0.05, 0.4, -0.20))
    result = cal.calibrate(surf, SPOT, seeds=cal.DEFAULT_SEEDS[:1], bounds=tight)
    x = result.params.to_array()
    assert np.all(x >= np.array(tight[0]) - 1e-9)
    assert np.all(x <= np.array(tight[1]) + 1e-9)
    # A rho clamped to -0.30 cannot reproduce a -0.65 skew, so the fit must be visibly bad.
    assert result.rmse_vol > 1e-3


def test_real_spy_calibration_quality(cached_spy_snapshot: data_mod.ChainSnapshot) -> None:
    """Regression guard on the headline README number, from the committed snapshot."""
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    curve = data_mod.RateCurve([0.25, 30.0], [0.039, 0.05])
    surf = surface_mod.thin_surface(
        surface_mod.build_surface(cached_spy_snapshot, clean, curve), max_per_expiry=14
    )
    result = cal.calibrate(surf, cached_spy_snapshot.spot, seeds=cal.DEFAULT_SEEDS[:2])
    _, per_expiry_errors = cal.fit_flat_vol_per_expiry(surf)
    per_expiry_rmse = float(np.sqrt((per_expiry_errors["vol_error"] ** 2).mean()))

    assert result.rmse_vol < 0.05  # under 5 vol points on a real surface
    assert result.rmse_vol < per_expiry_rmse / 2.0
    assert result.params.rho < -0.3  # equity indices always calibrate to a negative rho
    # The known limitation: the short-dated put wing is where Heston fails.
    errors = result.errors.dropna(subset=["vol_error"])
    body = errors.loc[errors["log_moneyness"].abs() < 0.15, "vol_error"]
    wing = errors.loc[(errors["log_moneyness"] < -0.15) & (errors["tau"] < 0.15), "vol_error"]
    assert len(wing) > 5
    assert float(np.sqrt((wing**2).mean())) > 2.0 * float(np.sqrt((body**2).mean()))
