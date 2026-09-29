"""Every analytic Greek, calls and puts, against finite differences of the price.

``test_blackscholes.py`` checks the Greeks at two hand-picked points. This file checks all
nine at 84 points per option type -- one day to three years, three standard deviations
either side of the forward, 10% and 40% vol -- using only second differences of the
*price* for the second-order Greeks. It also checks that the check has teeth: the
classic unit bugs (vega per vol point, theta per day) and a sign slip in the put branch
of charm must all be caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optpricing import blackscholes as bs
from optpricing import greeks_check as gc
from optpricing.types import OptionType

# Tolerances are set from the error model in the module docstring, not tuned to pass:
# first derivatives should be good to ~eps_eff^(2/3) ~ 1e-9..1e-8, second derivatives to
# ~eps_eff^(1/2) ~ 1e-7..1e-6 times curvature factors that grow in the wings. Measured
# worst cases are ~6e-8 and ~7e-4 (one-day ATM volga, which is nearly zero there).
FIRST_ORDER_RTOL = 1e-6
SECOND_ORDER_RTOL = 5e-3


@pytest.fixture(scope="module")
def errors() -> pd.DataFrame:
    return gc.error_table(gc.default_grid())


def test_grid_covers_calls_puts_moneyness_and_near_expiry(errors: pd.DataFrame) -> None:
    assert set(errors["option_type"]) == {"call", "put"}
    assert set(errors["greek"]) == set(gc.GREEK_NAMES)
    assert errors["tau_days"].min() == pytest.approx(1.0)
    assert errors["tau_days"].max() == pytest.approx(3 * 365.0)
    assert errors["z"].min() == pytest.approx(-3.0)
    assert errors["z"].max() == pytest.approx(3.0)
    # Nothing in the grid is so far out that the Greeks underflow to zero.
    assert (errors.loc[errors["greek"] == "gamma", "analytic"] > 1e-6).all()


@pytest.mark.parametrize("greek", gc.FIRST_ORDER)
def test_first_order_greeks_match(errors: pd.DataFrame, greek: str) -> None:
    sub = errors.loc[errors["greek"] == greek]
    worst = sub.loc[sub["rel_error"].idxmax()]
    assert worst["rel_error"] < FIRST_ORDER_RTOL, worst.to_dict()


@pytest.mark.parametrize("greek", gc.SECOND_ORDER)
def test_second_order_greeks_match(errors: pd.DataFrame, greek: str) -> None:
    sub = errors.loc[errors["greek"] == greek]
    worst = sub.loc[sub["rel_error"].idxmax()]
    assert worst["rel_error"] < SECOND_ORDER_RTOL, worst.to_dict()


def test_summary_has_one_row_per_greek_and_side(errors: pd.DataFrame) -> None:
    summary = gc.summarise(errors)
    assert len(summary) == 2 * len(gc.GREEK_NAMES)
    assert (summary["points"] == 84).all()
    assert list(summary["greek"].iloc[:2]) == ["delta", "delta"]


def test_scaled_steps_beat_textbook_steps_near_expiry() -> None:
    """The noise-scaled step should matter exactly where the docstring says it does."""
    one_day = [p for p in gc.default_grid(taus=(1 / 365,), sigmas=(0.10,)) if abs(p.z) <= 1.0]
    eps = float(np.finfo(np.float64).eps)
    scaled = gc.error_table(one_day)
    textbook = gc.error_table(one_day, first=eps ** (1 / 3), second=eps ** (1 / 4))
    for greek in ("gamma", "volga"):
        ours = scaled.loc[scaled["greek"] == greek, "rel_error"].max()
        theirs = textbook.loc[textbook["greek"] == greek, "rel_error"].max()
        assert ours < theirs / 3


def test_step_sweep_is_v_shaped() -> None:
    p = gc.GreekPoint(100.0, 101.0, 1 / 365, 0.04, 0.2, 0.015, OptionType.CALL)
    sweep = gc.step_sweep(p, "gamma")
    best = sweep.loc[sweep["rel_error"].idxmin(), "step"]
    # Both ends are much worse than the bottom: round-off on the left, truncation right.
    assert sweep["rel_error"].iloc[0] > 100 * sweep["rel_error"].min()
    assert sweep["rel_error"].iloc[-1] > 100 * sweep["rel_error"].min()
    # And the default second-derivative step lands within a decade of the optimum.
    _, second = gc.default_steps(p)
    assert 0.1 < second / best < 10.0
    with pytest.raises(ValueError, match="unknown Greek"):
        gc.step_sweep(p, "speed")


@pytest.mark.parametrize(
    ("name", "broken", "greek"),
    [
        ("vega per vol point", lambda f: lambda *a, **k: f(*a, **k) / 100.0, "vega"),
        ("theta per day", lambda f: lambda *a, **k: f(*a, **k) / 365.0, "theta"),
        ("volga sign", lambda f: lambda *a, **k: -f(*a, **k), "volga"),
    ],
)
def test_check_catches_unit_bugs(monkeypatch, name, broken, greek) -> None:
    monkeypatch.setattr(bs, greek, broken(getattr(bs, greek)))
    errs = gc.error_table(gc.default_grid(taus=(0.25,), sigmas=(0.2,)))
    assert errs.loc[errs["greek"] == greek, "rel_error"].max() > 0.5, name


def test_check_catches_a_put_only_sign_slip_in_charm(monkeypatch) -> None:
    """The original suite checked charm only for calls, so a put-branch slip would pass it."""
    real = bs.charm

    def charm_with_put_bug(s, k, t, r, vol, option_type=OptionType.CALL, q=0.0):
        out = real(s, k, t, r, vol, option_type, q)
        if OptionType(option_type) is OptionType.PUT:
            # Flip the sign of the dividend term only: q e^{-q tau} N(-d1).
            d1, _ = bs.d1_d2(s, k, t, r, vol, q)
            from scipy.stats import norm

            out = out + 2.0 * q * np.exp(-q * t) * norm.cdf(-d1)
        return out

    monkeypatch.setattr(bs, "charm", charm_with_put_bug)
    errs = gc.error_table(gc.default_grid(taus=(0.25, 1.0), sigmas=(0.2,)))
    charm = errs.loc[errs["greek"] == "charm"]
    assert charm.loc[charm["option_type"] == "call", "rel_error"].max() < SECOND_ORDER_RTOL
    assert charm.loc[charm["option_type"] == "put", "rel_error"].max() > 10 * SECOND_ORDER_RTOL
