"""Discrete cash dividends in the lattice: the escrowed-dividend model.

The checks, in order of how much they would catch:

* **European limit** -- at expiry every dividend has gone ex, so the European lattice
  price must be Black-Scholes on the escrowed spot ``S - PV(dividends)``.
* **No-dividend limit** -- with no dividend inside the option's life the lattice must be
  the continuous-yield code, bit for bit.
* **Textbook exercise** -- an American call on a stock with no continuous yield is
  exercised, if at all, only just before an ex-date, and only when the dividend beats
  the interest on the strike until expiry, ``D > K(1 - e^{-r(T - t_d)})`` (Hull).
* **Closed form** -- with one dividend the escrowed model has an exact American-call
  price (Roll-Geske-Whaley); the lattice must converge to it.
* **End to end** -- a chain priced with discrete dividends must come back exactly
  through the surface built with them, and not through the continuous-yield surface.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize
from scipy.stats import norm

from conftest import NY, SPY_TEST_CURVE, reference_vol
from optpricing import american, binomial, parity
from optpricing import blackscholes as bs
from optpricing import data as data_mod
from optpricing import dividends as dv
from optpricing import surface as surface_mod
from optpricing.dividends import DividendSchedule
from optpricing.types import ExerciseStyle, OptionType

SPOT = 500.0
QUARTERLY = DividendSchedule(
    np.array([0.1, 0.35, 0.6, 0.85, 1.1, 1.35, 1.6]), np.full(7, 2.5)
)  # ~2% a year, paid quarterly
#: SPY's pattern: quarterly, going ex at 09:30 on the quarterly expiry days, 6.5 hours
#: before the 16:00 expiry (day counts from an 11:00 valuation, as in the chain below).
EX_DAYS = (91, 182, 273, 365, 456, 547, 640)
SPY_LIKE = DividendSchedule(
    np.array([(d * 24.0 - 1.5) / (24.0 * 365.0) for d in EX_DAYS]), np.full(len(EX_DAYS), 2.5)
)


# --- the lattice -------------------------------------------------------------------------


TAU_640 = (640 * 24.0 + 5.0) / (24.0 * 365.0)  # 16:00 on day 640: 6.5 h after an ex-date


@pytest.mark.parametrize("tau", [1.3, TAU_640])  # the second ends in a stub after the ex-date
@pytest.mark.parametrize("side", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("q", [0.0, -0.01, 0.02])
def test_european_lattice_is_black_scholes_on_the_escrowed_spot(side, q, tau) -> None:
    k = np.array([380.0, 450.0, 500.0, 540.0, 620.0])
    vols = np.array([0.32, 0.24, 0.2, 0.18, 0.21])
    r = 0.04
    s_star = american.escrowed_spot(SPOT, tau, r, SPY_LIKE)
    assert s_star == pytest.approx(SPOT - SPY_LIKE.present_value(tau, r))
    n_divs = len(SPY_LIKE.within(tau))
    assert n_divs in {5, 7}
    assert s_star < SPOT - n_divs * 2.5 * np.exp(-r * tau)
    exact = np.asarray(bs.price(s_star, k, tau, r, vols, side, q))
    errors = []
    for steps in (201, 801):
        lat = american.lattice_prices(
            SPOT, k, tau, r, vols, side, q, dividends=SPY_LIKE, steps=steps
        )
        errors.append(np.max(np.abs(lat.european - exact)))
        assert np.all(lat.american >= lat.european - 1e-12)
    if tau == 1.3:
        assert errors[0] < 2e-3  # Leisen-Reimer: O(1/n^2)
        assert errors[1] < errors[0] / 8
    else:
        # The stub: the lattice stops 6.5 h before expiry at the ex-date and the rest is
        # priced in closed form. That smooths the payoff over less than one node spacing,
        # which LR's strike centring cannot resolve, so the European value loses its
        # O(1/n^2) and is off by ~1e-4 of the price. The premium is a difference on one
        # lattice and does not inherit it (next test).
        assert errors[0] < 0.02
        assert errors[1] < errors[0]


@pytest.mark.parametrize("side", [OptionType.CALL, OptionType.PUT])
def test_premium_with_an_ex_date_inside_the_final_step_is_converged_at_201_steps(side) -> None:
    """SPY's case: 21 months, the last ex-date 6.5 hours before expiry.

    Reference: a 6,401-step lattice, fine enough that its last node before the ex-date is
    at most 0.1 day early, so it needs no stub.
    """
    tau = TAU_640
    k = np.array([450.0, 500.0, 530.0, 560.0])
    vols = np.full(4, 0.18)
    args = (SPOT, k, tau, 0.04, -0.01, vols, side)
    coarse = american.early_exercise_premium(*args, 201, dividends=SPY_LIKE)
    fine = american.early_exercise_premium(*args, 6401, dividends=SPY_LIKE)
    assert tau - SPY_LIKE.times[-1] > tau / 6401  # the reference really has no stub
    assert np.all(fine > 0.1)
    assert np.max(np.abs(coarse / fine - 1.0)) < 0.02


@pytest.mark.parametrize("method", ["lr", "crr"])
@pytest.mark.parametrize("side", [OptionType.CALL, OptionType.PUT])
def test_no_dividends_is_the_continuous_yield_lattice_bit_for_bit(side, method) -> None:
    k = np.array([380.0, 450.0, 500.0, 540.0, 620.0])
    vols = np.array([0.32, 0.24, 0.2, 0.18, 0.21])
    tau, r, q = 0.3, 0.04, 0.015
    base = american.lattice_prices(SPOT, k, tau, r, vols, side, q, None, 151, method)
    # Empty; all already paid; all after expiry (0.35 > tau): nothing in the option's life.
    for divs in (
        DividendSchedule.empty(),
        DividendSchedule(np.array([-0.2, 0.0]), np.array([3.0, 3.0])),
        DividendSchedule(np.array([0.35, 0.6]), np.array([3.0, 3.0])),
    ):
        got = american.lattice_prices(SPOT, k, tau, r, vols, side, q, divs, 151, method)
        assert np.array_equal(got.american, base.american)
        assert np.array_equal(got.european, base.european)
        prem = american.early_exercise_premium(SPOT, k, tau, r, q, vols, side, 151, method, divs)
        assert np.array_equal(
            prem, american.early_exercise_premium(SPOT, k, tau, r, q, vols, side, 151, method)
        )
    # ...and that lattice is still the scalar tree of optpricing.binomial.
    for i, (ki, v) in enumerate(zip(k, vols, strict=True)):
        ref = binomial.price(SPOT, ki, tau, r, v, side, ExerciseStyle.AMERICAN, 151, method, q)
        assert base.american[i] == pytest.approx(ref, abs=1e-10)


def test_american_call_exercises_only_just_before_an_ex_date() -> None:
    """The textbook result, read off the lattice's own exercise decisions.

    Each dividend (12) beats the interest on the strike until the next ex-date or expiry,
    ``K(1 - e^{-r(t_{i+1} - t_i)})`` (at most 9.9 here), so both are worth exercising for.
    """
    tau, r, steps = 1.0, 0.05, 401
    divs = DividendSchedule(np.array([0.3, 0.8]), np.array([12.0, 12.0]))
    k = np.array([300.0, 350.0, 400.0])  # deep in the money
    lat = american.lattice_prices(
        SPOT, k, tau, r, np.full(3, 0.2), "call", 0.0, divs, steps, record_exercise=True
    )
    dt_step = tau / steps
    last_before = {int(np.ceil(t / dt_step)) - 1 for t in divs.times}
    for row in lat.exercise_steps:
        assert set(np.flatnonzero(row)) == last_before
    assert lat.exercise_steps.shape == (3, steps + 1)
    assert np.all(lat.step_times[sorted(last_before)] < divs.times)
    assert np.all(divs.times - lat.step_times[sorted(last_before)] <= dt_step)
    assert np.all(lat.american > lat.european + 1.0)
    # Without the dividends the same call is never exercised at all.
    none = american.lattice_prices(
        SPOT, k, tau, r, np.full(3, 0.2), "call", 0.0, None, steps, record_exercise=True
    )
    assert not none.exercise_steps.any()
    assert np.allclose(none.american, none.european, rtol=0, atol=1e-10)


def test_dividend_smaller_than_interest_on_the_strike_is_never_worth_exercising_for() -> None:
    r"""Hull: never exercise before :math:`t_d` if :math:`D \le K(1 - e^{-r(T-t_d)})`."""
    tau, r, t_d, k = 1.0, 0.05, 0.5, 400.0
    threshold = k * (1.0 - np.exp(-r * (tau - t_d)))  # 9.88
    below = DividendSchedule(np.array([t_d]), np.array([0.98 * threshold]))
    above = DividendSchedule(np.array([t_d]), np.array([1.5 * threshold]))
    args = (SPOT, np.array([k]), tau, r, np.array([0.2]), "call", 0.0)
    lat_below = american.lattice_prices(*args, below, 801, record_exercise=True)
    lat_above = american.lattice_prices(*args, above, 801, record_exercise=True)
    assert not lat_below.exercise_steps.any()
    assert lat_below.american[0] == pytest.approx(lat_below.european[0], abs=1e-10)
    assert lat_above.exercise_steps.any()
    assert lat_above.american[0] > lat_above.european[0] + 0.5


def _bivariate_normal(a: float, b: float, rho: float) -> float:
    def integrand(x: float) -> float:
        return float(norm.pdf(x) * norm.cdf((b - rho * x) / np.sqrt(1.0 - rho * rho)))

    return float(integrate.quad(integrand, -np.inf, a, epsabs=1e-13, epsrel=1e-12)[0])


def _roll_geske_whaley(s: float, k: float, tau: float, r: float, sig: float, d: float, t1: float):
    """American call, one cash dividend, escrowed model (Hull, Technical Note 4)."""
    s_star = s - d * np.exp(-r * t1)

    def ex_dividend_call(x: float) -> float:
        return float(bs.price(x, k, tau - t1, r, sig, "call"))

    s_bar = optimize.brentq(lambda x: ex_dividend_call(x) - (x + d - k), 1e-6, 100 * k, xtol=1e-12)
    a1 = (np.log(s_star / k) + (r + 0.5 * sig**2) * tau) / (sig * np.sqrt(tau))
    a2 = a1 - sig * np.sqrt(tau)
    b1 = (np.log(s_star / s_bar) + (r + 0.5 * sig**2) * t1) / (sig * np.sqrt(t1))
    b2 = b1 - sig * np.sqrt(t1)
    rho = -np.sqrt(t1 / tau)
    return (
        s_star * norm.cdf(b1)
        + s_star * _bivariate_normal(a1, -b1, rho)
        - k * np.exp(-r * tau) * _bivariate_normal(a2, -b2, rho)
        - (k - d) * np.exp(-r * t1) * norm.cdf(b2)
    )


@pytest.mark.parametrize("strike", [80.0, 100.0])
def test_one_dividend_american_call_converges_to_roll_geske_whaley(strike) -> None:
    s, tau, r, sig, d = 100.0, 1.0, 0.05, 0.25, 4.0
    errors = []
    for steps in (201, 1001):
        # Put the ex-date just after a node, so "the last node before it" is exact.
        t1 = (steps // 2 + 1e-9) * tau / steps
        exact = _roll_geske_whaley(s, strike, tau, r, sig, d, t1)
        lat = american.lattice_prices(
            s,
            np.array([strike]),
            tau,
            r,
            np.array([sig]),
            "call",
            0.0,
            DividendSchedule(np.array([t1]), np.array([d])),
            steps,
        )
        errors.append(abs(float(lat.american[0]) - exact))
        assert exact > float(bs.price(s - d * np.exp(-r * t1), strike, tau, r, sig, "call"))
    assert errors[1] < 5e-4
    assert errors[1] < errors[0]


def test_lattice_rejects_bad_inputs() -> None:
    k, v = np.array([500.0]), np.array([0.2])
    with pytest.raises(ValueError, match="unknown lattice"):
        american.lattice_prices(SPOT, k, 1.0, 0.04, v, "put", method="jr")
    with pytest.raises(ValueError, match="tau"):
        american.lattice_prices(SPOT, k, 0.0, 0.04, v, "put")
    huge = DividendSchedule(np.array([0.5]), np.array([600.0]))
    with pytest.raises(ValueError, match="worth more than the spot"):
        american.lattice_prices(SPOT, k, 1.0, 0.04, v, "put", dividends=huge)
    assert american.escrowed_spot(SPOT, 1.0, 0.04, None) == SPOT


# --- the fixed point and the surface --------------------------------------------------------


@pytest.mark.parametrize("side", [OptionType.CALL, OptionType.PUT])
def test_deamericanise_round_trip_with_cash_dividends(side) -> None:
    tau, r, q = TAU_640, 0.045, -0.01
    s_star = american.escrowed_spot(SPOT, tau, r, SPY_LIKE)
    fwd = s_star * np.exp((r - q) * tau)
    ks = np.linspace(-0.3, -0.01, 12) if side is OptionType.PUT else np.linspace(0.0, 0.3, 12)
    k = fwd * np.exp(ks)
    true_vol = reference_vol(ks, tau)
    european = np.asarray(bs.price(s_star, k, tau, r, true_vol, side, q))
    premium = american.early_exercise_premium(
        SPOT, k, tau, r, q, true_vol, side, dividends=SPY_LIKE
    )
    res = american.deamericanise(european + premium, SPOT, k, tau, r, q, side, dividends=SPY_LIKE)
    assert res.converged.all()
    assert np.max(np.abs(res.vol - true_vol)) < 1e-8
    if side is OptionType.CALL:
        assert premium.max() > 0.05  # the dividends give these calls a premium


RATE, RESIDUAL = 0.042, 0.0
EXPIRY_DAYS = (21, 91, 365, 640)


def _dividend_chain() -> data_mod.ChainSnapshot:
    """An American chain priced with quarterly cash dividends and no continuous yield."""
    asof = dt.datetime(2026, 9, 18, 11, 0, tzinfo=NY)
    rows: list[dict[str, object]] = []
    for days in EXPIRY_DAYS:
        expiry = (asof + dt.timedelta(days=days)).date()
        tau = data_mod.year_fraction(asof, expiry)
        s_star = american.escrowed_spot(SPOT, tau, RATE, SPY_LIKE)
        forward = s_star * np.exp((RATE - RESIDUAL) * tau)
        strikes = np.unique(np.round(forward * np.exp(np.linspace(-0.3, 0.2, 31)), 0))
        vols = reference_vol(np.log(strikes / forward), tau)
        for side in (OptionType.CALL, OptionType.PUT):
            mids = np.asarray(bs.price(s_star, strikes, tau, RATE, vols, side, RESIDUAL))
            mids = mids + american.early_exercise_premium(
                SPOT, strikes, tau, RATE, RESIDUAL, vols, side, dividends=SPY_LIKE
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


@pytest.fixture(scope="module")
def dividend_chain() -> data_mod.ChainSnapshot:
    return _dividend_chain()


def _build(snap, dividends, **kwargs: object) -> surface_mod.SurfaceBuild:
    clean, _ = data_mod.clean_chain(snap)
    curve = data_mod.RateCurve([0.01, 30.0], [RATE, RATE])
    return surface_mod.build_surface_detailed(snap, clean, curve, dividends=dividends, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def dividend_builds(dividend_chain):
    return {name: _build(dividend_chain, d) for name, d in (("discrete", SPY_LIKE), ("cont", None))}


def test_discrete_surface_recovers_a_discrete_dividend_chain(dividend_builds) -> None:
    b = dividend_builds["discrete"]
    assert b.converged and b.dividend_model == "discrete"
    assert b.unconverged_quotes == 0 and b.dropped_by_correction == 0
    fwd = b.forwards
    true_fwd = [
        american.escrowed_spot(SPOT, t, RATE, SPY_LIKE) * np.exp(RATE * t) for t in fwd["tau"]
    ]
    assert np.max(np.abs(fwd["forward"] / true_fwd - 1.0)) < 1e-6
    assert np.max(np.abs(fwd["residual_yield"])) < 1e-6  # no carry beyond the dividends
    assert np.allclose(fwd["pv_dividends"], [SPY_LIKE.present_value(t, RATE) for t in fwd["tau"]])
    surf = b.surface
    k_true = np.log(
        surf["strike"] / surf["expiry"].map(dict(zip(fwd["expiry"], true_fwd, strict=True)))
    )
    err = surf["implied_vol"] - reference_vol(k_true.to_numpy(), surf["tau"].to_numpy())
    assert np.max(np.abs(err)) < 1e-6
    # The out-of-the-money calls past an ex-date carry a premium; nothing before one.
    calls = surf.loc[surf["option_type"] == "call"]
    assert calls.loc[calls["tau"] < 0.1, "ee_premium"].max() == 0.0
    assert calls.loc[calls["tau"] > 1.0, "ee_premium"].max() > 0.01


def test_continuous_surface_misreads_a_discrete_dividend_chain(dividend_chain) -> None:
    """The falsification control: the continuous-yield lattice cannot price this chain."""
    gaps = {}
    for name, divs in (("discrete", SPY_LIKE), ("cont", None)):
        fitted = _build(dividend_chain, divs).forwards
        both = _build(
            dividend_chain, divs, otm_only=False, max_abs_log_moneyness=0.05, forwards=fitted
        )
        gaps[name] = surface_mod.call_put_gap(both.surface).set_index("days")["gap_vol_points"]
    assert np.max(np.abs(gaps["discrete"])) < 1e-4
    assert abs(gaps["cont"].iloc[-1]) > 20 * np.max(np.abs(gaps["discrete"]))


def test_empty_schedule_builds_the_continuous_surface(dividend_chain, dividend_builds) -> None:
    empty = _build(dividend_chain, DividendSchedule.empty())
    cont = dividend_builds["cont"]
    pd.testing.assert_frame_equal(empty.surface, cont.surface)
    assert np.array_equal(empty.forwards["forward"], cont.forwards["forward"])
    assert (cont.forwards["pv_dividends"] == 0.0).all()
    assert np.allclose(cont.forwards["residual_yield"], cont.forwards["dividend_yield"])


def test_european_build_ignores_dividends(dividend_chain) -> None:
    a = _build(dividend_chain, SPY_LIKE, exercise="european")
    b = _build(dividend_chain, None, exercise="european")
    pd.testing.assert_frame_equal(a.surface, b.surface)
    assert a.dividend_model == "continuous"


def test_parity_module_reproduces_the_discrete_surface_forward(
    dividend_chain, dividend_builds
) -> None:
    b = dividend_builds["discrete"]
    clean, _ = data_mod.clean_chain(dividend_chain)
    curve = data_mod.RateCurve([0.01, 30.0], [RATE, RATE])
    fwd = data_mod.implied_forward_curve(clean, SPOT, curve)
    result = parity.run_parity_analysis(
        dividend_chain, clean, curve, fwd, b.surface, dividends=SPY_LIKE
    )
    merged = result.forwards.merge(b.forwards, on="expiry")
    assert np.max(np.abs(merged["forward_adjusted"] / merged["forward"] - 1.0)) < 1e-6
    assert np.max(np.abs(result.forwards["residual_yield"])) < 1e-6
    # Every in-window pair is priced to within its spread once the premia are right.
    inside = result.residuals.loc[result.residuals["in_window"]]
    assert not inside["violation_american"].any()


# --- the committed SPY snapshot ---------------------------------------------------------


def test_spy_discrete_surface_with_the_committed_dividends(cached_spy_snapshot, spy_build) -> None:
    """The real path: committed projection -> discrete surface, against the continuous one."""
    schedule, frame, _ = dv.load_committed_schedule(cached_spy_snapshot.asof)
    assert len(schedule) == len(frame) >= 7
    clean, _ = data_mod.clean_chain(cached_spy_snapshot)
    disc = surface_mod.build_surface_detailed(
        cached_spy_snapshot, clean, SPY_TEST_CURVE, dividends=schedule
    )
    assert disc.converged and disc.unconverged_quotes == 0
    f = disc.forwards.merge(spy_build.forwards, on="expiry", suffixes=("", "_cont"))
    no_divs = f["pv_dividends"] == 0.0
    # Before the first ex-date the two lattices are the same lattice (the forwards agree to
    # the outer loop's tolerance: the two runs stop after different numbers of passes).
    assert no_divs.sum() >= 5
    assert np.allclose(f.loc[no_divs, "forward"], f.loc[no_divs, "forward_cont"], rtol=1e-6)
    # After it, the calls' pre-ex-date premium pulls the parity forward down.
    assert (f.loc[~no_divs, "forward"] < f.loc[~no_divs, "forward_cont"]).all()
    calls = disc.surface.loc[disc.surface["option_type"] == "call"]
    assert calls.loc[calls["tau"] < float(schedule.times[0]), "ee_premium"].max() == 0.0
    assert calls.loc[calls["tau"] > 1.5, "ee_premium"].max() > 0.1
