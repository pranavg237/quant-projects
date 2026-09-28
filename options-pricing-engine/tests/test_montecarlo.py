"""Monte Carlo pricing and variance reduction.

Three kinds of check:

1. **Unbiasedness** -- the MC price sits inside its own confidence interval around the
   analytic answer.
2. **Calibration of the error bars** -- over many independent replications the realised
   RMSE matches the reported standard error. This is the check that catches the classic
   antithetic bug (treating the ``Z`` and ``-Z`` legs as independent), which leaves the
   *price* correct and only the *error bar* wrong.
3. **Variance reduction actually reduces variance** -- at matched effective sample counts.
"""

from __future__ import annotations

import numpy as np
import pytest

from optpricing import blackscholes as bs
from optpricing import montecarlo as mc
from optpricing.types import OptionType

ATM = (100.0, 100.0, 1.0, 0.05, 0.20)


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize(
    ("antithetic", "control"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_european_mc_is_unbiased(option_type: OptionType, antithetic: bool, control: bool) -> None:
    exact = float(bs.price(*ATM, option_type))
    res = mc.european_price(
        *ATM, option_type, 200_000, antithetic=antithetic, control_variate=control, seed=7
    )
    lo, hi = res.confidence_interval(0.999)
    assert lo < exact < hi


def test_variance_reduction_at_matched_effective_samples() -> None:
    """Each scheme is given the same number of *effective* samples, not the same draws."""
    n = 200_000
    ses = {}
    for label, anti, cv in [
        ("plain", False, False),
        ("antithetic", True, False),
        ("control", False, True),
        ("both", True, True),
    ]:
        draws = 2 * n if anti else n
        res = mc.european_price(
            *ATM, OptionType.CALL, draws, antithetic=anti, control_variate=cv, seed=7
        )
        assert res.n_paths == n
        ses[label] = res.std_error

    assert ses["antithetic"] < ses["plain"]
    assert ses["control"] < ses["plain"]
    assert ses["both"] < min(ses["antithetic"], ses["control"])
    # The headline number quoted in the README.
    variance_reduction = (ses["plain"] / ses["both"]) ** 2
    assert variance_reduction > 40.0


def test_reported_standard_error_is_calibrated() -> None:
    """Realised RMSE over independent runs must match the reported SE to within ~20%."""
    exact = float(bs.price(*ATM, OptionType.CALL))
    for anti, cv in [(False, False), (True, False), (True, True)]:
        prices, ses = [], []
        for rep in range(40):
            res = mc.european_price(
                *ATM,
                OptionType.CALL,
                40_000 * (2 if anti else 1),
                antithetic=anti,
                control_variate=cv,
                seed=1000 + 13 * rep,
            )
            prices.append(res.price)
            ses.append(res.std_error)
        rmse = float(np.sqrt(np.mean((np.array(prices) - exact) ** 2)))
        assert rmse / float(np.mean(ses)) == pytest.approx(1.0, abs=0.25)


def test_control_variate_diagnostics() -> None:
    """Without antithetics, beta is the regression of the payoff on S_T: 0 < beta < 1.

    A call payoff has slope ``1{S_T > K}`` in ``S_T``, so the least-squares slope is a
    probability-weighted average of 0 and 1 and must land strictly inside the unit
    interval. (With antithetics on, beta is *not* bounded this way: folding makes the
    control nearly deterministic, shrinking its variance far faster than its covariance
    with the payoff, and beta comes out around 2.5.)
    """
    res = mc.european_price(
        *ATM, OptionType.CALL, 50_000, antithetic=False, control_variate=True, seed=3
    )
    assert res.beta is not None
    assert res.control_corr is not None
    assert 0.8 < res.control_corr < 1.0
    assert 0.0 < res.beta < 1.0

    folded = mc.european_price(*ATM, OptionType.CALL, 50_000, control_variate=True, seed=3)
    assert folded.beta is not None
    assert folded.beta > 1.0

    plain = mc.european_price(*ATM, OptionType.CALL, 50_000, control_variate=False, seed=3)
    assert plain.beta is None and plain.control_corr is None


def test_zero_beta_reproduces_the_plain_estimator() -> None:
    """beta = 0 must switch the control off exactly, which pins the adjustment algebra."""
    with_control = mc.european_price(
        *ATM, OptionType.CALL, 20_000, control_variate=True, beta=0.0, seed=8
    )
    plain = mc.european_price(*ATM, OptionType.CALL, 20_000, control_variate=False, seed=8)
    assert with_control.price == pytest.approx(plain.price, abs=1e-12)
    assert with_control.std_error == pytest.approx(plain.std_error, abs=1e-12)


def test_explicit_beta_is_honoured() -> None:
    res = mc.european_price(*ATM, OptionType.CALL, 20_000, control_variate=True, beta=0.5, seed=5)
    assert res.beta == 0.5


def test_seeding_is_reproducible_and_generators_are_accepted() -> None:
    a = mc.european_price(*ATM, OptionType.CALL, 20_000, seed=42)
    b = mc.european_price(*ATM, OptionType.CALL, 20_000, seed=42)
    c = mc.european_price(*ATM, OptionType.CALL, 20_000, seed=43)
    assert a.price == b.price
    assert a.price != c.price
    gen = np.random.default_rng(42)
    assert mc.european_price(*ATM, OptionType.CALL, 20_000, seed=gen).price == pytest.approx(
        a.price
    )


def test_antithetic_pairs_are_exact_negations() -> None:
    terminal = mc.simulate_terminal_gbm(100.0, 1.0, 0.05, 0.2, 10_000, antithetic=True, seed=1)
    half = terminal.size // 2
    # S = S0 exp(drift + diff*Z), so the geometric mean of a (Z, -Z) pair is exp(drift).
    drift = (0.05 - 0.5 * 0.04) * 1.0
    pair_product = terminal[:half] * terminal[half:]
    assert np.allclose(pair_product, 100.0**2 * np.exp(2 * drift))


def test_terminal_distribution_moments() -> None:
    """E[S_T] = S0 e^{(r-q)T} and Var[ln S_T] = sigma^2 T, to sampling accuracy."""
    n = 400_000
    terminal = mc.simulate_terminal_gbm(
        100.0, 2.0, 0.05, 0.3, n, dividend_yield=0.02, antithetic=False, seed=11
    )
    assert float(terminal.mean()) == pytest.approx(100.0 * np.exp(0.03 * 2.0), rel=0.01)
    assert float(np.log(terminal).var(ddof=1)) == pytest.approx(0.09 * 2.0, rel=0.02)


def test_path_simulation_shape_and_exactness() -> None:
    paths = mc.simulate_gbm_paths(100.0, 1.0, 0.05, 0.2, 5_000, 12, antithetic=True, seed=2)
    assert paths.shape == (5_000, 13)
    assert np.all(paths[:, 0] == 100.0)
    # Each step is an exact lognormal transition, so log-increments are exactly normal.
    increments = np.diff(np.log(paths), axis=1)
    assert float(increments.std(ddof=1)) == pytest.approx(0.2 * np.sqrt(1 / 12), rel=0.02)


def test_geometric_asian_closed_form_matches_its_own_monte_carlo() -> None:
    """The analytic geometric Asian price is validated against a direct simulation."""
    s, k, tau, r, sigma, m = 100.0, 100.0, 1.0, 0.05, 0.25, 12
    analytic = mc.geometric_asian_price(s, k, tau, r, sigma, m, OptionType.CALL)
    paths = mc.simulate_gbm_paths(s, tau, r, sigma, 400_000, m, antithetic=True, seed=17)
    geo = np.exp(np.log(paths[:, 1:]).mean(axis=1))
    payoff = np.maximum(geo - k, 0.0)
    half = payoff.size // 2
    folded = 0.5 * (payoff[:half] + payoff[half:])
    simulated = float(np.exp(-r * tau) * folded.mean())
    se = float(np.exp(-r * tau) * folded.std(ddof=1) / np.sqrt(folded.size))
    assert abs(simulated - analytic) < 4 * se


def test_geometric_asian_is_cheaper_than_the_vanilla() -> None:
    """Averaging reduces variance, so an Asian call is worth less than the European."""
    asian = mc.geometric_asian_price(*ATM, 12, OptionType.CALL)
    european = float(bs.price(*ATM, OptionType.CALL))
    assert 0 < asian < european


def test_arithmetic_asian_control_variate_is_worth_it() -> None:
    plain = mc.asian_price(*ATM, OptionType.CALL, 12, 100_000, control_variate=False, seed=9)
    controlled = mc.asian_price(*ATM, OptionType.CALL, 12, 100_000, control_variate=True, seed=9)
    assert controlled.control_corr is not None
    assert controlled.control_corr > 0.99
    assert (plain.std_error / controlled.std_error) ** 2 > 100.0
    # The two estimates must still agree.
    assert abs(plain.price - controlled.price) < 4 * plain.std_error
    # Arithmetic >= geometric by AM-GM, so the arithmetic Asian is worth more.
    assert controlled.price > mc.geometric_asian_price(*ATM, 12, OptionType.CALL)


def test_asian_put_prices() -> None:
    res = mc.asian_price(*ATM, OptionType.PUT, 12, 50_000, control_variate=True, seed=4)
    assert res.price > 0
    assert res.price < float(bs.price(*ATM, OptionType.PUT))


def test_mc_result_confidence_interval_and_repr() -> None:
    res = mc.MCResult(price=10.0, std_error=0.1, n_paths=1000)
    lo, hi = res.confidence_interval(0.95)
    assert lo == pytest.approx(10.0 - 1.959964 * 0.1, abs=1e-5)
    assert hi == pytest.approx(10.0 + 1.959964 * 0.1, abs=1e-5)
    assert "MCResult" in repr(res)
    with pytest.raises(ValueError, match="level must be"):
        res.confidence_interval(1.5)


def test_invalid_simulation_arguments() -> None:
    with pytest.raises(ValueError, match="n_paths must be"):
        mc.simulate_terminal_gbm(100.0, 1.0, 0.05, 0.2, 0)
    with pytest.raises(ValueError, match="tau must be"):
        mc.simulate_terminal_gbm(100.0, -1.0, 0.05, 0.2, 10)
    with pytest.raises(ValueError, match="n_steps must be"):
        mc.simulate_gbm_paths(100.0, 1.0, 0.05, 0.2, 10, 0)
    with pytest.raises(ValueError, match="n_paths must be"):
        mc.simulate_gbm_paths(100.0, 1.0, 0.05, 0.2, 0, 5)
    with pytest.raises(ValueError, match="n_fixings must be"):
        mc.geometric_asian_price(100.0, 100.0, 1.0, 0.05, 0.2, 0)


def test_geometric_asian_degenerate_inputs() -> None:
    assert mc.geometric_asian_price(110.0, 100.0, 0.0, 0.05, 0.2, 12) == pytest.approx(10.0)
    assert mc.geometric_asian_price(110.0, 100.0, 1.0, 0.05, 0.0, 12) == pytest.approx(
        10.0 * np.exp(-0.05)
    )
