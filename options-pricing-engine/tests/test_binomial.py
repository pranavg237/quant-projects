"""Binomial lattice pricing.

The European tests pin convergence to Black-Scholes; the American tests use a published
textbook value plus structural properties (an American option is never worth less than its
European twin, and an American call on a non-dividend payer is worth exactly the same).
"""

from __future__ import annotations

import numpy as np
import pytest

from optpricing import binomial as bn
from optpricing import blackscholes as bs
from optpricing.binomial import TreeMethod
from optpricing.types import ExerciseStyle, OptionType

ATM = (100.0, 100.0, 1.0, 0.05, 0.20)


@pytest.mark.parametrize("method", [TreeMethod.CRR, TreeMethod.JR, TreeMethod.LR])
@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_european_tree_converges_to_black_scholes(
    method: TreeMethod, option_type: OptionType
) -> None:
    exact = float(bs.price(*ATM, option_type))
    steps = 201 if method is TreeMethod.LR else 4001
    tree = bn.price(*ATM, option_type, ExerciseStyle.EUROPEAN, steps, method)
    assert tree == pytest.approx(exact, abs=1e-3)


def test_leisen_reimer_beats_crr_at_equal_cost() -> None:
    """Leisen-Reimer at 101 steps beats CRR at 2001 -- 20x fewer nodes, 25x less error."""
    exact = float(bs.price(*ATM, OptionType.CALL))
    lr_err = abs(bn.price(*ATM, OptionType.CALL, ExerciseStyle.EUROPEAN, 101, "lr") - exact)
    crr_err = abs(bn.price(*ATM, OptionType.CALL, ExerciseStyle.EUROPEAN, 2001, "crr") - exact)
    assert lr_err < crr_err
    assert lr_err < 1e-4


def test_crr_convergence_order_is_one_over_n() -> None:
    """Doubling the steps roughly halves the CRR error (averaging out the sawtooth)."""
    exact = float(bs.price(*ATM, OptionType.CALL))

    def mean_err(n: int) -> float:
        errs = [
            abs(bn.price(*ATM, OptionType.CALL, ExerciseStyle.EUROPEAN, n + j, "crr") - exact)
            for j in range(8)
        ]
        return float(np.mean(errs))

    assert mean_err(200) / mean_err(400) == pytest.approx(2.0, rel=0.35)


def test_hull_american_put_five_step_example() -> None:
    """Hull's five-step American put: S=K=50, r=10%, sigma=40%, T=5/12 -> 4.49."""
    price = bn.price(
        50.0, 50.0, 5 / 12, 0.10, 0.40, OptionType.PUT, ExerciseStyle.AMERICAN, 5, "crr"
    )
    assert price == pytest.approx(4.49, abs=0.01)


def test_american_put_matches_published_benchmark() -> None:
    """S=K=100, r=5%, sigma=20%, T=1, no dividends: the American put is ~6.090."""
    price = bn.price(*ATM, OptionType.PUT, ExerciseStyle.AMERICAN, 5001, "crr")
    assert price == pytest.approx(6.0900, abs=2e-3)


def test_american_call_equals_european_without_dividends() -> None:
    """Early exercise of a call is never optimal when q = 0, so the premium must be zero."""
    american = bn.price(*ATM, OptionType.CALL, ExerciseStyle.AMERICAN, 1501, "crr")
    european = bn.price(*ATM, OptionType.CALL, ExerciseStyle.EUROPEAN, 1501, "crr")
    assert american == pytest.approx(european, abs=1e-10)


def test_american_call_exceeds_european_with_dividends() -> None:
    args = (100.0, 100.0, 1.0, 0.02, 0.25)
    american = bn.price(*args, OptionType.CALL, ExerciseStyle.AMERICAN, 1501, "crr", 0.08)
    european = bn.price(*args, OptionType.CALL, ExerciseStyle.EUROPEAN, 1501, "crr", 0.08)
    assert american > european + 1e-4


@pytest.mark.parametrize("option_type", [OptionType.CALL, OptionType.PUT])
def test_american_never_below_european_or_intrinsic(option_type: OptionType) -> None:
    for spot in (70.0, 100.0, 130.0):
        args = (spot, 100.0, 0.75, 0.04, 0.3)
        american = bn.price(*args, option_type, ExerciseStyle.AMERICAN, 601, "crr", 0.03)
        european = bn.price(*args, option_type, ExerciseStyle.EUROPEAN, 601, "crr", 0.03)
        intrinsic = max(option_type.sign * (spot - 100.0), 0.0)
        assert american >= european - 1e-9
        assert american >= intrinsic - 1e-9


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [
        # CRR and Leisen-Reimer solve for p given (u, d), so the martingale condition holds
        # to machine precision. Jarrow-Rudd instead fixes p = 1/2 and solves for (u, d),
        # which is risk-neutral only to O(dt^2) per step -- that residual is precisely why
        # JR converges at O(1/n) rather than faster. sigma^4 dt^2 / 12 ~ 3e-8 here.
        (TreeMethod.CRR, 1e-12),
        (TreeMethod.LR, 1e-12),
        (TreeMethod.JR, 1e-6),
    ],
)
def test_tree_is_risk_neutral(method: TreeMethod, tolerance: float) -> None:
    """p*u + (1-p)*d must equal the one-step growth factor, or the tree has arbitrage."""
    steps = 101
    tau, rate, q, sigma = 1.0, 0.05, 0.02, 0.25
    u, d, p, discount = bn.tree_parameters(100.0, 95.0, tau, rate, sigma, steps, method, q)
    dt = tau / steps
    growth = np.exp((rate - q) * dt)
    assert p * u + (1 - p) * d == pytest.approx(growth, rel=tolerance)
    assert discount == pytest.approx(np.exp(-rate * dt), rel=1e-12)
    assert 0.0 <= p <= 1.0
    assert d < growth < u  # no dominance either way

    if method is TreeMethod.JR:
        # Pin the size of the JR bias rather than merely tolerating it.
        bias = abs(p * u + (1 - p) * d - growth) / growth
        assert bias == pytest.approx(sigma**4 * dt**2 / 12.0, rel=0.05)


def test_leisen_reimer_requires_odd_steps() -> None:
    with pytest.raises(ValueError, match="odd number of steps"):
        bn.tree_parameters(100.0, 100.0, 1.0, 0.05, 0.2, 100, TreeMethod.LR)
    # The public pricer silently bumps to the next odd count rather than failing.
    assert bn.price(*ATM, OptionType.CALL, ExerciseStyle.EUROPEAN, 100, "lr") > 0


def test_invalid_tree_arguments() -> None:
    with pytest.raises(ValueError, match="steps must be"):
        bn.tree_parameters(100.0, 100.0, 1.0, 0.05, 0.2, 0)
    with pytest.raises(ValueError, match="tau must be"):
        bn.tree_parameters(100.0, 100.0, 0.0, 0.05, 0.2, 10)
    with pytest.raises(ValueError, match="positive spot and strike"):
        bn.tree_parameters(0.0, 100.0, 1.0, 0.05, 0.2, 11, TreeMethod.LR)
    with pytest.raises(ValueError, match="outside"):
        # Tiny volatility against a huge carry makes the tree arbitrageable.
        bn.tree_parameters(100.0, 100.0, 1.0, 5.0, 0.01, 10, TreeMethod.CRR)


def test_degenerate_tree_inputs() -> None:
    assert bn.price(110.0, 100.0, 0.0, 0.05, 0.2, OptionType.CALL) == pytest.approx(10.0)
    assert bn.price(110.0, 100.0, 1.0, 0.05, 0.0, OptionType.CALL) == pytest.approx(
        110.0 - 100.0 * np.exp(-0.05)
    )
    # A zero-vol American put on a spot below the strike exercises immediately.
    assert bn.price(90.0, 100.0, 1.0, 0.05, 0.0, OptionType.PUT, "american") == pytest.approx(
        max(10.0, 100.0 * np.exp(-0.05) - 90.0)
    )


def test_exercise_boundary_shape() -> None:
    times, boundary = bn.american_exercise_boundary(100.0, 100.0, 1.0, 0.05, 0.2, "put", 400)
    assert times.shape == boundary.shape == (401,)
    assert boundary[-1] == pytest.approx(100.0)
    finite = boundary[np.isfinite(boundary)]
    # The boundary rises towards the strike as expiry approaches and never exceeds it.
    assert np.all(finite <= 100.0 + 1e-9)
    assert finite[-1] > finite[0]
    # Within one step parity the boundary is monotone. Across parities it alternates by
    # about +/-0.8 here, because a CRR lattice at step k only has nodes at S*u^(2j-k): odd
    # and even steps sit on interleaved grids, so the "most extreme node that exercises"
    # flips between them. That is lattice granularity, not a boundary that moves.
    for parity in (0, 1):
        assert np.all(np.diff(finite[parity::2]) >= -1e-9)
    assert np.max(np.abs(np.diff(finite))) < 1.5


def test_exercise_boundary_for_a_call_on_a_dividend_payer() -> None:
    _, boundary = bn.american_exercise_boundary(
        100.0, 100.0, 1.0, 0.02, 0.2, "call", 300, dividend_yield=0.10
    )
    finite = boundary[np.isfinite(boundary)]
    assert finite.size > 0
    # A call exercises from *above*, so its boundary sits at or above the strike.
    assert np.all(finite >= 100.0 - 1e-9)


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValueError):
        bn.price(*ATM, OptionType.CALL, ExerciseStyle.EUROPEAN, 100, "trinomial")
    with pytest.raises(ValueError, match="unknown exercise style"):
        bn.price(*ATM, OptionType.CALL, "bermudan")
