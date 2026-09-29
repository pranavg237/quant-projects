r"""Check every analytic Black-Scholes Greek against finite differences of the *price*.

A passing unit test says "within tolerance at the points I happened to pick". This module
sweeps a grid that deliberately includes the places Greeks go wrong -- one day to expiry,
three standard deviations in and out of the money, calls *and* puts, low and high vol --
and returns the actual errors, so the worst case is reported rather than asserted.
``scripts/run_analysis.py`` writes the summary to ``results/greeks_fd.md``.

**Independence.** Every finite difference is taken on :func:`blackscholes.price` and
nothing else. Second-order Greeks (gamma, vanna, volga, charm) are second differences of
the *price*, including the mixed partials, not first differences of the analytic delta or
vega -- otherwise an error in ``delta`` would be inherited by the check on ``vanna``.

**Step sizes.** A central difference has truncation error :math:`O(h^2)` and round-off
error :math:`O(\text{noise}/h^m)` for an :math:`m`-th derivative. Two things set the step:

1. *Scale.* Each input is bumped by a multiple of the distance over which the price
   actually changes shape, not by a fixed amount. A one-day option's price curves over a
   spot range of :math:`S\sigma\sqrt\tau` (about 0.5% of spot at 10% vol), so
   ``h = 0.01`` in spot, fine at one year, is 2% of the relevant scale at one day.

   ========  =================================  ==========================================
   variable  scale                              why
   ========  =================================  ==========================================
   spot      :math:`S\sigma\sqrt{\tau}`         one standard deviation of the move
   strike    :math:`K\sigma\sqrt{\tau}`         the same, in strike
   vol       :math:`\sigma`                     price depends on :math:`\sigma\sqrt\tau`
   tau       :math:`\tau`                       price depends on :math:`\sqrt\tau`
   rate      :math:`\min(\sigma/\sqrt\tau, 1)`  :math:`d_1` moves by one per that much
   ========  =================================  ==========================================

2. *Noise.* The price is computed as :math:`Se^{-q\tau}N(d_1) - Ke^{-r\tau}N(d_2)`, a
   difference of two terms of size :math:`S`, so its round-off is about
   :math:`\epsilon S` -- not :math:`\epsilon V`. Measured on the natural price scale
   :math:`S\sigma\sqrt\tau` that is a relative noise of
   :math:`\epsilon_{\text{eff}} = \epsilon/(\sigma\sqrt\tau)`, which is 200x machine
   epsilon for a one-day 10%-vol option. Balancing truncation against that noise gives
   relative steps :math:`\epsilon_{\text{eff}}^{1/3}` for first derivatives and
   :math:`\epsilon_{\text{eff}}^{1/4}` for second derivatives.

The plain textbook choice (:math:`\epsilon^{1/3}`, :math:`\epsilon^{1/4}`) under-steps
short-dated options and was measured to lose about an order of magnitude on one-day
volga and gamma. :func:`step_sweep` produces the V-shaped error-vs-step curve that
justifies the choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import blackscholes as bs
from .types import OptionType

__all__ = [
    "FIRST_ORDER",
    "GREEK_NAMES",
    "SECOND_ORDER",
    "GreekPoint",
    "analytic_greeks",
    "default_grid",
    "default_steps",
    "error_table",
    "finite_difference_greeks",
    "step_sweep",
    "summarise",
]

_EPS = float(np.finfo(np.float64).eps)

FIRST_ORDER = ("delta", "vega", "theta", "rho", "dual_delta")
SECOND_ORDER = ("gamma", "vanna", "volga", "charm")
GREEK_NAMES = FIRST_ORDER + SECOND_ORDER

#: Relative-error floor, as a fraction of the largest value of that Greek across the
#: smile. Only matters where a Greek crosses zero (see :func:`error_table`).
REL_FLOOR = 1e-3


@dataclass(frozen=True)
class GreekPoint:
    """One set of Black-Scholes inputs."""

    spot: float
    strike: float
    tau: float
    rate: float
    sigma: float
    dividend_yield: float
    option_type: OptionType

    @property
    def total_vol(self) -> float:
        r""":math:`\sigma\sqrt\tau`."""
        return float(self.sigma * np.sqrt(self.tau))

    @property
    def z(self) -> float:
        r"""Standardised moneyness :math:`\ln(K/F)/(\sigma\sqrt\tau)`."""
        fwd = self.spot * np.exp((self.rate - self.dividend_yield) * self.tau)
        return float(np.log(self.strike / fwd) / self.total_vol)


def _scales(p: GreekPoint) -> dict[str, float]:
    sd = p.total_vol
    return {
        "spot": p.spot * sd,
        "strike": p.strike * sd,
        "sigma": p.sigma,
        "tau": p.tau,
        "rate": float(min(p.sigma / np.sqrt(p.tau), 1.0)),
    }


def default_steps(p: GreekPoint) -> tuple[float, float]:
    r"""Relative steps ``(first, second)`` from the noise balance in the module docstring."""
    eps_eff = _EPS / min(p.total_vol, 1.0)
    return eps_eff ** (1.0 / 3.0), eps_eff ** (1.0 / 4.0)


def finite_difference_greeks(
    p: GreekPoint, first: float | None = None, second: float | None = None
) -> dict[str, float]:
    r"""All nine Greeks by central differences of :func:`blackscholes.price` alone.

    Args:
        p: Inputs.
        first: Relative step for first derivatives, as a multiple of each variable's
            natural scale. ``None`` uses :func:`default_steps`.
        second: The same for second and mixed derivatives.

    Returns:
        Greeks in the units of :mod:`optpricing.blackscholes`: vega and volga per unit of
        vol, theta and charm per year and as derivatives in calendar time :math:`t` (so
        minus the :math:`\tau` derivative).
    """
    d_first, d_second = default_steps(p)
    first = d_first if first is None else first
    second = d_second if second is None else second
    base = {"spot": p.spot, "strike": p.strike, "tau": p.tau, "rate": p.rate, "sigma": p.sigma}

    def v(**bump: float) -> float:
        a = {k: base[k] + bump.get(k, 0.0) for k in base}
        value = bs.price(
            a["spot"], a["strike"], a["tau"], a["rate"], a["sigma"], p.option_type, p.dividend_yield
        )
        return float(np.asarray(value))

    scale = _scales(p)

    def diff1(var: str) -> float:
        h = first * scale[var]
        return (v(**{var: h}) - v(**{var: -h})) / (2.0 * h)

    def diff2(var: str) -> float:
        h = second * scale[var]
        return (v(**{var: h}) - 2.0 * v() + v(**{var: -h})) / (h * h)

    def mixed(a: str, b: str) -> float:
        ha, hb = second * scale[a], second * scale[b]
        return (
            v(**{a: ha, b: hb})
            - v(**{a: ha, b: -hb})
            - v(**{a: -ha, b: hb})
            + v(**{a: -ha, b: -hb})
        ) / (4.0 * ha * hb)

    return {
        "delta": diff1("spot"),
        "vega": diff1("sigma"),
        "theta": -diff1("tau"),
        "rho": diff1("rate"),
        "dual_delta": diff1("strike"),
        "gamma": diff2("spot"),
        "vanna": mixed("spot", "sigma"),
        "volga": diff2("sigma"),
        "charm": -mixed("spot", "tau"),
    }


def analytic_greeks(p: GreekPoint) -> dict[str, float]:
    """The analytic Greeks from :mod:`optpricing.blackscholes`, as plain floats."""
    g = bs.greeks(p.spot, p.strike, p.tau, p.rate, p.sigma, p.option_type, p.dividend_yield)
    return {name: float(np.asarray(getattr(g, name))) for name in GREEK_NAMES}


def default_grid(
    z_values: tuple[float, ...] = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    taus: tuple[float, ...] = (1 / 365, 7 / 365, 30 / 365, 0.25, 1.0, 3.0),
    sigmas: tuple[float, ...] = (0.10, 0.40),
    spot: float = 100.0,
    rate: float = 0.04,
    dividend_yield: float = 0.015,
) -> list[GreekPoint]:
    r"""Calls and puts across moneyness, maturity (down to one day) and volatility.

    Moneyness is set in **standard deviations**, :math:`z = \ln(K/F)/(\sigma\sqrt\tau)`,
    not as a fixed percentage of spot. A one-day option 20% out of the money at 10% vol
    is 38 standard deviations out, worth about :math:`10^{-300}`, and every Greek
    underflows to zero -- "agreeing" there tests nothing. :math:`|z| \le 3` spans the
    strikes that carry real optionality at every maturity. Rates and dividends are
    non-zero so the carry terms in theta, rho and charm are exercised.
    """
    points: list[GreekPoint] = []
    for opt in (OptionType.CALL, OptionType.PUT):
        for sigma in sigmas:
            for tau in taus:
                fwd = spot * np.exp((rate - dividend_yield) * tau)
                for z in z_values:
                    strike = float(fwd * np.exp(z * sigma * np.sqrt(tau)))
                    points.append(GreekPoint(spot, strike, tau, rate, sigma, dividend_yield, opt))
    return points


def error_table(
    points: list[GreekPoint], first: float | None = None, second: float | None = None
) -> pd.DataFrame:
    r"""Analytic vs finite-difference value of every Greek at every point.

    The relative error is :math:`|a - f| / \max(|a|,\ 10^{-3} s)` where :math:`s` is the
    largest :math:`|a|` for that Greek over the moneyness points sharing a maturity, vol
    and option type. The floor matters only where a Greek crosses zero -- volga near the
    money (:math:`d_1 d_2 \approx 0`), vanna at :math:`d_2 = 0`, charm and put theta
    somewhere in the smile. Without it a correct Greek evaluated at 1e-12 reports a huge
    relative error that says nothing about the formula.

    Returns:
        Long frame with one row per (point, Greek).
    """
    rows: list[dict[str, object]] = []
    for p in points:
        fd = finite_difference_greeks(p, first, second)
        an = analytic_greeks(p)
        for name in GREEK_NAMES:
            rows.append(
                {
                    "option_type": str(p.option_type),
                    "tau_days": p.tau * 365.0,
                    "sigma": p.sigma,
                    "z": round(p.z, 6),
                    "strike": p.strike,
                    "greek": name,
                    "order": 1 if name in FIRST_ORDER else 2,
                    "analytic": an[name],
                    "finite_difference": fd[name],
                }
            )
    df = pd.DataFrame(rows)
    df["abs_error"] = (df["analytic"] - df["finite_difference"]).abs()
    scale = df.groupby(["option_type", "tau_days", "sigma", "greek"])["analytic"].transform(
        lambda s: s.abs().max()
    )
    denom = np.maximum(df["analytic"].abs(), REL_FLOOR * scale)
    df["rel_error"] = df["abs_error"] / denom
    return df


def summarise(errors: pd.DataFrame) -> pd.DataFrame:
    """Worst absolute and relative error per Greek and option type, and where it occurs."""
    rows: list[dict[str, object]] = []
    for name in GREEK_NAMES:
        for opt in ("call", "put"):
            g = errors.loc[(errors["greek"] == name) & (errors["option_type"] == opt)]
            if g.empty:
                continue
            worst = g.iloc[int(np.argmax(g["rel_error"].to_numpy()))]
            rows.append(
                {
                    "greek": name,
                    "option_type": opt,
                    "points": len(g),
                    "max_abs_error": float(g["abs_error"].max()),
                    "max_rel_error": float(g["rel_error"].max()),
                    "median_rel_error": float(g["rel_error"].median()),
                    "worst_at": (
                        f"{worst['tau_days']:.0f}d, z={worst['z']:+.0f}, vol={worst['sigma']:.0%}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def step_sweep(p: GreekPoint, greek: str, multipliers: np.ndarray | None = None) -> pd.DataFrame:
    """Relative error of one finite-difference Greek as the relative step is varied.

    The same multiplier is used for first- and second-derivative steps, so the curve for
    each Greek is traced on its own; plot it against :func:`default_steps` to see where
    the chosen step sits. The curve is V-shaped -- round-off on the left, truncation on
    the right.

    Raises:
        ValueError: on an unknown Greek name.
    """
    if greek not in GREEK_NAMES:
        raise ValueError(f"unknown Greek {greek!r}")
    mults = np.logspace(-8, -1, 36) if multipliers is None else np.asarray(multipliers)
    exact = analytic_greeks(p)[greek]
    out = []
    for m in mults:
        fd = finite_difference_greeks(p, first=float(m), second=float(m))[greek]
        out.append({"step": float(m), "rel_error": abs(fd - exact) / abs(exact)})
    return pd.DataFrame(out)
