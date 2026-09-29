"""End-to-end analysis: download a chain, build the surface, calibrate Heston, plot.

Run with::

    python scripts/run_analysis.py --ticker SPY

Everything is cached under ``data/``, so a second run is offline and reproduces the same
numbers. Results are written to ``results/`` as JSON and as a Markdown table that is
pasted straight into the README.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optpricing import (  # noqa: E402
    binomial,
    blackscholes,
    calibration,
    data,
    greeks_check,
    heston,
    implied_vol,
    montecarlo,
    parity,
    plotting,
    surface,
)
from optpricing.types import ExerciseStyle, OptionType  # noqa: E402


def _shown(path: Path) -> str:
    """``path`` relative to the project when it is inside it, else absolute."""
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(path)


def _to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Render a DataFrame as a GitHub Markdown table without pulling in ``tabulate``."""

    def cell(v: Any) -> str:
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def _benchmark_pricers() -> dict[str, Any]:
    """Cross-check the three European pricers and the two Heston formulations."""
    s, k, tau, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0
    exact = float(blackscholes.price(s, k, tau, r, sigma, OptionType.CALL, q))

    crr = binomial.price(s, k, tau, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, 2001, "crr")
    lr = binomial.price(s, k, tau, r, sigma, OptionType.CALL, ExerciseStyle.EUROPEAN, 101, "lr")

    rows: list[dict[str, Any]] = []
    for label, anti, cv in [
        ("plain", False, False),
        ("antithetic", True, False),
        ("control variate", False, True),
        ("antithetic + control", True, True),
    ]:
        draws = 400_000 if anti else 200_000
        res = montecarlo.european_price(
            s,
            k,
            tau,
            r,
            sigma,
            OptionType.CALL,
            draws,
            antithetic=anti,
            control_variate=cv,
            seed=20260918,
        )
        rows.append(
            {
                "scheme": label,
                "price": res.price,
                "std_error": res.std_error,
                "effective_samples": res.n_paths,
                "error_vs_bs": res.price - exact,
            }
        )
    plain_se = rows[0]["std_error"]
    for row in rows:
        row["variance_reduction"] = (plain_se / row["std_error"]) ** 2

    asian_plain = montecarlo.asian_price(
        s, k, tau, r, sigma, OptionType.CALL, 12, 200_000, control_variate=False, seed=11
    )
    asian_cv = montecarlo.asian_price(
        s, k, tau, r, sigma, OptionType.CALL, 12, 200_000, control_variate=True, seed=11
    )

    return {
        "black_scholes": exact,
        "binomial_crr_2001": crr,
        "binomial_lr_101": lr,
        "binomial_crr_error": crr - exact,
        "binomial_lr_error": lr - exact,
        "monte_carlo": rows,
        "asian": {
            "plain_se": asian_plain.std_error,
            "control_se": asian_cv.std_error,
            "variance_reduction": (asian_plain.std_error / asian_cv.std_error) ** 2,
            "control_corr": asian_cv.control_corr,
        },
        "american_put_premium": (
            binomial.price(s, k, tau, r, sigma, OptionType.PUT, ExerciseStyle.AMERICAN, 3001, "crr")
            - float(blackscholes.price(s, k, tau, r, sigma, OptionType.PUT, q))
        ),
    }


#: Tolerances asserted by tests/test_greeks_check.py, drawn on the figure for reference.
GREEKS_TOLERANCES = (1e-6, 5e-3)


def _greeks_check() -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Every analytic Greek against finite differences of the price, over a stress grid."""
    errors = greeks_check.error_table(greeks_check.default_grid())
    summary = greeks_check.summarise(errors)
    eps = float(np.finfo(np.float64).eps)
    textbook = greeks_check.error_table(
        greeks_check.default_grid(), first=eps ** (1 / 3), second=eps ** (1 / 4)
    )
    summary["max_rel_error_textbook_step"] = [
        float(
            textbook.loc[
                (textbook["greek"] == row.greek) & (textbook["option_type"] == row.option_type),
                "rel_error",
            ].max()
        )
        for row in summary.itertuples()
    ]
    # One-day, 20% vol, one standard deviation out of the money: where step choice bites.
    tau, sigma = 1 / 365, 0.20
    strike = float(100.0 * np.exp(0.025 * tau) * np.exp(sigma * np.sqrt(tau)))
    point = greeks_check.GreekPoint(100.0, strike, tau, 0.04, sigma, 0.015, OptionType.CALL)
    first, second = greeks_check.default_steps(point)
    sweep_names = ("delta", "gamma", "volga", "charm")
    sweeps = {name: greeks_check.step_sweep(point, name) for name in sweep_names}
    chosen = {name: first if name in greeks_check.FIRST_ORDER else second for name in sweep_names}
    figure = plotting.plot_greeks_fd(errors, sweeps, chosen, GREEKS_TOLERANCES)
    return errors, summary, figure


def _greeks_markdown(summary: pd.DataFrame, n_points: int) -> str:
    table = summary.rename(
        columns={
            "max_abs_error": "max abs error",
            "max_rel_error": "max rel error",
            "median_rel_error": "median rel error",
            "worst_at": "worst case at",
            "max_rel_error_textbook_step": "max rel error, textbook step",
        }
    )
    return (
        "# Analytic Greeks vs central finite differences of the price\n\n"
        "Generated by `python scripts/run_analysis.py` "
        "(`optpricing.greeks_check`). Do not edit by hand.\n\n"
        f"Grid: {n_points} points per option type -- calls and puts; 1, 7, 30, 91, 365 and "
        "1,095 days; strikes at z = -3..+3 standard deviations from the forward "
        "(z = ln(K/F) / (sigma sqrt(tau))); vol 10% and 40%; r = 4%, q = 1.5%. "
        "Second-order Greeks are second differences of the *price*, not first "
        "differences of the analytic delta or vega.\n\n"
        "Relative error = |analytic - FD| / max(|analytic|, 1e-3 x the largest |value| of "
        "that Greek across the smile at the same maturity/vol/side). The floor only "
        "matters where a Greek crosses zero (volga near the money, vanna at d2 = 0).\n\n"
        "Steps are scaled to each variable's natural scale (spot: S sigma sqrt(tau); "
        "vol: sigma; tau: tau) and to the round-off of the price formula, "
        "eps_eff = eps / (sigma sqrt(tau)): eps_eff^(1/3) for first derivatives, "
        "eps_eff^(1/4) for second. The last column repeats the check with the textbook "
        "eps^(1/3), eps^(1/4) for comparison. Tests assert max rel error < "
        f"{GREEKS_TOLERANCES[0]:.0e} (first order) and < {GREEKS_TOLERANCES[1]:.0e} "
        "(second order).\n\n" + _to_markdown(table, floatfmt="{:.1e}")
    )


def _parity_markdown(result: parity.ParityResult, meta: str) -> str:
    dist = result.distribution.copy()
    dist["violation_rate"] = 100.0 * dist["violation_rate"]
    dist = dist.rename(columns={"violation_rate": "violation %"})
    by_exp = result.by_expiry.copy()
    by_exp["expiry"] = by_exp["expiry"].astype(str)
    fwd = result.forwards[
        [
            "expiry",
            "tau",
            "n_pairs",
            "forward_raw",
            "forward_adjusted",
            "forward_shift_bp",
            "dispersion_raw",
            "dispersion_adjusted",
            "dividend_yield_raw",
            "dividend_yield_adjusted",
        ]
    ].copy()
    fwd["expiry"] = fwd["expiry"].astype(str)
    fwd["dividend_yield_raw"] *= 100.0
    fwd["dividend_yield_adjusted"] *= 100.0
    fwd = fwd.rename(
        columns={"dividend_yield_raw": "q_raw %", "dividend_yield_adjusted": "q_adjusted %"}
    )
    gap = result.vol_gap.copy()
    gap["expiry"] = gap["expiry"].astype(str)
    stale = result.stale.copy()
    stale["expiry"] = stale["expiry"].astype(str)
    upper = result.upper_bound.copy()
    upper["expiry"] = upper["expiry"].astype(str)
    n_upper = int(upper["upper_violations"].sum())
    n_deep = int(upper["upper_violations_deep_itm_call"].sum())
    upper_total = (
        f"\nTotal: {n_upper} violations, {n_deep} of them at K < 0.95 S "
        f"({100.0 * n_deep / max(n_upper, 1):.0f}%).\n"
    )
    return (
        "# Put-call parity on the committed SPY snapshot\n\n"
        "Generated by `python scripts/run_analysis.py` (`optpricing.parity`). "
        "Do not edit by hand.\n\n"
        f"{meta}\n\n"
        "A pair is every (expiry, strike) with a two-sided, uncrossed quote on both the call "
        "and the put, from the *raw* chain. A pair violates parity *beyond the spread* when "
        "the theoretical synthetic forward lies outside [C_bid - P_ask, C_ask - P_bid]; "
        "`excess` is the distance outside, in dollars. *In-window* strikes (within 10% of "
        "spot) are the ones the forward is fitted to; the rest are out of sample. "
        "`European parity` uses the plain parity forward (the pre-correction pipeline); "
        "`American-adjusted` uses the forward with early-exercise premia removed, priced at "
        "the vols of the de-Americanised surface -- the forward the surface itself uses.\n\n"
        "## Violations beyond the bid-ask spread\n\n"
        + _to_markdown(dist)
        + "\n## Per expiry\n\n"
        + _to_markdown(by_exp)
        + "\n## Forward: European parity vs American-adjusted\n\n"
        "`dispersion` is the interquartile range, in dollars, of the per-strike forward "
        "estimates K + (C - P)/D (American-adjusted: K + (C - P - e_C + e_P)/D) over the "
        "in-window pairs. A correct model of parity makes it small.\n\n"
        + _to_markdown(fwd)
        + "\n## Call-minus-put implied vol at strikes within 1% of the forward (vol points)\n\n"
        "`european`: raw mids, European-parity forward. `american`: mids minus each leg's "
        "early-exercise premium (priced at the surface vol), American-adjusted forward. "
        "The same gap measured on the surfaces themselves, with each quote's premium "
        "solved by its own fixed point, is in `exercise_comparison.md`.\n\n"
        + _to_markdown(gap)
        + "\n## Quotes arbitrageable on their own (no model, no forward)\n\n"
        "`ask_below_intrinsic`: calls offered below S - K or puts below K - S (SPY options "
        "are American, so these could be bought and exercised at once). `implied_spot`: "
        "the spot that would make those quotes fair. `strike_monotonicity_violations`: "
        "adjacent strikes where a call bid exceeds the lower strike's call ask, or the "
        "put mirror image.\n\n"
        + _to_markdown(stale)
        + "\n## American upper bound C - P <= S - K e^(-r tau) (no forward, no dividends)\n\n"
        + _to_markdown(upper)
        + upper_total
    )


def _validation_checks() -> dict[str, Any]:
    """Cross-checks quoted in the README, measured here so every number is traceable.

    The unit tests assert looser tolerances (so they survive platform differences); this
    records the values actually achieved on this machine.
    """
    out: dict[str, Any] = {}
    strikes = np.array([60.0, 80.0, 90.0, 100.0, 110.0, 120.0, 150.0])

    # Heston, Black-Scholes limit: xi = 0, v0 = theta = sigma^2 (same grid as the tests).
    sigma = 0.25
    limit = heston.HestonParams(v0=sigma**2, kappa=2.0, theta=sigma**2, xi=0.0, rho=-0.5)
    worst = 0.0
    for tau in (0.05, 0.25, 1.0, 3.0):
        for opt in (OptionType.CALL, OptionType.PUT):
            h = np.asarray(heston.price(100.0, strikes, tau, 0.03, limit, opt, 0.01))
            b = np.asarray(blackscholes.price(100.0, strikes, tau, 0.03, sigma, opt, 0.01))
            worst = max(worst, float(np.max(np.abs(h - b))))
    out["heston_bs_limit_max_abs"] = worst

    # Heston, Gil-Pelaez vs Lewis on an equity-like parameter set (same grid as the tests).
    equity = heston.HestonParams(v0=0.04, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
    worst = 0.0
    for tau in (0.02, 0.1, 0.5, 2.0, 5.0):
        a = np.asarray(heston.price(100.0, strikes, tau, 0.025, equity, OptionType.CALL))
        b = np.asarray(
            heston.price(100.0, strikes, tau, 0.025, equity, OptionType.CALL, method="lewis")
        )
        worst = max(worst, float(np.max(np.abs(a - b))))
    out["heston_gil_pelaez_vs_lewis_max_abs"] = worst

    # American call = European call without dividends (benchmark case, 3,001-step CRR).
    args = (100.0, 100.0, 1.0, 0.05, 0.2, OptionType.CALL)
    american = binomial.price(*args, ExerciseStyle.AMERICAN, 3001, "crr")
    european = binomial.price(*args, ExerciseStyle.EUROPEAN, 3001, "crr")
    out["american_minus_european_call_no_dividend"] = abs(american - european)

    # Implied vol round trip over a stress grid: 5 days to 3 years, 5% to 150% vol,
    # strikes 60%-160% of spot, calls and puts.
    rows = []
    for opt in (OptionType.CALL, OptionType.PUT):
        for tau in (5 / 365, 30 / 365, 1.0, 3.0):
            for vol in (0.05, 0.2, 0.5, 1.5):
                k = 100.0 * np.array([0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.6])
                px = np.asarray(blackscholes.price(100.0, k, tau, 0.03, vol, opt, 0.01))
                vega = np.asarray(blackscholes.vega(100.0, k, tau, 0.03, vol, 0.01))
                res = implied_vol.implied_vol(
                    px, 100.0, k, tau, 0.03, opt, 0.01, return_diagnostics=True
                )
                assert isinstance(res, implied_vol.ImpliedVolResult)
                for i in range(k.size):
                    rows.append(
                        {
                            "vega": float(vega[i]),
                            "price": float(px[i]),
                            "error": abs(float(res.vol[i]) - vol),
                            "iterations": float(res.iterations[i]),
                        }
                    )
    grid = pd.DataFrame(rows).dropna()
    # Quotes worth less than ~1e-14 of notional carry no information about vol at all.
    usable = grid.loc[grid["vega"] > 1e-8]
    out["iv_grid_quotes"] = len(grid)
    out["iv_usable_quotes"] = len(usable)
    out["iv_min_vega_usable"] = float(usable["vega"].min())
    out["iv_max_abs_vol_error"] = float(usable["error"].max())
    out["iv_max_iterations"] = int(usable["iterations"].max())
    worst_row = usable.iloc[int(np.argmax(usable["error"].to_numpy()))]
    out["iv_worst_price"] = float(worst_row["price"])
    out["iv_worst_vega"] = float(worst_row["vega"])
    out["iv_worst_roundoff_floor"] = float(
        np.finfo(np.float64).eps * worst_row["price"] / worst_row["vega"]
    )
    # The worst case above is set by round-off in the *price* (about eps * price / vega),
    # not by the solver: excluding only quotes with vega < 1e-6 shows the solver itself.
    solid = grid.loc[grid["vega"] >= 1e-6]
    out["iv_quotes_vega_ge_1e-6"] = len(solid)
    out["iv_max_abs_vol_error_vega_ge_1e-6"] = float(solid["error"].max())
    out["iv_max_iterations_vega_ge_1e-6"] = int(solid["iterations"].max())

    # Why a price-only stop is not enough: a 1e-10 price tolerance on a five-day
    # 10%-out-of-the-money call leaves tol / vega of vol uncertainty.
    out["five_day_10pct_otm_call"] = [
        {
            "sigma": s,
            "vega": float(blackscholes.vega(100.0, 110.0, 5 / 365, 0.03, s, 0.01)),
            "vol_error_from_1e-10_price_tol": 1e-10
            / float(blackscholes.vega(100.0, 110.0, 5 / 365, 0.03, s, 0.01)),
        }
        for s in (0.10, 0.15, 0.20)
    ]
    return out


def _band_table(surf: pd.DataFrame) -> pd.DataFrame:
    """Median bid-to-ask implied-vol width (vol points) by expiry and moneyness bucket."""
    df = surf.assign(
        band=100.0 * (surf["iv_ask"] - surf["iv_bid"]),
        days=(surf["tau"] * 365.0).round().astype(int),
        k_bucket=pd.cut(
            surf["log_moneyness"],
            [-np.inf, -0.3, -0.15, -0.05, 0.05, 0.15, np.inf],
            labels=["k<-0.30", "-0.30..-0.15", "-0.15..-0.05", "|k|<0.05", "0.05..0.15", "k>0.15"],
        ),
    )
    table = df.pivot_table(
        index="days", columns="k_bucket", values="band", aggfunc="median", observed=False
    )
    table.columns = [str(c) for c in table.columns]
    return table.reset_index()


def _validation_markdown(checks: dict[str, Any], bands: pd.DataFrame, meta: str) -> str:
    otm = pd.DataFrame(checks["five_day_10pct_otm_call"])
    return (
        "# Numerical cross-checks and surface diagnostics\n\n"
        "Generated by `python scripts/run_analysis.py`. Do not edit by hand. The unit tests "
        "assert looser tolerances than these so they pass across platforms; this file "
        "records what is actually achieved.\n\n"
        "## Pricer cross-checks\n\n"
        "| check | max abs difference |\n| --- | --- |\n"
        f"| Heston (xi = 0, v0 = theta) vs Black-Scholes, 4 maturities x 7 strikes x "
        f"call/put | {checks['heston_bs_limit_max_abs']:.1e} |\n"
        f"| Heston Gil-Pelaez vs Lewis, 5 maturities (0.02-5y) x 7 strikes | "
        f"{checks['heston_gil_pelaez_vs_lewis_max_abs']:.1e} |\n"
        f"| American vs European call, no dividends, 3,001-step CRR (S=K=100, 1y, "
        f"r=5%, vol 20%) | {checks['american_minus_european_call_no_dividend']:.1e} |\n\n"
        "## Implied-vol round trip\n\n"
        f"Grid: calls and puts; 5 days, 30 days, 1y, 3y; vol 5%, 20%, 50%, 150%; strikes "
        f"60-160% of spot ({checks['iv_grid_quotes']} quotes). Quotes with vega below 1e-8 "
        "carry no usable vol information and are excluded, leaving "
        f"{checks['iv_usable_quotes']} (smallest vega {checks['iv_min_vega_usable']:.1e}).\n\n"
        f"- worst absolute vol error: {checks['iv_max_abs_vol_error']:.1e}\n"
        f"- most iterations: {checks['iv_max_iterations']}\n\n"
        f"The worst case has price {checks['iv_worst_price']:.2f} and vega "
        f"{checks['iv_worst_vega']:.1e}: a price is only representable to about eps x price, "
        f"which is {checks['iv_worst_roundoff_floor']:.1e} of vol at that vega, so no solver "
        "can do better. Restricted to the "
        f"{checks['iv_quotes_vega_ge_1e-6']} quotes with vega >= 1e-6:\n\n"
        f"- worst absolute vol error: {checks['iv_max_abs_vol_error_vega_ge_1e-6']:.1e}\n"
        f"- most iterations: {checks['iv_max_iterations_vega_ge_1e-6']}\n\n"
        "Why the solver also needs a volatility-space stop: the vol error a 1e-10 *price* "
        "tolerance leaves on a five-day 10%-out-of-the-money call (S=100, K=110):\n\n"
        + _to_markdown(otm, floatfmt="{:.1e}")
        + "\n## Bid-to-ask implied-vol width on the SPY surface (vol points, median)\n\n"
        f"{meta}\n\n" + _to_markdown(bands, floatfmt="{:.2f}")
    )


#: Lattice steps for the early-exercise premium; the pipeline re-checks at about twice this.
TREE_STEPS = 201
FINE_STEPS = 2 * TREE_STEPS - 1  # odd, as Leisen-Reimer needs


def _fit_models(surf: pd.DataFrame, spot: float, per_expiry: int, verbose: bool) -> dict[str, Any]:
    """Heston, the two Black-Scholes benchmarks and the strike holdout on one surface."""
    thin = surface.thin_surface(surf, per_expiry)
    started = time.time()
    fit = calibration.calibrate(thin, spot, verbose=verbose)
    elapsed = time.time() - started
    _, global_errors = calibration.fit_global_flat_vol(thin)
    flat_vols, per_expiry_errors = calibration.fit_flat_vol_per_expiry(thin)
    rows = []
    for name, n_params, errors in [
        ("Black-Scholes, one vol", 1, global_errors),
        ("Black-Scholes, one vol per expiry", int(thin["tau"].nunique()), per_expiry_errors),
        ("Heston", 5, fit.errors),
    ]:
        err = errors["vol_error"].dropna().to_numpy(dtype=np.float64)
        rows.append(
            {
                "model": name,
                "free_parameters": n_params,
                "rmse_vol_points": float(np.sqrt(np.mean(err**2)) * 100),
                "mae_vol_points": float(np.mean(np.abs(err)) * 100),
                "max_abs_vol_points": float(np.max(np.abs(err)) * 100),
            }
        )
    wing = fit.errors.loc[
        (fit.errors["log_moneyness"] < -0.15) & (fit.errors["tau"] < 0.15), "vol_error"
    ].dropna()
    body = fit.errors.loc[fit.errors["log_moneyness"].between(-0.15, 0.15), "vol_error"].dropna()
    holdout = calibration.cross_validate(thin, spot)
    holdout_df = pd.DataFrame(
        [
            {
                "model": h.model,
                "in_sample_rmse": h.in_sample_rmse,
                "out_of_sample_rmse": h.out_of_sample_rmse,
                "degradation": h.degradation,
            }
            for h in holdout
        ]
    )
    return {
        "thin": thin,
        "fit": fit,
        "seconds": elapsed,
        "comparison": pd.DataFrame(rows),
        "per_expiry": flat_vols,
        "body_rmse": float(np.sqrt((body**2).mean()) * 100),
        "n_body": len(body),
        "wing_rmse": float(np.sqrt((wing**2).mean()) * 100),
        "n_wing": len(wing),
        "holdout": holdout_df,
    }


def _rmse_split(errors: pd.DataFrame) -> dict[str, float]:
    """RMSE (vol points) overall, in the body and in the short-dated put wing."""
    e = errors.dropna(subset=["vol_error"])
    body = e.loc[e["log_moneyness"].between(-0.15, 0.15), "vol_error"]
    wing = e.loc[(e["log_moneyness"] < -0.15) & (e["tau"] < 0.15), "vol_error"]
    return {
        name: float(np.sqrt((x**2).mean()) * 100)
        for name, x in (("all", e["vol_error"]), ("body", body), ("wing", wing))
    }


def _cross_scores(fits: dict[str, dict[str, Any]], spot: float) -> pd.DataFrame:
    """Each surface's Heston parameters scored on each surface's calibration quotes.

    Separates what changed in the *data* from what changed in the *fit*: if the old
    parameters already score better on the new quotes in some region, that region's
    quotes moved; if only the new parameters do, the optimiser found a different
    compromise.
    """
    rows = []
    for data_mode in ("european", "american"):
        for param_mode in ("european", "american"):
            errors = calibration.surface_errors(
                fits[param_mode]["fit"].params, fits[data_mode]["thin"], spot
            )
            split = _rmse_split(errors)
            rows.append(
                {
                    "quotes": data_mode,
                    "parameters": param_mode,
                    "rmse_all": split["all"],
                    "rmse_body": split["body"],
                    "rmse_short_put_wing": split["wing"],
                }
            )
    return pd.DataFrame(rows)


def _butterfly_tolerance_table(builds: dict[str, surface.SurfaceBuild]) -> pd.DataFrame:
    """Mid-price butterfly violations at several dollar tolerances, per surface.

    The zero-tolerance count is sensitive to how exactly-flat runs of tick-quantised mids
    (a deep-wing put quoted at $0.055 on ten consecutive strikes) are treated: their
    butterfly is 0 to round-off, and removing a premium that rises with strike tips them
    negative by millionths of a dollar. Counting at a hundredth and a tenth of a cent
    shows whether a change is that or real.
    """
    rows = []
    for mode, build in builds.items():
        flies: list[np.ndarray] = []
        for tau, g in build.surface.groupby("tau"):
            sl = g.sort_values("strike")
            k = sl["strike"].to_numpy(dtype=np.float64)
            c = np.asarray(
                blackscholes.price(
                    sl["forward"].to_numpy(dtype=np.float64),
                    k,
                    float(tau),  # type: ignore[arg-type]
                    0.0,
                    sl["implied_vol"].to_numpy(dtype=np.float64),
                    OptionType.CALL,
                    0.0,
                )
            )
            lam = (k[2:] - k[1:-1]) / (k[2:] - k[:-2])
            flies.append(lam * c[:-2] + (1.0 - lam) * c[2:] - c[1:-1])
        fly = np.concatenate(flies)
        row: dict[str, Any] = {"surface": mode, "triples": fly.size}
        for label, tol in (("< -1e-12", 1e-12), ("< -$0.0001", 1e-4), ("< -$0.001", 1e-3)):
            row[label] = int((fly < -tol).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _surface_summary(build: surface.SurfaceBuild) -> dict[str, Any]:
    arb = surface.arbitrage_report(build.surface)
    term = surface.atm_term_structure(build.surface)
    return {
        "n_quotes": len(build.surface),
        "arbitrage": arb,
        "atm_first": float(term["atm_vol"].iloc[0]),
        "atm_last": float(term["atm_vol"].iloc[-1]),
        "term": term,
    }


def _exercise_markdown(
    builds: dict[str, surface.SurfaceBuild],
    summaries: dict[str, dict[str, Any]],
    fits: dict[str, dict[str, Any]],
    gaps: dict[str, pd.DataFrame],
    steps_check: dict[str, float],
    parity_consistency: float,
    cross: pd.DataFrame,
    meta: str,
) -> str:
    """Before/after table: the European-parity pipeline against the American-corrected one."""
    modes = ("european", "american")
    heads = {"european": "European parity (old)", "american": "American-corrected (new)"}

    def row(label: str, values: list[str]) -> str:
        return f"| {label} | " + " | ".join(values) + " |\n"

    def model_rmse(mode: str, name: str) -> float:
        cmp = fits[mode]["comparison"]
        return float(cmp.loc[cmp["model"] == name, "rmse_vol_points"].iloc[0])

    def holdout(mode: str, name: str, col: str) -> float:
        h = fits[mode]["holdout"]
        return float(h.loc[h["model"] == name, col].iloc[0])

    out = (
        "# American exercise in the surface: before and after\n\n"
        "Generated by `python scripts/run_analysis.py`. Do not edit by hand.\n\n"
        f"{meta}\n\n"
        "Both columns are computed in the same run, from the same cleaned quotes, rate curve, "
        "thinning and calibration settings. *European parity* is `build_surface(..., "
        'exercise="european")`: quotes inverted as quoted, forward from plain put-call '
        "parity. *American-corrected* is the default: each quote's early-exercise premium "
        f"({TREE_STEPS}-step Leisen-Reimer lattice, continuous dividend yield) is removed "
        "by a per-quote fixed point, and the forward is re-solved from de-Americanised "
        "parity, the two iterated together. Vol errors are model minus market, in vol "
        "points; *body* is |k| < 0.15, the *short-dated put wing* k < -0.15 and "
        "tau < 0.15 years.\n\n"
        "## Headline\n\n"
        "| | " + " | ".join(heads[m] for m in modes) + " |\n| --- | --- | --- |\n"
    )
    fmt = "{:.2f}".format
    out += row("surface quotes", [str(summaries[m]["n_quotes"]) for m in modes])
    out += row("calibration quotes (thinned)", [str(len(fits[m]["thin"])) for m in modes])
    out += row("Heston RMSE, all", [fmt(model_rmse(m, "Heston")) for m in modes])
    out += row(
        "Heston RMSE, body",
        [f"{fits[m]['body_rmse']:.2f} ({fits[m]['n_body']} quotes)" for m in modes],
    )
    out += row(
        "Heston RMSE, short-dated put wing",
        [f"{fits[m]['wing_rmse']:.2f} ({fits[m]['n_wing']} quotes)" for m in modes],
    )
    out += row(
        "Heston MAE / max",
        [
            f"{fits[m]['fit'].mae_vol * 100:.2f} / {fits[m]['fit'].max_abs_vol_error * 100:.2f}"
            for m in modes
        ],
    )
    for name in ("Black-Scholes, one vol per expiry", "Black-Scholes, one vol"):
        out += row(f"{name}, RMSE", [fmt(model_rmse(m, name)) for m in modes])
    for name in ("Heston", "Black-Scholes, one vol per expiry", "Black-Scholes, one vol"):
        out += row(
            f"holdout {name}: in / out of sample",
            [
                f"{holdout(m, name, 'in_sample_rmse'):.2f} / "
                f"{holdout(m, name, 'out_of_sample_rmse'):.2f}"
                for m in modes
            ],
        )
    names = ("v0", "kappa", "theta", "xi", "rho")
    for name in names:
        out += row(
            f"Heston {name}",
            [
                f"{getattr(fits[m]['fit'].params, name):.4f} "
                f"(se {fits[m]['fit'].std_errors.get(name, float('nan')):.4f})"
                for m in modes
            ],
        )
    out += row(
        "Feller ratio 2 kappa theta / xi^2",
        [f"{fits[m]['fit'].params.feller_ratio:.3f}" for m in modes],
    )
    out += row(
        "multi-start RMSE range",
        [
            f"{min(fits[m]['fit'].seed_rmses) * 100:.4f}-{max(fits[m]['fit'].seed_rmses) * 100:.4f}"
            for m in modes
        ],
    )
    out += row(
        "corr(kappa, xi) / corr(kappa, theta)",
        [
            f"{fits[m]['fit'].correlations.loc['kappa', 'xi']:+.2f} / "
            f"{fits[m]['fit'].correlations.loc['kappa', 'theta']:+.2f}"
            if not fits[m]["fit"].correlations.empty
            else "n/a"
            for m in modes
        ],
    )
    for label, attr in (
        ("calendar violations", "calendar"),
        ("butterfly violations, zero tolerance", "zero"),
        ("butterfly violations, net of spread", "net"),
    ):
        vals = []
        for m in modes:
            a = summaries[m]["arbitrage"]
            if attr == "calendar":
                vals.append(f"{a.calendar_violations}/{a.calendar_checks}")
            elif attr == "zero":
                vals.append(
                    f"{a.butterfly_violations_zero_tol}/{a.butterfly_checks} "
                    f"({a.butterfly_rate_zero_tol:.1%})"
                )
            else:
                vals.append(
                    f"{a.butterfly_violations}/{a.butterfly_checks} ({a.butterfly_rate:.1%})"
                )
        out += row(label, vals)
    out += row(
        "ATM vol, first / last expiry",
        [f"{summaries[m]['atm_first']:.1%} / {summaries[m]['atm_last']:.1%}" for m in modes],
    )

    fwd = builds["american"].forwards.copy()
    days = (fwd["tau"] * 365.0).round().astype(int)
    gap_e = gaps["european"].set_index("expiry")["gap_vol_points"]
    gap_a = gaps["american"].set_index("expiry")["gap_vol_points"]
    atm_e = summaries["european"]["term"].set_index("expiry")["atm_vol"]
    atm_a = summaries["american"]["term"].set_index("expiry")["atm_vol"]
    by_expiry = pd.DataFrame(
        {
            "expiry": fwd["expiry"].astype(str),
            "days": days,
            "forward_old": fwd["forward_european"],
            "forward_new": fwd["forward"],
            "shift_bp": fwd["forward_shift_bp"],
            "q_old %": 100.0 * fwd["dividend_yield_european"],
            "q_new %": 100.0 * fwd["dividend_yield"],
            "gap_old": fwd["expiry"].map(gap_e).to_numpy(),
            "gap_new": fwd["expiry"].map(gap_a).to_numpy(),
            "atm_old %": 100.0 * fwd["expiry"].map(atm_e).to_numpy(),
            "atm_new %": 100.0 * fwd["expiry"].map(atm_a).to_numpy(),
        }
    )
    surf = builds["american"].surface
    prem = surf.assign(days=(surf["tau"] * 365.0).round().astype(int)).pivot_table(
        index="days", columns="option_type", values="ee_premium", aggfunc=["median", "max"]
    )
    prem.columns = [f"{col[1]} premium {col[0]} $" for col in prem.columns.to_list()]
    prem = prem.reset_index()
    b = builds["american"]
    history = ", ".join(f"{h:.1e}" for h in b.forward_history)
    out += (
        "\n## By expiry\n\n"
        "`shift_bp`: American-corrected forward over the European-parity one. `q`: implied "
        "dividend yield r - ln(F/S)/tau with Treasury discounting. `gap`: median call-minus-put "
        "implied vol (vol points) at strikes within 1% of the forward, both legs inverted the "
        "way the surface inverts them (old: raw mids at the European forward; new: each "
        "leg de-Americanised by its own fixed point at the corrected forward). `atm`: "
        "vega-weighted vol within |k| < 0.05.\n\n"
        + _to_markdown(by_expiry)
        + "\n## Butterflies on mid prices, by tolerance\n\n"
        "Undiscounted call prices repriced from each quote's implied vol, as in "
        "`surface.arbitrage_report`. The zero-tolerance column is the one in the headline "
        "table; runs of identical tick-quantised mids have a butterfly of exactly zero, and "
        "removing a premium that rises with strike tips them negative by millionths of a "
        "dollar.\n\n"
        + _to_markdown(_butterfly_tolerance_table(builds))
        + "\n## Where the change in fit comes from\n\n"
        "Each surface's Heston parameters scored on each surface's calibration quotes "
        "(RMSE, vol points). Rows with matching `quotes` and `parameters` reproduce the "
        "headline. If the old parameters scored better on the new quotes in a region, "
        "the *quotes* there moved; if only the new parameters do, the optimiser found a "
        "different compromise across the surface.\n\n"
        + _to_markdown(cross)
        + "\n## Early-exercise premium removed from the surface quotes\n\n"
        "Per expiry, over the out-of-the-money quotes the surface keeps. Calls carry none "
        "wherever the implied dividend yield is not positive: with a continuous yield an "
        "American call is then never exercised early.\n\n"
        + _to_markdown(prem, floatfmt="{:.3f}")
        + "\n## Convergence of the correction\n\n"
        f"- forward -> surface -> forward passes: {b.outer_iterations} "
        f"(largest relative forward change per pass: {history}); converged: {b.converged}\n"
        f"- per-quote fixed point: at most {b.max_fixed_point_iterations} iterations to "
        f"1e-8 in vol; largest observed contraction (|step n+1| / |step n|) "
        f"{b.max_contraction:.3f}; unconverged quotes: {b.unconverged_quotes}; quotes lost "
        f"because the corrected price fell to the European floor: {b.dropped_by_correction}\n"
        f"- lattice resolution: re-running the correction at {FINE_STEPS} steps moves "
        f"surface vols by at most {steps_check['max_vol_diff_points']:.4f} vol points "
        f"(median {steps_check['median_vol_diff_points']:.5f}) and forwards by at most "
        f"{steps_check['max_forward_diff_bp']:.2f} bp\n"
        f"- self-consistency: the parity module's American-adjusted forward, given this "
        f"surface, matches the surface's own forward to {parity_consistency:.1e} "
        "(largest relative difference)\n"
    )
    return out


def main() -> int:  # noqa: PLR0915  (a top-level pipeline reads better in one piece)
    """Run the full pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="underlying symbol")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    parser.add_argument("--max-expiries", type=int, default=12)
    parser.add_argument("--quotes-per-expiry", type=int, default=40, help="calibration thinning")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    parser.add_argument(
        "--exercise",
        choices=["american", "european"],
        default="american",
        help="treat quotes as American (de-Americanise; default) or European (old pipeline)",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="do not calibrate the other exercise treatment for exercise_comparison.md",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.set_option("display.width", 200)
    results: dict[str, Any] = {}

    print("=" * 78)
    print("1. Cross-checking the pricers against each other")
    print("=" * 78)
    bench = _benchmark_pricers()
    results["pricer_benchmarks"] = bench
    print(f"  Black-Scholes            {bench['black_scholes']:.8f}")
    print(
        f"  CRR, 2001 steps          {bench['binomial_crr_2001']:.8f}  "
        f"({bench['binomial_crr_error']:+.2e})"
    )
    print(
        f"  Leisen-Reimer, 101 steps {bench['binomial_lr_101']:.8f}  "
        f"({bench['binomial_lr_error']:+.2e})"
    )
    for row in bench["monte_carlo"]:
        print(
            f"  MC {row['scheme']:<21} {row['price']:.6f}  SE {row['std_error']:.5f}  "
            f"variance reduction {row['variance_reduction']:5.1f}x"
        )
    print(
        f"  Arithmetic Asian: control variate cuts variance "
        f"{bench['asian']['variance_reduction']:.0f}x (rho={bench['asian']['control_corr']:.4f})"
    )
    print(f"  American put early-exercise premium: {bench['american_put_premium']:.4f}")

    print()
    print("=" * 78)
    print("1b. Analytic Greeks vs central finite differences of the price")
    print("=" * 78)
    started = time.time()
    greek_errors, greek_summary, greeks_fd_figure = _greeks_check()
    print(
        "  "
        + greek_summary.drop(columns="points")
        .to_string(index=False, float_format=lambda x: f"{x:9.1e}")
        .replace("\n", "\n  ")
    )
    print(f"  ({time.time() - started:.1f}s, {len(greek_errors)} Greek evaluations)")
    results["greeks_fd"] = greek_summary.to_dict(orient="records")

    print()
    print("=" * 78)
    print(f"2. Market data: {args.ticker}")
    print("=" * 78)
    snapshot = data.load_chain(
        args.ticker, force_refresh=args.refresh, max_expiries=args.max_expiries
    )
    # Price the chain with the rate curve from the same day, not today's.
    rate_curve = data.load_rate_curve(force_refresh=args.refresh, asof=snapshot.asof.date())
    print(f"  {rate_curve}")
    print(
        f"  as of {snapshot.asof:%Y-%m-%d %H:%M %Z}   spot {snapshot.spot:.2f}   "
        f"{len(snapshot.quotes)} raw quotes   {len(snapshot.expiries())} expiries"
    )
    clean, report = data.clean_chain(snapshot)
    print("  " + str(report).replace("\n", "\n  "))

    fwd_fixed = data.implied_forward_curve(clean, snapshot.spot, rate_curve, method="fixed-rate")
    fwd_reg = data.implied_forward_curve(clean, snapshot.spot, method="regression")
    comparison = fwd_fixed.merge(fwd_reg, on=["expiry", "tau"], suffixes=("", "_reg"))
    print("\n  Forward extraction, fixed-rate vs joint regression:")
    print(
        "  "
        + comparison[["tau", "forward", "forward_reg", "rate", "rate_reg", "r_squared_reg"]]
        .to_string(index=False, float_format=lambda x: f"{x:9.4f}")
        .replace("\n", "\n  ")
    )
    results["forward_curve"] = fwd_fixed.assign(expiry=fwd_fixed["expiry"].astype(str)).to_dict(
        orient="records"
    )
    results["regression_rate_range"] = [float(fwd_reg["rate"].min()), float(fwd_reg["rate"].max())]

    print()
    print("=" * 78)
    print("3. Implied-volatility surface")
    print("=" * 78)
    other = "european" if args.exercise == "american" else "american"
    started = time.time()
    builds = {
        ex: surface.build_surface_detailed(
            snapshot, clean, rate_curve, exercise=ex, tree_steps=TREE_STEPS
        )
        for ex in ("european", "american")
    }
    build = builds[args.exercise]
    surf = build.surface
    amer = builds["american"]
    print(f"  surface built as {args.exercise} ({time.time() - started:.1f}s for both)")
    print(
        f"  de-Americanisation: {amer.outer_iterations} forward passes, per-quote fixed point "
        f"<= {amer.max_fixed_point_iterations} iterations, contraction <= "
        f"{amer.max_contraction:.3f}, {amer.unconverged_quotes} unconverged, "
        f"{amer.dropped_by_correction} dropped"
    )
    print(
        "  "
        + amer.forwards[["tau", "forward_european", "forward", "forward_shift_bp"]]
        .to_string(index=False, float_format=lambda x: f"{x:9.3f}")
        .replace("\n", "\n  ")
    )
    fine = surface.build_surface_detailed(
        snapshot, clean, rate_curve, exercise="american", tree_steps=FINE_STEPS
    )
    joined = amer.surface.merge(
        fine.surface, on=["expiry", "strike", "option_type"], suffixes=("", "_fine")
    )
    vol_diff = 100.0 * (joined["implied_vol"] - joined["implied_vol_fine"]).abs()
    steps_check = {
        "max_vol_diff_points": float(vol_diff.max()),
        "median_vol_diff_points": float(vol_diff.median()),
        "max_forward_diff_bp": float(
            1e4 * np.max(np.abs(fine.forwards["forward"] / amer.forwards["forward"] - 1.0))
        ),
    }
    print(
        f"  {FINE_STEPS} vs {TREE_STEPS} lattice steps: vols move <= "
        f"{steps_check['max_vol_diff_points']:.4f} vol pts, forwards <= "
        f"{steps_check['max_forward_diff_bp']:.2f} bp"
    )
    gaps = {
        ex: surface.call_put_gap(
            surface.build_surface_detailed(
                snapshot,
                clean,
                rate_curve,
                otm_only=False,
                max_abs_log_moneyness=0.05,
                exercise=ex,
                tree_steps=TREE_STEPS,
                forwards=builds[ex].forwards,
            ).surface
        )
        for ex in ("european", "american")
    }
    print(f"  {len(surf)} quotes inverted across {surf['tau'].nunique()} expiries")
    term = surface.atm_term_structure(surf)
    print(
        "  " + term.to_string(index=False, float_format=lambda x: f"{x:9.4f}").replace("\n", "\n  ")
    )
    arb = surface.arbitrage_report(surf)
    print("  " + str(arb).replace("\n", "\n  "))
    results["surface"] = {
        "exercise": args.exercise,
        "n_quotes": len(surf),
        "n_expiries": int(surf["tau"].nunique()),
        "atm_term_structure": term.assign(expiry=term["expiry"].astype(str)).to_dict(
            orient="records"
        ),
        "arbitrage": {
            "calendar_violations": arb.calendar_violations,
            "calendar_checks": arb.calendar_checks,
            "butterfly_violations": arb.butterfly_violations,
            "butterfly_violations_zero_tol": arb.butterfly_violations_zero_tol,
            "butterfly_checks": arb.butterfly_checks,
        },
        "cleaning": report.to_frame().to_dict(orient="records"),
        "forwards": builds[args.exercise]
        .forwards.assign(expiry=lambda d: d["expiry"].astype(str))
        .to_dict(orient="records"),
        "de_americanisation": {
            "outer_iterations": amer.outer_iterations,
            "forward_history": amer.forward_history,
            "converged": amer.converged,
            "max_fixed_point_iterations": amer.max_fixed_point_iterations,
            "max_contraction": amer.max_contraction,
            "unconverged_quotes": amer.unconverged_quotes,
            "dropped_by_correction": amer.dropped_by_correction,
            "tree_steps": TREE_STEPS,
            "doubled_steps_check": steps_check,
        },
        "call_put_gap": {
            ex: g.assign(expiry=g["expiry"].astype(str)).to_dict(orient="records")
            for ex, g in gaps.items()
        },
    }

    print()
    print("=" * 78)
    print("3b. Put-call parity")
    print("=" * 78)
    started = time.time()
    # The American-adjusted theory needs de-Americanised vols whichever surface is primary.
    parity_result = parity.run_parity_analysis(
        snapshot, clean, rate_curve, fwd_fixed, amer.surface, steps=TREE_STEPS
    )
    consistency = parity_result.forwards.merge(amer.forwards, on="expiry")
    parity_consistency = float(
        np.max(np.abs(consistency["forward_adjusted"] / consistency["forward"] - 1.0))
    )
    print(f"  parity's American forward vs the surface's: max rel diff {parity_consistency:.1e}")
    print(
        "  "
        + parity_result.distribution.to_string(
            index=False, float_format=lambda x: f"{x:8.3f}"
        ).replace("\n", "\n  ")
    )
    print(
        "\n  "
        + parity_result.forwards[
            [
                "tau",
                "forward_raw",
                "forward_adjusted",
                "forward_shift_bp",
                "dispersion_raw",
                "dispersion_adjusted",
            ]
        ]
        .to_string(index=False, float_format=lambda x: f"{x:9.3f}")
        .replace("\n", "\n  ")
    )
    print(
        "\n  "
        + parity_result.vol_gap.to_string(index=False, float_format=lambda x: f"{x:7.2f}").replace(
            "\n", "\n  "
        )
    )
    stale = parity_result.stale
    print(
        f"\n  quotes with ask below intrinsic: {int(stale['ask_below_intrinsic'].sum())}; "
        f"strike-monotonicity violations: {int(stale['strike_monotonicity_violations'].sum())}; "
        f"American upper-bound violations: "
        f"{int(parity_result.upper_bound['upper_violations'].sum())}"
    )
    print(f"  ({time.time() - started:.1f}s)")

    def _records(df: pd.DataFrame) -> Any:
        return df.assign(expiry=df["expiry"].astype(str)).to_dict(orient="records")

    results["parity"] = {
        "distribution": parity_result.distribution.to_dict(orient="records"),
        "by_expiry": _records(parity_result.by_expiry),
        "forwards": _records(parity_result.forwards),
        "call_put_vol_gap": _records(parity_result.vol_gap),
        "stale_quotes": _records(parity_result.stale),
        "american_upper_bound": _records(parity_result.upper_bound),
    }

    print()
    print("=" * 78)
    print("4. Heston calibration")
    print("=" * 78)
    fitted = _fit_models(surf, snapshot.spot, args.quotes_per_expiry, verbose=True)
    thin, fit, elapsed = fitted["thin"], fitted["fit"], fitted["seconds"]
    per_expiry = fitted["per_expiry"]
    comparison_df = fitted["comparison"]
    model_rows = comparison_df.to_dict(orient="records")
    print(f"  calibrated on {len(thin)} thinned quotes ({elapsed:.1f}s)")
    print("  " + str(fit).replace("\n", "\n  "))
    print()
    print(
        "  "
        + comparison_df.to_string(index=False, float_format=lambda x: f"{x:8.2f}").replace(
            "\n", "\n  "
        )
    )
    print(
        f"\n  RMSE in the body (|k| < 0.15):                "
        f"{fitted['body_rmse']:5.2f} vol pts  ({fitted['n_body']} quotes)"
    )
    print(
        f"  RMSE in the short-dated put wing (k < -0.15, tau < 0.15): "
        f"{fitted['wing_rmse']:5.2f} vol pts  ({fitted['n_wing']} quotes)"
    )

    print()
    print("  Parameter uncertainty (asymptotic, local to this optimum):")
    print("  " + fit.uncertainty_summary().replace("\n", "\n  ").strip())
    if not fit.correlations.empty:
        print("\n  Parameter correlations:")
        print("  " + fit.correlations.round(2).to_string().replace("\n", "\n  "))

    print()
    print("  Out-of-sample: fit on alternate strikes, score on the rest")
    holdout_df = fitted["holdout"]
    print(
        "  "
        + holdout_df.to_string(index=False, float_format=lambda x: f"{x:8.2f}").replace(
            "\n", "\n  "
        )
    )
    results["holdout"] = holdout_df.to_dict(orient="records")

    results["heston"] = {
        "params": asdict(fit.params),
        "std_errors": fit.std_errors,
        "correlations": fit.correlations.to_dict() if not fit.correlations.empty else {},
        "feller_ratio": fit.params.feller_ratio,
        "rmse_vol_points": fit.rmse_vol * 100,
        "mae_vol_points": fit.mae_vol * 100,
        "max_abs_vol_points": fit.max_abs_vol_error * 100,
        "n_quotes": fit.n_quotes,
        "seed_rmse_vol_points": [x * 100 for x in fit.seed_rmses],
        "seconds": elapsed,
        "body_rmse_vol_points": fitted["body_rmse"],
        "short_dated_put_wing_rmse_vol_points": fitted["wing_rmse"],
        "per_expiry_flat_vols": per_expiry.to_dict(orient="records"),
    }
    results["model_comparison"] = model_rows

    fits: dict[str, dict[str, Any]] = {args.exercise: fitted}
    if not args.skip_comparison:
        print()
        print(f"  Same calibration on the {other} surface, for exercise_comparison.md")
        started = time.time()
        fits[other] = _fit_models(
            builds[other].surface, snapshot.spot, args.quotes_per_expiry, verbose=False
        )
        alt = fits[other]
        print(
            f"  ({time.time() - started:.1f}s) Heston RMSE {alt['fit'].rmse_vol * 100:.2f} "
            f"(body {alt['body_rmse']:.2f}, short-dated put wing {alt['wing_rmse']:.2f}) "
            f"vs {fit.rmse_vol * 100:.2f} on the {args.exercise} surface"
        )

    print()
    print("=" * 78)
    print("5. Figures")
    print("=" * 78)
    figures = {
        "greeks": plotting.plot_greeks_panel(),
        "greeks_fd": greeks_fd_figure,
        "binomial_convergence": plotting.plot_binomial_convergence(),
        "mc_convergence": plotting.plot_mc_convergence(),
        "exercise_boundary": plotting.plot_exercise_boundary(),
        "smiles": plotting.plot_smiles(surf),
        "smiles_by_expiry": plotting.plot_smile_grid(surf),
        "parity_residuals": plotting.plot_parity_residuals(parity_result.residuals),
        "term_structure": plotting.plot_term_structure(surf),
        "surface_heatmap": plotting.plot_surface_heatmap(surf),
        "surface_3d": plotting.plot_surface_3d(surf),
        "heston_fit": plotting.plot_heston_fit(fit.errors),
        "heston_errors": plotting.plot_heston_fit_errors(fit.errors, fit.params),
    }
    for path in plotting.save_all(figures, args.out):
        print(f"  wrote {_shown(path)}")

    args.results.mkdir(parents=True, exist_ok=True)
    results["meta"] = {
        "ticker": snapshot.ticker,
        "asof": snapshot.asof.isoformat(),
        "spot": snapshot.spot,
    }
    (args.results / "model_comparison.md").write_text(_to_markdown(comparison_df))
    (args.results / "holdout.md").write_text(_to_markdown(holdout_df))
    surf.to_csv(args.results / "surface.csv", index=False)
    (args.results / "greeks_fd.md").write_text(
        _greeks_markdown(greek_summary, int(greek_summary["points"].iloc[0]))
    )
    meta = (
        f"Data: {snapshot.ticker} option chain from Yahoo Finance, captured "
        f"{snapshot.asof:%Y-%m-%d %H:%M} ET, spot {snapshot.spot:.2f}; discounting from the "
        f"same day's Treasury curve. {len(parity_result.residuals)} matched call/put pairs."
    )
    (args.results / "parity.md").write_text(_parity_markdown(parity_result, meta))
    if len(fits) == 2:
        summaries = {ex: _surface_summary(b) for ex, b in builds.items()}
        cross = _cross_scores(fits, snapshot.spot)
        (args.results / "exercise_comparison.md").write_text(
            _exercise_markdown(
                builds, summaries, fits, gaps, steps_check, parity_consistency, cross, meta
            )
        )
        results["exercise_cross_scores"] = cross.to_dict(orient="records")
        results["exercise_comparison"] = {
            ex: {
                "heston_params": asdict(f["fit"].params),
                "heston_rmse_vol_points": f["fit"].rmse_vol * 100,
                "body_rmse_vol_points": f["body_rmse"],
                "short_dated_put_wing_rmse_vol_points": f["wing_rmse"],
                "model_comparison": f["comparison"].to_dict(orient="records"),
                "holdout": f["holdout"].to_dict(orient="records"),
                "n_quotes": summaries[ex]["n_quotes"],
                "arbitrage": asdict(summaries[ex]["arbitrage"]),
            }
            for ex, f in fits.items()
        }
    checks = _validation_checks()
    results["validation"] = checks
    (args.results / "validation.md").write_text(
        _validation_markdown(checks, _band_table(surf), meta)
    )
    (args.results / "results.json").write_text(json.dumps(results, indent=2, default=str))
    parity_result.residuals.to_csv(args.results / "parity_pairs.csv", index=False)
    print(f"  wrote {_shown(args.results / 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
