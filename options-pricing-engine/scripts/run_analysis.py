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
    return (
        "# Put-call parity on the committed SPY snapshot\n\n"
        "Generated by `python scripts/run_analysis.py` (`optpricing.parity`). "
        "Do not edit by hand.\n\n"
        f"{meta}\n\n"
        "A pair is every (expiry, strike) with a two-sided, uncrossed quote on both the call "
        "and the put, from the *raw* chain. A pair violates parity *beyond the spread* when "
        "the theoretical synthetic forward lies outside [C_bid - P_ask, C_ask - P_bid]; "
        "`excess` is the distance outside, in dollars. *In-window* strikes (within 10% of "
        "spot) are the ones the pipeline's forward was fitted to; the rest are out of "
        "sample.\n\n"
        "## Violations beyond the bid-ask spread\n\n"
        + _to_markdown(dist)
        + "\n## Per expiry\n\n"
        + _to_markdown(by_exp)
        + "\n## Forward: pipeline (European parity) vs American-adjusted\n\n"
        "`dispersion` is the interquartile range, in dollars, of the per-strike forward "
        "estimates K + (C - P)/D (American-adjusted: K + (C - P - e_C + e_P)/D) over the "
        "in-window pairs. A correct model of parity makes it small.\n\n"
        + _to_markdown(fwd)
        + "\n## Call-minus-put implied vol at strikes within 1% of the forward (vol points)\n\n"
        "`pipeline`: raw mids, the pipeline's forward. `american`: mids minus each leg's "
        "early-exercise premium, American-adjusted forward.\n\n"
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
    )


def main() -> int:  # noqa: PLR0915  (a top-level pipeline reads better in one piece)
    """Run the full pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY", help="underlying symbol")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and re-download")
    parser.add_argument("--max-expiries", type=int, default=12)
    parser.add_argument("--quotes-per-expiry", type=int, default=40, help="calibration thinning")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
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
    surf = surface.build_surface(snapshot, clean, rate_curve)
    print(f"  {len(surf)} quotes inverted across {surf['tau'].nunique()} expiries")
    term = surface.atm_term_structure(surf)
    print(
        "  " + term.to_string(index=False, float_format=lambda x: f"{x:9.4f}").replace("\n", "\n  ")
    )
    arb = surface.arbitrage_report(surf)
    print("  " + str(arb).replace("\n", "\n  "))
    results["surface"] = {
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
    }

    print()
    print("=" * 78)
    print("3b. Put-call parity")
    print("=" * 78)
    started = time.time()
    parity_result = parity.run_parity_analysis(snapshot, clean, rate_curve, fwd_fixed, surf)
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
    thin = surface.thin_surface(surf, args.quotes_per_expiry)
    print(f"  calibrating on {len(thin)} thinned quotes")
    started = time.time()
    fit = calibration.calibrate(thin, snapshot.spot, verbose=True)
    elapsed = time.time() - started
    print(f"  ({elapsed:.1f}s)")
    print("  " + str(fit).replace("\n", "\n  "))

    _, global_errors = calibration.fit_global_flat_vol(thin)
    per_expiry, per_expiry_errors = calibration.fit_flat_vol_per_expiry(thin)
    model_rows = []
    for name, n_params, errors in [
        ("Black-Scholes, one vol", 1, global_errors),
        ("Black-Scholes, one vol per expiry", int(thin["tau"].nunique()), per_expiry_errors),
        ("Heston", 5, fit.errors),
    ]:
        err = errors["vol_error"].dropna().to_numpy(dtype=np.float64)
        model_rows.append(
            {
                "model": name,
                "free_parameters": n_params,
                "rmse_vol_points": float(np.sqrt(np.mean(err**2)) * 100),
                "mae_vol_points": float(np.mean(np.abs(err)) * 100),
                "max_abs_vol_points": float(np.max(np.abs(err)) * 100),
            }
        )
    comparison_df = pd.DataFrame(model_rows)
    print()
    print(
        "  "
        + comparison_df.to_string(index=False, float_format=lambda x: f"{x:8.2f}").replace(
            "\n", "\n  "
        )
    )

    wing = fit.errors.loc[
        (fit.errors["log_moneyness"] < -0.15) & (fit.errors["tau"] < 0.15), "vol_error"
    ].dropna()
    body = fit.errors.loc[fit.errors["log_moneyness"].between(-0.15, 0.15), "vol_error"].dropna()
    print(
        f"\n  RMSE in the body (|k| < 0.15):                "
        f"{np.sqrt((body**2).mean()) * 100:5.2f} vol pts  ({len(body)} quotes)"
    )
    print(
        f"  RMSE in the short-dated put wing (k < -0.15, tau < 0.15): "
        f"{np.sqrt((wing**2).mean()) * 100:5.2f} vol pts  ({len(wing)} quotes)"
    )

    print()
    print("  Parameter uncertainty (asymptotic, local to this optimum):")
    print("  " + fit.uncertainty_summary().replace("\n", "\n  ").strip())
    if not fit.correlations.empty:
        print("\n  Parameter correlations:")
        print("  " + fit.correlations.round(2).to_string().replace("\n", "\n  "))

    print()
    print("  Out-of-sample: fit on alternate strikes, score on the rest")
    holdout = calibration.cross_validate(thin, snapshot.spot)
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
        "body_rmse_vol_points": float(np.sqrt((body**2).mean()) * 100),
        "short_dated_put_wing_rmse_vol_points": float(np.sqrt((wing**2).mean()) * 100),
        "per_expiry_flat_vols": per_expiry.to_dict(orient="records"),
    }
    results["model_comparison"] = model_rows

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
    (args.results / "results.json").write_text(json.dumps(results, indent=2, default=str))
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
    parity_result.residuals.to_csv(args.results / "parity_pairs.csv", index=False)
    print(f"  wrote {_shown(args.results / 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
