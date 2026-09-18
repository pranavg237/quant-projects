"""Full study: validate against the paper, compare strategies, sweep parameters, plot.

Run with::

    python scripts/run_analysis.py

Everything is seeded, so the numbers in the README reproduce exactly. Results are written
to ``results/`` as JSON and Markdown, figures to ``figures/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mmsim import plotting  # noqa: E402
from mmsim.avellaneda_stoikov import AvellanedaStoikovParams, HorizonMode  # noqa: E402
from mmsim.calibration import fit_as_params_to_book, volatility_signature  # noqa: E402
from mmsim.engine import simulate_reference  # noqa: E402
from mmsim.experiments import (  # noqa: E402
    build_policy_set,
    compare_policies_book,
    compare_policies_reference,
    sensitivity_sweep,
)
from mmsim.flow import FlowConfig  # noqa: E402
from mmsim.metrics import pnl_decomposition  # noqa: E402
from mmsim.strategies import AvellanedaStoikovPolicy, average_optimal_spread  # noqa: E402
from mmsim.types import MarketConfig  # noqa: E402

#: Avellaneda & Stoikov (2008) Table 1 parameters.
PAPER = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0, horizon=1.0)

#: Published Table 1 values, for the reproduction check.
PAPER_TABLE_1 = {
    "Inventory": {"spread": 1.49, "profit": 65.0, "std_profit": 6.6, "std_final_q": 2.0},
    "Symmetric": {"spread": 1.49, "profit": 68.4, "std_profit": 13.4, "std_final_q": 8.4},
}


def _to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a Markdown table without pulling in ``tabulate``."""

    def cell(v: Any) -> str:
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def _banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def reproduce_paper(n_runs: int) -> dict[str, Any]:
    """Reproduce Table 1 of Avellaneda & Stoikov (2008)."""
    avg_spread = average_optimal_spread(PAPER)
    print(f"  Average A-S spread, analytic : {avg_spread:.4f}   (paper: 1.49)")

    rows: list[dict[str, Any]] = []
    for fill_model in ("linear", "exact"):
        comparison = compare_policies_reference(
            build_policy_set(PAPER, inventory_limit=3.0)[:2],
            PAPER,
            n_runs=n_runs,
            n_steps=200,
            fill_model=fill_model,
        )
        for _, row in comparison.metrics.iterrows():
            label = "Inventory" if "Avellaneda" in str(row["policy"]) else "Symmetric"
            reference = PAPER_TABLE_1[label]
            rows.append(
                {
                    "fill_model": fill_model,
                    "strategy": label,
                    "spread": row["mean_spread_captured"],
                    "profit": row["mean_pnl"],
                    "std_profit": row["std_pnl"],
                    "std_final_q": row["std_final_inventory"],
                    "paper_profit": reference["profit"],
                    "paper_std_profit": reference["std_profit"],
                    "paper_std_final_q": reference["std_final_q"],
                }
            )
    frame = pd.DataFrame(rows)
    print()
    print(
        "  "
        + frame.to_string(index=False, float_format=lambda x: f"{x:8.2f}").replace("\n", "\n  ")
    )
    return {"table": frame.to_dict(orient="records"), "analytic_spread": avg_spread}


def horizon_study(n_runs: int) -> list[dict[str, Any]]:
    """How inventory control decays at the horizon, and what the stationary fix does."""
    finite = AvellanedaStoikovParams(
        gamma=0.1,
        kappa=1.5,
        arrival_rate=140.0,
        sigma=2.0,
        horizon=1.0,
        horizon_mode=HorizonMode.FINITE,
    )
    stationary = AvellanedaStoikovParams(
        gamma=0.1,
        kappa=1.5,
        arrival_rate=140.0,
        sigma=2.0,
        horizon=0.5,
        horizon_mode=HorizonMode.STATIONARY,
    )
    market = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)

    rows: list[dict[str, Any]] = []
    for label, params in (("finite T=1", finite), ("stationary h=0.5", stationary)):
        policy = AvellanedaStoikovPolicy(params)
        inventories = np.array(
            [
                simulate_reference(policy, market, n_steps=200, seed=1000 + i).inventory
                for i in range(n_runs)
            ]
        )
        row: dict[str, Any] = {"mode": label}
        for frac in (0.25, 0.50, 0.75, 1.00):
            row[f"std_q_at_t={frac:.2f}"] = float(inventories[:, int(frac * 200)].std(ddof=1))
        rows.append(row)
    frame = pd.DataFrame(rows)
    print(
        "  "
        + frame.to_string(index=False, float_format=lambda x: f"{x:7.2f}").replace("\n", "\n  ")
    )
    return [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]


def main() -> int:
    """Run the whole study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-reference", type=int, default=1000)
    parser.add_argument("--runs-book", type=int, default=120)
    parser.add_argument("--runs-sweep", type=int, default=300)
    parser.add_argument("--book-steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    pd.set_option("display.width", 220)
    results: dict[str, Any] = {}

    _banner("1. Reproducing Avellaneda & Stoikov (2008), Table 1")
    results["paper_reproduction"] = reproduce_paper(args.runs_reference)

    _banner("2. The finite-horizon model loses inventory control at the bell")
    results["horizon_study"] = horizon_study(args.runs_reference // 2)

    _banner("3. Strategy comparison in the idealised model")
    stationary = AvellanedaStoikovParams(
        gamma=0.1,
        kappa=1.5,
        arrival_rate=140.0,
        sigma=2.0,
        horizon=0.5,
        horizon_mode=HorizonMode.STATIONARY,
    )
    sim_market = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=2.0)
    reference_comparison = compare_policies_reference(
        build_policy_set(stationary, inventory_limit=3.0), sim_market, n_runs=args.runs_reference
    )
    display = [
        "policy",
        "mean_pnl",
        "std_pnl",
        "sharpe",
        "mean_abs_inventory",
        "std_final_inventory",
        "max_abs_inventory",
        "mean_trades",
        "mean_spread_captured",
        "max_drawdown",
    ]
    print(
        "  "
        + reference_comparison.metrics[display]
        .to_string(index=False, float_format=lambda x: f"{x:8.3f}")
        .replace("\n", "\n  ")
    )
    print()
    print("  Paired tests (common random numbers):")
    print(
        "  "
        + reference_comparison.paired_tests.to_string(
            index=False, float_format=lambda x: f"{x:9.4f}"
        ).replace("\n", "\n  ")
    )
    results["reference_comparison"] = reference_comparison.metrics.to_dict(orient="records")
    results["reference_paired_tests"] = reference_comparison.paired_tests.to_dict(orient="records")

    _banner("4. The same strategies in a full order book with informed flow")
    market = MarketConfig()
    flow = FlowConfig(informed_fraction=0.15, info_impact_ticks=2.0)
    params, fill_fit, sigma = fit_as_params_to_book(
        flow, market, gamma=0.5, risk_horizon=0.1, n_steps=4000, seed=1
    )
    print(
        f"  Estimated from the book: sigma={sigma:.4f}, A={fill_fit.arrival_rate:.0f}, "
        f"kappa={fill_fit.kappa:.2f} per price unit, fit R^2={fill_fit.r_squared:.3f}"
    )
    print(
        f"  => 1/kappa = {1.0 / fill_fit.kappa / market.tick_size:.2f} ticks; "
        f"A-S quotes {average_optimal_spread(params) / market.tick_size:.2f} ticks wide"
    )
    print(
        f"  Expected adverse selection: {flow.expected_adverse_selection_per_fill:.2f} "
        f"ticks per fill (informed fraction {flow.informed_fraction:.0%} "
        f"x {flow.info_impact_ticks:.0f} tick impact)"
    )

    signatures = {
        "no informed flow": volatility_signature(
            FlowConfig(informed_fraction=0.0), market, n_steps=8000, seed=3
        ),
        "15% informed, gradual": volatility_signature(flow, market, n_steps=8000, seed=3),
        "15% informed, instant": volatility_signature(
            FlowConfig(informed_fraction=0.15, info_impact_ticks=2.0, info_impact_speed=1.0),
            market,
            n_steps=8000,
            seed=3,
        ),
    }
    print("\n  Volatility signature (sigma estimated at each sampling horizon):")
    signature_table = pd.DataFrame(
        {name: frame.set_index("block_steps")["sigma"] for name, frame in signatures.items()}
    )
    print(
        "  " + signature_table.to_string(float_format=lambda x: f"{x:8.4f}").replace("\n", "\n  ")
    )
    results["volatility_signature"] = {
        name: frame.to_dict(orient="records") for name, frame in signatures.items()
    }

    book_comparison = compare_policies_book(
        build_policy_set(params, inventory_limit=6.0),
        flow,
        market,
        n_runs=args.runs_book,
        n_steps=args.book_steps,
    )
    print()
    print(
        "  "
        + book_comparison.metrics[display]
        .to_string(index=False, float_format=lambda x: f"{x:8.3f}")
        .replace("\n", "\n  ")
    )
    print()
    print(
        "  "
        + book_comparison.paired_tests.to_string(
            index=False, float_format=lambda x: f"{x:9.4f}"
        ).replace("\n", "\n  ")
    )

    decompositions = []
    for name, runs in book_comparison.runs.items():
        parts = pd.DataFrame([pnl_decomposition(r) for r in runs]).mean()
        decompositions.append(
            {
                "policy": name,
                "spread_capture": float(parts["spread_capture"]),
                "inventory_pnl": float(parts["inventory_pnl"]),
                "total": float(parts["total"]),
                "max_reconciliation_error": float(
                    pd.DataFrame([pnl_decomposition(r) for r in runs])["reconciliation_error"]
                    .abs()
                    .max()
                ),
            }
        )
    decomposition_frame = pd.DataFrame(decompositions)
    print()
    print("  PnL decomposition (mean per run):")
    print(
        "  "
        + decomposition_frame.to_string(index=False, float_format=lambda x: f"{x:10.5f}").replace(
            "\n", "\n  "
        )
    )

    results["book_comparison"] = book_comparison.metrics.to_dict(orient="records")
    results["book_paired_tests"] = book_comparison.paired_tests.to_dict(orient="records")
    results["pnl_decomposition"] = decomposition_frame.to_dict(orient="records")
    results["fill_fit"] = {
        "arrival_rate": fill_fit.arrival_rate,
        "kappa": fill_fit.kappa,
        "r_squared": fill_fit.r_squared,
        "sigma": sigma,
        "distances_ticks": (fill_fit.distances / market.tick_size).tolist(),
        "intensities": fill_fit.intensities.tolist(),
    }
    results["as_params_from_book"] = asdict(params) | {"horizon_mode": str(params.horizon_mode)}

    _banner("5. Adverse selection: does it change the ranking?")
    adverse_rows: list[dict[str, Any]] = []
    for informed in (0.0, 0.10, 0.20, 0.30):
        cfg = FlowConfig(informed_fraction=informed, info_impact_ticks=2.0)
        local_params, _, _ = fit_as_params_to_book(
            cfg, market, gamma=0.5, risk_horizon=0.1, n_steps=2500, seed=2
        )
        comparison = compare_policies_book(
            build_policy_set(local_params, inventory_limit=6.0)[:2],
            cfg,
            market,
            n_runs=max(args.runs_book // 3, 20),
            n_steps=args.book_steps,
        )
        for _, row in comparison.metrics.iterrows():
            adverse_rows.append(
                {
                    "informed_fraction": informed,
                    "expected_cost_ticks": cfg.expected_adverse_selection_per_fill,
                    "policy": row["policy"],
                    "mean_pnl": row["mean_pnl"],
                    "sharpe": row["sharpe"],
                    "std_final_inventory": row["std_final_inventory"],
                    "mean_trades": row["mean_trades"],
                }
            )
    adverse_frame = pd.DataFrame(adverse_rows)
    print(
        "  "
        + adverse_frame.to_string(index=False, float_format=lambda x: f"{x:8.3f}").replace(
            "\n", "\n  "
        )
    )
    results["adverse_selection"] = adverse_frame.to_dict(orient="records")

    _banner("6. Parameter sensitivity")
    sweeps: dict[str, pd.DataFrame] = {}

    def build_gamma(value: float) -> tuple[Any, AvellanedaStoikovParams]:
        p = AvellanedaStoikovParams(
            gamma=value,
            kappa=1.5,
            arrival_rate=140.0,
            sigma=2.0,
            horizon=0.5,
            horizon_mode=HorizonMode.STATIONARY,
        )
        return AvellanedaStoikovPolicy(p), sim_market

    def build_sigma(value: float) -> tuple[Any, AvellanedaStoikovParams]:
        p = AvellanedaStoikovParams(
            gamma=0.1,
            kappa=1.5,
            arrival_rate=140.0,
            sigma=value,
            horizon=0.5,
            horizon_mode=HorizonMode.STATIONARY,
        )
        world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=140.0, sigma=value)
        return AvellanedaStoikovPolicy(p), world

    def build_intensity(value: float) -> tuple[Any, AvellanedaStoikovParams]:
        p = AvellanedaStoikovParams(
            gamma=0.1,
            kappa=1.5,
            arrival_rate=value,
            sigma=2.0,
            horizon=0.5,
            horizon_mode=HorizonMode.STATIONARY,
        )
        world = AvellanedaStoikovParams(gamma=0.1, kappa=1.5, arrival_rate=value, sigma=2.0)
        return AvellanedaStoikovPolicy(p), world

    for parameter, values, builder in (
        ("gamma", np.geomspace(0.01, 3.0, 9), build_gamma),
        ("sigma", np.linspace(0.5, 5.0, 9), build_sigma),
        ("arrival_rate", np.geomspace(20.0, 600.0, 9), build_intensity),
    ):
        frame = sensitivity_sweep(
            list(values), builder, parameter, n_runs=args.runs_sweep, n_steps=200
        )
        sweeps[parameter] = frame
        print(f"\n  {parameter}:")
        print(
            "  "
            + frame[
                [
                    parameter,
                    "mean_pnl",
                    "std_pnl",
                    "sharpe",
                    "mean_abs_inventory",
                    "mean_trades",
                    "mean_spread_captured",
                ]
            ]
            .to_string(index=False, float_format=lambda x: f"{x:8.3f}")
            .replace("\n", "\n  ")
        )
        results[f"sweep_{parameter}"] = frame.to_dict(orient="records")

    _banner("7. Figures")
    figures = {
        "reservation_price": plotting.plot_reservation_price(PAPER),
        "inventory_paths": plotting.plot_inventory_paths(reference_comparison),
        "pnl_distribution": plotting.plot_pnl_distribution(reference_comparison),
        "risk_return": plotting.plot_risk_return(reference_comparison),
        "sensitivity": plotting.plot_sensitivity(sweeps),
        "fill_intensity": plotting.plot_fill_intensity_fit(fill_fit, market),
        "volatility_signature": plotting.plot_volatility_signature(signatures),
        "book_risk_return": plotting.plot_risk_return(book_comparison),
        "book_inventory_paths": plotting.plot_inventory_paths(book_comparison),
        "markout": plotting.plot_markout(book_comparison.runs),
        "session": plotting.plot_quotes_and_inventory(
            book_comparison.runs[next(iter(book_comparison.runs))][0]
        ),
    }
    for path in plotting.save_all(figures, args.out):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (args.results / "reference_comparison.md").write_text(
        _to_markdown(reference_comparison.metrics[display])
    )
    (args.results / "book_comparison.md").write_text(_to_markdown(book_comparison.metrics[display]))
    (args.results / "adverse_selection.md").write_text(_to_markdown(adverse_frame))
    print(f"  wrote {(args.results / 'results.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
