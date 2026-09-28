"""How long an estimation window does mean-variance need to beat 1/N?

Two experiments, both offline from the committed snapshots:

1. **Simulation** (:mod:`portopt.estimation_error`): calibrate to this universe, draw
   IID normal samples of each window length, and score each estimated portfolio by its
   *true* Sharpe ratio. Two calibrations of the true means, fixed before looking at the
   output: the full-sample means themselves, and "every asset has the same Sharpe ratio".
2. **Real data, paired**: the walk-forward long-only max-Sharpe with 24- to 120-month
   windows, all scored on the *same* out-of-sample months (those available to the longest
   window), against 1/N on those months, with the paired Sharpe test.

Run with::

    python scripts/estimation_window.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from portopt import backtest as bt  # noqa: E402
from portopt import metrics as mx  # noqa: E402
from portopt import plotting  # noqa: E402
from portopt import strategies as st  # noqa: E402
from portopt.data import (  # noqa: E402
    ETF_UNIVERSE,
    drop_partial_last_month,
    load_prices,
    load_risk_free,
    to_returns,
)
from portopt.estimation_error import WindowExperiment, run_window_experiment  # noqa: E402

PERIODS = 12.0
SIM_WINDOWS = [24, 36, 60, 120, 240, 480, 960, 1920, 3840]
REAL_WINDOWS = [24, 36, 60, 90, 120]
#: An illustrative true Sharpe edge over 1/N, for the "how long to detect it" column.
DETECT_EDGE = 0.10


def _to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(v: Any) -> str:
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *body]) + "\n"


def _simulate(excess: pd.DataFrame, n_reps: int, seed: int) -> dict[str, WindowExperiment]:
    sigma = excess.cov(ddof=1)
    vol = excess.std(ddof=1)
    mean_sharpe = float((excess.mean() / vol).mean())
    calibrations = {
        "True means = full-sample means": excess.mean(),
        "True means = equal Sharpe for every asset": mean_sharpe * vol,
    }
    experiments = {}
    for label, mu in calibrations.items():
        started = time.perf_counter()
        experiments[label] = run_window_experiment(mu, sigma, SIM_WINDOWS, n_reps, seed)
        print(f"  {label}: {time.perf_counter() - started:.0f}s")
    return experiments


def _real_paired(returns: pd.DataFrame, rf: pd.Series, cost_bps: float) -> pd.DataFrame:
    longest = max(REAL_WINDOWS)
    first_scored = returns.index[longest]
    # 1/N ignores the window; run it from the shortest so its initial purchase from cash
    # falls before the scored months, as it does for every max-Sharpe run but the longest.
    equal = bt.walk_forward(
        returns, bt.equal_weight_builder, "1/N", lookback=min(REAL_WINDOWS), cost_bps=cost_bps
    )
    equal_returns = equal.returns.loc[first_scored:]
    rows = []
    for window in REAL_WINDOWS:
        run = bt.walk_forward(
            returns,
            st.make_max_sharpe(PERIODS, st.sample_estimator),
            f"max-Sharpe {window}m",
            lookback=window,
            cost_bps=cost_bps,
            periods_per_year=PERIODS,
        )
        scored = run.returns.loc[first_scored:]
        test = mx.sharpe_difference_test(scored, equal_returns, rf, PERIODS)
        # Standard error of the annualised Sharpe difference implied by the paired test,
        # and how many months at that per-month noise level a true edge of DETECT_EDGE
        # would need before it reached p = 0.05 about half the time (SE ~ 1/sqrt(months)).
        standard_error = test.difference / test.z if test.z else float("nan")
        months_needed = test.n_periods * (1.96 * standard_error / DETECT_EDGE) ** 2
        rows.append(
            {
                "window_months": window,
                "assets_per_obs": returns.shape[1] / window,
                "mv_sharpe": test.sharpe_a,
                "equal_weight_sharpe": test.sharpe_b,
                "difference": test.difference,
                "se_difference": standard_error,
                "p_value": test.p_value,
                "months_to_detect_0.10": months_needed,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs["first"] = f"{equal_returns.index[0]:%Y-%m}"
    frame.attrs["last"] = f"{equal_returns.index[-1]:%Y-%m}"
    frame.attrs["n_months"] = len(equal_returns)
    return frame


def main() -> int:
    """Run both experiments and write results/ and figures/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200, help="draws per window")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    pd.set_option("display.width", 200)

    history = load_prices(ETF_UNIVERSE)
    returns = to_returns(drop_partial_last_month(history.prices), "monthly")
    rf = load_risk_free().reindex(returns.index)
    excess = returns.sub(rf, axis=0)
    print(
        f"Calibration sample: {returns.index[0]:%Y-%m} to {returns.index[-1]:%Y-%m}, "
        f"{len(returns)} months, {returns.shape[1]} assets"
    )

    print(f"\n1. Simulation, {args.reps} draws per window")
    experiments = _simulate(excess, args.reps, args.seed)
    sim_rows = []
    output: dict[str, Any] = {"simulation": {}, "n_reps": args.reps, "seed": args.seed}
    for label, experiment in experiments.items():
        crossovers = {
            name: experiment.crossover(name) for name in experiment.table["portfolio"].unique()
        }
        print(
            f"\n  {label}: 1/N true Sharpe {experiment.equal_weight_sharpe:.3f}, "
            f"true tangency {experiment.tangency_sharpe:.3f}"
        )
        print("  " + experiment.table.round(3).to_string(index=False).replace("\n", "\n  "))
        print(f"  Shortest window whose mean beats 1/N: {crossovers}")
        sim_rows.append(experiment.table.assign(calibration=label))
        output["simulation"][label] = {
            "equal_weight_sharpe": experiment.equal_weight_sharpe,
            "tangency_sharpe": experiment.tangency_sharpe,
            "crossover_window": crossovers,
            "table": experiment.table.to_dict(orient="records"),
        }

    print("\n2. Real data, paired on a common out-of-sample period")
    paired = _real_paired(returns, rf, args.cost_bps)
    print(
        f"  {paired.attrs['first']} to {paired.attrs['last']} "
        f"({paired.attrs['n_months']} months), long-only max-Sharpe (sample) vs 1/N, "
        f"{args.cost_bps:.0f}bp costs"
    )
    print("  " + paired.round(3).to_string(index=False).replace("\n", "\n  "))
    output["real_paired"] = {
        "first_month": paired.attrs["first"],
        "last_month": paired.attrs["last"],
        "n_months": paired.attrs["n_months"],
        "table": paired.to_dict(orient="records"),
    }

    args.results.mkdir(parents=True, exist_ok=True)
    sim_table = pd.concat(sim_rows)[
        [
            "calibration",
            "window",
            "portfolio",
            "mean_sharpe",
            "p10_sharpe",
            "p90_sharpe",
            "share_beating_1N",
        ]
    ]
    header = "\n".join(
        f"- {label}: 1/N true Sharpe {e.equal_weight_sharpe:.3f}, "
        f"true tangency Sharpe {e.tangency_sharpe:.3f}"
        for label, e in experiments.items()
    )
    (args.results / "estimation_window.md").write_text(
        "## Simulation\n\n"
        f"{args.reps} IID normal draws per window, seed {args.seed}. Calibrated to "
        f"{returns.index[0]:%Y-%m} to {returns.index[-1]:%Y-%m} monthly excess returns.\n\n"
        f"{header}\n\n" + _to_markdown(sim_table) + "\n## Real data, paired\n\n"
        f"{paired.attrs['first']} to {paired.attrs['last']} ({paired.attrs['n_months']} "
        f"months), long-only max-Sharpe (sample) vs 1/N, {args.cost_bps:.0f}bp costs. "
        "months_to_detect_0.10: months of data at this noise level before a true 0.10 Sharpe "
        "edge would reach p = 0.05 about half the time. "
        "p_value: paired Jobson-Korkie/Memmel test.\n\n" + _to_markdown(paired)
    )
    (args.results / "estimation_window.json").write_text(json.dumps(output, indent=2, default=str))
    figure = plotting.plot_estimation_window(experiments)
    for path in plotting.save_all({"estimation_window": figure}, args.out):
        print(f"  wrote {path}")
    print(f"  wrote {args.results / 'estimation_window.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
