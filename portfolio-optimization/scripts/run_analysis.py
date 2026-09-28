"""Full portfolio study: estimate, optimise, walk forward, and report honestly.

Run with::

    python scripts/run_analysis.py

Prices are cached under ``data/``, so a rerun is offline and the numbers reproduce.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from portopt import backtest as bt  # noqa: E402
from portopt import covariance as cov  # noqa: E402
from portopt import metrics as mx  # noqa: E402
from portopt import plotting  # noqa: E402
from portopt import riskparity as rp  # noqa: E402
from portopt import strategies as st  # noqa: E402
from portopt.blacklitterman import View, black_litterman  # noqa: E402
from portopt.data import (  # noqa: E402
    ETF_UNIVERSE,
    MARKET_WEIGHTS,
    PriceHistory,
    drop_partial_last_month,
    load_benchmark,
    load_prices,
    load_risk_free,
    to_returns,
)
from portopt.hrp import (  # noqa: E402
    correlation_distance,
    hierarchical_risk_parity,
    quasi_diagonal_order,
)
from portopt.optimizers import efficient_frontier, max_sharpe, min_variance  # noqa: E402
from portopt.types import Constraints  # noqa: E402

PERIODS = 12.0


def _to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as Markdown without pulling in ``tabulate``."""

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
    print("=" * 100)
    print(title)
    print("=" * 100)


def _show(frame: pd.DataFrame, fmt: str = "{:8.3f}") -> None:
    """Print an indented, fixed-width table."""

    def formatter(value: float) -> str:
        return fmt.format(value)

    print("  " + frame.to_string(index=False, float_format=formatter).replace("\n", "\n  "))


def _load_inputs(
    refresh: bool,
) -> tuple[PriceHistory, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load prices, monthly returns, the risk-free series and the SPY benchmark.

    Returns everything on one monthly index, with the partial final month removed.
    """
    history = load_prices(ETF_UNIVERSE, force_refresh=refresh)
    prices = drop_partial_last_month(history.prices)
    returns = to_returns(prices, "monthly")
    print(f"  {history}")
    if len(prices) < len(history.prices):
        print(f"  Dropped the partial final month; last full month is {prices.index[-1]:%Y-%m}")
    print(f"  {len(returns)} monthly observations")

    # Cash earns the 13-week T-bill rate, known at the start of each month. Every Sharpe
    # ratio below is on returns in excess of it.
    risk_free = load_risk_free(force_refresh=refresh)
    missing_rf = returns.index.difference(risk_free.index)
    if len(missing_rf):
        raise RuntimeError(f"risk-free series does not cover {list(missing_rf[:3])}")
    rf = risk_free.reindex(returns.index)
    print(f"  Risk-free: 13-week T-bill (^IRX), mean {rf.mean() * PERIODS:.2%}/yr over the sample")

    # Buy-and-hold SPY over the same months is the market benchmark.
    spy_prices = load_benchmark("SPY", force_refresh=refresh).loc[: prices.index[-1]]
    spy_returns = to_returns(spy_prices.to_frame(), "monthly")["SPY"].reindex(returns.index)
    if spy_returns.isna().any():
        raise RuntimeError("SPY benchmark does not cover the sample")
    return history, prices, returns, rf, spy_returns


def _p_values_against(
    runs: dict[str, bt.BacktestResult], reference: str, rf: pd.Series
) -> dict[str, float]:
    """Paired test of equal Sharpe ratio between every run and ``reference``."""
    base = runs[reference].returns
    return {
        name: mx.sharpe_difference_test(run.returns, base, rf, PERIODS).p_value
        for name, run in runs.items()
        if name != reference
    }


def main() -> int:
    """Run the whole study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback", type=int, default=60, help="estimation window in months")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--borrow-bps", type=float, default=50.0, help="annual stock-loan fee on shorts"
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "figures")
    parser.add_argument("--results", type=Path, default=REPO_ROOT / "results")
    args = parser.parse_args()
    pd.set_option("display.width", 260)
    results: dict[str, Any] = {}

    _banner("1. Universe")
    history, prices, returns, rf, spy_returns = _load_inputs(args.refresh)
    excess_assets = returns.sub(rf, axis=0)
    asset_stats = pd.DataFrame(
        {
            "return": (1.0 + returns).prod() ** (PERIODS / len(returns)) - 1.0,
            "volatility": returns.std(ddof=1) * np.sqrt(PERIODS),
            "sharpe": excess_assets.mean() / excess_assets.std(ddof=1) * np.sqrt(PERIODS),
        }
    )
    _show(asset_stats.reset_index().rename(columns={"index": "asset", "Ticker": "asset"}))
    correlation = returns.corr()
    upper = np.triu_indices(len(correlation), 1)
    mean_correlation = float(correlation.to_numpy()[upper].mean())
    print(f"\n  Mean pairwise correlation: {mean_correlation:.3f}")
    results["universe"] = {
        "start": str(history.start.date()),
        "end": str(prices.index[-1].date()),
        "risk_free": "13-week T-bill (^IRX), previous month-end yield / 12",
        "mean_risk_free_annual": float(rf.mean() * PERIODS),
        "benchmark": "SPY buy-and-hold, dividends reinvested",
        "n_assets": len(history.tickers),
        "n_months": len(returns),
        "mean_correlation": mean_correlation,
        "asset_stats": asset_stats.reset_index().to_dict(orient="records"),
    }

    _banner("2. Covariance estimation: why shrinkage")
    rows = []
    diagnostics_by_window: dict[str, cov.CovarianceDiagnostics] = {}
    for window in (24, 36, 60, 90, 120, 180, len(returns)):
        sub = returns.iloc[-window:]
        sample = cov.sample_covariance(sub, PERIODS)
        shrunk_id, delta_id = cov.ledoit_wolf(sub, "identity", PERIODS)
        shrunk_cc, delta_cc = cov.ledoit_wolf(sub, "constant_correlation", PERIODS)
        rows.append(
            {
                "n_observations": window,
                "assets_per_obs": len(returns.columns) / window,
                "delta_identity": delta_id,
                "delta_constant_correlation": delta_cc,
                "condition_sample": cov.diagnose(sample, window).condition_number,
                "condition_shrunk": cov.diagnose(shrunk_cc, window).condition_number,
                "effective_rank_sample": cov.diagnose(sample, window).effective_rank,
                "effective_rank_shrunk": cov.diagnose(shrunk_cc, window).effective_rank,
            }
        )
        if window == args.lookback:
            diagnostics_by_window = {
                "sample": cov.diagnose(sample, window),
                "Ledoit-Wolf (identity)": cov.diagnose(shrunk_id, window),
                "Ledoit-Wolf (const. corr.)": cov.diagnose(shrunk_cc, window),
                "exponentially weighted": cov.diagnose(
                    cov.exponentially_weighted_covariance(sub, 36.0, PERIODS), window
                ),
            }
    shrinkage_frame = pd.DataFrame(rows)
    _show(shrinkage_frame, "{:10.4f}")
    results["shrinkage"] = shrinkage_frame.to_dict(orient="records")

    _banner("3. In-sample portfolios (the part everyone shows)")
    full_cov, _ = cov.ledoit_wolf(returns, "constant_correlation", PERIODS)
    full_mu = (1.0 + returns.mean()) ** PERIODS - 1.0
    long_only = Constraints()
    portfolios = {
        "Min variance": min_variance(full_cov, long_only, full_mu),
        "Max Sharpe": max_sharpe(full_mu, full_cov, long_only),
        "Risk parity (ERC)": rp.equal_risk_contribution(full_cov, full_mu),
        "Inverse volatility": rp.inverse_volatility(full_cov, full_mu),
        "HRP": hierarchical_risk_parity(full_cov, full_mu),
    }
    for name, result in portfolios.items():
        print(f"  {name:22s} {result}")
    weight_table = pd.DataFrame({k: v.weights for k, v in portfolios.items()})
    print()
    print("  " + weight_table.round(4).to_string().replace("\n", "\n  "))
    results["in_sample_weights"] = weight_table.to_dict()

    contributions = {
        name: rp.risk_contributions(portfolios[name].weights, full_cov)
        for name in ("Inverse volatility", "Risk parity (ERC)", "HRP")
    }
    print(f"\n  Risk contribution shares (equal share would be {1 / len(full_cov):.3f}):")
    share_table = pd.DataFrame({k: v / v.sum() for k, v in contributions.items()})
    print("  " + share_table.round(4).to_string().replace("\n", "\n  "))

    _banner("4. Walk-forward: the part that matters")
    strategies = st.default_strategies(PERIODS, long_only)
    backtests = bt.compare_strategies(
        returns,
        strategies,
        lookback=args.lookback,
        rebalance_every=1,
        cost_bps=args.cost_bps,
        periods_per_year=PERIODS,
        borrow_bps=args.borrow_bps,
    )
    oos_index = next(iter(backtests.values())).returns.index
    benchmark = bt.buy_and_hold(spy_returns.loc[oos_index], "SPY buy-and-hold (benchmark)")
    summary = mx.summarise({**backtests, benchmark.name: benchmark}, risk_free=rf)
    p_vs_equal = _p_values_against(
        {**backtests, benchmark.name: benchmark}, "Equal weight (1/N)", rf
    )
    summary["p_vs_1N"] = summary["strategy"].map(p_vs_equal)
    display = [
        "strategy",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "sharpe_lower",
        "sharpe_upper",
        "max_drawdown",
        "calmar",
        "annual_turnover",
        "annual_cost_drag",
        "gross_sharpe",
        "p_vs_1N",
    ]
    _show(summary[display])
    print(
        f"\n  {summary['n_periods'].iloc[0]} out-of-sample months "
        f"({summary['n_periods'].iloc[0] / 12:.1f} years), "
        f"{args.lookback}-month estimation window, {args.cost_bps:.0f}bp one-way costs"
    )
    n_significant = int((summary["p_vs_1N"] < 0.05).sum())
    print(
        f"  Sample: {oos_index[0]:%Y-%m} to {oos_index[-1]:%Y-%m}. "
        f"p_vs_1N is a paired Jobson-Korkie/Memmel test of equal Sharpe against 1/N;\n"
        f"  {n_significant} of {summary['p_vs_1N'].notna().sum()} strategies differ from 1/N "
        "at the 5% level."
    )
    results["walk_forward"] = summary.to_dict(orient="records")
    results["walk_forward_sample"] = {
        "first_month": str(oos_index[0].date()),
        "last_month": str(oos_index[-1].date()),
        "n_months": len(oos_index),
        "cost_bps_one_way": args.cost_bps,
        "borrow_bps_annual": args.borrow_bps,
    }

    _banner("5. The leverage experiment: what actually breaks Markowitz")
    sweep_rows = []
    caps: list[float] = [1.0, 1.5, 2.0, 3.0, 5.0, 8.0, float("inf")]
    for cap in caps:
        if cap == 1.0:
            constraints = Constraints()
            label = "1.0 (long-only)"
        elif np.isinf(cap):
            constraints = Constraints(long_only=False, min_weight=-1e18)
            label = "uncapped"
        else:
            constraints = Constraints(long_only=False, min_weight=-1e18, max_leverage=cap)
            label = f"{cap:.1f}x"
        run = bt.walk_forward(
            returns,
            st.make_max_sharpe(PERIODS, st.sample_estimator, constraints),
            f"max-Sharpe, leverage {label}",
            lookback=args.lookback,
            cost_bps=args.cost_bps,
            periods_per_year=PERIODS,
            borrow_bps=args.borrow_bps,
        )
        metrics = mx.evaluate(run, rf)
        sweep_rows.append({"leverage_cap": cap, "label": label, **metrics.as_dict()})
    sweep = pd.DataFrame(sweep_rows)
    _show(
        sweep[
            [
                "label",
                "annual_return",
                "annual_volatility",
                "sharpe",
                "max_drawdown",
                "annual_turnover",
                "annual_borrow_drag",
                "mean_leverage",
                "max_leverage",
                "worst_period",
                "is_ruined",
            ]
        ]
    )
    results["leverage_sweep"] = (
        sweep.drop(columns=["leverage_cap"])
        .assign(leverage_cap=[str(c) for c in sweep["leverage_cap"]])
        .to_dict(orient="records")
    )

    _banner("6. Lookback sensitivity: the same optimiser, different amounts of data")
    lookback_rows = []
    for lookback in (24, 36, 60, 90, 120):
        for label, constraints in (
            ("long-only", Constraints()),
            ("long-short", Constraints(long_only=False, min_weight=-1e18)),
        ):
            run = bt.walk_forward(
                returns,
                st.make_max_sharpe(PERIODS, st.sample_estimator, constraints),
                f"{label} ({lookback}m)",
                lookback=lookback,
                cost_bps=args.cost_bps,
                periods_per_year=PERIODS,
                borrow_bps=args.borrow_bps,
            )
            metrics = mx.evaluate(run, rf.reindex(run.returns.index))
            lookback_rows.append(
                {
                    "lookback_months": lookback,
                    "constraint": label,
                    "assets_per_obs": len(returns.columns) / lookback,
                    "sharpe": metrics.sharpe,
                    "annual_return": metrics.annual_return,
                    "max_drawdown": metrics.max_drawdown,
                    "annual_turnover": metrics.annual_turnover,
                    "mean_leverage": metrics.mean_leverage,
                    "is_ruined": metrics.is_ruined,
                }
            )
    lookback_frame = pd.DataFrame(lookback_rows)
    _show(lookback_frame)
    results["lookback_sensitivity"] = lookback_frame.to_dict(orient="records")

    _banner("7. Cost sensitivity: which edges survive implementation")
    cost_rows = []
    for cost in (0.0, 5.0, 10.0, 25.0, 50.0):
        runs = bt.compare_strategies(
            returns, strategies, lookback=args.lookback, cost_bps=cost, periods_per_year=PERIODS
        )
        for name, run in runs.items():
            cost_rows.append(
                {"cost_bps": cost, "strategy": name, "sharpe": mx.evaluate(run, rf).sharpe}
            )
    cost_frame = pd.DataFrame(cost_rows).pivot(
        index="strategy", columns="cost_bps", values="sharpe"
    )
    print("  Sharpe by one-way cost (bps):")
    print("  " + cost_frame.round(3).to_string().replace("\n", "\n  "))
    results["cost_sensitivity"] = cost_frame.reset_index().to_dict(orient="records")

    _banner("8. Black-Litterman")
    market = pd.Series(MARKET_WEIGHTS).reindex(full_cov.index)
    market = market / market.sum()
    prior_only = black_litterman(full_cov, market)
    print(
        "  With no views the posterior equals the prior exactly "
        f"(max |tilt| = {prior_only.tilt.abs().max():.2e}),"
    )
    print("  and optimising it reproduces the market portfolio. That is the implementation check.")
    views = [
        View({"XLK": 1.0, "XLP": -1.0}, 0.04, 0.5),
        View({"TLT": 1.0}, 0.01, 0.4),
    ]
    posterior = black_litterman(full_cov, market, views)
    print(f"\n  Views: {'; '.join(str(v) for v in views)}")
    tilt_frame = pd.DataFrame(
        {
            "equilibrium": prior_only.prior_returns,
            "posterior": posterior.posterior_returns,
            "tilt_bps": posterior.tilt * 10_000,
        }
    )
    print("  " + tilt_frame.round(4).to_string().replace("\n", "\n  "))
    print(
        "\n  Note XLY and XLI move despite no view mentioning them: the update propagates "
        "through\n  the covariance, which is what stops Black-Litterman portfolios being "
        "concentrated."
    )
    results["black_litterman"] = {
        "max_tilt_no_views": float(prior_only.tilt.abs().max()),
        "risk_aversion": posterior.risk_aversion,
        "tilts": tilt_frame.to_dict(orient="records"),
    }

    _banner("9. Figures")
    frontier = efficient_frontier(full_mu, full_cov, 40, long_only)
    distance = correlation_distance(correlation)
    from scipy.cluster.hierarchy import linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform  # noqa: PLC0415

    link = linkage(squareform(distance.to_numpy(), checks=False), "single")
    order = [str(correlation.index[i]) for i in quasi_diagonal_order(np.asarray(link))]

    headline = {
        k: v
        for k, v in backtests.items()
        if k
        in (
            "Equal weight (1/N)",
            "Risk parity (ERC)",
            "Hierarchical Risk Parity",
            "Markowitz max-Sharpe (sample)",
        )
    }
    figures = {
        "equity_curves": plotting.plot_equity_curves(headline),
        "drawdowns": plotting.plot_drawdowns(headline),
        "risk_return": plotting.plot_risk_return_scatter(summary),
        "efficient_frontier": plotting.plot_efficient_frontier(
            frontier,
            {"Min variance": portfolios["Min variance"], "Max Sharpe": portfolios["Max Sharpe"]},
            asset_stats,
        ),
        "leverage_sweep": plotting.plot_leverage_sweep(sweep),
        "shrinkage": plotting.plot_shrinkage_intensity(shrinkage_frame),
        "correlation_clustered": plotting.plot_correlation_heatmap(correlation, order),
        "risk_contributions": plotting.plot_risk_contributions(contributions),
        "turnover_vs_sharpe": plotting.plot_turnover_vs_sharpe(summary),
        "covariance_diagnostics": plotting.plot_covariance_diagnostics(diagnostics_by_window),
        "weights_over_time": plotting.plot_weights_over_time(
            backtests["Markowitz max-Sharpe (sample)"]
        ),
    }
    for path in plotting.save_all(figures, args.out):
        print(f"  wrote {path.relative_to(REPO_ROOT)}")

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "results.json").write_text(json.dumps(results, indent=2, default=str))
    (args.results / "walk_forward.md").write_text(_to_markdown(summary[display]))
    (args.results / "leverage_sweep.md").write_text(
        _to_markdown(
            sweep[
                [
                    "label",
                    "sharpe",
                    "annual_turnover",
                    "annual_borrow_drag",
                    "max_drawdown",
                    "mean_leverage",
                    "max_leverage",
                    "is_ruined",
                ]
            ]
        )
    )
    print(f"  wrote {(args.results / 'results.json').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
