"""Command line: ``ffmodel analyze | test-portfolios | factors``."""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import data, report
from .asset_pricing import fama_macbeth, grs_test
from .assets import build_portfolio, download_returns, load_returns_csv
from .attribution import attribute_returns
from .factor_stats import factor_summary, spanning_regressions
from .models import MODELS, get_model
from .regression import compare_models, fit_many, rolling_regression, summarize

PORTFOLIO = "Portfolio"
COV_LABELS = {"hac": "Newey-West (HAC)", "robust": "White (HC1)", "ols": "classical OLS"}
SORT_NAMES = {
    "ME": "size (ME)",
    "BM": "book-to-market",
    "BEME": "book-to-market",
    "OP": "profitability (OP)",
    "INV": "investment (INV)",
    "PRIOR": "prior return",
}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ffmodel", description="Fama-French factor models.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--freq", choices=["monthly", "daily"], default="monthly")
    common.add_argument("--start", help="first date, e.g. 1990-01")
    common.add_argument("--end", help="last date, e.g. 2024-12")
    common.add_argument("--refresh", action="store_true", help="re-download French data even if the cache is fresh")
    common.add_argument("--out", help="report directory (default: reports/<command>-<timestamp>)")
    common.add_argument("--no-report", action="store_true", help="print results only; write no files")

    a = sub.add_parser("analyze", parents=[common], help="factor regressions for stocks, funds or portfolios")
    source = a.add_mutually_exclusive_group(required=True)
    source.add_argument("--tickers", nargs="+", help="Yahoo Finance tickers")
    source.add_argument("--csv", help="CSV with a date column followed by one column per asset")
    a.add_argument("--prices", action="store_true", help="the CSV holds prices rather than returns")
    a.add_argument("--percent", action="store_true", help="the CSV returns are in percent")
    a.add_argument("--excess", action="store_true", help="returns are already net of the risk-free rate")
    a.add_argument("--weights", help="also analyze a portfolio, e.g. AAPL=0.6,MSFT=0.4 (rebalanced every period)")
    a.add_argument("--model", default="ff5", choices=list(MODELS))
    a.add_argument("--compare", action="store_true", help="also fit every model on a common sample")
    a.add_argument("--cov", default="hac", choices=list(COV_LABELS), help="standard errors (default: hac)")
    a.add_argument("--lags", type=int, help="Newey-West lags (default: 4*(T/100)^(2/9))")
    a.add_argument("--rolling", type=int, metavar="WINDOW", help="also estimate over a rolling window (periods)")
    a.set_defaults(func=cmd_analyze)

    t = sub.add_parser("test-portfolios", parents=[common], help="GRS and Fama-MacBeth tests on French portfolios")
    t.add_argument("--dataset", default="25_Portfolios_5x5", help="French library file name without _CSV.zip")
    t.add_argument("--weighting", default="value", choices=["value", "equal"])
    t.add_argument("--model", default="ff3", choices=list(MODELS))
    t.add_argument("--compare", action="store_true", help="test every model on a common sample")
    t.add_argument("--beta-window", type=int, help="rolling beta window for Fama-MacBeth (default: full sample)")
    t.add_argument("--fm-lags", type=int, default=0, help="Newey-West lags for Fama-MacBeth errors (default: 0)")
    t.set_defaults(func=cmd_test_portfolios)

    f = sub.add_parser("factors", parents=[common], help="summary statistics and spanning tests for the factors")
    f.add_argument("--model", default="ff6", choices=list(MODELS))
    f.set_defaults(func=cmd_factors)
    return parser


def _open_report(args, title: str) -> Optional[report.Report]:
    if args.no_report:
        return None
    return report.Report(args.out or f"reports/{args.command}-{datetime.now():%Y%m%d-%H%M%S}", title)


def _show(rep: Optional[report.Report], title: str, df: pd.DataFrame, name: str, level: int = 3,
          decimals: Optional[int] = None) -> None:
    print(f"\n{title}\n{report.format_table(df, decimals).to_string()}")
    if rep:
        rep.heading(title, level)
        rep.table(df, name, decimals)


def _note(rep: Optional[report.Report], text: str) -> None:
    print(text)
    if rep:
        rep.text(text)


def _parse_weights(text: str) -> Dict[str, float]:
    weights = {}
    for part in text.split(","):
        ticker, _, weight = part.partition("=")
        if not weight:
            raise ValueError(f"bad weight {part!r}; expected TICKER=WEIGHT")
        weights[ticker.strip()] = float(weight)
    return weights


def cmd_analyze(args) -> int:
    if args.tickers:
        returns = download_returns(args.tickers, args.start, args.end, args.freq)
        source = "Yahoo Finance (split- and dividend-adjusted closes)"
    else:
        returns = load_returns_csv(args.csv, prices=args.prices, percent=args.percent, frequency=args.freq)
        returns = returns.loc[args.start:args.end]
        source = args.csv
    if args.weights:
        by_upper = {str(c).upper(): c for c in returns.columns}
        weights = {}
        for holding, weight in _parse_weights(args.weights).items():
            if holding.upper() not in by_upper:
                raise ValueError(f"portfolio holding {holding!r} is not among the assets")
            weights[by_upper[holding.upper()]] = weight
        returns[PORTFOLIO] = build_portfolio(returns, weights)

    models = list(MODELS) if args.compare else [args.model]
    factor_sets = {m: data.load_factors(m, args.freq, args.start, args.end, refresh=args.refresh) for m in models}
    factors = factor_sets[args.model]
    spec = get_model(args.model)
    fit_kw = dict(excess=args.excess, cov=args.cov, lags=args.lags)
    results = fit_many(returns, factors, model=args.model, **fit_kw)
    if not results:
        raise ValueError("none of the assets has enough data overlapping the factors")

    rep = _open_report(args, f"{spec.label} analysis")
    _note(rep, f"Factors: Kenneth R. French Data Library ({args.freq}). Asset returns: {source}. "
               f"Standard errors: {COV_LABELS[args.cov]}. Alphas and returns are annualized.")
    _show(rep, f"{spec.label} regressions", summarize(results), "summary", level=2)

    for name, res in results.items():
        print("\n" + "=" * 100 + "\n" + res.summary())
        if rep:
            start, end = res.period
            rep.heading(str(name), 2)
            rep.text(f"{start:%b %Y} to {end:%b %Y}, {res.nobs} observations. Alpha {res.alpha_annual:+.2%} a year "
                     f"(t = {res.tvalues['alpha']:.2f}), R² {res.rsquared:.3f}, adjusted R² {res.rsquared_adj:.3f}, "
                     f"residual volatility {res.resid_vol_annual:.2%}, information ratio {res.information_ratio:.2f}.")
            rep.table(res.table(), f"{name}-coefficients", decimals=4)
            rep.figure(report.plot_loadings(res), f"{name}-loadings", f"{name} factor loadings")

        attribution = attribute_returns(res, factors)
        _show(rep, f"{name}: return attribution", attribution.summary, f"{name}-attribution")
        if rep:
            rep.figure(report.plot_attribution(attribution, str(name)), f"{name}-attribution-chart",
                       f"{name} return attribution")
            rep.figure(report.plot_cumulative_fit(attribution, str(name)), f"{name}-cumulative",
                       f"{name} actual vs factor-explained return", data=attribution.contributions)

        if args.compare:
            table, _ = compare_models(returns[name], factor_sets, **fit_kw)
            table.index = [get_model(m).label for m in table.index]
            _show(rep, f"{name}: model comparison (common sample)", table, f"{name}-models")

        if args.rolling:
            try:
                rolled = rolling_regression(returns[name], factors, args.model, args.rolling, excess=args.excess)
            except ValueError as exc:
                print(f"(rolling estimates skipped for {name}: {exc})")
                continue
            shown = rolled.rename(columns={"alpha": "alpha (ann.)"})
            shown["alpha (ann.)"] *= res.periods_per_year
            stats = pd.DataFrame({"latest": shown.iloc[-1], "mean": shown.mean(), "min": shown.min(), "max": shown.max()}).T
            _show(rep, f"{name}: rolling {args.rolling}-period estimates", stats, f"{name}-rolling-summary")
            if rep:
                rep.figure(report.plot_rolling(rolled, res.factors, str(name), args.model, args.rolling,
                                               res.periods_per_year),
                           f"{name}-rolling", f"{name} rolling estimates", data=rolled)

    if rep:
        print(f"\nReport: {rep.save()}")
    return 0


def _sort_layout(dataset: str, n: int):
    """Grid shape and sort names for datasets like 25_Portfolios_ME_OP_5x5, else None."""
    match = re.search(r"Portfolios_?(.*?)_?(\d+)x(\d+)", dataset, flags=re.IGNORECASE)
    if not match:
        return None
    rows, cols = int(match.group(2)), int(match.group(3))
    if rows * cols != n:
        return None
    keys = [k for k in match.group(1).upper().split("_") if k] or ["ME", "BM"]
    if len(keys) != 2:
        return None
    return (rows, cols), SORT_NAMES.get(keys[0], keys[0]), SORT_NAMES.get(keys[1], keys[1])


def cmd_test_portfolios(args) -> int:
    raw = data.load_portfolios(args.dataset, args.freq, args.weighting, args.start, args.end, refresh=args.refresh)
    models = list(MODELS) if args.compare else [args.model]
    factor_sets = {m: data.load_factors(m, args.freq, args.start, args.end, refresh=args.refresh) for m in models}
    index = raw.dropna().index
    for factors in factor_sets.values():
        index = index.intersection(factors.index)
    if len(index) == 0:
        raise ValueError("the portfolios and factors have no complete periods in common")
    raw = raw.loc[index]

    rep = _open_report(args, f"Asset pricing tests: {args.dataset}")
    _note(rep, f"{raw.shape[1]} {args.weighting}-weighted portfolios ({args.dataset}), {args.freq}, "
               f"{index[0]:%b %Y} to {index[-1]:%b %Y} ({len(index)} periods). Returns are in excess of the "
               f"T-bill rate; alphas and premia are annualized.")

    grs_rows, alpha_cols, predicted, pvalues, fm_results, alphas = {}, {}, {}, {}, {}, {}
    realized = None
    ppy = 12
    for m, factors in factor_sets.items():
        spec = get_model(m)
        factors = factors.loc[index]
        excess = raw.sub(factors["RF"], axis=0)
        cols = list(spec.factors)
        grs = grs_test(excess, factors[cols])
        ppy = grs.periods_per_year
        grs_rows[spec.label] = grs.summary()
        alpha_cols[f"{spec.label} alpha (ann.)"] = grs.alphas * ppy
        alpha_cols[f"{spec.label} t(alpha)"] = grs.alpha_tstats
        alphas[spec.label] = grs.alphas
        realized = excess.mean() * ppy
        predicted[spec.label] = realized - grs.alphas * ppy
        pvalues[spec.label] = grs.pvalue
        fm_results[spec.label] = fama_macbeth(excess, factors[cols], beta_window=args.beta_window, lags=args.fm_lags)

    _show(rep, "GRS test: are all alphas jointly zero?", pd.DataFrame(grs_rows).T, "grs", level=2)
    _note(rep, "A|alpha| is the average absolute alpha; A|alpha| / A|r| compares it with the spread in average "
               "returns the model has to explain (Fama and French 2015). SR(alpha) is the maximum Sharpe ratio "
               "attainable from the alphas.")
    _show(rep, "Time-series alphas by portfolio", pd.DataFrame(alpha_cols), "alphas", level=2)

    if rep:
        rep.heading("Fama-MacBeth risk premia", 2)
    for label, fm in fm_results.items():
        _show(rep, f"Fama-MacBeth: {label}", fm.table(), f"fama-macbeth-{label}")
        errors = "Newey-West" if args.fm_lags != 0 else "classic Fama-MacBeth"
        betas = f"rolling {args.beta_window}-period betas" if args.beta_window else "full-sample betas"
        _note(rep, f"{fm.nobs} cross-sections, {betas}, {errors} t-stats"
                   f"{' with Shanken (1992) correction' if fm.tstats_shanken is not None else ''}. "
                   f"Cross-sectional R² {fm.cs_rsquared:.3f}. For traded factors the premium should be close to "
                   f"the factor's mean.")

    if rep:
        rep.heading("Charts", 2)
        rep.figure(report.plot_pricing(realized, predicted, pvalues, f"{args.dataset}: model-implied vs. realized returns"),
                   "pricing", "Model-implied vs realized average excess returns",
                   data=pd.DataFrame({"realized (ann.)": realized, **{f"{k} implied (ann.)": v for k, v in predicted.items()}}))
        layout = _sort_layout(args.dataset, raw.shape[1])
        if layout:
            shape, row_name, col_name = layout
            rep.figure(report.plot_alpha_heatmaps(alphas, shape, row_name, col_name, ppy, f"{args.dataset}: alphas"),
                       "alpha-heatmap", "Alpha heatmap")
        print(f"\nReport: {rep.save()}")
    return 0


def cmd_factors(args) -> int:
    factors = data.load_factors(args.model, args.freq, args.start, args.end, refresh=args.refresh)
    spec = get_model(args.model)
    rep = _open_report(args, f"{spec.label} factors")
    _note(rep, f"{args.freq.capitalize()} factor returns, {factors.index[0]:%b %Y} to {factors.index[-1]:%b %Y} "
               f"({len(factors)} periods). Source: Kenneth R. French Data Library.")
    _show(rep, "Summary statistics", factor_summary(factors), "summary", level=2)
    _show(rep, "Correlations", factors.drop(columns="RF").corr(), "correlations", level=2)
    if len(spec.factors) > 1:
        _show(rep, "Spanning regressions: each factor on the others", spanning_regressions(factors), "spanning", level=2)
        _note(rep, "A factor with an insignificant alpha here is explained by the others and adds little to the model.")
    if rep:
        rep.figure(report.plot_factor_growth(factors, f"{spec.label} factors: cumulative returns"), "growth",
                   "Cumulative factor returns", data=factors)
        print(f"\nReport: {rep.save()}")
    return 0
