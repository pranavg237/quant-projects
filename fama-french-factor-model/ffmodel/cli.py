"""Command line: ``ffmodel analyze | test-portfolios | factors``."""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pandas as pd

from . import data, report, snapshot
from .asset_pricing import fama_macbeth, grs_test
from .assets import build_portfolio, download_returns, load_returns_csv
from .attribution import attribute_returns
from .factor_stats import factor_summary, spanning_regressions
from .models import MODELS, get_model
from .regression import (
    compare_models,
    compare_standard_errors,
    fit_many,
    holm_adjust,
    rolling_regression,
    stability_test,
    summarize,
)

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


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``ffmodel`` command. Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """The argument parser for the ``analyze``, ``test-portfolios`` and ``factors`` commands."""
    parser = argparse.ArgumentParser(prog="ffmodel", description="Fama-French factor models.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--freq", choices=["monthly", "daily"], default="monthly")
    common.add_argument("--start", help="first date, e.g. 1990-01")
    common.add_argument("--end", help="last date, e.g. 2024-12")
    common.add_argument("--refresh", action="store_true", help="re-download French data even if the cache is fresh")
    common.add_argument("--out", help="report directory (default: reports/<command>-<timestamp>)")
    common.add_argument("--no-report", action="store_true", help="print results only; write no files")
    common.add_argument("--data-dir", help="read French library files from this snapshot directory (no download); "
                                           "see the snapshot command")

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

    s = sub.add_parser("snapshot", help="download French files and Yahoo returns into a directory for offline runs")
    s.add_argument("--tickers", nargs="+", default=[], help="Yahoo Finance tickers (omit for French files only)")
    s.add_argument("--portfolios", nargs="+", default=[], metavar="DATASET",
                   help="also save these French test-portfolio files, e.g. 25_Portfolios_5x5")
    s.add_argument("--freq", choices=["monthly", "daily"], default="monthly")
    s.add_argument("--start", help="first date, e.g. 1990-01")
    s.add_argument("--end", help="last date, e.g. 2024-12")
    s.add_argument("--out", required=True, help="snapshot directory")
    s.set_defaults(func=cmd_snapshot)

    f = sub.add_parser("factors", parents=[common], help="summary statistics and spanning tests for the factors")
    f.add_argument("--model", default="ff6", choices=list(MODELS))
    f.set_defaults(func=cmd_factors)
    return parser


def _open_report(args: argparse.Namespace, title: str) -> Optional[report.Report]:
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


def cmd_analyze(args: argparse.Namespace) -> int:
    """Factor regressions, attribution and rolling estimates for tickers or a CSV."""
    if args.tickers:
        returns = download_returns(args.tickers, args.start, args.end, args.freq)
        source = "Yahoo Finance (split- and dividend-adjusted closes)"
    else:
        snapshot.verify_returns_file(args.csv)
        returns = load_returns_csv(args.csv, prices=args.prices, percent=args.percent, frequency=args.freq)
        returns = returns.loc[args.start:args.end]
        source = args.csv
        manifest = snapshot.read_manifest(Path(args.csv).parent)
        if manifest and "returns" in manifest:
            source = (f"{manifest['returns']['source']}, saved snapshot `{Path(args.csv).parent.name}` "
                      f"(downloaded {manifest['downloaded_utc'][:10]})")
    if args.weights:
        by_upper = {str(c).upper(): c for c in returns.columns}
        weights = {}
        for holding, weight in _parse_weights(args.weights).items():
            if holding.upper() not in by_upper:
                raise ValueError(f"portfolio holding {holding!r} is not among the assets")
            weights[by_upper[holding.upper()]] = weight
        returns[PORTFOLIO] = build_portfolio(returns, weights)

    models = list(MODELS) if args.compare else [args.model]
    factor_sets = {m: _load_factors(args, m) for m in models}
    factors = factor_sets[args.model]
    spec = get_model(args.model)
    fit_kw = dict(excess=args.excess, cov=args.cov, lags=args.lags)
    results = fit_many(returns, factors, model=args.model, **fit_kw)
    if not results:
        raise ValueError("none of the assets has enough data overlapping the factors")

    rep = _open_report(args, f"{spec.label} analysis")
    first = next(iter(results.values()))
    lag_note = ""
    if first.hac_lags is not None:
        lag_note = (f" with {first.hac_lags} lags" if args.lags is not None else
                    f" with {first.hac_lags} lags (rule floor(4(T/100)^(2/9)) at T = {first.nobs})")
    _note(rep, f"Factors: Kenneth R. French Data Library ({args.freq}{_factor_source(args)}). Asset returns: {source}. "
               f"Standard errors: {COV_LABELS[args.cov]}{lag_note}; p-values use the normal distribution. "
               f"Alphas and returns are annualized.")
    _show(rep, f"{spec.label} regressions", summarize(results), "summary", level=2)
    _significance_section(rep, results, args)

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
        if args.cov == "hac" and args.lags is None:
            _show(rep, f"{name}: t-statistics under different standard errors", compare_standard_errors(res),
                  f"{name}-standard-errors")

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
            # Short windows use HC3 whatever --cov says: Newey-West is far too narrow with ~36 observations
            # (see scripts/window_se_simulation.py). --cov ols keeps classical errors.
            window_se = "ols" if args.cov == "ols" else "hc3"
            errors = "HC3, MacKinnon-White" if window_se == "hc3" else COV_LABELS[window_se]
            try:
                rolled = rolling_regression(returns[name], factors, args.model, args.rolling, excess=args.excess,
                                            se=window_se)
            except ValueError as exc:
                print(f"(rolling estimates skipped for {name}: {exc})")
                continue
            shown = rolled[[c for c in rolled.columns if not c.startswith("se(")]].rename(columns={"alpha": "alpha (ann.)"})
            shown["alpha (ann.)"] *= res.periods_per_year
            stats = pd.DataFrame({"latest": shown.iloc[-1], "mean": shown.mean(), "min": shown.min(), "max": shown.max()}).T
            _show(rep, f"{name}: rolling {args.rolling}-period estimates", stats, f"{name}-rolling-summary")
            _note(rep, f"{len(rolled)} overlapping windows of {args.rolling} periods; the first ends "
                       f"{rolled.index[0]:%b %Y} and the last {rolled.index[-1]:%b %Y}. Shaded bands in the chart are "
                       f"estimate ± 1.96 standard errors ({errors}) for each window on its own: they are pointwise, "
                       f"not joint, and neighbouring windows share {args.rolling - 1} of {args.rolling} observations.")
            if rep:
                rep.figure(report.plot_rolling(rolled, res.factors, str(name), args.model, args.rolling,
                                               res.periods_per_year),
                           f"{name}-rolling", f"{name} rolling estimates", data=rolled)
            try:
                stability = stability_test(returns[name], factors, args.model, args.rolling, excess=args.excess,
                                           se=window_se)
            except ValueError as exc:
                print(f"(stability test skipped for {name}: {exc})")
                continue
            info = stability.attrs
            shown = stability.astype(object)
            for col in ("lowest block", "highest block"):
                shown.loc["alpha", col] = f"{stability.loc['alpha', col] * res.periods_per_year:.1%}"
            shown = shown.rename(index={"alpha": "alpha (ann.)"})
            _show(rep, f"{name}: did the exposures change? ({info['blocks']} separate {args.rolling}-period blocks)",
                  shown, f"{name}-stability")
            _note(rep, f"Blocks run from {info['start']:%b %Y} to {info['end']:%b %Y} and do not overlap. Each row tests "
                       f"that one coefficient is the same in all {info['blocks']} blocks (Wald chi-squared, "
                       f"{info['blocks'] - 1} degrees of freedom, {errors} within each block). A small p-value "
                       f"says the exposure moved by more than estimation noise; a large one says the rolling "
                       f"chart's wiggles are consistent with a constant exposure.")

    if rep:
        print(f"\nReport: {rep.save()}")
    return 0


def _load_factors(args: argparse.Namespace, model: str) -> pd.DataFrame:
    return data.load_factors(model, args.freq, args.start, args.end, refresh=args.refresh, data_dir=args.data_dir)


def _factor_source(args: argparse.Namespace) -> str:
    if not args.data_dir:
        return ""
    manifest = snapshot.read_manifest(args.data_dir)
    when = f", downloaded {manifest['downloaded_utc'][:10]}" if manifest else ""
    return f"; saved snapshot `{Path(args.data_dir).name}`{when}"


def _french_source(args: argparse.Namespace) -> str:
    """'Kenneth R. French Data Library', plus the snapshot and its download date when --data-dir is given."""
    snap = _factor_source(args)
    return "Kenneth R. French Data Library" + (f" ({snap[2:]})" if snap else "")


def _significance_section(rep: Optional[report.Report], results: Dict, args: argparse.Namespace) -> None:
    """Alpha significance across assets, with a Holm correction for testing several alphas at once."""
    if len(results) < 2:
        return
    table = pd.DataFrame({
        "alpha (ann.)": {n: r.alpha_annual for n, r in results.items()},
        "t(alpha)": {n: float(r.tvalues["alpha"]) for n, r in results.items()},
        "p-value": {n: float(r.pvalues["alpha"]) for n, r in results.items()},
    })
    table["Holm p-value"] = holm_adjust(table["p-value"])
    _show(rep, "Alphas: one test per asset vs. the whole family", table, "alpha-tests", level=2)
    _note(rep, f"{len(table)} alphas are tested here. The Holm column adjusts each p-value so that the chance of "
               f"any false rejection across all {len(table)} stays at the nominal level (valid even though the assets "
               f"are correlated). Standard errors: {COV_LABELS[args.cov]}.")


def cmd_snapshot(args: argparse.Namespace) -> int:
    """Save factor files and Yahoo returns so a later analysis can run offline."""
    manifest = snapshot.save_snapshot(args.out, args.tickers, args.start, args.end, args.freq, args.portfolios)
    saved = f"{len(manifest['french_library'])} French library files"
    r = manifest.get("returns")
    if r:
        saved += (f" and {r['periods']} periods of returns for {', '.join(r['tickers'])} "
                  f"({r['first']} to {r['last']})")
    print(f"Saved {saved} to {args.out}")
    return 0


def _sort_layout(dataset: str, n: int) -> Optional[Tuple[Tuple[int, int], str, str]]:
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


def cmd_test_portfolios(args: argparse.Namespace) -> int:
    """GRS and Fama-MacBeth tests on a set of French test portfolios."""
    raw = data.load_portfolios(args.dataset, args.freq, args.weighting, args.start, args.end, refresh=args.refresh,
                               data_dir=args.data_dir)
    models = list(MODELS) if args.compare else [args.model]
    factor_sets = {m: _load_factors(args, m) for m in models}
    index = raw.dropna().index
    for factors in factor_sets.values():
        index = index.intersection(factors.index)
    if len(index) == 0:
        raise ValueError("the portfolios and factors have no complete periods in common")
    raw = raw.loc[index]

    rep = _open_report(args, f"Asset pricing tests: {args.dataset}")
    _note(rep, f"{raw.shape[1]} {args.weighting}-weighted portfolios ({args.dataset}), {args.freq}, "
               f"{index[0]:%b %Y} to {index[-1]:%b %Y} ({len(index)} periods). Returns are in excess of the "
               f"T-bill rate; alphas and premia are annualized. Source: {_french_source(args)}.")

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


def cmd_factors(args: argparse.Namespace) -> int:
    """Summary statistics, correlations and spanning regressions for the factors themselves."""
    factors = _load_factors(args, args.model)
    spec = get_model(args.model)
    rep = _open_report(args, f"{spec.label} factors")
    _note(rep, f"{args.freq.capitalize()} factor returns, {factors.index[0]:%b %Y} to {factors.index[-1]:%b %Y} "
               f"({len(factors)} periods). Source: {_french_source(args)}.")
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
