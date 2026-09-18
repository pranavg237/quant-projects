"""Tearsheet tests: the HTML is well formed, the numbers in it are the computed ones,
and the verdict does not flatter a strategy that lost."""

from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest

from quantbt import metrics
from quantbt.report import TearsheetInputs, build_tearsheet, charts, write_tearsheet
from quantbt.report.charts import Series, factor_bars, monthly_heatmap, time_series_chart
from quantbt.report.palette import DARK, LIGHT, css_variables
from quantbt.report.tearsheet import rolling_sharpe


def _returns(mean: float, sd: float, n: int = 1500, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, sd, n), index=pd.bdate_range("2012-01-02", periods=n))


def _inputs(
    strategy_mean: float = 0.0006, bench_mean: float = 0.0003, **kwargs: object
) -> TearsheetInputs:
    r = _returns(strategy_mean, 0.008, seed=1)
    b = _returns(bench_mean, 0.010, seed=2)
    factors = pd.DataFrame(
        {
            "alpha_annual": [0.02, 0.018],
            "alpha_t": [1.1, 0.9],
            "beta_Mkt-RF": [0.45, 0.47],
            "t_Mkt-RF": [9.1, 9.0],
            "beta_SMB": [np.nan, -0.12],
            "t_SMB": [np.nan, -2.4],
            "r2": [0.40, 0.41],
            "r2_adj": [0.40, 0.41],
            "nobs": [1500, 1500],
        },
        index=pd.Index(["capm", "ff5"], name="model"),
    )
    overfit = pd.Series(
        {
            "sharpe": 1.1,
            "sharpe_se_annual": 0.25,
            "psr": 0.97,
            "dsr": 0.8,
            "n_trials": 12,
            "min_track_record_periods": 900.0,
            "bootstrap_ci_lo": 0.6,
            "bootstrap_ci_hi": 1.6,
            "bootstrap_p_sharpe_le_0": 0.01,
            "pbo": 0.42,
            "flags": "none",
        }
    )
    folds = pd.DataFrame(
        {
            "train_start": ["2012-01-02"],
            "test_start": ["2017-01-02"],
            "short": [20],
            "test_obj": [0.8],
        }
    )
    base: dict[str, object] = {
        "name": "demo",
        "description": "A demonstration strategy.",
        "returns": r,
        "benchmark_returns": b,
        "benchmark_name": "SPY",
        "rf": pd.Series(0.00002, index=r.index),
        "weights": pd.DataFrame({"X": np.ones(len(r))}, index=r.index),
        "factors": factors,
        "overfit": overfit,
        "folds": folds,
        "caveats": ("Universe chosen today.",),
        "spec": {"grid": {"short": [10, 20]}},
        "costs": "5 bps slippage",
    }
    base.update(kwargs)
    return TearsheetInputs(**base)  # type: ignore[arg-type]


def test_rolling_sharpe_matches_metrics_on_the_final_window() -> None:
    r = _returns(0.0005, 0.01, 800, seed=3)
    rs = rolling_sharpe(r, window=252)
    assert rs.iloc[:251].isna().all()
    expected = metrics.sharpe(r.iloc[-252:])
    assert rs.iloc[-1] == pytest.approx(expected, rel=1e-10)
    constant = pd.Series(0.0, index=r.index)
    assert rolling_sharpe(constant).isna().all()  # zero volatility, not an infinite Sharpe


def test_tearsheet_is_well_formed_html_with_valid_svg() -> None:
    html = build_tearsheet(_inputs())
    assert html.startswith("<!doctype html>")
    assert html.count("<svg") == 5  # equity, drawdown, rolling Sharpe, heatmap, factor bars
    for svg in re.findall(r"<svg.*?</svg>", html, flags=re.S):
        ElementTree.fromstring(svg)  # raises on malformed markup
    # every chart colour is a token, so light and dark mode both work
    assert 'fill="#' not in html and 'stroke="#' not in html
    assert "prefers-color-scheme: dark" in html
    assert 'data-theme="dark"' in html


def test_tearsheet_numbers_match_the_metrics() -> None:
    inputs = _inputs()
    summary = metrics.summary(inputs.returns, rf=inputs.rf, weights=inputs.weights)
    html = build_tearsheet(inputs)
    assert f"{summary['cagr'] * 100:.1f}%" in html
    assert f"{summary['sharpe']:,.2f}" in html
    assert f"{summary['max_drawdown'] * 100:.1f}%" in html
    assert "1,500 trading days" in html
    assert "5 bps slippage" in html


def test_verdict_is_honest_about_losing() -> None:
    winner = build_tearsheet(_inputs(strategy_mean=0.0012, bench_mean=0.0002))
    assert "Beats SPY out of sample" in winner
    loser = build_tearsheet(_inputs(strategy_mean=0.0004, bench_mean=0.0012))
    assert "Loses to buy-and-hold SPY" in loser
    assert "verdict warning" in loser
    dud = build_tearsheet(_inputs(strategy_mean=-0.0008, bench_mean=0.0004))
    assert "Does not work." in dud
    assert "verdict critical" in dud
    weak = _inputs(strategy_mean=0.0009, bench_mean=0.0002)
    assert weak.overfit is not None
    weak_overfit = weak.overfit.copy()
    weak_overfit["psr"] = 0.6
    html = build_tearsheet(_inputs(strategy_mean=0.0009, bench_mean=0.0002, overfit=weak_overfit))
    assert "not convincingly" in html


def test_tearsheet_reports_factor_alpha_significance() -> None:
    inputs = _inputs()
    html = build_tearsheet(inputs)
    assert "not statistically distinguishable from zero" in html  # t = 0.9
    assert inputs.factors is not None
    significant = inputs.factors.copy()
    significant.loc["ff5", "alpha_t"] = 3.4
    html2 = build_tearsheet(_inputs(factors=significant))
    assert "survives the factor controls" in html2


def test_tearsheet_degrades_without_optional_panels() -> None:
    html = build_tearsheet(
        _inputs(factors=None, overfit=None, folds=None, weights=None, spec=None, caveats=())
    )
    assert "<svg" in html
    assert "Factor exposures" not in html
    assert "Is this luck?" not in html
    assert "Walk-forward folds" not in html
    assert "Caveats" not in html
    for svg in re.findall(r"<svg.*?</svg>", html, flags=re.S):
        ElementTree.fromstring(svg)


def test_time_series_chart_encodes_hover_data() -> None:
    r = _returns(0.0005, 0.01, 300, seed=4)
    svg = time_series_chart(
        [Series("a", metrics.equity_curve(r)), Series("b", metrics.equity_curve(-r), "series_2")],
        label="test",
        value_format="growth",
        baseline=1.0,
    )
    root = ElementTree.fromstring(svg)
    points = json.loads(root.attrib["data-points"])
    assert len(points["dates"]) == 300
    assert [s["name"] for s in points["series"]] == ["a", "b"]
    assert len(points["series"][0]["y"]) == 300
    assert points["format"] == "growth"
    assert root.attrib["data-kind"] == "timeseries"
    assert svg.count('stroke-width="2"') >= 2  # 2px lines per the mark spec


def test_charts_handle_empty_and_degenerate_input() -> None:
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    assert "No data" in time_series_chart([Series("x", empty)], label="e")
    assert "No data" in monthly_heatmap(empty, label="e")
    flat = pd.Series(1.0, index=pd.bdate_range("2020-01-01", periods=30))
    svg = time_series_chart([Series("flat", flat)], label="flat")
    ElementTree.fromstring(svg)
    gappy = pd.Series([1.0, np.nan, 1.2], index=pd.bdate_range("2020-01-01", periods=3))
    ElementTree.fromstring(time_series_chart([Series("g", gappy)], label="g"))
    no_betas = pd.DataFrame({"alpha_annual": [0.1], "alpha_t": [1.0], "r2": [0.2]}, index=["ff5"])
    assert "No factor loadings" in factor_bars(no_betas, label="x")


def test_monthly_heatmap_marks_positive_and_negative_months() -> None:
    idx = pd.bdate_range("2021-01-01", periods=400)
    r = pd.Series(0.0, index=idx)
    r.loc["2021-01-05"] = 0.05  # a clearly positive January
    r.loc["2021-03-05"] = -0.05  # a clearly negative March
    svg = monthly_heatmap(r, label="m")
    root = ElementTree.fromstring(svg)
    cells = {
        c.attrib["data-label"]: c
        for c in root.iter("{http://www.w3.org/2000/svg}rect")
        if "data-label" in c.attrib
    }
    assert cells["2021-01"].attrib["fill"] == "var(--positive)"
    assert cells["2021-03"].attrib["fill"] == "var(--negative)"
    assert cells["2021-01"].attrib["data-value"] == "5.0%"


def test_legend_present_for_two_series_absent_for_one() -> None:
    s = pd.Series([1.0, 1.1], index=pd.bdate_range("2020-01-01", periods=2))
    assert charts.legend([Series("only", s)]) == ""
    two = charts.legend([Series("a", s), Series("b", s, "series_2")])
    assert "a" in two and "b" in two and "swatch" in two


def test_palette_defines_the_same_roles_in_both_modes() -> None:
    assert set(LIGHT) == set(DARK)
    css = css_variables()
    for role in ("surface", "text", "series_1", "series_2", "positive", "negative"):
        assert f"--{role}:" in css
    assert css.count("color-scheme: dark;") == 2  # media query scope and explicit toggle


def test_write_tearsheet_creates_the_file(tmp_path: Path) -> None:
    out = write_tearsheet(_inputs(), tmp_path / "nested" / "demo.html")
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>") and text.rstrip().endswith("</html>")


def test_rolling_sharpe_blanks_no_risk_windows() -> None:
    """A year spent in cash has a tiny, near-constant return and an enormous raw Sharpe.

    Plotting that spike says "this was cash", not "this was brilliant", and it flattens
    every other year on the axis, so those windows are left blank.
    """
    n = 800
    idx = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(5)
    rf = pd.Series(8e-5, index=idx)  # about 2% a year
    cash = rng.normal(8e-5, 2e-7, 400)  # in cash: a tiny, nearly constant return
    risky = rng.normal(0.0004, 0.01, n - 400)  # risk taken only in the second half
    r = pd.Series(np.concatenate([cash, risky]), index=idx)
    raw = rolling_sharpe(r, window=252, rf=None, min_ann_vol=0.0)
    assert raw.iloc[399] > 10  # the artefact the chart used to show
    clean = rolling_sharpe(r, window=252, rf=rf)
    assert np.isnan(clean.iloc[399])  # cash window is blank
    assert np.isfinite(clean.iloc[-1])  # a genuinely risky window still reports


def test_converging_end_labels_are_dropped_not_stacked() -> None:
    idx = pd.bdate_range("2020-01-01", periods=200)
    a = pd.Series(np.linspace(1.0, 2.0, 200), index=idx)
    b = pd.Series(np.linspace(1.5, 2.0001, 200), index=idx)  # ends within a hair of a
    together = time_series_chart(
        [Series("a", a), Series("b", b, "series_2")], label="converging", value_format="growth"
    )
    apart = time_series_chart(
        [Series("a", a), Series("b", b * 0.5, "series_2")], label="separate", value_format="growth"
    )
    root_together = ElementTree.fromstring(together)
    root_apart = ElementTree.fromstring(apart)
    ns = "{http://www.w3.org/2000/svg}"
    # axis labels are present in both; only the series end-labels differ
    assert len(root_together.findall(f"{ns}text")) < len(root_apart.findall(f"{ns}text"))


def test_decimation_preserves_extremes_and_shrinks_the_file() -> None:
    """Thinning a 20-year daily series must not smooth away the bottom of a drawdown."""
    from quantbt.report.charts import MAX_PLOT_POINTS, decimate_positions

    n = 5000
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(11)
    values = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    values.iloc[3333] = -99.0  # a single catastrophic bar
    values.iloc[4444] = 99.0

    keep = decimate_positions([values.to_numpy()], n)
    assert len(keep) <= MAX_PLOT_POINTS
    assert 3333 in keep and 4444 in keep  # both extremes survive
    assert keep[0] == 0 and keep[-1] == n - 1
    assert keep == sorted(keep)
    assert decimate_positions([values.to_numpy()], 100) == list(range(100))  # no thinning needed

    big = time_series_chart([Series("x", values)], label="big", value_format="ratio")
    ElementTree.fromstring(big)
    assert "-99" in big or "99" in big  # the spike is drawn, not averaged away
    # the tooltip payload shrinks with the path
    points = json.loads(ElementTree.fromstring(big).attrib["data-points"])
    assert len(points["dates"]) == len(keep)
    assert len(big) < 200_000


def test_decimation_keeps_series_aligned() -> None:
    """All series on one chart share the kept positions, so the crosshair stays truthful."""
    n = 4000
    idx = pd.bdate_range("2008-01-01", periods=n)
    rng = np.random.default_rng(12)
    a = pd.Series(np.cumsum(rng.normal(0, 0.01, n)), index=idx)
    b = pd.Series(np.cumsum(rng.normal(0, 0.02, n)), index=idx)
    svg = time_series_chart(
        [Series("a", a), Series("b", b, "series_2")], label="two", value_format="ratio"
    )
    points = json.loads(ElementTree.fromstring(svg).attrib["data-points"])
    assert len({len(s["v"]) for s in points["series"]}) == 1
    assert len(points["series"][0]["v"]) == len(points["dates"])
    # a sampled value is a real observation on the date the tooltip will show
    k = len(points["dates"]) // 2
    date = pd.Timestamp(points["dates"][k])
    assert points["series"][0]["v"][k] == pytest.approx(float(a.loc[date]), abs=1e-6)


def test_decimation_budget_accounts_for_series_count() -> None:
    from quantbt.report.charts import MAX_PLOT_POINTS, decimate_positions

    n = 6000
    rng = np.random.default_rng(13)
    arrays = [np.cumsum(rng.normal(0, 0.01, n)) for _ in range(3)]
    assert len(decimate_positions(arrays, n)) <= MAX_PLOT_POINTS
    assert len(decimate_positions(arrays[:1], n)) <= MAX_PLOT_POINTS
