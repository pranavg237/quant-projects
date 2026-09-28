"""HTML tearsheet: one self-contained file per strategy.

The page leads with the honest verdict (does this beat its benchmark out of sample, and
is the Sharpe distinguishable from luck), then the five standard panels: equity curve,
drawdown, rolling Sharpe, monthly returns and factor exposures. Every table the charts
summarise is also rendered as text, so nothing is gated behind colour or hover, and the
caveats that apply to the run are printed rather than buried in a commit message.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantbt import metrics
from quantbt.report import charts
from quantbt.report.charts import Series
from quantbt.report.palette import FONT_STACK, css_variables
from quantbt.validation.bootstrap import excess_returns

ROLLING_WINDOW = 252


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float, places: int = 1) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value * 100:.{places}f}%"


def _num(value: float, places: int = 2) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:,.{places}f}"


MIN_ROLLING_VOL = 0.01  # annualised; below this the strategy is holding cash, not risk


def rolling_sharpe(
    returns: pd.Series,
    window: int = ROLLING_WINDOW,
    ppy: int = 252,
    rf: pd.Series | float | None = None,
    min_ann_vol: float = MIN_ROLLING_VOL,
) -> pd.Series:
    """Annualised Sharpe of excess returns over a trailing window.

    Two details matter for a long/flat strategy. The ratio is computed on returns in
    excess of ``rf``, and windows in which the strategy took essentially no risk are left
    blank rather than plotted. A year spent entirely in Treasury bills has a tiny, almost
    constant return and therefore an enormous raw Sharpe; plotting it produces a spike
    that says "this was cash", not "this was brilliant", and it flattens every other year
    on the axis.
    """
    excess = returns if rf is None else excess_returns(returns, rf)
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    ann_vol = std * np.sqrt(ppy)
    out: pd.Series = (mean / std * np.sqrt(ppy)).replace([np.inf, -np.inf], np.nan)
    return out.where(ann_vol >= min_ann_vol)


@dataclass(frozen=True)
class TearsheetInputs:
    """Everything the page renders. Assembled from a research run or a single backtest."""

    name: str
    description: str
    returns: pd.Series
    benchmark_returns: pd.Series
    benchmark_name: str
    rf: pd.Series
    weights: pd.DataFrame | None = None
    summary: pd.Series | None = None
    folds: pd.DataFrame | None = None
    factors: pd.DataFrame | None = None
    overfit: pd.Series | None = None
    caveats: tuple[str, ...] = ()
    spec: dict[str, Any] | None = None
    costs: str = ""


def _verdict(inputs: TearsheetInputs, summary: pd.Series) -> tuple[str, str, str]:
    """A one-line judgement: (status class, headline, detail)."""
    sharpe = float(summary.get("sharpe", np.nan))
    bench = metrics.summary(inputs.benchmark_returns, rf=inputs.rf)
    bench_sharpe = float(bench["sharpe"])
    psr = float(inputs.overfit["psr"]) if inputs.overfit is not None else np.nan
    beats = sharpe > bench_sharpe
    convincing = np.isfinite(psr) and psr >= 0.95

    if not np.isfinite(sharpe) or sharpe <= 0:
        return (
            "critical",
            "Does not work.",
            f"Out-of-sample Sharpe {_num(sharpe)} is at or below zero. "
            f"Buy-and-hold {inputs.benchmark_name} returned {_num(bench_sharpe)}.",
        )
    if beats and convincing:
        return (
            "good",
            f"Beats {inputs.benchmark_name} out of sample.",
            f"Sharpe {_num(sharpe)} against {_num(bench_sharpe)} for buy-and-hold, and the "
            f"probabilistic Sharpe ratio of {_num(psr)} clears 0.95. "
            "Check the factor panel before calling any of it alpha.",
        )
    if beats:
        return (
            "warning",
            f"Edges out {inputs.benchmark_name}, but not convincingly.",
            f"Sharpe {_num(sharpe)} against {_num(bench_sharpe)}, with a probabilistic Sharpe "
            f"ratio of {_num(psr)} (below the 0.95 threshold). The sample cannot separate this "
            "from luck.",
        )
    return (
        "warning",
        f"Loses to buy-and-hold {inputs.benchmark_name}.",
        f"Sharpe {_num(sharpe)} against {_num(bench_sharpe)}. It is a real return series, "
        "just not a better one than doing nothing.",
    )


def _stat_tiles(summary: pd.Series, inputs: TearsheetInputs) -> str:
    bench = metrics.summary(inputs.benchmark_returns, rf=inputs.rf)
    tiles = [
        ("CAGR", _pct(float(summary["cagr"])), f"benchmark {_pct(float(bench['cagr']))}"),
        ("Sharpe", _num(float(summary["sharpe"])), f"benchmark {_num(float(bench['sharpe']))}"),
        (
            "Max drawdown",
            _pct(float(summary["max_drawdown"])),
            f"benchmark {_pct(float(bench['max_drawdown']))}",
        ),
        (
            "Annualised vol",
            _pct(float(summary["ann_vol"])),
            f"benchmark {_pct(float(bench['ann_vol']))}",
        ),
        ("Sortino", _num(float(summary["sortino"])), "downside deviation"),
        ("Calmar", _num(float(summary["calmar"])), "CAGR over max drawdown"),
    ]
    if "turnover" in summary.index:
        tiles.append(("Turnover", f"{_num(float(summary['turnover']), 1)}x", "one-way, per year"))
    if "exposure" in summary.index:
        tiles.append(("Exposure", _pct(float(summary["exposure"]), 0), "average gross"))
    cells = "".join(
        f'<div class="tile"><div class="tile-label">{_esc(label)}</div>'
        f'<div class="tile-value">{_esc(value)}</div>'
        f'<div class="tile-note">{_esc(note)}</div></div>'
        for label, value, note in tiles
    )
    return f'<div class="tiles">{cells}</div>'


def _summary_table(summary: pd.Series, inputs: TearsheetInputs) -> str:
    bench = metrics.summary(inputs.benchmark_returns, rf=inputs.rf)
    percent_keys = {
        "total_return",
        "cagr",
        "ann_vol",
        "max_drawdown",
        "best_period",
        "worst_period",
        "positive_periods",
        "exposure",
        "time_in_market",
        "hit_rate",
    }
    rows = []
    for key in [
        "start",
        "end",
        "periods",
        "total_return",
        "cagr",
        "ann_vol",
        "sharpe",
        "sortino",
        "calmar",
        "max_drawdown",
        "max_dd_duration_days",
        "max_dd_recovered",
        "best_period",
        "worst_period",
        "positive_periods",
        "turnover",
        "exposure",
        "time_in_market",
        "trades",
        "hit_rate",
        "profit_factor",
    ]:
        if key not in summary.index:
            continue
        value = summary[key]
        other = bench.get(key, None)

        def render(v: Any, k: str = key) -> str:
            if v is None or isinstance(v, str):
                return _esc(v if v is not None else "")
            if isinstance(v, pd.Timestamp):
                return _esc(v.date())
            if isinstance(v, (bool, np.bool_)):
                return "yes" if v else "no"
            if isinstance(v, (int, np.integer)):
                return f"{int(v):,}"
            if k in percent_keys:
                return _pct(float(v))
            return _num(float(v))

        label = key.replace("_", " ")
        rows.append(
            f"<tr><th>{_esc(label)}</th><td>{render(value)}</td>"
            f"<td class='muted'>{render(other) if other is not None else ''}</td></tr>"
        )
    return (
        "<table class='data'><thead><tr><th>Metric</th><th>Strategy</th>"
        f"<th>{_esc(inputs.benchmark_name)} buy &amp; hold</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _frame_table(frame: pd.DataFrame, *, float_places: int = 3, index: bool = False) -> str:
    if frame is None or frame.empty:
        return "<p class='empty'>No rows.</p>"
    display = frame.copy()
    if index:
        display = display.reset_index()
    head = "".join(f"<th>{_esc(str(c).replace('_', ' '))}</th>" for c in display.columns)
    body = []
    for _, row in display.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(
                    f"<td>{'' if not np.isfinite(value) else f'{value:,.{float_places}f}'}</td>"
                )
            else:
                cells.append(f"<td>{_esc(value)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<table class='data'><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


def _overfit_block(overfit: pd.Series | None) -> str:
    if overfit is None:
        return ""
    flags = str(overfit.get("flags", "none"))
    status = "good" if flags == "none" else "warning"
    items = [
        ("Sharpe", _num(float(overfit["sharpe"]))),
        ("Standard error (annual)", _num(float(overfit["sharpe_se_annual"]))),
        (
            "Bootstrap 95% interval",
            f"{_num(float(overfit['bootstrap_ci_lo']))} to {_num(float(overfit['bootstrap_ci_hi']))}",
        ),
        ("P(Sharpe <= 0) by bootstrap", _num(float(overfit["bootstrap_p_sharpe_le_0"]))),
        ("Probabilistic Sharpe ratio", _num(float(overfit["psr"]))),
    ]
    if np.isfinite(float(overfit.get("dsr", np.nan))):
        items.append(("Deflated Sharpe ratio", _num(float(overfit["dsr"]))))
    if np.isfinite(float(overfit.get("pbo", np.nan))):
        items.append(("Probability of backtest overfitting", _num(float(overfit["pbo"]))))
    rows = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in items)
    flag_html = (
        "<p class='flag good'>No overfitting flags raised.</p>"
        if flags == "none"
        else "".join(f"<p class='flag warning'>{_esc(f.strip())}</p>" for f in flags.split(";"))
    )
    return (
        f"<div class='panel-body {status}'>"
        f"<table class='data compact'><tbody>{rows}</tbody></table>{flag_html}</div>"
    )


HOVER_SCRIPT = """
(function () {
  const fmt = (v, kind) => {
    if (v === null || v === undefined) return 'n/a';
    if (kind === 'percent') return (v * 100).toFixed(2) + '%';
    if (kind === 'growth') return v.toFixed(2) + 'x';
    return v.toFixed(2);
  };
  document.querySelectorAll('svg[data-kind="timeseries"]').forEach((svg) => {
    const data = JSON.parse(svg.getAttribute('data-points'));
    const cross = svg.querySelector('.crosshair');
    const dots = cross.querySelectorAll('circle');
    const line = cross.querySelector('line');
    const tip = document.createElement('div');
    tip.className = 'tooltip';
    document.body.appendChild(tip);
    const n = data.dates.length;
    const move = (event) => {
      const box = svg.getBoundingClientRect();
      const scale = box.width / svg.viewBox.baseVal.width;
      const localX = (event.clientX - box.left) / scale;
      const t = (localX - data.x0) / (data.x1 - data.x0);
      const i = Math.max(0, Math.min(n - 1, Math.round(t * (n - 1))));
      const x = data.x0 + (data.x1 - data.x0) * (i / Math.max(n - 1, 1));
      line.setAttribute('x1', x);
      line.setAttribute('x2', x);
      let rows = '';
      data.series.forEach((s, k) => {
        const y = s.y[i];
        if (y === null) { dots[k].style.display = 'none'; return; }
        dots[k].style.display = '';
        dots[k].setAttribute('cx', x);
        dots[k].setAttribute('cy', y);
        rows += '<div><span class="swatch" style="background: var(--' + s.color +
                ')"></span>' + s.name + ' <b>' + fmt(s.v[i], data.format) + '</b></div>';
      });
      cross.style.display = '';
      tip.innerHTML = '<div class="tip-date">' + data.dates[i] + '</div>' + rows;
      tip.style.display = 'block';
      tip.style.left = Math.min(event.clientX + 14, window.innerWidth - tip.offsetWidth - 12) + 'px';
      tip.style.top = (event.clientY + 14) + 'px';
    };
    const hide = () => { cross.style.display = 'none'; tip.style.display = 'none'; };
    svg.addEventListener('mousemove', move);
    svg.addEventListener('mouseleave', hide);
  });
  const cellTip = document.createElement('div');
  cellTip.className = 'tooltip';
  document.body.appendChild(cellTip);
  document.querySelectorAll('svg[data-kind="heatmap"] .cell').forEach((cell) => {
    cell.addEventListener('mousemove', (event) => {
      cellTip.innerHTML = '<div class="tip-date">' + cell.dataset.label + '</div><b>' +
                          cell.dataset.value + '</b>';
      cellTip.style.display = 'block';
      cellTip.style.left = (event.clientX + 14) + 'px';
      cellTip.style.top = (event.clientY + 14) + 'px';
    });
    cell.addEventListener('mouseleave', () => { cellTip.style.display = 'none'; });
  });
  document.querySelectorAll('[data-toggle]').forEach((button) => {
    button.addEventListener('click', () => {
      const target = document.getElementById(button.dataset.toggle);
      const open = target.hasAttribute('hidden');
      if (open) { target.removeAttribute('hidden'); } else { target.setAttribute('hidden', ''); }
      button.textContent = (open ? 'Hide' : 'Show') + ' ' + button.dataset.name;
    });
  });
})();
"""


def _styles() -> str:
    return f"""{css_variables()}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 32px 16px 64px; background: var(--page); color: var(--text);
      font-family: {FONT_STACK}; font-size: 15px; line-height: 1.55;
    }}
    .wrap {{ max-width: 1120px; margin: 0 auto; }}
    header h1 {{ font-size: 28px; margin: 0 0 4px; }}
    header p.sub {{ color: var(--text_secondary); margin: 0 0 8px; }}
    header p.meta {{ color: var(--muted); font-size: 13px; margin: 0; }}
    .verdict {{
      margin: 20px 0 28px; padding: 16px 18px; border-radius: 10px;
      background: var(--surface); border: 1px solid var(--border); border-left-width: 4px;
    }}
    .verdict.good {{ border-left-color: var(--good); }}
    .verdict.warning {{ border-left-color: #eda100; }}
    .verdict.critical {{ border-left-color: var(--critical); }}
    .verdict h2 {{ font-size: 18px; margin: 0 0 4px; }}
    .verdict p {{ margin: 0; color: var(--text_secondary); }}
    .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
    .tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }}
    .tile-label {{ color: var(--muted); font-size: 12px; }}
    .tile-value {{ font-size: 26px; font-weight: 600; margin: 2px 0; }}
    .tile-note {{ color: var(--muted); font-size: 12px; }}
    section {{ margin-top: 28px; background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 18px 20px 20px; }}
    section h2 {{ font-size: 17px; margin: 0 0 2px; }}
    section p.note {{ color: var(--text_secondary); font-size: 13px; margin: 0 0 12px; }}
    .chart {{ width: 100%; height: auto; display: block; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin: 0 0 10px; font-size: 13px; color: var(--text_secondary); }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
    .swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
    table.data {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }}
    table.data th, table.data td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); font-variant-numeric: tabular-nums; }}
    table.data th:first-child, table.data thead th {{ text-align: left; }}
    table.data thead th {{ color: var(--muted); font-weight: 500; }}
    table.data td.muted, table.data .muted {{ color: var(--muted); }}
    table.compact th {{ font-weight: 500; color: var(--text_secondary); }}
    .toggle {{ background: none; border: 1px solid var(--border); color: var(--text_secondary); border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer; font-family: inherit; }}
    .toggle:hover {{ background: var(--page); }}
    .panel-head {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }}
    .flag {{ margin: 10px 0 0; font-size: 13px; padding-left: 12px; border-left: 3px solid var(--muted); }}
    .flag.good {{ border-left-color: var(--good); color: var(--text_secondary); }}
    .flag.warning {{ border-left-color: #eda100; color: var(--text_secondary); }}
    ul.caveats {{ margin: 8px 0 0; padding-left: 20px; color: var(--text_secondary); font-size: 13px; }}
    .tooltip {{
      position: fixed; display: none; z-index: 20; pointer-events: none;
      background: var(--surface); color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 10px; font-size: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.18);
    }}
    .tooltip .tip-date {{ color: var(--muted); margin-bottom: 3px; }}
    .tooltip .swatch {{ margin-right: 6px; }}
    .empty {{ color: var(--muted); font-size: 13px; }}
    footer {{ margin-top: 32px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 640px) {{ body {{ padding: 20px 16px 48px; }} .tile-value {{ font-size: 22px; }} }}"""


def build_tearsheet(inputs: TearsheetInputs) -> str:
    """Render the complete HTML document for one strategy."""
    returns = inputs.returns.dropna()
    bench = inputs.benchmark_returns.reindex(returns.index).fillna(0.0)
    summary = inputs.summary
    if summary is None:
        summary = metrics.summary(returns, rf=inputs.rf, weights=inputs.weights, benchmark=bench)

    growth = metrics.equity_curve(returns)
    bench_growth = metrics.equity_curve(bench)
    equity_series = [
        Series(inputs.name, growth, "series_1"),
        Series(f"{inputs.benchmark_name} buy & hold", bench_growth, "series_2"),
    ]
    dd_series = [Series("Drawdown", metrics.drawdown_series(growth), "series_1", fill=True)]
    rs_series = [
        Series(inputs.name, rolling_sharpe(returns, rf=inputs.rf), "series_1"),
        Series(
            f"{inputs.benchmark_name} buy & hold",
            rolling_sharpe(bench, rf=inputs.rf),
            "series_2",
        ),
    ]

    status, headline, detail = _verdict(inputs, summary)
    parts: list[str] = []

    parts.append(
        f"<header><h1>{_esc(inputs.name)}</h1>"
        f"<p class='sub'>{_esc(inputs.description)}</p>"
        f"<p class='meta'>Out of sample {_esc(returns.index[0].date())} to "
        f"{_esc(returns.index[-1].date())} &middot; {len(returns):,} trading days"
        f"{' &middot; ' + _esc(inputs.costs) if inputs.costs else ''}</p></header>"
    )
    parts.append(
        f"<div class='verdict {status}'><h2>{_esc(headline)}</h2><p>{_esc(detail)}</p></div>"
    )
    parts.append(_stat_tiles(summary, inputs))

    parts.append(
        "<section><h2>Equity curve</h2>"
        "<p class='note'>Growth of 1 unit of capital, net of commission and slippage. "
        "Cash earns the Treasury-bill rate when the strategy is out of the market.</p>"
        + charts.legend(equity_series)
        + charts.time_series_chart(
            equity_series, label="Equity curve", value_format="growth", baseline=1.0
        )
        + "</section>"
    )
    parts.append(
        "<section><h2>Drawdown</h2>"
        "<p class='note'>Loss from the running peak. The deepest point and the longest time "
        "spent under water matter more than the average.</p>"
        + charts.time_series_chart(
            dd_series, label="Drawdown", value_format="percent", baseline=0.0
        )
        + "</section>"
    )
    parts.append(
        "<section><h2>Rolling Sharpe ratio</h2>"
        f"<p class='note'>Excess of the Treasury-bill rate, annualised over a trailing "
        f"{ROLLING_WINDOW}-day window. A line hovering near zero is what an edge that is "
        "not there looks like. Gaps are windows where the strategy held cash and took "
        "too little risk for the ratio to mean anything.</p>"
        + charts.legend(rs_series)
        + charts.time_series_chart(
            rs_series, label="Rolling Sharpe", value_format="ratio", baseline=0.0
        )
        + "</section>"
    )
    parts.append(
        "<section><h2>Monthly returns</h2>"
        "<p class='note'>Compounded return per calendar month, blue for gains and red for "
        "losses; the right-hand column is the calendar year. Values are percentages.</p>"
        + charts.monthly_heatmap(returns, label="Monthly returns")
        + "</section>"
    )

    if inputs.factors is not None and not inputs.factors.empty:
        model = "ff5" if "ff5" in inputs.factors.index else str(inputs.factors.index[0])
        row: dict[str, float] = {
            str(k): float(v)
            for k, v in inputs.factors.loc[model].items()
            if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool)
        }
        alpha, alpha_t = row["alpha_annual"], row["alpha_t"]
        reading = (
            "Alpha is not statistically distinguishable from zero."
            if abs(alpha_t) < 2
            else "Alpha survives the factor controls at the conventional two-standard-error bar."
        )
        parts.append(
            "<section><h2>Factor exposures</h2>"
            f"<p class='note'>Returns regressed on the Fama-French five factors with "
            f"Newey-West standard errors. Annualised alpha {_pct(alpha)} "
            f"(t = {_num(alpha_t)}), R-squared {_num(row['r2'])}. {reading}</p>"
            + charts.factor_bars(inputs.factors, label="Factor betas", model=model)
            + "<div class='panel-head'><p class='note'>Every model side by side.</p>"
            "<button class='toggle' data-toggle='factor-table' data-name='factor table'>"
            "Show factor table</button></div>"
            f"<div id='factor-table' hidden>{_frame_table(inputs.factors, index=True)}</div>"
            "</section>"
        )

    if inputs.overfit is not None:
        parts.append(
            "<section><h2>Is this luck?</h2>"
            "<p class='note'>The probabilistic Sharpe ratio asks whether the observed Sharpe is "
            "distinguishable from zero given the sample length and the shape of the return "
            "distribution. The deflated version and the probability of backtest overfitting "
            "additionally account for how many parameter settings were tried.</p>"
            + _overfit_block(inputs.overfit)
            + "</section>"
        )

    if inputs.folds is not None and not inputs.folds.empty:
        parts.append(
            "<section><div class='panel-head'><div><h2>Walk-forward folds</h2>"
            "<p class='note'>Parameters were chosen on each training window and scored on the "
            "following window only. Parameters that jump around from fold to fold are a sign "
            "the objective surface is noise.</p></div>"
            "<button class='toggle' data-toggle='fold-table' data-name='folds'>Show folds</button>"
            f"</div><div id='fold-table' hidden>{_frame_table(inputs.folds)}</div></section>"
        )

    parts.append(
        "<section><div class='panel-head'><div><h2>All metrics</h2>"
        "<p class='note'>Every number the charts summarise, as text.</p></div>"
        "<button class='toggle' data-toggle='metric-table' data-name='metrics'>Show metrics</button>"
        f"</div><div id='metric-table' hidden>{_summary_table(summary, inputs)}</div></section>"
    )

    caveats = list(inputs.caveats)
    if inputs.spec:
        grid = inputs.spec.get("grid")
        if grid:
            caveats.append(f"Parameter grid searched on each training window: {json.dumps(grid)}.")
    if caveats:
        parts.append(
            "<section><h2>Caveats</h2><ul class='caveats'>"
            + "".join(f"<li>{_esc(c)}</li>" for c in caveats)
            + "</ul></section>"
        )

    parts.append(
        "<footer>Generated by quantbt. Signals are computed at the close and filled at the "
        "next open with slippage and commission; parameters are chosen on training windows "
        "only. Past performance in a backtest is not evidence of future returns.</footer>"
    )

    return (
        "<!doctype html>\n<html lang='en'>\n<head>\n<meta charset='utf-8'>\n"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>\n"
        f"<title>{_esc(inputs.name)} tearsheet</title>\n<style>\n{_styles()}\n</style>\n"
        "</head>\n<body>\n<div class='wrap'>\n"
        + "\n".join(parts)
        + f"\n</div>\n<script>{HOVER_SCRIPT}</script>\n</body>\n</html>\n"
    )


def write_tearsheet(inputs: TearsheetInputs, path: str | Path) -> Path:
    """Render a self-contained HTML tearsheet to ``path`` and return the path written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_tearsheet(inputs), encoding="utf-8")
    return out
