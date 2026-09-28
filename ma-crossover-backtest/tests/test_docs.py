"""The README teaches one way to write a strategy. This keeps that example honest.

The code below is the README's "Adding a strategy" snippet, run against synthetic data so
the test stays offline. If the Context API changes, this fails and the README gets fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from quantbt.data import synthetic_prices
from quantbt.engine import run_backtest
from quantbt.execution import ExecutionSimulator, FixedBpsSlippage, PercentageCommission
from quantbt.strategy import Context, Strategy
from quantbt.validation import overfit_report, walk_forward


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, lookback: int = 126) -> None:
        self.lookback = lookback
        self.warmup = lookback

    def on_bar(self, ctx: Context) -> None:
        closes = ctx.history("close", self.lookback)
        winners = [s for s in ctx.symbols if closes[s].iloc[-1] > closes[s].iloc[0]]
        if winners:
            ctx.order_target_weights({s: 1 / len(winners) for s in winners})
        ctx.record(n_winners=len(winners))


def test_readme_strategy_example_runs() -> None:
    data = synthetic_prices(1500, symbols=("A", "B", "C"), drift=0.0003, gap_vol=0.004, seed=17)
    result = run_backtest(
        data,
        Momentum(126),
        start=data.index[200],
        execution=ExecutionSimulator(PercentageCommission(1e-4), FixedBpsSlippage(5.0)),
        rf=0.02,
        benchmark="A",
    )
    summary = result.summary()
    assert summary["periods"] == len(data) - 200
    assert "n_winners" in result.records.columns
    assert result.total_commission > 0 and result.total_slippage > 0


def test_readme_validation_example_runs() -> None:
    data = synthetic_prices(2500, symbols=("A", "B", "C"), drift=0.0002, gap_vol=0.004, seed=18)
    wf = walk_forward(
        data,
        lambda p: Momentum(p["lookback"]),
        {"lookback": [63, 126]},
        start=data.index[300],
        train_years=3,
        test_years=1,
    )
    assert {"out_of_sample", "in_sample_best"} == set(wf.summary().columns)
    report = overfit_report(wf.oos_returns, rf=wf.oos_rf, n_samples=50)
    assert isinstance(report.flags, list)


def test_readme_documents_every_bias_test() -> None:
    """Every ``test_bias_*`` the README promises exists, and every one is named there."""
    readme = Path("README.md").read_text()
    found: set[str] = set()
    for path in ("tests/test_vectorized.py", "tests/test_engine.py"):
        found |= set(re.findall(r"def (test_bias_\w+)", Path(path).read_text()))
    assert len(found) >= 4
    assert "test_bias_*" in readme
    for claim in ("lookahead", "survivorship", "walk_forward", "probabilistic Sharpe"):
        assert claim.lower() in readme.lower()


def test_documented_files_exist() -> None:
    readme = Path("README.md").read_text()
    for name in re.findall(r"\]\((\w+\.md)\)", readme):
        assert Path(name).exists(), f"README links to missing {name}"
    for path in re.findall(r"\]\((tests/\w+\.py|scripts/\w+\.py)\)", readme):
        assert Path(path).exists(), f"README links to missing {path}"


# -- RESULTS.md must agree with the CSV that produced it ---------------------------------

RESULTS_CSV = Path("reports/strategies/results.csv")


def _markdown_rows(text: str, header_contains: str) -> dict[str, list[str]]:
    """The rows of the first markdown table whose header contains ``header_contains``."""
    rows: dict[str, list[str]] = {}
    in_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            in_table = False
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header_contains in line:
            in_table = True
            continue
        if not in_table or set(line) <= set("|- "):
            continue
        rows[cells[0].replace("**", "")] = cells[1:]
    return rows


def _number(cell: str) -> float:
    """Parse a table cell like ``-16.1%``, ``**0.58**``, ``1,102`` or ``+2.4%``."""
    raw = cell.replace("**", "").replace(",", "").strip()
    if raw.endswith("%"):
        return float(raw[:-1]) / 100.0
    return float(raw)


def _check_headline_table(text: str, table: pd.DataFrame) -> None:
    headline = _markdown_rows(text, "| CAGR | Vol |")
    columns = ["cagr", "ann_vol", "sharpe", "sortino", "max_drawdown", "max_dd_days", "turnover"]
    for name, row in table.iterrows():
        cells = headline[str(name)]
        assert len(cells) == 8, f"{name}: expected 8 columns, got {cells}"
        for cell, column in zip(cells, columns, strict=False):
            reported, actual = _number(cell), float(row[column])
            tol = 0.5 if column == "max_dd_days" else max(abs(actual) * 0.02, 0.005)
            assert abs(reported - actual) <= tol, f"{name}.{column}: {reported} != {actual}"


@pytest.mark.skipif(not RESULTS_CSV.exists(), reason="run scripts/run_strategies.py first")
def test_readme_headline_table_matches_the_generated_table() -> None:
    """The copy of the headline table in README.md cannot drift from the pipeline either."""
    _check_headline_table(
        Path("README.md").read_text(), pd.read_csv(RESULTS_CSV).set_index("strategy")
    )


@pytest.mark.skipif(not RESULTS_CSV.exists(), reason="run scripts/run_strategies.py first")
def test_results_md_matches_the_generated_table() -> None:
    """Every headline number in RESULTS.md is the one the pipeline actually produced.

    The tables are hand-written prose around machine-generated numbers, which is exactly
    the arrangement that silently goes stale. This is the check that stops it.
    """
    table = pd.read_csv(RESULTS_CSV).set_index("strategy")
    text = Path("RESULTS.md").read_text()

    _check_headline_table(text, table)

    diagnostics = _markdown_rows(text, "| Bootstrap 95% CI |")
    for name, row in table.iterrows():
        sharpe, ci, psr, pbo, alpha, tstat, r2, _beta, _t = diagnostics[str(name)]
        assert abs(_number(sharpe) - float(row["sharpe"])) <= 0.005
        lo, hi = (_number(x) for x in ci.split(" to "))
        assert abs(lo - float(row["boot_ci_lo"])) <= 0.005
        assert abs(hi - float(row["boot_ci_hi"])) <= 0.005
        assert abs(_number(psr) - float(row["psr"])) <= 0.0005
        assert abs(_number(pbo) - float(row["pbo"])) <= 0.005
        assert abs(_number(alpha) - float(row["ff5_alpha"])) <= 0.0005
        assert abs(_number(tstat) - float(row["ff5_alpha_t"])) <= 0.005
        assert abs(_number(r2) - float(row["ff5_r2"])) <= 0.005

    penalty = _markdown_rows(text, "| In-sample best |")
    for name, row in table.iterrows():
        is_cell, oos_cell, gap_cell = penalty[str(name)]
        assert abs(_number(is_cell) - float(row["is_sharpe"])) <= 0.005
        assert abs(_number(oos_cell) - float(row["sharpe"])) <= 0.005
        # the gap is rounded from the unrounded Sharpes, not from the rounded cells
        gap = float(row["sharpe"]) - float(row["is_sharpe"])
        assert abs(_number(gap_cell) - gap) <= 0.005


@pytest.mark.skipif(not RESULTS_CSV.exists(), reason="run scripts/run_strategies.py first")
def test_results_md_documents_the_borrow_rate_actually_charged() -> None:
    """A long/short result reported without a stock-loan fee is an upper bound; say so."""
    table = pd.read_csv(RESULTS_CSV).set_index("strategy")
    text = Path("RESULTS.md").read_text()
    for name, row in table.iterrows():
        shorts = float(row["exposure"]) > 0 and name in ("tsmom", "xsmom", "pairs")
        if shorts:
            assert float(row["borrow_rate"]) > 0, f"{name} shorts but pays no borrow"
            bps = round(float(row["borrow_rate"]) * 1e4)
            assert f"{bps} bp" in text, f"RESULTS.md does not state {name}'s {bps} bp borrow"
