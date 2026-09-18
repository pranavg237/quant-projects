"""The README teaches one way to write a strategy. This keeps that example honest.

The code below is the README's "Adding a strategy" snippet, run against synthetic data so
the test stays offline. If the Context API changes, this fails and the README gets fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

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
