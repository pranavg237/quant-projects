import numpy as np
import pandas as pd
import pytest
from conftest import make_factors

from ffmodel import cli, data
from ffmodel.models import get_model


@pytest.fixture
def offline(monkeypatch):
    factors = make_factors(T=240, start="2000-01-31")

    def fake_load_factors(model="ff5", frequency="monthly", start=None, end=None, refresh=False, data_dir=None):
        return factors.loc[start:end, list(get_model(model).factors) + ["RF"]]

    def fake_load_portfolios(dataset="25_Portfolios_5x5", frequency="monthly", weighting="value",
                             start=None, end=None, refresh=False, data_dir=None):
        rng = np.random.default_rng(3)
        betas = rng.uniform(0.5, 1.5, size=(25, 3))
        r = factors[["Mkt-RF", "SMB", "HML"]].to_numpy() @ betas.T + rng.normal(0, 0.02, (len(factors), 25)) + 0.003
        return pd.DataFrame(r, index=factors.index, columns=[f"P{i}" for i in range(25)]).loc[start:end]

    monkeypatch.setattr(data, "load_factors", fake_load_factors)
    monkeypatch.setattr(data, "load_portfolios", fake_load_portfolios)
    return factors


def write_returns(factors, path):
    rng = np.random.default_rng(0)
    n = len(factors)
    frame = pd.DataFrame(
        {
            "A": 0.004 + 1.2 * factors["Mkt-RF"] + rng.normal(0, 0.01, n),
            "B": 0.001 + 0.8 * factors["Mkt-RF"] + 0.5 * factors["HML"] + rng.normal(0, 0.01, n),
        },
        index=factors.index,
    )
    frame.to_csv(path)


def test_analyze_writes_full_report(offline, tmp_path, capsys):
    write_returns(offline, tmp_path / "returns.csv")
    out = tmp_path / "report"
    code = cli.main(["analyze", "--csv", str(tmp_path / "returns.csv"), "--model", "ff3", "--compare",
                     "--rolling", "36", "--weights", "A=0.5,b=0.5", "--out", str(out)])
    assert code == 0
    text = (out / "report.md").read_text()
    assert "## Portfolio" in text and "Carhart 4-factor" in text
    assert len(list(out.glob("*.png"))) == 3 * 4  # loadings, attribution, cumulative, rolling per asset
    assert (out / "summary.csv").exists()
    assert "Fama-French 3-factor regressions" in capsys.readouterr().out
    assert "Newey-West (HAC) with 4 lags" in text  # T = 240
    assert "Alphas: one test per asset vs. the whole family" in text
    assert "A: t-statistics under different standard errors" in text
    assert "the first ends Dec 2002 and the last Dec 2019" in text
    assert "HC3, MacKinnon-White" in text  # short windows never use Newey-West
    rolled = pd.read_csv(out / "a-rolling.csv", index_col=0)
    assert {"se(alpha)", "se(Mkt-RF)"} <= set(rolled.columns)
    assert "se(alpha)" not in pd.read_csv(out / "a-rolling-summary.csv", index_col=0).columns
    assert "A: did the exposures change? (6 separate 36-period blocks)" in text
    assert (out / "portfolio-stability.csv").exists()


def test_test_portfolios_command(offline, tmp_path):
    out = tmp_path / "tp"
    assert cli.main(["test-portfolios", "--compare", "--out", str(out)]) == 0
    text = (out / "report.md").read_text()
    assert "GRS test" in text and "Fama-MacBeth: Fama-French 6-factor" in text
    assert (out / "grs.csv").exists()
    assert (out / "alpha-heatmap.png").exists() and (out / "pricing.png").exists()


def test_factors_command_without_report(offline, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["factors", "--model", "ff6", "--no-report"]) == 0
    assert "Spanning regressions" in capsys.readouterr().out
    assert not (tmp_path / "reports").exists()


def test_bad_portfolio_holding_is_a_clean_error(offline, tmp_path, capsys):
    write_returns(offline, tmp_path / "returns.csv")
    code = cli.main(["analyze", "--csv", str(tmp_path / "returns.csv"), "--weights", "ZZZ=1", "--no-report"])
    assert code == 1
    assert "ZZZ" in capsys.readouterr().err


def test_sort_layout():
    assert cli._sort_layout("25_Portfolios_5x5", 25) == ((5, 5), "size (ME)", "book-to-market")
    assert cli._sort_layout("25_Portfolios_ME_OP_5x5", 25) == ((5, 5), "size (ME)", "profitability (OP)")
    assert cli._sort_layout("10_Industry_Portfolios", 10) is None
