import io
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

from ffmodel import data
from ffmodel.assets import build_portfolio, load_returns_csv, prices_to_returns

FF3_TEXT = (
    "This file was created using the 202607 CRSP database.\r\n"
    "The 1-month TBill rate data until 202405 are from Ibbotson Associates.\r\n"
    "\r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "192607,   2.89,  -2.42,  -2.75,   0.22\r\n"
    "192608,   2.64,  -1.44,   4.13,   0.25\r\n"
    "192609,   0.38,  -1.20,   0.12,   0.23\r\n"
    "\r\n"
    " Annual Factors: January-December \r\n"
    ",Mkt-RF,SMB,HML,RF\r\n"
    "  1927,  29.47,  -2.04,  -3.54,   3.12\r\n"
    "\r\n"
    "Copyright 2026 Eugene F. Fama and Kenneth R. French\r\n"
)

MOM_TEXT = (
    "Missing data are indicated by -99.99 or -999.\r\n"
    "\r\n"
    ",Mom   \r\n"
    "192608,   1.00\r\n"
    "192609,  -0.50\r\n"
    "192610,   0.70\r\n"
)

PORTFOLIO_TEXT = (
    "It contains value- and equal-weighted returns for portfolios formed on ME and BEME.\r\n"
    "\r\n"
    "\r\n"
    "  Average Value Weighted Returns -- Monthly\r\n"
    ",SMALL LoBM,SMALL HiBM,BIG LoBM,BIG HiBM,\r\n"
    "192607,   1.00, -99.99,   3.00,   4.00,\r\n"
    "192608,   2.00,   2.50,   3.50,   4.50,\r\n"
    "\r\n"
    "\r\n"
    "  Average Equal Weighted Returns -- Monthly\r\n"
    ",SMALL LoBM,SMALL HiBM,BIG LoBM,BIG HiBM,\r\n"
    "192607,   9.00,   9.00,   9.00,   9.00,\r\n"
    "\r\n"
    "  Number of Firms in Portfolios\r\n"
    ",SMALL LoBM,SMALL HiBM,BIG LoBM,BIG HiBM,\r\n"
    "192607,   100,   20,   30,   40,\r\n"
)


def test_parse_splits_tables_and_dates():
    tables = data.parse_french_csv(FF3_TEXT)
    assert [t.frequency for t in tables] == ["monthly", "annual"]
    monthly, annual = tables
    assert list(monthly.data.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert monthly.data.index[0] == pd.Timestamp("1926-07-31")
    assert monthly.data.loc["1926-08-31", "HML"] == 4.13
    assert annual.title == "Annual Factors: January-December"
    assert annual.data.index[0] == pd.Timestamp("1927-12-31")


def test_parse_handles_trailing_commas_and_missing_values():
    tables = data.parse_french_csv(PORTFOLIO_TEXT)
    assert [t.title for t in tables] == [
        "Average Value Weighted Returns -- Monthly",
        "Average Equal Weighted Returns -- Monthly",
        "Number of Firms in Portfolios",
    ]
    value = tables[0].data
    assert list(value.columns) == ["SMALL LoBM", "SMALL HiBM", "BIG LoBM", "BIG HiBM"]
    assert np.isnan(value.iloc[0, 1])


@pytest.fixture
def fake_library(monkeypatch):
    files = {
        "F-F_Research_Data_Factors": FF3_TEXT,
        "F-F_Momentum_Factor": MOM_TEXT,
        "25_Portfolios_5x5": PORTFOLIO_TEXT,
    }
    monkeypatch.setattr(data, "fetch_dataset", lambda name, refresh=False, data_dir=None: data.parse_french_csv(files[name]))


def test_load_factors_joins_momentum(fake_library):
    carhart = data.load_factors("carhart")
    assert list(carhart.columns) == ["Mkt-RF", "SMB", "HML", "MOM", "RF"]
    assert list(carhart.index) == [pd.Timestamp("1926-08-31"), pd.Timestamp("1926-09-30")]
    assert carhart.loc["1926-08-31", "Mkt-RF"] == pytest.approx(0.0264)
    assert carhart.loc["1926-09-30", "MOM"] == pytest.approx(-0.005)

    capm = data.load_factors("capm", start="1926-09")
    assert list(capm.columns) == ["Mkt-RF", "RF"]
    assert len(capm) == 1


def test_load_portfolios_picks_weighting(fake_library):
    value = data.load_portfolios("25_Portfolios_5x5", weighting="value")
    assert value.loc["1926-08-31", "BIG HiBM"] == pytest.approx(0.045)
    equal = data.load_portfolios("25_Portfolios_5x5", weighting="equal")
    assert (equal.iloc[0] == 0.09).all()
    with pytest.raises(ValueError):
        data.load_portfolios("25_Portfolios_5x5", weighting="cap")


def test_fetch_uses_cache_until_stale_or_refresh(monkeypatch, tmp_path):
    calls = []

    def fake_download(name, path):
        calls.append(name)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{name}.csv", FF3_TEXT)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())

    monkeypatch.setattr(data, "_download", fake_download)
    tables = data.fetch_dataset("F-F_Research_Data_Factors", cache_dir=tmp_path)
    data.fetch_dataset("F-F_Research_Data_Factors", cache_dir=tmp_path)
    assert len(calls) == 1 and len(tables) == 2
    data.fetch_dataset("F-F_Research_Data_Factors", cache_dir=tmp_path, refresh=True)
    assert len(calls) == 2

    def failing_download(name, path):
        raise data.requests.ConnectionError("offline")

    monkeypatch.setattr(data, "_download", failing_download)
    with pytest.warns(UserWarning, match="cached copy"):
        assert len(data.fetch_dataset("F-F_Research_Data_Factors", cache_dir=tmp_path, refresh=True)) == 2
    with pytest.raises(RuntimeError, match="Could not download"):
        data.fetch_dataset("Something_Else", cache_dir=tmp_path)


def test_prices_to_monthly_returns_drops_partial_month():
    days = pd.bdate_range("2024-01-02", "2024-04-10")
    prices = pd.DataFrame({"X": np.linspace(100, 130, len(days))}, index=days)
    monthly = prices_to_returns(prices, "monthly")
    assert list(monthly.index) == [pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-31")]
    feb_ret = prices.loc["2024-02"].iloc[-1] / prices.loc["2024-01"].iloc[-1] - 1
    assert monthly.iloc[0, 0] == pytest.approx(feb_ret["X"])


def test_load_returns_csv_and_portfolio(tmp_path):
    path = tmp_path / "r.csv"
    path.write_text("date,A,B\n2020-01-01,1.0,2.0\n2020-02-01,-1.0,4.0\n")
    returns = load_returns_csv(str(path), percent=True)
    assert list(returns.index) == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    port = build_portfolio(returns, {"A": 0.5, "B": 0.5})
    np.testing.assert_allclose(port.to_numpy(), [0.015, 0.015])
    with pytest.raises(KeyError):
        build_portfolio(returns, {"C": 1.0})

    daily = tmp_path / "d.csv"
    daily.write_text("date,A\n2020-01-02,1.0\n2020-01-03,2.0\n")
    with pytest.raises(ValueError, match="daily"):
        load_returns_csv(str(daily))


FF5_TEXT = (
    ",Mkt-RF,SMB,HML,RMW,CMA,RF\r\n"
    "192607,   2.89,  -2.42,  -2.75,   0.10,   0.20,   0.22\r\n"
    "192608,   2.64,  -1.44,   4.13,   0.30,   0.40,   0.25\r\n"
    "192609,   0.38,  -1.20,   0.12,   0.50,   0.60,   0.23\r\n"
)


def test_snapshot_round_trip_runs_offline(monkeypatch, tmp_path):
    from ffmodel import snapshot

    texts = {"F-F_Research_Data_Factors": FF3_TEXT, "F-F_Research_Data_5_Factors_2x3": FF5_TEXT,
             "F-F_Momentum_Factor": MOM_TEXT}

    def fake_download(name, path):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{name}.csv", texts[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(buffer.getvalue())

    monkeypatch.setattr(data, "_download", fake_download)
    returns = pd.DataFrame({"SPY": [0.01, -0.02]}, index=pd.to_datetime(["1926-08-31", "1926-09-30"]))
    monkeypatch.setattr(snapshot, "download_returns", lambda tickers, start, end, frequency: returns)

    snap = tmp_path / "snap"
    manifest = snapshot.save_snapshot(snap, ["SPY"], "1926-08", "1926-09")
    assert set(manifest["french_library"]) == {f"{name}_CSV.zip" for name in texts}
    assert manifest["french_library"]["F-F_Momentum_Factor_CSV.zip"]["first"] == "1926-08"
    assert manifest["returns"]["periods"] == 2 and manifest["returns"]["last"] == "1926-09-30"
    assert snapshot.read_manifest(snap)["downloaded_utc"] == manifest["downloaded_utc"]
    assert snapshot.read_manifest(tmp_path) is None

    def no_network(name, path):
        raise AssertionError("reading a snapshot must not download anything")

    monkeypatch.setattr(data, "_download", no_network)
    ff6 = data.load_factors("ff6", data_dir=snap)
    assert list(ff6.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM", "RF"]
    assert ff6.loc["1926-09-30", "CMA"] == pytest.approx(0.006)
    loaded = load_returns_csv(str(snap / "returns.csv"))
    np.testing.assert_allclose(loaded["SPY"].to_numpy(), returns["SPY"].to_numpy())
    assert list(loaded.index) == list(returns.index)
    with pytest.raises(RuntimeError, match="not in the data snapshot"):
        data.load_portfolios("25_Portfolios_5x5", data_dir=snap)
