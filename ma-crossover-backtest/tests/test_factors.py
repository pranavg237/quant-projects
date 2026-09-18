from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantbt.factors import (
    compare_models,
    factor_regression,
    french,
    load_factors,
    parse_french_csv,
)
from quantbt.factors.regression import newey_west_lags

FF3_TEXT = """This file was created by CMPT_ME_BEME_RETS using the 202508 CRSP database.
The 1-month TBill return is from Ibbotson and Associates, Inc.

,Mkt-RF,SMB,HML,RF
20200102,    0.85,   -0.40,   -1.00,    0.006
20200103,   -0.67,    0.10,   -0.20,    0.006
20200106,    0.36,   -0.20,  -99.99,    0.006
20200107,   -0.25,    0.30,    0.15,    0.006

  Annual Factors: January-December

,Mkt-RF,SMB,HML,RF
2020,   23.66,   12.28,  -46.62,    0.45
"""

FF3_MONTHLY_TEXT = """Some description

,Mkt-RF,SMB,HML,RF
202001,   -0.11,   -3.11,   -6.27,    0.13
202002,   -8.13,    1.09,   -3.95,    0.12
"""

MOM_TEXT = """Momentum factor

,Mom
20200102,    0.50
20200103,    0.20
20200106,   -0.30
20200107,    0.10
"""


def _zip_bytes(text: str, name: str = "table.csv") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, text)
    return buf.getvalue()


def test_parse_french_csv_tables_and_missing_values() -> None:
    tables = parse_french_csv(FF3_TEXT)
    assert [t.frequency for t in tables] == ["daily", "annual"]
    daily = tables[0].data
    assert list(daily.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert daily.index[0] == pd.Timestamp("2020-01-02")
    assert daily.loc["2020-01-02", "Mkt-RF"] == pytest.approx(0.0085)
    assert bool(daily["HML"].isna().loc["2020-01-06"])
    assert "CRSP database" in tables[0].title
    assert "Annual Factors" in tables[1].title
    assert tables[1].data.index[0] == pd.Timestamp("2020-12-31")
    monthly = parse_french_csv(FF3_MONTHLY_TEXT)[0]
    assert monthly.frequency == "monthly"
    assert monthly.data.index[0] == pd.Timestamp("2020-01-31")
    mom = parse_french_csv(MOM_TEXT)[0].data
    assert list(mom.columns) == ["MOM"]


def test_load_factors_downloads_once_and_joins_momentum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(name: str) -> bytes:
        calls.append(name)
        if "Momentum" in name:
            return _zip_bytes(MOM_TEXT)
        return _zip_bytes(FF3_TEXT)

    monkeypatch.setattr(french, "_download", fake_download)
    ff3 = load_factors("ff3", "daily", cache_dir=tmp_path)
    assert list(ff3.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(ff3) == 3  # the -99.99 row is dropped
    assert calls == ["F-F_Research_Data_Factors_daily"]
    load_factors("ff3", "daily", cache_dir=tmp_path)  # cached
    assert len(calls) == 1
    carhart = load_factors("carhart", "daily", cache_dir=tmp_path, start="2020-01-03")
    assert list(carhart.columns) == ["Mkt-RF", "SMB", "HML", "MOM", "RF"]
    assert carhart.index[0] == pd.Timestamp("2020-01-03")
    assert calls[-1] == "F-F_Momentum_Factor_daily"
    load_factors("ff3", "daily", cache_dir=tmp_path, refresh=True)
    assert calls.count("F-F_Research_Data_Factors_daily") == 2
    with pytest.raises(ValueError, match="unknown model"):
        load_factors("ff9", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="frequency"):
        load_factors("ff3", "weekly", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="no monthly"):
        load_factors("ff3", "monthly", cache_dir=tmp_path)


def test_fetch_dataset_rejects_non_zip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(french, "_download", lambda name: b"<html>error</html>")
    with pytest.raises(zipfile.BadZipFile):
        french.fetch_dataset("X", cache_dir=tmp_path)


def _synthetic(n: int = 2500, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    f = pd.DataFrame(
        {
            "Mkt-RF": rng.normal(0.0003, 0.01, n),
            "SMB": rng.normal(0.0, 0.005, n),
            "HML": rng.normal(0.0, 0.005, n),
            "RMW": rng.normal(0.0, 0.004, n),
            "CMA": rng.normal(0.0, 0.004, n),
            "MOM": rng.normal(0.0, 0.006, n),
            "RF": 0.0001,
        },
        index=idx,
    )
    alpha = 0.0002
    r = alpha + 0.8 * f["Mkt-RF"] - 0.5 * f["SMB"] + 0.3 * f["HML"] + rng.normal(0, 0.004, n)
    return (r + f["RF"]).rename("strat"), f


def test_regression_recovers_known_coefficients() -> None:
    r, f = _synthetic()
    res = factor_regression(r, f, "ff3")
    assert res.alpha == pytest.approx(0.0002, abs=3e-4)
    assert res.betas["Mkt-RF"] == pytest.approx(0.8, abs=0.03)
    assert res.betas["SMB"] == pytest.approx(-0.5, abs=0.06)
    assert res.betas["HML"] == pytest.approx(0.3, abs=0.06)
    assert res.rsquared > 0.8
    assert res.nobs == 2500
    assert res.lags == newey_west_lags(2500) == int(np.floor(4 * 25 ** (2 / 9)))
    assert res.periods_per_year == 252
    assert res.alpha_annual == pytest.approx(res.alpha * 252)
    assert abs(res.tvalues["Mkt-RF"]) > 50
    assert res.information_ratio == pytest.approx(res.alpha_annual / res.residual_vol_annual)
    table = res.table()
    assert list(table.columns) == ["coef", "std_err", "t", "p"] and "alpha" in table.index
    s = res.to_series()
    assert s["model"] == "ff3" and "beta_HML" in s.index and s["nobs"] == 2500
    text = res.summary()
    assert "FF3" in text and "alpha" in text and "Mkt-RF" in text
    # explicit lag count and excess-return input give the same betas
    res2 = factor_regression(r - f["RF"], f, "ff3", excess=True, lags=5, name="x")
    assert res2.lags == 5 and res2.name == "x"
    np.testing.assert_allclose(res2.betas, res.betas)
    assert res.start == pd.Timestamp("2010-01-01") and res.end == r.index[-1]


def test_hac_errors_widen_under_autocorrelation() -> None:
    rng = np.random.default_rng(3)
    n = 3000
    idx = pd.bdate_range("2010-01-01", periods=n)
    f = pd.DataFrame({"Mkt-RF": rng.normal(0, 0.01, n), "RF": 0.0}, index=idx)
    # AR(1) residual with strong persistence
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.8 * e[i - 1] + rng.normal(0, 0.003)
    r = (0.5 * f["Mkt-RF"] + e).rename("ar")
    ols = factor_regression(r, f, "capm", lags=0)
    hac = factor_regression(r, f, "capm", lags=20)
    assert hac.std_errors["alpha"] > 1.5 * ols.std_errors["alpha"]


def test_regression_validation_and_compare_models() -> None:
    r, f = _synthetic(300)
    with pytest.raises(ValueError, match="lacks columns"):
        factor_regression(r, f.drop(columns=["HML"]), "ff3")
    with pytest.raises(ValueError, match="overlapping"):
        factor_regression(r.iloc[:5], f, "ff3")
    table = compare_models(r, f, ("capm", "ff3", "ff5", "ff6"))
    assert list(table.index) == ["capm", "ff3", "ff5", "ff6"]
    assert "beta_MOM" in table.columns and bool(table["beta_SMB"].isna().loc["capm"])
    partial = compare_models(r, f.drop(columns=["MOM"]), ("ff3", "ff6"))
    assert list(partial.index) == ["ff3"]
    with pytest.raises(ValueError, match="no model"):
        compare_models(r, f[["Mkt-RF", "RF"]], ("ff3",))
    custom = factor_regression(r, f, "custom", factor_names=["Mkt-RF", "MOM"])
    assert custom.factors == ["Mkt-RF", "MOM"]
