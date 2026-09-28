"""Download, caching and fallback paths, exercised against a stub ``yfinance``.

These paths are the ones that break silently in production -- a renamed Yahoo column, an
expiry that 404s, an offline machine -- and they are exactly the ones a live-network test
cannot pin down, because it passes or fails for reasons outside the repo. A stub module
injected into ``sys.modules`` makes them deterministic.
"""

from __future__ import annotations

import datetime as dt
import sys
import types
import warnings
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from optpricing import data as data_mod

NY = ZoneInfo("America/New_York")


class _Chain:
    def __init__(self, calls: pd.DataFrame, puts: pd.DataFrame) -> None:
        self.calls = calls
        self.puts = puts


class _StubTicker:
    """Mimics the slice of the yfinance API this module actually uses."""

    def __init__(
        self,
        expiries: list[str],
        spot: float = 500.0,
        bad_expiries: frozenset[str] = frozenset(),
        empty_history: bool = False,
        empty_puts: bool = False,
    ) -> None:
        self.options = tuple(expiries)
        self._spot = spot
        self._bad = bad_expiries
        self._empty_history = empty_history
        self._empty_puts = empty_puts

    def history(self, period: str = "5d", auto_adjust: bool = False) -> pd.DataFrame:
        if self._empty_history:
            return pd.DataFrame()
        return pd.DataFrame({"Close": [self._spot - 1.0, self._spot]})

    def option_chain(self, expiry: str) -> _Chain:
        if expiry in self._bad:
            raise RuntimeError(f"upstream 404 for {expiry}")
        strikes = np.array([480.0, 490.0, 500.0, 510.0, 520.0])
        frame = pd.DataFrame(
            {
                "contractSymbol": [f"STUB{expiry}{k:.0f}" for k in strikes],
                "strike": strikes,
                "lastPrice": np.full(5, 10.0),
                "bid": np.full(5, 9.9),
                "ask": np.full(5, 10.1),
                "volume": np.full(5, 25.0),
                "openInterest": np.full(5, 300.0),
                "impliedVolatility": np.full(5, 0.2),
                "lastTradeDate": pd.to_datetime(["2026-09-18"] * 5, utc=True),
            }
        )
        puts = pd.DataFrame() if self._empty_puts else frame.copy()
        return _Chain(frame, puts)


def _install_stub(monkeypatch: pytest.MonkeyPatch, ticker_factory) -> None:
    module = types.ModuleType("yfinance")
    module.Ticker = ticker_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", module)


def _future_expiries(n: int, start_days: int = 7, step_days: int = 30) -> list[str]:
    today = dt.datetime.now(tz=NY).date()
    return [(today + dt.timedelta(days=start_days + i * step_days)).isoformat() for i in range(n)]


def test_download_chain_shapes_the_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expiries = _future_expiries(4)
    _install_stub(monkeypatch, lambda ticker: _StubTicker(expiries))
    snapshot = data_mod.download_chain("stub", max_expiries=4, cache_dir=tmp_path)

    assert snapshot.ticker == "STUB"
    assert snapshot.spot == pytest.approx(500.0)
    assert len(snapshot.quotes) == 4 * 2 * 5
    # Yahoo's column names are normalised to the repo's vocabulary.
    assert {"last", "open_interest", "yf_implied_vol", "contract", "tau", "mid"} <= set(
        snapshot.quotes.columns
    )
    assert set(snapshot.quotes["option_type"]) == {"call", "put"}
    assert np.allclose(snapshot.quotes["mid"], 10.0)
    assert (snapshot.quotes["tau"] > 0).all()
    # And the download is cached, so a rerun is offline.
    cached = tmp_path / "raw" / f"STUB_{snapshot.asof.date().isoformat()}.csv"
    assert cached.exists()


def test_download_skips_a_broken_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """One 404 expiry must not take the whole pull down with it."""
    expiries = _future_expiries(3)
    _install_stub(
        monkeypatch,
        lambda ticker: _StubTicker(expiries, bad_expiries=frozenset({expiries[1]})),
    )
    with pytest.warns(UserWarning, match="skipping expiry"):
        snapshot = data_mod.download_chain("stub", cache_dir=tmp_path)
    assert len(snapshot.expiries()) == 2


def test_download_raises_when_there_is_no_price_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub(monkeypatch, lambda ticker: _StubTicker(_future_expiries(2), empty_history=True))
    with pytest.raises(RuntimeError, match="no price history"):
        data_mod.download_chain("stub", cache_dir=tmp_path)


def test_download_raises_when_every_expiry_is_filtered_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # All expiries inside the min_tau_days window.
    _install_stub(
        monkeypatch, lambda ticker: _StubTicker(_future_expiries(3, start_days=1, step_days=1))
    )
    with pytest.raises(RuntimeError, match="tau window"):
        data_mod.download_chain("stub", cache_dir=tmp_path, min_tau_days=30.0)


def test_download_raises_when_every_expiry_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expiries = _future_expiries(2)
    _install_stub(
        monkeypatch, lambda ticker: _StubTicker(expiries, bad_expiries=frozenset(expiries))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="no option quotes"):
            data_mod.download_chain("stub", cache_dir=tmp_path)


def test_download_tolerates_a_missing_put_side(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub(monkeypatch, lambda ticker: _StubTicker(_future_expiries(2), empty_puts=True))
    snapshot = data_mod.download_chain("stub", cache_dir=tmp_path)
    assert set(snapshot.quotes["option_type"]) == {"call"}


def test_expiries_are_spread_in_log_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Twenty weeklies plus a LEAP: the selection must reach the back end, not stop at 5 weeks.

    Taking the ``n`` nearest expiries is the obvious implementation and gives a "term
    structure" with no term in it.
    """
    today = dt.datetime.now(tz=NY).date()
    weeklies = [(today + dt.timedelta(days=7 + 7 * i)).isoformat() for i in range(20)]
    longer = [(today + dt.timedelta(days=d)).isoformat() for d in (200, 400, 700)]
    _install_stub(monkeypatch, lambda ticker: _StubTicker(weeklies + longer))

    snapshot = data_mod.download_chain("stub", max_expiries=6, cache_dir=tmp_path)
    chosen = snapshot.expiries()
    assert len(chosen) <= 6
    span_days = (chosen[-1] - chosen[0]).days
    assert span_days > 500  # the back end is represented
    # And the front end is too, so the short-dated smile is not lost either.
    assert (chosen[0] - today).days < 30


def test_load_chain_falls_back_to_a_cache_when_the_download_fails(
    synthetic_snapshot: data_mod.ChainSnapshot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A requested refresh that fails offline must degrade to the cache, not crash."""
    (tmp_path / "snapshots").mkdir(parents=True)
    stale = dt.datetime.now(tz=NY) - dt.timedelta(days=3)
    data_mod.ChainSnapshot("FALL", stale, 321.0, synthetic_snapshot.quotes).to_csv(
        tmp_path / "snapshots" / f"FALL_{stale.date().isoformat()}.csv"
    )

    def _explode(ticker: str):
        raise ConnectionError("no network")

    _install_stub(monkeypatch, _explode)
    with pytest.warns(UserWarning, match="falling back to cached snapshot"):
        loaded = data_mod.load_chain("FALL", cache_dir=tmp_path, force_refresh=True)
    assert loaded.spot == pytest.approx(321.0)


def test_load_chain_force_refresh_bypasses_todays_cache(
    synthetic_snapshot: data_mod.ChainSnapshot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    today = dt.datetime.now(tz=NY)
    (tmp_path / "raw").mkdir(parents=True)
    data_mod.ChainSnapshot("STUB", today, 111.0, synthetic_snapshot.quotes).to_csv(
        tmp_path / "raw" / f"STUB_{today.date().isoformat()}.csv"
    )
    _install_stub(monkeypatch, lambda ticker: _StubTicker(_future_expiries(2), spot=999.0))

    assert data_mod.load_chain("STUB", cache_dir=tmp_path).spot == pytest.approx(111.0)
    refreshed = data_mod.load_chain("STUB", cache_dir=tmp_path, force_refresh=True)
    assert refreshed.spot == pytest.approx(999.0)


def test_load_chain_uses_the_committed_snapshot_without_touching_the_network(
    synthetic_snapshot: data_mod.ChainSnapshot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression test: an old snapshot used to be ignored in favour of a live download,
    so the analysis silently changed with the date it was run (and failed on weekends,
    when most live quotes are one-sided)."""
    (tmp_path / "snapshots").mkdir(parents=True)
    old = dt.datetime(2026, 9, 18, 15, 30, tzinfo=NY)
    data_mod.ChainSnapshot("SNAP", old, 654.0, synthetic_snapshot.quotes).to_csv(
        tmp_path / "snapshots" / "SNAP_2026-09-18.csv"
    )

    def _must_not_download(ticker: str):
        raise AssertionError("load_chain went to the network despite a cached snapshot")

    _install_stub(monkeypatch, _must_not_download)
    assert data_mod.load_chain("SNAP", cache_dir=tmp_path).spot == pytest.approx(654.0)


def test_load_rate_curve_matches_the_requested_date(tmp_path: Path) -> None:
    (tmp_path / "snapshots").mkdir(parents=True)
    for day, rate in (("2026-09-18", 0.040), ("2026-09-25", 0.050)):
        pd.DataFrame({"tenor": [0.25, 10.0], "rate": [rate, rate]}).to_csv(
            tmp_path / "snapshots" / f"ratecurve_{day}.csv", index=False
        )
    matched = data_mod.load_rate_curve(cache_dir=tmp_path, asof=dt.date(2026, 9, 18))
    assert float(matched.rate(1.0)[0]) == pytest.approx(0.040)
    latest = data_mod.load_rate_curve(cache_dir=tmp_path)
    assert float(latest.rate(1.0)[0]) == pytest.approx(0.050)
    with pytest.warns(UserWarning, match="no rate curve cached"):
        data_mod.load_rate_curve(cache_dir=tmp_path, asof=dt.date(2020, 1, 1))


class _StubIndexTicker:
    """Yahoo publishes Treasury yields as index levels in percent."""

    _LEVELS = {"^IRX": 4.0, "^FVX": 4.5, "^TNX": 4.75, "^TYX": 5.0}

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol

    def history(self, period: str = "5d", auto_adjust: bool = False) -> pd.DataFrame:
        return pd.DataFrame({"Close": [self._LEVELS[self._symbol]]})


def test_load_rate_curve_converts_to_continuous_compounding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_stub(monkeypatch, _StubIndexTicker)
    curve = data_mod.load_rate_curve(cache_dir=tmp_path, force_refresh=True)
    assert np.allclose(curve.tenors, [0.25, 5.0, 10.0, 30.0])
    # Bond-equivalent 4.00% -> continuously compounded ln(1.04) = 3.922%.
    assert float(curve.rate(0.25)[0]) == pytest.approx(np.log(1.04), abs=1e-12)
    assert float(curve.rate(30.0)[0]) == pytest.approx(np.log(1.05), abs=1e-12)
    # Continuous compounding is always below the bond-equivalent quote.
    assert float(curve.rate(10.0)[0]) < 0.0475
    # ...and it is cached, so the second call is offline.
    cached = tmp_path / "raw" / f"ratecurve_{dt.datetime.now(tz=NY).date().isoformat()}.csv"
    assert cached.exists()
    assert np.allclose(data_mod.load_rate_curve(cache_dir=tmp_path).rate([1.0]), curve.rate([1.0]))


def test_load_rate_curve_falls_back_to_an_older_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "raw").mkdir(parents=True)
    pd.DataFrame({"tenor": [0.25, 10.0], "rate": [0.031, 0.041]}).to_csv(
        tmp_path / "raw" / "ratecurve_2020-01-01.csv", index=False
    )

    def _explode(symbol: str):
        raise ConnectionError("offline")

    _install_stub(monkeypatch, _explode)
    with pytest.warns(UserWarning):
        curve = data_mod.load_rate_curve(cache_dir=tmp_path, force_refresh=True)
    assert float(curve.rate(0.25)[0]) == pytest.approx(0.031)


def test_load_rate_curve_falls_back_to_the_hard_coded_curve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _explode(symbol: str):
        raise ConnectionError("offline")

    _install_stub(monkeypatch, _explode)
    with pytest.warns(UserWarning, match="hard-coded fallback"):
        curve = data_mod.load_rate_curve(cache_dir=tmp_path, force_refresh=True)
    assert np.allclose(curve.tenors, sorted(data_mod.FALLBACK_RATES))
    assert 0.03 < float(curve.rate(1.0)[0]) < 0.06


def test_load_rate_curve_skips_a_nonsense_quote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _PartlyBroken(_StubIndexTicker):
        def history(self, period: str = "5d", auto_adjust: bool = False) -> pd.DataFrame:
            if self._symbol == "^TNX":
                return pd.DataFrame()
            if self._symbol == "^TYX":
                return pd.DataFrame({"Close": [-99.0]})  # a quote no curve should accept
            return super().history(period, auto_adjust)

    _install_stub(monkeypatch, _PartlyBroken)
    curve = data_mod.load_rate_curve(cache_dir=tmp_path, force_refresh=True)
    assert np.allclose(curve.tenors, [0.25, 5.0])
