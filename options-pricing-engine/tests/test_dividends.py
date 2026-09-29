"""Dividend history, the SPY ex-date rule, the projection, and schedules."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import sys
import types
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from optpricing import dividends as dv

NY = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[1]
DIVIDEND_DIR = REPO_ROOT / "data" / "dividends"


def _history(dates: list[str], amounts: list[float]) -> pd.DataFrame:
    stamps = pd.to_datetime([f"{d} 09:30" for d in dates]).tz_localize(NY)
    return dv.history_frame(pd.Series(amounts, index=stamps))


# ------------------------------------------------------------------------ ex-date rule


@pytest.mark.parametrize(
    ("year", "month", "expected"),
    [
        (2026, 9, dt.date(2026, 9, 18)),  # plain third Friday
        (2026, 6, dt.date(2026, 6, 18)),  # Juneteenth on the Friday -> Thursday
        (2027, 6, dt.date(2027, 6, 17)),  # June 19 is a Saturday, observed Friday 18th
        (2008, 3, dt.date(2008, 3, 20)),  # Good Friday -> Thursday
        (2021, 6, dt.date(2021, 6, 18)),  # before Juneteenth was an exchange holiday
        (2028, 6, dt.date(2028, 6, 16)),
    ],
)
def test_quarterly_ex_date_rule(year: int, month: int, expected: dt.date) -> None:
    assert dv.quarterly_ex_date(year, month) == expected


def test_quarterly_ex_date_rejects_non_quarter_month() -> None:
    with pytest.raises(ValueError, match="March, June"):
        dv.quarterly_ex_date(2026, 5)


def test_easter_known_dates() -> None:
    assert dv.easter_sunday(2008) == dt.date(2008, 3, 23)
    assert dv.easter_sunday(2026) == dt.date(2026, 4, 5)
    assert dv.easter_sunday(2028) == dt.date(2028, 4, 16)


def test_third_friday() -> None:
    assert dv.third_friday(2026, 12) == dt.date(2026, 12, 18)
    assert dv.third_friday(2027, 10) == dt.date(2027, 10, 15)


def test_rule_check_flags_off_cycle_dividends() -> None:
    h = _history(["2004-11-15", "2026-09-18", "2026-06-19"], [3.0, 1.9, 1.9])
    check = dv.ex_date_rule_check(h, since_year=2000)
    assert check["matches"].tolist() == [False, False, True]  # special; wrong day; ok


# ------------------------------------------------------------------------- projection


def _four_quarters() -> pd.DataFrame:
    return _history(
        ["2025-09-19", "2025-12-19", "2026-03-20", "2026-06-18", "2026-09-18", "2026-12-18"],
        [1.80, 1.99, 1.79, 1.90, 1.88, 9.99],
    )


def test_projection_repeats_last_four_on_rule_dates_without_look_ahead() -> None:
    asof = dt.datetime(2026, 9, 18, 11, 0, tzinfo=NY)
    proj = dv.project_dividends(_four_quarters(), asof, dt.date(2027, 12, 17))
    assert proj["ex_date"].tolist() == [
        dt.date(2026, 12, 18),
        dt.date(2027, 3, 19),
        dt.date(2027, 6, 17),
        dt.date(2027, 9, 17),
        dt.date(2027, 12, 17),
    ]
    # The 9.99 goes ex after asof, so it must not be used; Decembers repeat 1.99.
    assert proj["amount"].tolist() == [1.99, 1.79, 1.90, 1.88, 1.99]
    assert all(t.hour == 9 and t.minute == 30 for t in proj["ex_time"])


def test_projection_before_the_morning_ex_time_excludes_that_dividend() -> None:
    asof = dt.datetime(2026, 9, 18, 9, 0, tzinfo=NY)  # before the 09:30 ex time
    proj = dv.project_dividends(_four_quarters(), asof, dt.date(2026, 9, 30))
    assert proj["ex_date"].tolist() == [dt.date(2026, 9, 18)]
    assert proj["amount"].tolist() == [1.80]


def test_projection_needs_one_dividend_per_quarter() -> None:
    h = _history(["2026-03-20", "2026-06-18", "2026-09-18"], [1.8, 1.9, 1.9])
    with pytest.raises(ValueError, match="one quarter each"):
        dv.project_dividends(h, dt.datetime(2026, 9, 19, tzinfo=NY), dt.date(2027, 1, 1))
    with pytest.raises(ValueError, match="timezone"):
        dv.project_dividends(h, dt.datetime(2026, 9, 19), dt.date(2027, 1, 1))


def test_trailing_growth() -> None:
    h = _history(
        [f"{y}-{m:02d}-15" for y in (2025, 2026) for m in (3, 6, 9, 12)], [1.0] * 4 + [1.1] * 4
    )
    assert dv.trailing_growth(h, dt.datetime(2027, 1, 1, tzinfo=NY)) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="eight"):
        dv.trailing_growth(h, dt.datetime(2026, 1, 1, tzinfo=NY))


def test_history_frame_reads_the_raw_csv_layout() -> None:
    raw = pd.DataFrame(
        {"Date": ["2026-06-18 09:30:00-04:00", "2026-03-20 09:30:00-04:00"], "Dividends": [2, 1]}
    )
    h = dv.history_frame(raw)
    assert h["ex_date"].tolist() == [dt.date(2026, 3, 20), dt.date(2026, 6, 18)]
    assert h["amount"].tolist() == [1.0, 2.0]


# -------------------------------------------------------------------------- schedules


def test_schedule_within_and_present_value() -> None:
    s = dv.DividendSchedule(np.array([-0.1, 0.25, 0.5, 1.0]), np.array([9.0, 1.0, 2.0, 3.0]))
    inside = s.within(0.75)
    assert inside.times.tolist() == [0.25, 0.5]
    expected = 1.0 * np.exp(-0.04 * 0.25) + 2.0 * np.exp(-0.04 * 0.5)
    assert s.present_value(0.75, 0.04) == pytest.approx(expected, rel=1e-15)
    assert len(dv.DividendSchedule.empty()) == 0
    assert dv.DividendSchedule.empty().present_value(1.0, 0.05) == 0.0


@pytest.mark.parametrize(
    ("times", "amounts", "match"),
    [
        ([0.1, 0.2], [1.0], "same length"),
        ([0.2, 0.1], [1.0, 1.0], "increasing"),
        ([0.1], [-1.0], "non-negative"),
        ([np.nan], [1.0], "finite"),
    ],
)
def test_schedule_validation(times: list[float], amounts: list[float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        dv.DividendSchedule(np.array(times), np.array(amounts))


def test_schedule_from_frame_uses_the_expiry_clock() -> None:
    asof = dt.datetime(2026, 9, 18, 11, 0, tzinfo=NY)
    frame = pd.DataFrame(
        {
            "ex_time": pd.to_datetime(["2026-09-18 09:30", "2026-12-18 09:30"]).tz_localize(NY),
            "amount": [1.9, 2.0],
        }
    )
    s = dv.DividendSchedule.from_frame(frame, asof)
    assert s.amounts.tolist() == [2.0]  # the morning's dividend is already ex
    hours = (91 * 24 - 1.5 + 1.0) / (365 * 24)  # 91 days, 11:00 EDT -> 09:30 EST
    assert s.times[0] == pytest.approx(hours, rel=1e-12)
    with pytest.raises(ValueError, match="timezone"):
        dv.DividendSchedule.from_frame(frame, asof.replace(tzinfo=None))


# ----------------------------------------------------------------- committed snapshot


def test_committed_snapshot_matches_its_manifest() -> None:
    manifest = json.loads((DIVIDEND_DIR / "SPY_dividends_manifest.json").read_text())
    asof = dt.datetime.fromisoformat(manifest["projection"]["asof"])
    schedule, frame, loaded = dv.load_committed_schedule(asof)
    assert loaded == manifest
    assert len(schedule) == len(frame) == manifest["files"][manifest["projection"]["file"]]["rows"]
    # The projection reaches the snapshot's longest expiry (21 months) and no further.
    assert frame["ex_date"].max() <= dt.date.fromisoformat(manifest["projection"]["until"])
    # Rebuilding the projection from the committed raw download reproduces it exactly.
    raw_name = next(n for n in manifest["files"] if n != manifest["projection"]["file"])
    history = dv.history_frame(pd.read_csv(DIVIDEND_DIR / raw_name))
    again = dv.project_dividends(
        history, asof, dt.date.fromisoformat(manifest["projection"]["until"])
    )
    assert again["amount"].tolist() == frame["amount"].tolist()
    assert again["ex_date"].tolist() == frame["ex_date"].tolist()


def test_committed_snapshot_refuses_tampering_and_wrong_asof(tmp_path: Path) -> None:
    shutil.copytree(DIVIDEND_DIR, tmp_path / "d")
    manifest = json.loads((tmp_path / "d" / "SPY_dividends_manifest.json").read_text())
    asof = dt.datetime.fromisoformat(manifest["projection"]["asof"])
    with pytest.raises(ValueError, match="not"):
        dv.load_committed_schedule(asof + dt.timedelta(days=1), tmp_path / "d")
    proj = tmp_path / "d" / manifest["projection"]["file"]
    proj.write_text(proj.read_text().replace("1.9", "2.9"))
    with pytest.raises(ValueError, match="sha256"):
        dv.load_committed_schedule(asof, tmp_path / "d")


def test_download_uses_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    series = pd.Series([1.0, 2.0], index=pd.to_datetime(["2026-03-20", "2026-06-18"]))

    class _Ticker:
        def __init__(self, symbol: str) -> None:
            self.dividends = series if symbol == "SPY" else pd.Series(dtype=float)

    stub = types.ModuleType("yfinance")
    stub.Ticker = _Ticker  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", stub)
    assert dv.download_dividends("SPY").tolist() == [1.0, 2.0]
    with pytest.raises(RuntimeError, match="no dividend history"):
        dv.download_dividends("NONE")
