r"""Discrete cash dividends: history, the SPY ex-date rule, a projection, and schedules.

SPY pays a cash dividend every quarter. The lattice in :mod:`optpricing.american` can
price American options on it with those dividends as *discrete* cash amounts (the
escrowed-dividend model) instead of a continuous yield; this module supplies them.

**History.** :func:`download_dividends` pulls ex-dates and amounts from Yahoo Finance
(``yfinance.Ticker(...).dividends``); ``scripts/download_dividends.py`` commits the raw
download with a manifest (source, time, sha256) so the pipeline never needs the network.

**The ex-date rule.** SPY goes ex-dividend on the third Friday of March, June, September
and December -- the quarterly option expiration day -- or on the business day before it
when that Friday is an exchange holiday (Good Friday; Juneteenth, observed, since 2022).
:func:`quarterly_ex_date` implements that, and :func:`ex_date_rule_check` measures how
many historical ex-dates it reproduces, so the rule is tested on the data rather than
asserted.

**The projection.** :func:`project_dividends` repeats the last four quarterly amounts
known at the valuation time, each on the same quarter of later years (the December
amount for future Decembers, and so on), on the rule's ex-dates. It is the simplest
projection that keeps SPY's seasonality (December is the largest quarter) and uses no
information after the valuation time. It assumes no growth; the history supplies the
number needed to judge that (the trailing-year growth, reported by the download script).

**Schedules.** A :class:`DividendSchedule` holds ex-dividend *times* in years from the
valuation time (ACT/365 on the same clock as :func:`optpricing.data.year_fraction`) and
cash amounts. An ex-date's time is 09:30 New York time, the open on the ex-date, which
is how Yahoo stamps it: a dividend going ex on an option's expiry day is before the
16:00 expiry and so is paid within the option's life.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .types import FloatArray

__all__ = [
    "DividendSchedule",
    "download_dividends",
    "easter_sunday",
    "ex_date_rule_check",
    "history_frame",
    "load_committed_schedule",
    "project_dividends",
    "quarterly_ex_date",
    "read_projection",
    "sha256_of",
    "third_friday",
    "trailing_growth",
]

NEW_YORK = ZoneInfo("America/New_York")
_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0
_EX_TIME = dt.time(9, 30)  # the open on the ex-date
_QUARTER_MONTHS = (3, 6, 9, 12)


@dataclass(frozen=True)
class DividendSchedule:
    """Cash dividends as ``(time, amount)`` pairs, time in years from the valuation time.

    Attributes:
        times: Ex-dividend times, years from valuation, increasing.
        amounts: Cash amount per share going ex at each time (non-negative).
    """

    times: FloatArray
    amounts: FloatArray

    def __post_init__(self) -> None:
        t = np.atleast_1d(np.asarray(self.times, dtype=np.float64))
        a = np.atleast_1d(np.asarray(self.amounts, dtype=np.float64))
        if t.shape != a.shape or t.ndim != 1:
            raise ValueError("times and amounts must be 1-d arrays of the same length")
        if not (np.all(np.isfinite(t)) and np.all(np.isfinite(a))):
            raise ValueError("dividend times and amounts must be finite")
        if np.any(a < 0.0):
            raise ValueError("dividend amounts must be non-negative")
        if np.any(np.diff(t) < 0.0):
            raise ValueError("dividend times must be increasing")
        object.__setattr__(self, "times", t)
        object.__setattr__(self, "amounts", a)

    @classmethod
    def empty(cls) -> DividendSchedule:
        """No dividends."""
        return cls(np.zeros(0), np.zeros(0))

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, asof: dt.datetime) -> DividendSchedule:
        """Schedule from a frame with ``ex_time`` (tz-aware) and ``amount``.

        Dividends that went ex at or before ``asof`` are dropped: the spot at ``asof``
        is already ex those dividends.
        """
        if asof.tzinfo is None:
            raise ValueError("asof must be timezone-aware")
        ex = pd.to_datetime(frame["ex_time"], utc=True)
        seconds = (ex - pd.Timestamp(asof).tz_convert("UTC")).dt.total_seconds()
        t = np.asarray(seconds.to_numpy(dtype=np.float64) / _SECONDS_PER_YEAR, dtype=np.float64)
        a: FloatArray = frame["amount"].to_numpy(dtype=np.float64)
        keep = t > 0.0
        order = np.argsort(t[keep], kind="stable")
        return cls(t[keep][order], a[keep][order])

    def within(self, tau: float) -> DividendSchedule:
        """The dividends that go ex strictly after now and strictly before ``tau``."""
        keep = (self.times > 0.0) & (self.times < tau)
        return DividendSchedule(self.times[keep], self.amounts[keep])

    def present_value(self, tau: float, rate: float) -> float:
        r""":math:`\sum_{0 < t_i < \tau} D_i e^{-r t_i}`: what is escrowed out of the spot."""
        d = self.within(tau)
        return float(np.sum(d.amounts * np.exp(-rate * d.times)))

    def __len__(self) -> int:
        return int(self.times.size)


# ---------------------------------------------------------------------------------------
# The quarterly ex-date rule
# ---------------------------------------------------------------------------------------


def third_friday(year: int, month: int) -> dt.date:
    """The third Friday of ``month``: the standard monthly option expiration day."""
    first = dt.date(year, month, 1)
    offset = (4 - first.weekday()) % 7  # Friday is weekday 4
    return first + dt.timedelta(days=offset + 14)


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter Sunday (the anonymous / Meeus-Jones-Butcher algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def _exchange_holidays(year: int) -> set[dt.date]:
    """The NYSE holidays that can fall on a quarterly third Friday (days 15-21).

    New Year, Independence Day and Christmas cannot (wrong days of the month); the
    Monday holidays cannot (wrong weekday). That leaves Good Friday and, since 2022,
    Juneteenth -- observed on the Friday when June 19 is a Saturday.
    """
    out = {easter_sunday(year) - dt.timedelta(days=2)}
    if year >= 2022:
        june19 = dt.date(year, 6, 19)
        if june19.weekday() == 5:
            out.add(june19 - dt.timedelta(days=1))
        elif june19.weekday() == 6:
            out.add(june19 + dt.timedelta(days=1))
        else:
            out.add(june19)
    return out


def quarterly_ex_date(year: int, month: int) -> dt.date:
    """SPY's ex-date for a quarter: the third Friday, or the business day before it."""
    if month not in _QUARTER_MONTHS:
        raise ValueError(f"SPY goes ex in March, June, September and December, not {month}")
    day = third_friday(year, month)
    holidays = _exchange_holidays(year)
    while day in holidays or day.weekday() >= 5:
        day -= dt.timedelta(days=1)
    return day


def _ex_time(day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, _EX_TIME, tzinfo=NEW_YORK))


# ---------------------------------------------------------------------------------------
# History, projection
# ---------------------------------------------------------------------------------------


def download_dividends(ticker: str = "SPY") -> pd.Series:
    """Every dividend Yahoo Finance has for ``ticker``: a Series indexed by ex-date time.

    Raises:
        RuntimeError: if the download comes back empty.
    """
    import yfinance as yf  # noqa: PLC0415  (local: keeps the network dep off the import path)

    series = yf.Ticker(ticker).dividends
    if series is None or len(series) == 0:
        raise RuntimeError(f"no dividend history returned for {ticker!r}")
    return pd.Series(series, dtype=np.float64)


def history_frame(raw: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Normalise a raw download (Series, or its CSV with ``Date, Dividends``) to a frame.

    Returns:
        ``ex_date`` (date), ``ex_time`` (tz-aware, New York), ``amount``; sorted.
    """
    if isinstance(raw, pd.Series):
        stamps = pd.Series(raw.index)
        amounts = raw.to_numpy(dtype=np.float64)
    else:
        stamps = pd.Series(raw["Date"])
        amounts = raw["Dividends"].to_numpy(dtype=np.float64)
    ts = pd.to_datetime(stamps, utc=True).dt.tz_convert(NEW_YORK)
    frame = pd.DataFrame(
        {"ex_date": ts.dt.date.to_numpy(), "ex_time": ts.to_numpy(), "amount": amounts}
    )
    frame["ex_time"] = pd.to_datetime(frame["ex_time"], utc=True).dt.tz_convert(NEW_YORK)
    return frame.sort_values("ex_time").reset_index(drop=True)


def ex_date_rule_check(history: pd.DataFrame, since_year: int = 2000) -> pd.DataFrame:
    """Compare each historical ex-date (from ``since_year``) with :func:`quarterly_ex_date`.

    Returns:
        ``ex_date, rule_date, matches`` per historical dividend.
    """
    h = history.loc[[d.year >= since_year for d in history["ex_date"]]]
    rule: list[dt.date | None] = []
    for d in h["ex_date"]:
        rule.append(quarterly_ex_date(d.year, d.month) if d.month in _QUARTER_MONTHS else None)
    out = pd.DataFrame({"ex_date": h["ex_date"].to_numpy(), "rule_date": rule})
    out["matches"] = out["ex_date"] == out["rule_date"]
    return out.reset_index(drop=True)


def trailing_growth(history: pd.DataFrame, asof: dt.datetime) -> float:
    """Sum of the last four dividends known at ``asof`` over the four before, minus one."""
    known = history.loc[history["ex_time"] <= pd.Timestamp(asof)]
    if len(known) < 8:
        raise ValueError("need eight dividends before asof for a trailing growth rate")
    last8 = known["amount"].to_numpy(dtype=np.float64)[-8:]
    return float(last8[4:].sum() / last8[:4].sum() - 1.0)


def project_dividends(history: pd.DataFrame, asof: dt.datetime, until: dt.date) -> pd.DataFrame:
    """Project SPY's dividends after ``asof`` through ``until`` (inclusive).

    Uses only dividends that went ex at or before ``asof`` (no look-ahead). Each future
    quarter gets the amount of the same quarter in the last four known dividends, on the
    :func:`quarterly_ex_date` of that quarter.

    Returns:
        ``ex_date, ex_time, amount, basis_ex_date`` (the historical dividend each amount
        repeats).

    Raises:
        ValueError: if the last four known dividends are not one per quarter.
    """
    if asof.tzinfo is None:
        raise ValueError("asof must be timezone-aware")
    known = history.loc[history["ex_time"] <= pd.Timestamp(asof)]
    last4 = known.tail(4)
    months = [d.month for d in last4["ex_date"]]
    if len(last4) < 4 or sorted(months) != list(_QUARTER_MONTHS):
        raise ValueError(
            f"the last four dividends before {asof:%Y-%m-%d} must cover one quarter each; "
            f"got months {months}"
        )
    by_month = {
        d.month: (d, float(a)) for d, a in zip(last4["ex_date"], last4["amount"], strict=True)
    }
    last_date: dt.date = last4["ex_date"].iloc[-1]
    rows: list[dict[str, object]] = []
    year, idx = last_date.year, _QUARTER_MONTHS.index(last_date.month)
    while True:
        idx += 1
        if idx == len(_QUARTER_MONTHS):
            idx, year = 0, year + 1
        month = _QUARTER_MONTHS[idx]
        day = quarterly_ex_date(year, month)
        if day > until:
            break
        basis, amount = by_month[month]
        rows.append(
            {"ex_date": day, "ex_time": _ex_time(day), "amount": amount, "basis_ex_date": basis}
        )
    frame = pd.DataFrame(rows, columns=["ex_date", "ex_time", "amount", "basis_ex_date"])
    if not frame.empty:
        frame["ex_time"] = pd.to_datetime(frame["ex_time"], utc=True).dt.tz_convert(NEW_YORK)
    return frame


# ---------------------------------------------------------------------------------------
# The committed snapshot
# ---------------------------------------------------------------------------------------

#: Where ``scripts/download_dividends.py`` commits the raw download and the projection.
DEFAULT_DIVIDEND_DIR = Path(__file__).resolve().parents[2] / "data" / "dividends"


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_projection(path: Path) -> pd.DataFrame:
    """Load a projection CSV written by ``scripts/download_dividends.py``."""
    frame = pd.read_csv(path)
    frame["ex_date"] = pd.to_datetime(frame["ex_date"]).dt.date
    frame["ex_time"] = pd.to_datetime(frame["ex_time"], utc=True).dt.tz_convert(NEW_YORK)
    return frame


def load_committed_schedule(
    asof: dt.datetime, directory: Path | None = None, ticker: str = "SPY"
) -> tuple[DividendSchedule, pd.DataFrame, dict[str, Any]]:
    """The committed dividend projection for a snapshot taken at ``asof``.

    Reads ``<ticker>_dividends_manifest.json``, checks that its projection was made for
    this ``asof`` and that every file's sha256 still matches, and returns the schedule.

    Returns:
        ``(schedule, projection_frame, manifest)``.

    Raises:
        FileNotFoundError: if there is no manifest.
        ValueError: if the projection is for another valuation time or a hash differs.
    """
    directory = directory or DEFAULT_DIVIDEND_DIR
    manifest_path = directory / f"{ticker.upper()}_dividends_manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())
    stamp = dt.datetime.fromisoformat(str(manifest["projection"]["asof"]))
    if stamp != asof:
        raise ValueError(
            f"the committed projection is for {stamp.isoformat()}, not {asof.isoformat()}"
        )
    for name, meta in manifest["files"].items():
        if sha256_of(directory / name) != meta["sha256"]:
            raise ValueError(f"{name} does not match its manifest sha256")
    frame = read_projection(directory / manifest["projection"]["file"])
    return DividendSchedule.from_frame(frame, asof), frame, manifest
