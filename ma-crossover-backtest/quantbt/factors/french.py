"""Download, parse and cache factor files from Kenneth French's data library.

Each library file is a zipped CSV containing several tables (daily/monthly/annual)
separated by prose. A table starts with a header row that begins with a comma; data
rows start with a date of 4, 6 or 8 digits. Values are in percent and ``-99.99`` /
``-999`` mark missing values. Everything returned here is in decimals with a tz-naive
``DatetimeIndex`` (month-end dates for monthly tables) so it joins directly onto the
price data.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantbt.data import DEFAULT_CACHE_DIR, normalize_index

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
MISSING_VALUES = (-99.99, -999.0)

FACTOR_FILES: dict[tuple[str, str], str] = {
    ("ff3", "daily"): "F-F_Research_Data_Factors_daily",
    ("ff3", "monthly"): "F-F_Research_Data_Factors",
    ("ff5", "daily"): "F-F_Research_Data_5_Factors_2x3_daily",
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3",
    ("mom", "daily"): "F-F_Momentum_Factor_daily",
    ("mom", "monthly"): "F-F_Momentum_Factor",
}
MODELS: dict[str, tuple[str, ...]] = {
    "capm": ("Mkt-RF",),
    "ff3": ("Mkt-RF", "SMB", "HML"),
    "carhart": ("Mkt-RF", "SMB", "HML", "MOM"),
    "ff5": ("Mkt-RF", "SMB", "HML", "RMW", "CMA"),
    "ff6": ("Mkt-RF", "SMB", "HML", "RMW", "CMA", "MOM"),
}

_DATA_ROW = re.compile(r"^\s*(\d{4}|\d{6}|\d{8})\s*,")
_FREQ_BY_WIDTH = {4: "annual", 6: "monthly", 8: "daily"}
_FORMAT_BY_WIDTH = {4: "%Y", 6: "%Y%m", 8: "%Y%m%d"}


@dataclass(frozen=True)
class FrenchTable:
    title: str
    frequency: str
    data: pd.DataFrame  # decimals


def parse_french_csv(text: str) -> list[FrenchTable]:
    """Split a French library CSV into its tables (values converted to decimals)."""
    lines = text.splitlines()
    tables: list[FrenchTable] = []
    paragraph: list[str] = []
    fresh = True
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith(","):
            columns = [c.strip() for c in line.split(",")[1:]]
            start = i + 1
            i = start
            while i < len(lines) and _DATA_ROW.match(lines[i]):
                i += 1
            if i > start:
                table = _build_table(" ".join(paragraph), columns, lines[start:i])
                if table is not None:
                    tables.append(table)
            paragraph, fresh = [], True
            continue
        if not line:
            fresh = True
        else:
            if fresh:
                paragraph, fresh = [], False
            paragraph.append(line)
        i += 1
    return tables


def _build_table(title: str, columns: list[str], block: list[str]) -> FrenchTable | None:
    frame = pd.read_csv(
        io.StringIO("\n".join(block)), header=None, dtype={0: str}, skipinitialspace=True
    )
    dates = frame.iloc[:, 0].astype(str).str.strip()
    width = len(dates.iloc[0])
    if width not in _FREQ_BY_WIDTH:
        return None
    n_values = frame.shape[1] - 1
    keep = [j for j, name in enumerate(columns[:n_values]) if name]
    values = frame.iloc[:, [j + 1 for j in keep]].astype(float)
    values.columns = [columns[j] for j in keep]
    values = values.mask(values.isin(MISSING_VALUES)) / 100.0
    index = pd.to_datetime(dates, format=_FORMAT_BY_WIDTH[width])
    if width == 6:
        index = index + pd.offsets.MonthEnd(0)
    elif width == 4:
        index = index + pd.offsets.YearEnd(0)
    values.index = normalize_index(pd.DatetimeIndex(index))
    values = values.rename(columns=lambda c: "MOM" if c.upper() in ("MOM", "WML") else c)
    return FrenchTable(title=title, frequency=_FREQ_BY_WIDTH[width], data=values)


def _download(name: str) -> bytes:
    import requests

    response = requests.get(f"{BASE_URL}{name}_CSV.zip", timeout=60)
    response.raise_for_status()
    return response.content


def fetch_dataset(
    name: str,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    max_age_days: float = 30.0,
) -> list[FrenchTable]:
    """Parsed tables of ``<name>_CSV.zip``, downloaded unless a fresh cached copy exists."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR / "french"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{name}_CSV.zip"
    stale = not path.exists() or time.time() - path.stat().st_mtime > max_age_days * 86400
    if refresh or stale:
        content = _download(name)
        zipfile.ZipFile(io.BytesIO(content)).close()  # raises BadZipFile on an error page
        path.write_bytes(content)
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(member).decode("latin-1")
    return parse_french_csv(text)


def _select(tables: Sequence[FrenchTable], frequency: str) -> FrenchTable:
    for t in tables:
        if t.frequency == frequency:
            return t
    raise ValueError(f"no {frequency} table; found {[t.frequency for t in tables]}")


def load_factors(
    model: str = "ff3",
    frequency: str = "daily",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Factor returns (decimals) for ``model`` plus ``RF``: columns ``MODELS[model] + ('RF',)``."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {list(MODELS)}")
    if frequency not in ("daily", "monthly"):
        raise ValueError("frequency must be 'daily' or 'monthly'")
    factors = MODELS[model]
    source = "ff5" if any(f in ("RMW", "CMA") for f in factors) else "ff3"
    parts = [
        _select(
            fetch_dataset(FACTOR_FILES[(source, frequency)], cache_dir=cache_dir, refresh=refresh),
            frequency,
        ).data
    ]
    if "MOM" in factors:
        parts.append(
            _select(
                fetch_dataset(
                    FACTOR_FILES[("mom", frequency)], cache_dir=cache_dir, refresh=refresh
                ),
                frequency,
            ).data
        )
    frame = pd.concat(parts, axis=1, join="inner")
    frame = frame.loc[:, ~frame.columns.duplicated()]
    out = frame[[*factors, "RF"]].dropna()
    return out.loc[start:end]
