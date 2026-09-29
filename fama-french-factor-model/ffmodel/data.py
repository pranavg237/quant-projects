"""Download, parse and cache datasets from Kenneth French's data library.

Every file in the library is a zipped CSV holding several tables (monthly,
annual, value- vs equal-weighted, ...), each introduced by a header row that
starts with a comma. ``parse_french_csv`` splits a file into those tables and
the ``load_*`` helpers pick the right one and convert percent to decimals.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import requests

from .models import get_model

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
CACHE_DIR = Path(os.environ.get("FFMODEL_CACHE", str(Path.home() / ".cache" / "ffmodel")))
CACHE_MAX_AGE_DAYS = 7
MISSING_VALUES = (-99.99, -999.0)
FREQUENCIES = ("monthly", "daily")

FACTOR_FILES = {
    ("ff3", "monthly"): "F-F_Research_Data_Factors",
    ("ff3", "daily"): "F-F_Research_Data_Factors_daily",
    ("ff5", "monthly"): "F-F_Research_Data_5_Factors_2x3",
    ("ff5", "daily"): "F-F_Research_Data_5_Factors_2x3_daily",
    ("mom", "monthly"): "F-F_Momentum_Factor",
    ("mom", "daily"): "F-F_Momentum_Factor_daily",
}

_DATA_ROW = re.compile(r"^\s*\d{4}(?:\d{2}){0,2}\s*,")
_FREQ_BY_WIDTH = {4: "annual", 6: "monthly", 8: "daily"}
_FORMAT_BY_WIDTH = {4: "%Y", 6: "%Y%m", 8: "%Y%m%d"}


@dataclass
class FrenchTable:
    """One table from a French library file, with values as published (usually percent)."""

    title: str
    frequency: str
    data: pd.DataFrame


def parse_french_csv(text: str) -> List[FrenchTable]:
    """Split the text of a French library CSV into its tables.

    A table's title is the last paragraph of text before its header row.
    """
    lines = text.splitlines()
    tables: List[FrenchTable] = []
    paragraph: List[str] = []
    new_paragraph = True
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
            paragraph, new_paragraph = [], True
            continue
        if not line:
            new_paragraph = True
        else:
            if new_paragraph:
                paragraph, new_paragraph = [], False
            paragraph.append(line)
        i += 1
    return tables


def _build_table(title: str, columns: List[str], block: List[str]) -> Optional[FrenchTable]:
    frame = pd.read_csv(io.StringIO("\n".join(block)), header=None, dtype={0: str}, skipinitialspace=True)
    dates = frame.iloc[:, 0].str.strip()
    width = len(dates.iloc[0])
    if width not in _FREQ_BY_WIDTH:
        return None
    # Rows can end with a trailing comma, which shows up as an unnamed column.
    keep = [j for j, name in enumerate(columns[: frame.shape[1] - 1]) if name]
    values = frame.iloc[:, [j + 1 for j in keep]].astype(float)
    values.columns = [columns[j] for j in keep]
    values = values.mask(values.isin(MISSING_VALUES))

    index = pd.to_datetime(dates, format=_FORMAT_BY_WIDTH[width])
    if width == 6:
        index = index + pd.offsets.MonthEnd(0)
    elif width == 4:
        index = index + pd.offsets.YearEnd(0)
    values.index = pd.DatetimeIndex(index, name="date")
    return FrenchTable(title, _FREQ_BY_WIDTH[width], values)


def fetch_dataset(
    name: str, refresh: bool = False, cache_dir: Optional[Path] = None, data_dir: Optional[Path] = None
) -> List[FrenchTable]:
    """Return the parsed tables of ``<name>_CSV.zip``, downloading it if the cache is stale.

    With ``data_dir`` (a saved snapshot, see ``save_snapshot``) the file is read from that
    directory and nothing is downloaded, so results are reproducible offline. If the
    directory has a ``manifest.json`` that lists the file, the file's SHA-256 must match it.
    """
    if data_dir is not None:
        path = Path(data_dir) / f"{name}_CSV.zip"
        if not path.exists():
            raise RuntimeError(f"{name}_CSV.zip is not in the data snapshot {data_dir}")
        verify_snapshot_file(path)
        return _read_zip(path)
    path = Path(cache_dir or CACHE_DIR) / f"{name}_CSV.zip"
    stale = not path.exists() or time.time() - path.stat().st_mtime > CACHE_MAX_AGE_DAYS * 86400
    if refresh or stale:
        try:
            _download(name, path)
        except (requests.RequestException, zipfile.BadZipFile) as exc:
            if not path.exists():
                raise RuntimeError(f"Could not download {name!r} from the French data library: {exc}") from exc
            warnings.warn(f"Could not refresh {name!r} ({exc}); using the cached copy.")
    return _read_zip(path)


def verify_snapshot_file(path: Path) -> None:
    """Raise if ``path`` is listed in its directory's ``manifest.json`` with a different SHA-256."""
    manifest = path.parent / "manifest.json"
    if not manifest.exists():
        return
    entry = json.loads(manifest.read_text()).get("french_library", {}).get(path.name)
    if entry is None:
        return
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(f"{path} does not match the SHA-256 in its manifest "
                           f"(file {digest}, manifest {entry['sha256']})")


def portfolio_file_name(dataset: str, frequency: str = "monthly") -> str:
    """The French library file name (without ``_CSV.zip``) of a test-portfolio set at ``frequency``."""
    if frequency == "daily" and not dataset.lower().endswith("_daily"):
        return dataset + "_Daily"
    return dataset


def _read_zip(path: Path) -> List[FrenchTable]:
    with zipfile.ZipFile(path) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        text = zf.read(member).decode("latin-1")
    return parse_french_csv(text)


def _download(name: str, path: Path) -> None:
    response = requests.get(f"{BASE_URL}{name}_CSV.zip", timeout=60)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(response.content)
        zipfile.ZipFile(tmp).close()  # raises BadZipFile on an error page
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def load_factors(
    model: str = "ff5",
    frequency: str = "monthly",
    start: Optional[str] = None,
    end: Optional[str] = None,
    refresh: bool = False,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Factor returns for ``model`` plus the risk-free rate ``RF``, in decimals.

    ``data_dir`` reads the French files from a saved snapshot instead of downloading them.
    """
    spec = get_model(model)
    _check_frequency(frequency)
    parts = [_factor_table(spec.source, frequency, refresh, data_dir)]
    if "MOM" in spec.factors:
        parts.append(_factor_table("mom", frequency, refresh, data_dir))
    factors = pd.concat(parts, axis=1, join="inner")[list(spec.factors) + ["RF"]]
    return factors.loc[start:end].dropna()


def load_portfolios(
    dataset: str = "25_Portfolios_5x5",
    frequency: str = "monthly",
    weighting: str = "value",
    start: Optional[str] = None,
    end: Optional[str] = None,
    refresh: bool = False,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Raw (not excess) returns of a French test-portfolio set, in decimals.

    ``dataset`` is the file name without ``_CSV.zip``, e.g. ``25_Portfolios_5x5``,
    ``6_Portfolios_2x3``, ``10_Industry_Portfolios`` or ``49_Industry_Portfolios``.
    """
    _check_frequency(frequency)
    if weighting not in ("value", "equal"):
        raise ValueError("weighting must be 'value' or 'equal'")
    name = portfolio_file_name(dataset, frequency)
    table = _select(fetch_dataset(name, refresh=refresh, data_dir=data_dir), frequency, (f"{weighting} weight", "return"))
    return table.data.loc[start:end] / 100.0


def _factor_table(key: str, frequency: str, refresh: bool, data_dir: Optional[Path] = None) -> pd.DataFrame:
    table = _select(fetch_dataset(FACTOR_FILES[(key, frequency)], refresh=refresh, data_dir=data_dir), frequency)
    return table.data.rename(columns=lambda c: "MOM" if c.upper() == "MOM" else c) / 100.0


def _select(tables: List[FrenchTable], frequency: str, title_contains: Sequence[str] = ()) -> FrenchTable:
    for table in tables:
        if table.frequency == frequency and all(s in table.title.lower() for s in title_contains):
            return table
    available = "; ".join(f"[{t.frequency}] {t.title[:70]}" for t in tables)
    raise ValueError(f"No {frequency} table matching {list(title_contains)}. Available: {available}")


def _check_frequency(frequency: str) -> None:
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}")
