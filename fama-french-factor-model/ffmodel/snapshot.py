"""Save the downloaded inputs of an analysis so it can be re-run offline.

A snapshot directory holds French library files exactly as downloaded
(``<name>_CSV.zip``: the factor files and any test-portfolio sets asked for), the
asset returns as ``returns.csv`` if tickers were given, and a ``manifest.json``
recording when and from where everything was downloaded, each French file's SHA-256
and its sample period. Pass the directory as ``data_dir`` (``--data-dir`` on the
command line) and ``returns.csv`` as the returns CSV to reproduce a report without
touching the network. Reading a file from a snapshot checks its hash against the
manifest.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from . import data
from .assets import download_returns

MANIFEST = "manifest.json"
RETURNS_FILE = "returns.csv"


def save_snapshot(
    directory: Union[str, Path],
    tickers: Sequence[str] = (),
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: str = "monthly",
    portfolios: Sequence[str] = (),
) -> Dict:
    """Download the factor files, any ``portfolios`` sets and Yahoo returns into ``directory``.

    ``portfolios`` are French test-portfolio file names without ``_CSV.zip`` (e.g.
    ``25_Portfolios_5x5``). With no ``tickers`` the snapshot holds French files only.
    ``start`` and ``end`` limit the returns; French files are always saved whole.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    downloaded = datetime.now(timezone.utc)

    french = {}
    for (_, freq), name in sorted(data.FACTOR_FILES.items()):
        if freq == frequency:
            french.update(_save_french_file(directory, name, frequency))
    for dataset in portfolios:
        name = data.portfolio_file_name(dataset, frequency)
        french.update(_save_french_file(directory, name, frequency, ("value weight", "return")))

    manifest: Dict = {
        "downloaded_utc": downloaded.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frequency": frequency,
        "french_library": french,
    }
    if tickers:
        returns = download_returns(tickers, start, end, frequency)
        returns.index.name = "date"
        returns.to_csv(directory / RETURNS_FILE)
        manifest["returns"] = {
            "file": RETURNS_FILE,
            "source": "Yahoo Finance via yfinance, split- and dividend-adjusted closes (auto_adjust=True)",
            "units": "simple returns, decimals, month-end dates" if frequency == "monthly" else "simple returns, decimals",
            "tickers": list(returns.columns),
            "first": f"{returns.index[0]:%Y-%m-%d}",
            "last": f"{returns.index[-1]:%Y-%m-%d}",
            "periods": len(returns),
        }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _save_french_file(directory: Path, name: str, frequency: str, title_contains: Sequence[str] = ()) -> Dict:
    """Download ``<name>_CSV.zip`` unchanged and describe it for the manifest."""
    path = directory / f"{name}_CSV.zip"
    data._download(name, path)
    table = data._select(data._read_zip(path), frequency, title_contains)
    period = "%Y-%m" if frequency == "monthly" else "%Y-%m-%d"
    entry = {
        "url": f"{data.BASE_URL}{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "first": f"{table.data.index[0]:{period}}",
        "last": f"{table.data.index[-1]:{period}}",
    }
    if title_contains:
        entry["period_of"] = table.title
    return {path.name: entry}


def read_manifest(directory: Union[str, Path]) -> Optional[Dict]:
    """The snapshot manifest in ``directory``, or None if there is none."""
    path = Path(directory) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None


def verify_snapshot(directory: Union[str, Path]) -> Dict[str, bool]:
    """Check every French file listed in the manifest against its SHA-256: {file name: matches}."""
    directory = Path(directory)
    manifest = read_manifest(directory)
    if manifest is None:
        raise RuntimeError(f"{directory} has no {MANIFEST}")
    return {
        name: (directory / name).exists()
        and hashlib.sha256((directory / name).read_bytes()).hexdigest() == entry["sha256"]
        for name, entry in manifest["french_library"].items()
    }
