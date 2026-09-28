"""Save the downloaded inputs of an analysis so it can be re-run offline.

A snapshot directory holds the French library factor files exactly as downloaded
(``<name>_CSV.zip``), the asset returns as ``returns.csv`` and a ``manifest.json``
recording when and from where everything was downloaded. Pass the directory as
``data_dir`` (``--data-dir`` on the command line) and ``returns.csv`` as the
returns CSV to reproduce a report without touching the network.
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
    tickers: Sequence[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: str = "monthly",
) -> Dict:
    """Download the factor files and Yahoo returns into ``directory`` and write a manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    downloaded = datetime.now(timezone.utc)

    french = {}
    for (key, freq), name in sorted(data.FACTOR_FILES.items()):
        if freq != frequency:
            continue
        path = directory / f"{name}_CSV.zip"
        data._download(name, path)
        table = data._select(data.fetch_dataset(name, data_dir=directory), frequency)
        french[path.name] = {
            "url": f"{data.BASE_URL}{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first": f"{table.data.index[0]:%Y-%m}",
            "last": f"{table.data.index[-1]:%Y-%m}",
        }

    returns = download_returns(tickers, start, end, frequency)
    returns.index.name = "date"
    returns.to_csv(directory / RETURNS_FILE)
    manifest = {
        "downloaded_utc": downloaded.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frequency": frequency,
        "french_library": french,
        "returns": {
            "file": RETURNS_FILE,
            "source": "Yahoo Finance via yfinance, split- and dividend-adjusted closes (auto_adjust=True)",
            "units": "simple returns, decimals, month-end dates" if frequency == "monthly" else "simple returns, decimals",
            "tickers": list(returns.columns),
            "first": f"{returns.index[0]:%Y-%m-%d}",
            "last": f"{returns.index[-1]:%Y-%m-%d}",
            "periods": len(returns),
        },
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(directory: Union[str, Path]) -> Optional[Dict]:
    """The snapshot manifest in ``directory``, or None if there is none."""
    path = Path(directory) / MANIFEST
    return json.loads(path.read_text()) if path.exists() else None
