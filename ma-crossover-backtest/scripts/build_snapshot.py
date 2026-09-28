"""Freeze the raw download cache into the committed snapshot under data/snapshot-<date>/.

Every reported number is produced from this snapshot, so it has to be exactly what the
pipeline reads and nothing that could change silently:

* prices/<SYMBOL>.csv.gz: Yahoo's raw columns at full precision (Open, Close, Adj Close,
  Volume, Dividends, Stock Splits). High and Low are dropped because no code reads them.
  History starts at the earliest date any committed script loads for that symbol
  (1998-01-01 for SPY, which the MA crossover walk-forward warms up from; 2003-01-01
  for everything else, two years of warm-up before the 2005 start) and runs to the
  download date, because Yahoo's split and dividend columns after the analysis end still
  feed the as-traded price reconstruction.
* french/<name>_CSV.zip: the Ken French daily factor files, byte-for-byte as downloaded.
* MANIFEST.json: source, download times, date ranges and a sha256 for every file.

Usage: python scripts/build_snapshot.py [--cache data/cache] [--out data/snapshot-2026-09-28]
Run scripts/download_data.py --live-data first to fill the cache.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

import pandas as pd

from quantbt.data import sha256_file
from quantbt.factors.french import FACTOR_FILES
from quantbt.research.universes import ALL_SYMBOLS

EARLIEST = {"SPY": "1998-01-01"}  # scripts/run_strategies.py, walk_forward_ma.py, ma_sensitivity.py
DEFAULT_START = "2003-01-01"  # 2005 strategy start minus the runner's 2-year warm-up
COLUMNS = ["Open", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
FRENCH = sorted({name for (_, freq), name in FACTOR_FILES.items() if freq == "daily"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="data/cache")
    parser.add_argument("--out", default="data/snapshot-2026-09-28")
    args = parser.parse_args()
    cache, out = Path(args.cache), Path(args.out)
    cache_manifest = json.loads((cache / "MANIFEST.json").read_text())
    (out / "prices").mkdir(parents=True, exist_ok=True)
    (out / "french").mkdir(parents=True, exist_ok=True)

    files: dict[str, dict[str, object]] = {}
    for symbol in sorted(ALL_SYMBOLS):
        raw = pd.read_csv(cache / f"{symbol}.csv", index_col=0, parse_dates=True)
        start = EARLIEST.get(symbol, DEFAULT_START)
        frame = raw.loc[start:, COLUMNS]
        frame.index.name = "Date"
        rel = f"prices/{symbol}.csv.gz"
        text = frame.to_csv(date_format="%Y-%m-%d").encode()
        # mtime=0 and a fixed level make the archive byte-identical on every rebuild
        (out / rel).write_bytes(gzip.compress(text, compresslevel=9, mtime=0))
        check = pd.read_csv(out / rel, index_col=0, parse_dates=True, float_precision="round_trip")
        pd.testing.assert_frame_equal(check, frame, check_exact=True, check_freq=False)
        files[rel] = {
            "sha256": sha256_file(out / rel),
            "rows": len(frame),
            "first_date": str(frame.index[0].date()),
            "last_date": str(frame.index[-1].date()),
            "source_first_date": str(raw.index[0].date()),
            "downloaded_at": cache_manifest[symbol]["downloaded_at"],
        }
    for name in FRENCH:
        rel = f"french/{name}_CSV.zip"
        shutil.copyfile(cache / "french" / f"{name}_CSV.zip", out / rel)
        files[rel] = {"sha256": sha256_file(out / rel)}

    manifest = {
        "source": {
            "prices": "Yahoo Finance daily bars via yfinance (auto_adjust=False, actions=True)",
            "factors": "Kenneth French data library, daily files",
        },
        "download_date": "2026-09-28",
        "analysis_end": "2025-08-29",
        "columns": COLUMNS,
        "dropped": "High and Low (no code reads them; they load as NaN)",
        "symbols": sorted(ALL_SYMBOLS),
        "files": files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"{len(files)} files, {total / 1e6:.2f} MB written to {out}")


if __name__ == "__main__":
    main()
