"""Download SPY's dividend history and commit it with a projection and a manifest.

Run with::

    python scripts/download_dividends.py            # downloads (needs the network)
    python scripts/download_dividends.py --from-raw data/dividends/SPY_dividends_<date>.csv

Writes to ``data/dividends/``:

* ``<TICKER>_dividends_<download date>.csv`` -- the raw ``yfinance`` download, as returned.
* ``<TICKER>_dividend_projection_<snapshot date>.csv`` -- the dividends projected from the
  option snapshot's valuation time through its longest expiry
  (:func:`optpricing.dividends.project_dividends`).
* ``<TICKER>_dividends_manifest.json`` -- source, download time, sha256 of both files, the
  projection rule and how well the ex-date rule reproduces the history.

``--from-raw`` rebuilds the projection and manifest from a committed raw file without the
network (the download time is then taken from the existing manifest).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from optpricing import data, dividends  # noqa: E402


def _latest_snapshot(ticker: str) -> Path:
    candidates = sorted((REPO_ROOT / "data" / "snapshots").glob(f"{ticker}_*.csv"))
    if not candidates:
        raise SystemExit(f"no committed {ticker} option snapshot in data/snapshots/")
    return candidates[-1]


def main() -> int:
    """Download (or reload), project, and write the snapshot. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--snapshot", type=Path, default=None, help="option snapshot CSV")
    parser.add_argument("--out", type=Path, default=dividends.DEFAULT_DIVIDEND_DIR)
    parser.add_argument("--from-raw", type=Path, default=None, help="reuse a raw download")
    args = parser.parse_args()
    ticker = args.ticker.upper()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / f"{ticker}_dividends_manifest.json"

    snap = data.ChainSnapshot.from_csv(args.snapshot or _latest_snapshot(ticker))
    until = max(snap.expiries())

    if args.from_raw is None:
        series = dividends.download_dividends(ticker)
        downloaded_at = dt.datetime.now(dt.UTC).replace(microsecond=0)
        raw_path = out / f"{ticker}_dividends_{downloaded_at:%Y-%m-%d}.csv"
        series.to_csv(raw_path)
        download = {
            "downloaded_at_utc": downloaded_at.isoformat(),
            "yfinance_version": metadata.version("yfinance"),
        }
    else:
        raw_path = args.from_raw
        previous = json.loads(manifest_path.read_text())
        download = {k: previous[k] for k in ("downloaded_at_utc", "yfinance_version")}

    history = dividends.history_frame(pd.read_csv(raw_path))
    projection = dividends.project_dividends(history, snap.asof, until)
    proj_path = out / f"{ticker}_dividend_projection_{snap.asof:%Y-%m-%d}.csv"
    projection.assign(ex_time=projection["ex_time"].map(lambda t: t.isoformat())).to_csv(
        proj_path, index=False
    )

    rule = dividends.ex_date_rule_check(history, since_year=2000)
    known = history.loc[history["ex_time"] <= pd.Timestamp(snap.asof)]
    manifest: dict[str, Any] = {
        "ticker": ticker,
        "source": f'Yahoo Finance via yfinance: Ticker("{ticker}").dividends',
        **download,
        "files": {
            raw_path.name: {
                "sha256": dividends.sha256_of(raw_path),
                "rows": len(history),
                "content": "raw download: ex-date (09:30 New York) and cash amount per share",
            },
            proj_path.name: {
                "sha256": dividends.sha256_of(proj_path),
                "rows": len(projection),
                "content": "projected ex-dates and amounts after the option snapshot",
            },
        },
        "projection": {
            "file": proj_path.name,
            "asof": snap.asof.isoformat(),
            "option_snapshot": (args.snapshot or _latest_snapshot(ticker)).name,
            "until": until.isoformat(),
            "rule": (
                "repeat the last four quarterly amounts known at asof, each on the same "
                "quarter of later years, no growth; ex-date = third Friday of Mar/Jun/Sep/"
                "Dec, or the business day before when it is Good Friday or Juneteenth "
                "(observed, from 2022)"
            ),
            "last_known_ex_dates": [str(d) for d in known["ex_date"].tail(4)],
            "trailing_year_growth": dividends.trailing_growth(history, snap.asof),
        },
        "ex_date_rule_check": {
            "since": 2000,
            "dividends": len(rule),
            "matched": int(rule["matches"].sum()),
            "mismatches": [
                {"ex_date": str(r.ex_date), "rule_date": str(r.rule_date)}
                for r in rule.loc[~rule["matches"]].itertuples()
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"raw download: {raw_path.name} ({len(history)} dividends)")
    print(f"projection through {until}:")
    print(projection.to_string(index=False))
    check = manifest["ex_date_rule_check"]
    print(f"ex-date rule reproduces {check['matched']}/{check['dividends']} ex-dates since 2000")
    print(f"trailing-year growth: {manifest['projection']['trailing_year_growth']:+.2%}")
    print(f"wrote {manifest_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
