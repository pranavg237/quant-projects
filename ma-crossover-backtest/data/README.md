# data/

## `snapshot-2026-09-28/` (committed): what every reported number is computed from

| | |
|---|---|
| Prices | Yahoo Finance daily bars via `yfinance` (`auto_adjust=False`, `actions=True`), downloaded 2026-09-28, 100 symbols, `prices/<SYMBOL>.csv.gz` |
| Columns | `Open, Close, Adj Close, Volume, Dividends, Stock Splits`, at full float64 precision. `High`/`Low` are not stored because no code reads them. They load as NaN, and a strategy asking for them gets an error |
| Dates | SPY from 1998-01-01, everything else from 2003-01-01 (the earliest date any committed script loads, warm-up included), through 2026-09-28. The analysis ends 2025-08-29; the later rows stay because post-end splits feed the as-traded price |
| Factors | Ken French daily files (FF3, FF5 2x3, momentum), byte-for-byte as downloaded, `french/` |
| Integrity | `MANIFEST.json` records source, download time, date range and a sha256 for every file |
| Size | 12.6 MB |

`quantbt.data.load_yahoo` and `quantbt.factors.load_factors` read the snapshot by default.
They check each file's hash and never touch the network. A symbol that is not in the
snapshot, a hash mismatch, an end date past the snapshot, or a start before a symbol's trim
date all raise instead of downloading or returning a shorter series.

Two full runs of `scripts/run_strategies.py` from the snapshot produce byte-identical
reports, apart from the recorded runtime.

## Live data and rebuilding the snapshot

Every script takes `--live-data` (or set `QUANTBT_LIVE_DATA=1`) to download from Yahoo and
Ken French into `cache/` instead. `cache/` is git-ignored. Results computed that way
will not match the committed ones: Yahoo restates adjusted prices, and one strategy
(`mean_reversion`) is sensitive to its seventh significant digit (see RESULTS.md,
"Reproducibility").

To freeze a new vintage:

```bash
python scripts/download_data.py            # always live: fills data/cache/
python scripts/build_snapshot.py --out data/snapshot-YYYY-MM-DD
```

Then point `SNAPSHOT_DIR` in `quantbt/data.py` at the new folder and regenerate every
report. `build_snapshot.py` is deterministic: the same cache gives byte-identical files.
