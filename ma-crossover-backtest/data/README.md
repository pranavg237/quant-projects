# data/

`cache/` (git-ignored) holds one CSV per symbol with the *raw* Yahoo Finance download
(unadjusted OHLC, `Adj Close`, volume, dividends and splits) plus `MANIFEST.json`
recording when each file was fetched and the date range it covers.

Results in this repository are reproducible only against the snapshot recorded in the
manifest: Yahoo back-adjusts its history every ex-dividend date, so a fresh download
shifts every adjusted price by a small constant. Re-run with `refresh=True` (or delete
the cache) to pull current data.
