"""Data layer: OHLCV containers, corporate-action adjustment and cached Yahoo downloads.

Conventions (these are asserted, not assumed):

* Every index is a tz-naive ``DatetimeIndex`` normalised to midnight, strictly increasing
  and unique. Yahoo daily bars are exchange-local dates; the Ken French factor files use
  the same convention, so the two join without any conversion. tz-aware input is
  rejected rather than silently converted, because a daily bar stamped at UTC midnight
  is *not* the same trading day as one stamped at New York midnight.
* ``open/high/low/close/volume`` are total-return adjusted (splits and dividends, the
  CRSP convention). They are what the portfolio marks to and fills against, so
  dividends are implicitly reinvested.
* ``split_adj_close`` is adjusted for splits only. Use it for anything that depends on
  the *level* of the price (dollar thresholds, round lots, price filters). A dividend
  back-adjustment factor depends on dividends paid *after* the bar, so a level-based
  rule computed on ``close`` reads the future. Scale-invariant rules (ratios of moving
  averages, returns, z-scores) are safe on either series.
* ``raw_close`` is the price as it traded on the day.

Yahoo's ``Close`` is already split-adjusted; ``Adj Close`` is split- and dividend-adjusted.
CSV input is expected to follow the same convention (see ``PriceData.from_frames``).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
# The committed, frozen inputs every reported number is produced from (see data/README.md).
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshot-2026-09-28"
LIVE_DATA_ENV = "QUANTBT_LIVE_DATA"
PRICE_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
ALL_FIELDS: tuple[str, ...] = (
    *PRICE_FIELDS,
    "raw_close",
    "split_adj_close",
    "dividends",
    "splits",
)

_YAHOO_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
)


class DataError(ValueError):
    """Raised when input data violates one of the conventions above."""


def normalize_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return ``index`` as a tz-naive, midnight-normalised, strictly increasing DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        try:
            index = pd.DatetimeIndex(pd.to_datetime(index))
        except (ValueError, TypeError) as exc:
            raise DataError(f"index is not datetime-like: {exc}") from exc
    if index.tz is not None:
        raise DataError(
            "tz-aware index rejected: convert to exchange-local dates and drop the tz first "
            "(e.g. idx.tz_convert('America/New_York').tz_localize(None))"
        )
    out = pd.DatetimeIndex(index.normalize())
    if out.has_duplicates:
        dupes = out[out.duplicated()].unique()[:3].tolist()
        raise DataError(f"duplicate dates in index, e.g. {dupes}")
    if not out.is_monotonic_increasing:
        raise DataError("index must be sorted ascending")
    out.name = "date"
    return out


@dataclass(frozen=True)
class PriceData:
    """Aligned OHLCV frames for one or more symbols (rows = dates, columns = symbols).

    Cells are ``NaN`` where a symbol has no bar (not yet listed, delisted, halted).
    """

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    raw_close: pd.DataFrame
    split_adj_close: pd.DataFrame
    dividends: pd.DataFrame
    splits: pd.DataFrame

    def __post_init__(self) -> None:
        index = normalize_index(self.close.index)
        columns = list(self.close.columns)
        if len(columns) != len(set(columns)):
            raise DataError("duplicate symbols")
        for f in fields(self):
            frame = getattr(self, f.name)
            if not isinstance(frame, pd.DataFrame):
                raise DataError(f"{f.name} must be a DataFrame")
            if list(frame.columns) != columns:
                raise DataError(f"{f.name} columns {list(frame.columns)} != {columns}")
            if not frame.index.equals(self.close.index):
                raise DataError(f"{f.name} index differs from close index")
            frame.index = index  # normalised, named
            if f.name != "splits" and f.name != "dividends" and (frame < 0).any().any():
                raise DataError(f"negative values in {f.name}")
        bad = (self.high < self.low).any()
        if bad.any():
            raise DataError(f"high < low for {list(bad[bad].index)}")

    # -- basic accessors -------------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    def __len__(self) -> int:
        return len(self.close)

    def field(self, name: str) -> pd.DataFrame:
        if name not in ALL_FIELDS:
            raise KeyError(f"unknown field {name!r}; choose from {ALL_FIELDS}")
        frame: pd.DataFrame = getattr(self, name)
        return frame

    def slice(
        self, start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None
    ) -> PriceData:
        """Rows with ``start <= date <= end`` (both inclusive, unlike Yahoo's exclusive end)."""
        kwargs = {f.name: getattr(self, f.name).loc[start:end] for f in fields(self)}
        return PriceData(**kwargs)

    def select(self, symbols: Sequence[str]) -> PriceData:
        missing = [s for s in symbols if s not in self.close.columns]
        if missing:
            raise KeyError(f"symbols not in data: {missing}")
        kwargs = {f.name: getattr(self, f.name)[list(symbols)] for f in fields(self)}
        return PriceData(**kwargs)

    def single(self, field_name: str = "close") -> pd.Series:
        """The one-column view of ``field_name`` for a single-symbol dataset."""
        if len(self.symbols) != 1:
            raise DataError(f"single() needs exactly one symbol, have {self.symbols}")
        return self.field(field_name).iloc[:, 0]

    # -- construction ----------------------------------------------------------------
    @classmethod
    def from_frames(cls, frames: Mapping[str, pd.DataFrame]) -> PriceData:
        """Build from ``{symbol: DataFrame}`` where each frame has Yahoo-style columns.

        Required: ``Open Close Volume``; ``High``/``Low`` load as NaN if absent. Optional:
        ``Adj Close`` (used directly if
        present), ``Dividends`` and ``Stock Splits`` (used to derive the adjustment when
        ``Adj Close`` is absent, and to reconstruct as-traded prices). ``Close`` is assumed
        split-adjusted, as Yahoo publishes it. Dates are outer-joined across symbols.
        """
        if not frames:
            raise DataError("no symbols")
        per_field: dict[str, dict[str, pd.Series]] = {name: {} for name in ALL_FIELDS}
        for symbol, raw in frames.items():
            adjusted = _adjust_one(raw)
            for name in ALL_FIELDS:
                per_field[name][symbol] = adjusted[name]
        symbols = list(frames)
        built = {name: pd.DataFrame(series)[symbols] for name, series in per_field.items()}
        index = built["close"].index.union(built["open"].index)
        built = {name: frame.reindex(index) for name, frame in built.items()}
        built["dividends"] = built["dividends"].fillna(0.0)
        built["splits"] = built["splits"].fillna(0.0)
        return cls(**built)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        """Inverse of :meth:`from_frames` (values are the *adjusted* series)."""
        out: dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            out[symbol] = pd.DataFrame(
                {
                    "Open": self.open[symbol],
                    "High": self.high[symbol],
                    "Low": self.low[symbol],
                    "Close": self.split_adj_close[symbol],
                    "Adj Close": self.close[symbol],
                    "Volume": self.volume[symbol],
                    "Dividends": self.dividends[symbol],
                    "Stock Splits": self.splits[symbol],
                }
            ).dropna(subset=["Close"])
        return out


def _adjust_one(raw: pd.DataFrame) -> dict[str, pd.Series]:
    """Split one Yahoo-style frame into the fields of :class:`PriceData`."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c) for c in frame.columns]
    missing = [c for c in ("Open", "Close", "Volume") if c not in frame.columns]
    if missing:
        raise DataError(f"missing columns {missing}; have {list(frame.columns)}")
    for column in ("High", "Low"):
        if column not in frame.columns:
            # The committed snapshot omits High/Low because no code reads them; they load
            # as NaN, and Context.history refuses to serve an all-NaN field.
            frame[column] = np.nan
    frame.index = normalize_index(frame.index)
    frame = frame.dropna(subset=["Close"])
    close = frame["Close"].astype(float)
    dividends = (
        frame["Dividends"].astype(float).fillna(0.0) if "Dividends" in frame else close * 0.0
    )
    splits = (
        frame["Stock Splits"].astype(float).fillna(0.0) if "Stock Splits" in frame else close * 0.0
    )

    if "Adj Close" in frame.columns:
        adj_close = frame["Adj Close"].astype(float)
        factor = (adj_close / close).astype(float)
    else:
        factor = dividend_adjustment_factor(close, dividends)
        adj_close = close * factor

    # Undo Yahoo's split adjustment to recover the as-traded price: every split after
    # date t multiplied the historical price by 1/ratio, so multiply it back.
    ratio = splits.where(splits > 0, 1.0)
    future_split = ratio[::-1].cumprod()[::-1] / ratio  # product of ratios strictly after t
    raw_close = close * future_split

    return {
        "open": frame["Open"].astype(float) * factor,
        "high": frame["High"].astype(float) * factor,
        "low": frame["Low"].astype(float) * factor,
        "close": adj_close,
        "volume": frame["Volume"].astype(float),
        "raw_close": raw_close,
        "split_adj_close": close,
        "dividends": dividends,
        "splits": splits,
    }


def dividend_adjustment_factor(close: pd.Series, dividends: pd.Series) -> pd.Series:
    """CRSP-style multiplicative factor turning split-adjusted prices into total-return prices.

    On an ex-date ``d`` with cash dividend ``D`` the factor for every bar before ``d`` is
    multiplied by ``1 - D / Close[d-1]``. The last bar has factor 1.
    """
    prev_close = close.shift(1)
    f = pd.Series(1.0, index=close.index)
    ex = dividends > 0
    f[ex] = 1.0 - dividends[ex] / prev_close[ex]
    f = f.fillna(1.0)
    # factor[t] = product of f over ex-dates strictly after t
    return f[::-1].cumprod()[::-1] / f


# ---------------------------------------------------------------------------------
# Yahoo Finance with an on-disk cache
# ---------------------------------------------------------------------------------

Downloader = Callable[[str], pd.DataFrame]


def yahoo_downloader(symbol: str) -> pd.DataFrame:
    """Full-history raw download for ``symbol`` (unadjusted, with dividend/split actions)."""
    import yfinance as yf  # imported lazily so tests never need it

    frame = yf.download(
        symbol,
        period="max",
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
    )
    if frame is None or len(frame) == 0:
        raise DataError(f"Yahoo returned no data for {symbol!r}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    keep = [c for c in _YAHOO_COLUMNS if c in frame.columns]
    out: pd.DataFrame = frame[keep].copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    return out


def _manifest_path(cache_dir: Path) -> Path:
    return cache_dir / "MANIFEST.json"


def _read_manifest(cache_dir: Path) -> dict[str, Any]:
    path = _manifest_path(cache_dir)
    if path.exists():
        loaded: dict[str, Any] = json.loads(path.read_text())
        return loaded
    return {}


def _write_manifest(cache_dir: Path, manifest: Mapping[str, Any]) -> None:
    _manifest_path(cache_dir).write_text(json.dumps(dict(manifest), indent=2, sort_keys=True))


def _cached_frame(
    symbol: str,
    cache_dir: Path,
    refresh: bool,
    downloader: Downloader,
    needed_end: pd.Timestamp | None,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{symbol.replace('/', '_')}.csv"
    if path.exists() and not refresh:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        last = pd.Timestamp(frame.index.max())
        if needed_end is None or last >= needed_end - pd.Timedelta(days=7):
            return frame
        warnings.warn(
            f"{symbol}: cache ends {last.date()} but {needed_end.date()} requested; re-downloading",
            stacklevel=3,
        )
    frame = downloader(symbol)
    frame.to_csv(path)
    manifest = _read_manifest(cache_dir)
    manifest[symbol] = {
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "first_date": str(pd.Timestamp(frame.index.min()).date()),
        "last_date": str(pd.Timestamp(frame.index.max()).date()),
        "rows": len(frame),
    }
    _write_manifest(cache_dir, manifest)
    return frame


def load_yahoo(
    symbols: str | Iterable[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    downloader: Downloader | None = None,
) -> PriceData:
    """Load daily bars for ``symbols`` from Yahoo Finance, caching the raw download.

    **By default this reads the committed snapshot** (:data:`SNAPSHOT_DIR`) and never
    touches the network, so every reported number reproduces offline. Live downloads
    happen only when :func:`use_live_data` was called (the scripts' ``--live-data`` flag),
    the ``QUANTBT_LIVE_DATA=1`` environment variable is set, or a ``cache_dir``,
    ``downloader`` or ``refresh`` is passed explicitly.

    The cache holds each symbol's full history, so changing ``start``/``end`` never
    triggers a download. ``end`` is inclusive. Pass a ``downloader`` to substitute
    another source (tests use a synthetic one).
    """
    symbol_list = [symbols] if isinstance(symbols, str) else list(symbols)
    if not symbol_list:
        raise DataError("no symbols requested")
    if cache_dir is None and downloader is None and not refresh and not live_data_enabled():
        return load_snapshot(symbol_list, start, end)
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    fetch = downloader or yahoo_downloader
    needed_end = pd.Timestamp(end) if end is not None else None
    frames = {s: _cached_frame(s, cache, refresh, fetch, needed_end) for s in symbol_list}
    return PriceData.from_frames(frames).slice(start, end)


# ---------------------------------------------------------------------------------
# The committed snapshot
# ---------------------------------------------------------------------------------

_LIVE: dict[str, bool] = {"enabled": False}


def use_live_data(enabled: bool = True) -> None:
    """Switch :func:`load_yahoo` and the French loader between the snapshot and live data."""
    _LIVE["enabled"] = enabled


def add_live_data_flag(parser: Any) -> None:
    """Add the scripts' shared ``--live-data`` switch to an ``argparse`` parser."""
    parser.add_argument(
        "--live-data",
        action="store_true",
        help="download from Yahoo and Ken French (into data/cache/) instead of reading the "
        "committed snapshot; results will then differ from the committed ones",
    )


def live_data_enabled() -> bool:
    return _LIVE["enabled"] or os.environ.get(LIVE_DATA_ENV, "") == "1"


def snapshot_manifest(snapshot_dir: str | Path | None = None) -> dict[str, Any]:
    """The snapshot's MANIFEST.json (source, dates, and a sha256 for every file)."""
    root = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    path = root / "MANIFEST.json"
    if not path.exists():
        raise DataError(
            f"no snapshot manifest at {path}; restore data/ from git or pass --live-data"
        )
    loaded: dict[str, Any] = json.loads(path.read_text())
    return loaded


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_file(relative: str, snapshot_dir: str | Path | None = None) -> Path:
    """Path of a file inside the snapshot, checked against the manifest's sha256.

    Raises :class:`DataError` if the file is not in the manifest, is missing on disk or
    does not match its recorded hash, so a silently edited input cannot produce a number.
    """
    root = Path(snapshot_dir) if snapshot_dir is not None else SNAPSHOT_DIR
    files: dict[str, Any] = snapshot_manifest(root)["files"]
    if relative not in files:
        raise DataError(
            f"{relative} is not in the snapshot at {root}; the snapshot only holds what the "
            "committed scripts read. Pass --live-data (or set QUANTBT_LIVE_DATA=1) to download."
        )
    path = root / relative
    if not path.exists():
        raise DataError(f"snapshot file missing: {path}")
    digest = sha256_file(path)
    if digest != files[relative]["sha256"]:
        raise DataError(f"snapshot file {path} does not match its manifest sha256")
    return path


def load_snapshot(
    symbols: Sequence[str],
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    *,
    snapshot_dir: str | Path | None = None,
) -> PriceData:
    """Load ``symbols`` from the committed snapshot. Never downloads; fails loudly instead.

    Each symbol's history in the snapshot starts at the earliest date any committed script
    reads. Asking for an earlier ``start`` (where the source has data) or a later ``end``
    than the snapshot holds raises rather than returning a silently shorter series.
    """
    frames: dict[str, pd.DataFrame] = {}
    files = snapshot_manifest(snapshot_dir)["files"]
    for symbol in symbols:
        rel = f"prices/{symbol.replace('/', '_')}.csv.gz"
        path = snapshot_file(rel, snapshot_dir)
        meta = files[rel]
        first, last = pd.Timestamp(meta["first_date"]), pd.Timestamp(meta["last_date"])
        trimmed = pd.Timestamp(meta["trimmed_before"])
        truncated = pd.Timestamp(meta["source_first_date"]) < first
        if start is not None and pd.Timestamp(start) < trimmed and truncated:
            raise DataError(
                f"{symbol}: snapshot starts {trimmed.date()} but {pd.Timestamp(start).date()} "
                "was requested; pass --live-data to use the full download"
            )
        if end is not None and pd.Timestamp(end) > last:
            raise DataError(
                f"{symbol}: snapshot ends {last.date()} but {pd.Timestamp(end).date()} was "
                "requested; pass --live-data to download newer data"
            )
        # round_trip parsing reads back exactly the float64 values that were written
        frames[symbol] = pd.read_csv(
            path, index_col=0, parse_dates=True, float_precision="round_trip"
        )
    return PriceData.from_frames(frames).slice(start, end)


def load_csv(path: str | Path, symbol: str | None = None) -> PriceData:
    """Load a single-symbol Yahoo-style CSV (``Date`` index, ``Open High Low Close Volume`` ...)."""
    p = Path(path)
    frame = pd.read_csv(p, index_col=0, parse_dates=True)
    return PriceData.from_frames({symbol or p.stem: frame})


def synthetic_prices(
    n: int = 1000,
    symbols: Sequence[str] = ("SYN",),
    *,
    start: str = "2015-01-01",
    drift: float = 0.0,
    vol: float = 0.01,
    seed: int = 0,
    initial: float = 100.0,
    gap_vol: float = 0.0,
) -> PriceData:
    """Geometric random walk OHLCV with an explicit open-to-close and close-to-open split.

    ``gap_vol`` controls the overnight (close-to-next-open) move; ``vol`` the intraday one.
    Useful for tests: no dividends, no splits, so every price field agrees.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    frames: dict[str, pd.DataFrame] = {}
    for k, symbol in enumerate(symbols):
        intraday = rng.normal(drift, vol, n)
        overnight = rng.normal(0.0, gap_vol, n)
        close = np.empty(n)
        open_ = np.empty(n)
        prev_close = initial * (1 + 0.1 * k)
        for i in range(n):
            open_[i] = prev_close * np.exp(overnight[i])
            close[i] = open_[i] * np.exp(intraday[i])
            prev_close = close[i]
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
        frames[symbol] = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
            },
            index=index,
        )
    return PriceData.from_frames(frames)
