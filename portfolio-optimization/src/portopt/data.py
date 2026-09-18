"""Price history download, local caching and return construction.

Prices come from Yahoo Finance via ``yfinance``. Every download is cached to CSV, and
:func:`load_prices` prefers the cache, so reruns are offline and reproducible after the
market has moved.

Three choices here matter more than they look.

**Adjusted closes, not raw closes.** A portfolio backtest that uses unadjusted prices
silently throws away every dividend, which on a 20-year run of sector ETFs is roughly
2% a year -- larger than most of the differences the backtest is trying to measure. It also
puts a fake -10% return on every ex-dividend date, which inflates estimated volatility.

**Simple returns, not log returns.** Portfolio return is the weighted average of simple
asset returns. Log returns do not aggregate that way across assets, and using them in a
mean-variance optimiser is a subtle, common and hard-to-spot error.

**The universe starts when its *last* member starts.** Assets are listed at different
times, and quietly letting the panel grow as new assets appear creates survivorship-style
distortions and makes covariance windows inconsistent. The default is a common start date
for the whole universe, with the cost stated: :func:`load_prices` reports which asset is
the binding constraint and how much history it costs.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ETF_UNIVERSE",
    "PriceHistory",
    "download_prices",
    "load_prices",
    "to_returns",
]

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"

#: A deliberately diversified ETF universe: nine GICS sectors, two Treasury maturities,
#: gold, developed and emerging international equity, and REITs.
#:
#: Sector ETFs are the right test bed for this project precisely because they are *hard*:
#: they have similar expected returns and correlations of 0.6-0.9, which is the regime in
#: which sample-covariance mean-variance optimisation falls apart most spectacularly. A
#: universe of genuinely dissimilar assets would make naive Markowitz look better than it
#: deserves.
ETF_UNIVERSE: tuple[str, ...] = (
    "XLB",  # materials
    "XLE",  # energy
    "XLF",  # financials
    "XLI",  # industrials
    "XLK",  # technology
    "XLP",  # consumer staples
    "XLU",  # utilities
    "XLV",  # health care
    "XLY",  # consumer discretionary
    "IYR",  # US real estate
    "EFA",  # developed international equity
    "EEM",  # emerging market equity
    "TLT",  # 20+ year Treasuries
    "IEF",  # 7-10 year Treasuries
    "GLD",  # gold
)


@dataclass
class PriceHistory:
    """Adjusted closing prices for a universe, plus provenance.

    Attributes:
        prices: Adjusted closes, dates on the index and tickers on the columns.
        downloaded_at: When the data was pulled.
        source: ``"yfinance"`` or ``"cache"``.
        binding_ticker: The asset whose listing date sets the common start, if any.
    """

    prices: pd.DataFrame
    downloaded_at: dt.datetime
    source: str = "yfinance"
    binding_ticker: str | None = None

    @property
    def tickers(self) -> list[str]:
        """Assets in the panel."""
        return list(self.prices.columns)

    @property
    def start(self) -> pd.Timestamp:
        """First date in the panel."""
        return pd.Timestamp(self.prices.index[0])

    @property
    def end(self) -> pd.Timestamp:
        """Last date in the panel."""
        return pd.Timestamp(self.prices.index[-1])

    def to_csv(self, path: Path) -> None:
        """Write the price panel to CSV."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.prices.to_csv(path)

    @classmethod
    def from_csv(cls, path: Path) -> PriceHistory:
        """Read a price panel written by :meth:`to_csv`."""
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        return cls(
            prices=prices,
            downloaded_at=dt.datetime.fromtimestamp(path.stat().st_mtime),
            source="cache",
        )

    def __str__(self) -> str:
        return (
            f"{len(self.tickers)} assets, {len(self.prices)} days, "
            f"{self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} (source: {self.source})"
        )


def download_prices(
    tickers: tuple[str, ...] | list[str] = ETF_UNIVERSE,
    start: str = "1990-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
) -> PriceHistory:
    """Download adjusted closes for a universe and cache them.

    Args:
        tickers: Symbols to fetch.
        start: Earliest date to request.
        end: Latest date, or ``None`` for today.
        cache_dir: Root of the data cache.

    Returns:
        A :class:`PriceHistory` starting on the first date every asset has a price.

    Raises:
        RuntimeError: if the download returns nothing usable.
    """
    import yfinance as yf  # noqa: PLC0415  (local: keeps the network dep off the import path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
    if raw is None or raw.empty:
        raise RuntimeError(f"no price data returned for {list(tickers)}")

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.reindex(columns=list(tickers)).dropna(axis=1, how="all")
    if prices.empty or prices.shape[1] < 2:
        raise RuntimeError("fewer than two assets returned usable prices")

    # The universe starts when its last member starts.
    first_valid = prices.apply(lambda col: col.first_valid_index())
    binding = str(first_valid.idxmax())
    common_start = first_valid.max()
    prices = prices.loc[common_start:].dropna(how="any")

    history = PriceHistory(
        prices=prices,
        downloaded_at=dt.datetime.now(),
        source="yfinance",
        binding_ticker=binding,
    )
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    history.to_csv(cache_dir / "raw" / f"prices_{dt.date.today().isoformat()}.csv")
    return history


def load_prices(
    tickers: tuple[str, ...] | list[str] = ETF_UNIVERSE,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    **download_kwargs: object,
) -> PriceHistory:
    """Return a price panel, preferring the local cache over the network.

    Search order: the most recent cache file that covers every requested ticker, then a
    fresh download. If the download fails and any cache exists, the cache is used with a
    warning -- an offline rerun should still reproduce the analysis rather than crash.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    candidates = sorted((cache_dir / "raw").glob("prices_*.csv")) if cache_dir.exists() else []
    candidates += sorted((cache_dir / "snapshots").glob("prices_*.csv"))
    candidates = sorted(candidates, key=lambda p: p.stem.split("_")[-1])

    def _usable(path: Path) -> PriceHistory | None:
        history = PriceHistory.from_csv(path)
        if set(tickers).issubset(history.prices.columns):
            history.prices = history.prices[list(tickers)].dropna(how="any")
            return history
        return None

    if not force_refresh:
        for path in reversed(candidates):
            history = _usable(path)
            if history is not None:
                return history

    try:
        return download_prices(tickers, cache_dir=cache_dir, **download_kwargs)  # type: ignore[arg-type]
    except Exception as exc:
        for path in reversed(candidates):
            history = _usable(path)
            if history is not None:
                warnings.warn(
                    f"download failed ({exc}); using cached prices from {path.name}",
                    stacklevel=2,
                )
                return history
        raise


def to_returns(
    prices: pd.DataFrame, frequency: str = "daily", drop_first: bool = True
) -> pd.DataFrame:
    """Convert a price panel to **simple** returns at the requested frequency.

    Args:
        prices: Adjusted closes.
        frequency: ``"daily"``, ``"weekly"`` or ``"monthly"``. Monthly resampling takes the
            last observation of each calendar month.
        drop_first: Drop the leading row of NaNs.

    Returns:
        Simple returns.

    Raises:
        ValueError: on an unknown frequency.
    """
    if frequency == "daily":
        sampled = prices
    elif frequency == "weekly":
        sampled = prices.resample("W-FRI").last()
    elif frequency == "monthly":
        sampled = prices.resample("ME").last()
    else:
        raise ValueError(f"unknown frequency {frequency!r}; expected daily/weekly/monthly")

    returns = sampled.pct_change(fill_method=None)
    if drop_first:
        returns = returns.iloc[1:]
    return returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")


def periods_per_year(frequency: str) -> float:
    """Number of return observations per year at a given frequency."""
    table = {"daily": 252.0, "weekly": 52.0, "monthly": 12.0}
    if frequency not in table:
        raise ValueError(f"unknown frequency {frequency!r}")
    return table[frequency]
