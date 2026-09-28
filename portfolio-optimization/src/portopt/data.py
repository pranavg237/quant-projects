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
    "MARKET_WEIGHTS",
    "PriceHistory",
    "download_prices",
    "drop_partial_last_month",
    "load_benchmark",
    "load_prices",
    "load_risk_free",
    "tbill_yield_to_monthly_rf",
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


#: A static approximation of global market-capitalisation weights for :data:`ETF_UNIVERSE`.
#:
#: Black-Litterman's prior is the market portfolio, so it needs market weights. Real ETF
#: assets under management would be circular (AUM reflects flows, not the underlying market)
#: and are not available through this data source, so a static approximation of the global
#: multi-asset market is used instead: roughly 55% US equity split across sectors in
#: proportion to their weight in the S&P 500, 12% developed international, 5% emerging, 4%
#: REITs, 20% Treasuries and 4% gold.
#:
#: These are **assumptions, not data**, and they are stated here rather than buried so the
#: reader can disagree with them. Using equal weights instead makes Black-Litterman with no
#: views mathematically identical to 1/N, which is a useful implementation check and a
#: useless backtest.
MARKET_WEIGHTS: dict[str, float] = {
    "XLB": 0.018,
    "XLE": 0.022,
    "XLF": 0.070,
    "XLI": 0.045,
    "XLK": 0.170,
    "XLP": 0.033,
    "XLU": 0.014,
    "XLV": 0.068,
    "XLY": 0.060,
    "IYR": 0.040,
    "EFA": 0.120,
    "EEM": 0.050,
    "TLT": 0.090,
    "IEF": 0.110,
    "GLD": 0.040,
}


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


def drop_partial_last_month(prices: pd.DataFrame, tolerance_days: int = 4) -> pd.DataFrame:
    """Drop the final calendar month if the data stops well before it ends.

    Monthly resampling takes the last observation in each month. If the data ends on the
    18th, that "month" is really 18 days, and treating it as a full monthly return
    misstates its volatility and its weight in every average. A month is treated as
    complete if its last observation is within ``tolerance_days`` of the calendar month
    end, which allows for weekends and a month-end holiday.

    Args:
        prices: Daily prices with a sorted ``DatetimeIndex``.
        tolerance_days: How close to the month end the last observation must be.

    Returns:
        ``prices`` without the trailing partial month, or unchanged if it is complete.
    """
    if prices.empty:
        return prices
    last = pd.Timestamp(prices.index[-1])
    month_end = last + pd.offsets.MonthEnd(0)
    if (month_end - last).days <= tolerance_days:
        return prices
    previous_month_end = last.to_period("M").start_time - pd.Timedelta(days=1)
    return prices.loc[:previous_month_end]


def tbill_yield_to_monthly_rf(daily_yield_pct: pd.Series) -> pd.Series:
    r"""Convert a daily 13-week T-bill yield (in percent) to a monthly risk-free return.

    The return earned on cash during month :math:`m` is set by the yield observed at the
    **end of month** :math:`m-1`, so the rate is known before the month starts and this
    introduces no lookahead. The quoted yield is annualised, so the monthly return is
    approximated as ``yield / 12``. That ignores the difference between the discount-basis
    quote and a compounded return, which is a few basis points a year.

    Args:
        daily_yield_pct: Daily ``^IRX`` closes, e.g. ``5.2`` for 5.2%.

    Returns:
        Per-month simple risk-free returns, indexed by calendar month end.
    """
    month_end_yield = daily_yield_pct.dropna().resample("ME").last()
    monthly = (month_end_yield / 100.0 / 12.0).shift(1)
    return monthly.dropna().rename("rf")


def _latest_snapshot(cache_dir: Path, prefix: str) -> Path | None:
    snapshots = sorted((cache_dir / "snapshots").glob(f"{prefix}_*.csv"))
    return snapshots[-1] if snapshots else None


def _download_close(ticker: str, start: str) -> pd.Series:
    import yfinance as yf  # noqa: PLC0415  (local: keeps the network dep off the import path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"no data returned for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return pd.Series(close, name=ticker, dtype=float)


def _load_cached_series(
    prefix: str, ticker: str, start: str, cache_dir: Path | None, force_refresh: bool
) -> pd.Series:
    """Read the latest committed snapshot of one series, or download and snapshot it."""
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = _latest_snapshot(cache_dir, prefix)
    if path is not None and not force_refresh:
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        return pd.Series(frame.iloc[:, 0], name=ticker)
    series = _download_close(ticker, start)
    out = cache_dir / "snapshots" / f"{prefix}_{dt.date.today().isoformat()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(out)
    return series


def load_benchmark(
    ticker: str = "SPY",
    cache_dir: Path | None = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Daily adjusted closes for the buy-and-hold benchmark, cached as a snapshot.

    Args:
        ticker: Benchmark symbol.
        cache_dir: Root of the data cache.
        force_refresh: Download even if a snapshot exists.

    Returns:
        Adjusted closes (dividends reinvested), so buy-and-hold returns are total returns.
    """
    return _load_cached_series(
        f"benchmark_{ticker}", ticker, "1993-01-01", cache_dir, force_refresh
    )


def load_risk_free(cache_dir: Path | None = None, force_refresh: bool = False) -> pd.Series:
    """Monthly risk-free returns from the 13-week T-bill yield (``^IRX``).

    Args:
        cache_dir: Root of the data cache.
        force_refresh: Download even if a snapshot exists.

    Returns:
        Per-month simple returns indexed by month end. See
        :func:`tbill_yield_to_monthly_rf` for the timing convention.
    """
    daily = _load_cached_series("tbill_13w", "^IRX", "1993-01-01", cache_dir, force_refresh)
    return tbill_yield_to_monthly_rf(daily)
