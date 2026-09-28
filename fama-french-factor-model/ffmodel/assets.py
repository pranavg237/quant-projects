"""Asset returns: from Yahoo Finance, from a CSV, or combined into a portfolio."""
from __future__ import annotations

import warnings
from typing import Mapping, Optional, Sequence

import pandas as pd


def prices_to_returns(prices: pd.DataFrame, frequency: str = "monthly") -> pd.DataFrame:
    """Simple returns from (total-return adjusted) prices.

    Monthly returns use month-end prices; a trailing partial month (data that
    stops more than a few days before month end) is dropped.
    """
    prices = prices.sort_index()
    if frequency == "monthly":
        last_day = prices.index.max()
        prices = prices.resample("ME").last()
        if last_day < prices.index[-1] - pd.Timedelta(days=4):
            prices = prices.iloc[:-1]
    elif frequency != "daily":
        raise ValueError("frequency must be 'monthly' or 'daily'")
    return prices.pct_change(fill_method=None).iloc[1:].dropna(how="all")


def download_returns(
    tickers: Sequence[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    frequency: str = "monthly",
) -> pd.DataFrame:
    """Total returns (dividend- and split-adjusted) from Yahoo Finance."""
    import yfinance as yf

    tickers = [t.upper() for t in tickers]
    # Fetch a little extra history so the first requested period has a base price.
    fetch_start = None
    if start is not None:
        pad = pd.DateOffset(months=1) if frequency == "monthly" else pd.DateOffset(days=7)
        fetch_start = (pd.Timestamp(start) - pad).strftime("%Y-%m-%d")
    # `end` may be a month ("2024-12", as the CLI documents) or a day. Yahoo wants a day and
    # treats it as exclusive, so ask for the day after the last day wanted.
    fetch_end = None
    if end is not None:
        last_day = pd.Period(end, freq="M" if len(str(end)) <= 7 else "D").end_time.normalize()
        fetch_end = (last_day + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(tickers, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"Yahoo Finance returned no data for {tickers}")
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)
    missing = [t for t in tickers if close[t].dropna().empty]
    if missing:
        warnings.warn(f"no price data for {missing}; dropping them")
        close = close.drop(columns=missing)
    if close.empty:
        raise RuntimeError(f"Yahoo Finance returned no usable prices for {tickers}")
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.columns.name = None
    returns = prices_to_returns(close, frequency)
    return returns.loc[start:end]


def load_returns_csv(path: str, prices: bool = False, percent: bool = False, frequency: str = "monthly") -> pd.DataFrame:
    """Read a CSV whose first column is dates and remaining columns are assets.

    Set ``prices=True`` if the columns are price levels, ``percent=True`` if
    returns are in percent. Monthly dates are moved to month end so they line
    up with the factor data.
    """
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().apply(pd.to_numeric, errors="coerce")
    if prices:
        return prices_to_returns(frame, frequency)
    if percent:
        frame = frame / 100.0
    if frequency == "monthly":
        frame.index = frame.index + pd.offsets.MonthEnd(0)
        if frame.index.has_duplicates:
            raise ValueError(f"{path} has several rows per month; is it daily data? Use frequency='daily'.")
    frame.index.name = "date"
    return frame


def build_portfolio(returns: pd.DataFrame, weights: Mapping[str, float], name: str = "Portfolio") -> pd.Series:
    """Return of a portfolio rebalanced to ``weights`` every period (NaN if any holding is missing)."""
    missing = [k for k in weights if k not in returns.columns]
    if missing:
        raise KeyError(f"no returns for portfolio holdings {missing}")
    w = pd.Series(weights, dtype=float)
    return returns[w.index].mul(w, axis=1).sum(axis=1, min_count=len(w)).rename(name)
