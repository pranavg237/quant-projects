"""
MA crossover signal logic - the same short/long moving-average crossover as
pranavg237/ma-crossover-backtest, pulled out into a pure function so it can
be unit tested without hitting any API, and reused by both a backtest and
this live/paper bot.
"""

from __future__ import annotations

import datetime as dt
import math
from enum import Enum
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = dt.time(16, 0)


class Signal(str, Enum):
    """Desired position for the next session."""

    LONG = "long"  # short MA above long MA -> want to be in the position
    FLAT = "flat"  # short MA below long MA -> want to be out


def compute_signal(closes: pd.Series, short_window: int, long_window: int) -> Signal:
    """Given a close-price series (oldest first), returns the *current*
    desired position: LONG if the short MA is above the long MA on the most
    recent bar, else FLAT. Requires at least `long_window` bars."""
    if len(closes) < long_window:
        raise ValueError(f"Need at least {long_window} bars, got {len(closes)}")

    ma_short = closes.rolling(short_window).mean().iloc[-1]
    ma_long = closes.rolling(long_window).mean().iloc[-1]
    return Signal.LONG if ma_short > ma_long else Signal.FLAT


def completed_bars(bars: list[dict[str, Any]], now: dt.datetime) -> list[dict[str, Any]]:
    """Drop today's daily bar if the session has not closed yet.

    During market hours Alpaca returns a bar for today whose "close" is really the
    latest trade. Treating it as a close would make the signal depend on what time
    the bot happened to run, and would use a price that the backtest never saw. Early
    closes (1pm) are handled conservatively: the bar is dropped until 4pm.

    Args:
        bars: Alpaca bar dicts, oldest first, each with an ISO timestamp under "t".
        now: The current time; must be timezone-aware.
    """
    if not bars:
        return bars
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_et = now.astimezone(MARKET_TZ)
    last_session = pd.Timestamp(bars[-1]["t"]).tz_convert(MARKET_TZ).date()
    if last_session == now_et.date() and now_et.time() < MARKET_CLOSE:
        return bars[:-1]
    return bars


def validate_bars(
    bars: list[dict[str, Any]], now: dt.datetime, long_window: int, max_age_days: int = 5
) -> str | None:
    """Return a reason the bars are unusable, or None if they are fine.

    Refuses to trade on:
      - too few bars for the long moving average (missing history, or a data outage);
      - a bar with a missing, non-numeric or non-positive close;
      - stale data: the newest completed bar is more than `max_age_days` calendar days
        old. Five days covers a normal weekend plus a holiday (e.g. a Thursday bar seen on
        the following Monday); anything older means the feed has stopped updating, and a
        signal computed from it would be a signal about the past.
    """
    if len(bars) < long_window:
        return (
            f"only {len(bars)} completed bars, need >= {long_window} "
            f"(market data may lag on a free/paper account)"
        )
    for i, bar in enumerate(bars):
        if not isinstance(bar.get("t"), str):
            return f"bar {i} has no timestamp"
        close = bar.get("c")
        if not isinstance(close, (int, float)) or not math.isfinite(close) or close <= 0:
            return f"bar {i} ({bar.get('t')}) has an invalid close: {close!r}"
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    last_session = pd.Timestamp(bars[-1]["t"]).tz_convert(MARKET_TZ).date()
    age = (now.astimezone(MARKET_TZ).date() - last_session).days
    if age > max_age_days:
        return f"stale bars: newest completed bar is {last_session} ({age} days old, max {max_age_days})"
    return None


def position_size(
    equity: float,
    price: float,
    risk_fraction: float,
    max_position_fraction: float = 0.25,
    buying_power: float | None = None,
) -> int:
    """Whole-share position size: risk_fraction of equity, capped at
    max_position_fraction of equity so one signal can't put the whole
    account into one name, and never more than the available buying power.
    Returns an integer share count (floor)."""
    if not (0 < risk_fraction <= 1) or not (0 < max_position_fraction <= 1):
        raise ValueError("risk_fraction and max_position_fraction must be in (0, 1]")
    if price <= 0:
        raise ValueError("price must be positive")
    target_fraction = min(risk_fraction, max_position_fraction)
    dollars = equity * target_fraction
    if buying_power is not None:
        dollars = min(dollars, max(buying_power, 0.0))
    return int(dollars // price)


def decide_order(
    current_qty: float, signal: Signal, target_qty: int, top_up_below: float = 0.5
) -> tuple[str, float] | None:
    """Compares current holdings to the desired signal and returns
    (side, qty) to submit, or None if already positioned correctly.

    target_qty is only used on the buy side; going LONG->FLAT always sells the full
    current position.

    If the signal is LONG and we hold something, but less than `top_up_below` of the
    target (e.g. a buy that was partially filled and then cancelled or expired), buy the
    difference. The threshold is deliberately wide so ordinary price drift in the target
    share count does not make the bot trade every day.
    """
    has_position = current_qty > 0
    if signal == Signal.LONG and not has_position:
        if target_qty <= 0:
            return None  # not enough equity/buying power for even 1 share
        return ("buy", target_qty)
    if signal == Signal.LONG and current_qty < top_up_below * target_qty:
        shortfall = math.floor(target_qty - current_qty)
        return ("buy", shortfall) if shortfall > 0 else None
    if signal == Signal.FLAT and has_position:
        return ("sell", current_qty)
    return None  # already in the right state
