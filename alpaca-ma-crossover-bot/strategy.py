"""
MA crossover signal logic - the same short/long moving-average crossover as
pranavg237/ma-crossover-backtest, pulled out into a pure function so it can
be unit tested without hitting any API, and reused by both a backtest and
this live/paper bot.
"""
from __future__ import annotations

import datetime as dt
from enum import Enum
from zoneinfo import ZoneInfo

import pandas as pd

MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = dt.time(16, 0)


class Signal(str, Enum):
    """Desired position for the next session."""

    LONG = "long"    # short MA above long MA -> want to be in the position
    FLAT = "flat"    # short MA below long MA -> want to be out


def compute_signal(closes: pd.Series, short_window: int, long_window: int) -> Signal:
    """Given a close-price series (oldest first), returns the *current*
    desired position: LONG if the short MA is above the long MA on the most
    recent bar, else FLAT. Requires at least `long_window` bars."""
    if len(closes) < long_window:
        raise ValueError(f"Need at least {long_window} bars, got {len(closes)}")

    ma_short = closes.rolling(short_window).mean().iloc[-1]
    ma_long = closes.rolling(long_window).mean().iloc[-1]
    return Signal.LONG if ma_short > ma_long else Signal.FLAT


def completed_bars(bars: list[dict], now: dt.datetime) -> list[dict]:
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


def position_size(equity: float, price: float, risk_fraction: float,
                  max_position_fraction: float = 0.25,
                  buying_power: float | None = None) -> int:
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


def decide_order(current_qty: float, signal: Signal, target_qty: int) -> tuple[str, float] | None:
    """Compares current holdings to the desired signal and returns
    (side, qty) to submit, or None if already positioned correctly.
    target_qty is only used when going from FLAT->LONG (how many shares to buy);
    going LONG->FLAT always sells the full current position."""
    has_position = current_qty > 0
    if signal == Signal.LONG and not has_position:
        if target_qty <= 0:
            return None  # not enough equity/buying power for even 1 share
        return ("buy", target_qty)
    if signal == Signal.FLAT and has_position:
        return ("sell", current_qty)
    return None  # already in the right state
