"""
MA crossover signal logic - the same short/long moving-average crossover as
pranavg237/ma-crossover-backtest, pulled out into a pure function so it can
be unit tested without hitting any API, and reused by both a backtest and
this live/paper bot.
"""
from __future__ import annotations

from enum import Enum

import pandas as pd


class Signal(str, Enum):
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


def position_size(equity: float, price: float, risk_fraction: float, max_position_fraction: float = 0.25) -> int:
    """Whole-share position size: risk_fraction of equity, capped at
    max_position_fraction of equity so one signal can't put the whole
    account into one name. Returns an integer share count (floor)."""
    if not (0 < risk_fraction <= 1) or not (0 < max_position_fraction <= 1):
        raise ValueError("risk_fraction and max_position_fraction must be in (0, 1]")
    target_fraction = min(risk_fraction, max_position_fraction)
    dollars = equity * target_fraction
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
