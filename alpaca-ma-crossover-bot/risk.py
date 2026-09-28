"""
Pre-trade risk checks.

`evaluate_order()` is a pure function: it takes the proposed order, the current position,
an account snapshot and the limits, and returns a `RiskDecision` listing every check with
a pass / fail / skip status and a human-readable reason. Nothing here talks to the network,
so every rule is unit-tested directly (tests/test_risk.py).

The one impure helper, `read_kill_switch()`, only reads an environment mapping and checks
whether a file exists. It is kept separate so the checks themselves stay pure.

Design choices:
- **Every check runs** (no short-circuit), so the log shows the full picture of why an
  order was blocked, not just the first reason.
- **Risk-reducing orders are only blocked by the kill switch.** A daily-loss breach
  should never stop the bot from selling; that would trap it in the losing position.
- **Fail closed.** If the account does not report the previous day's equity, the
  daily-loss check cannot be evaluated, so risk-increasing orders are blocked.
- **The kill switch never flattens.** It halts all order submission, buys and sells alike.
  See `read_kill_switch()` for why it does not liquidate.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CheckStatus = Literal["pass", "fail", "skip"]

# Values of BOT_KILL_SWITCH that mean "off". Anything else non-empty engages it (fail safe:
# a typo such as BOT_KILL_SWITCH=ture should halt trading, not silently allow it).
_KILL_SWITCH_OFF_VALUES = frozenset({"", "0", "false", "no", "off"})


@dataclass(frozen=True)
class RiskLimits:
    """Configurable pre-trade limits. Defaults are deliberately conservative."""

    max_position_pct: float = 0.25          # max position value as a fraction of equity
    max_position_notional: float = 50_000.0  # max position value in dollars
    max_daily_loss_pct: float = 0.02        # block new risk once equity is down this much vs last_equity

    def __post_init__(self) -> None:
        if not (0 < self.max_position_pct <= 1):
            raise ValueError("max_position_pct must be in (0, 1]")
        if not (self.max_position_notional > 0 and math.isfinite(self.max_position_notional)):
            raise ValueError("max_position_notional must be a positive finite number")
        if not (0 < self.max_daily_loss_pct < 1):
            raise ValueError("max_daily_loss_pct must be in (0, 1)")


@dataclass(frozen=True)
class AccountSnapshot:
    """The account fields the checks need, parsed from GET /v2/account."""

    equity: float
    last_equity: float | None  # equity at the previous trading day's close; None if unavailable
    buying_power: float


@dataclass(frozen=True)
class OrderIntent:
    """An order the strategy wants to place. `price` is the reference price used for sizing."""

    symbol: str
    side: str   # "buy" or "sell"
    qty: float
    price: float


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"check": self.name, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    risk_increasing: bool
    checks: tuple[CheckResult, ...]

    @property
    def rejections(self) -> list[CheckResult]:
        return [c for c in self.checks if c.status == "fail"]

    def as_log(self) -> list[dict[str, str]]:
        return [c.as_dict() for c in self.checks]


def read_kill_switch(env: Mapping[str, str], kill_file: Path | None) -> str | None:
    """Return the reason the kill switch is engaged, or None if it is off.

    Two independent triggers, either of which halts ALL order submission:
      - environment variable BOT_KILL_SWITCH set to anything other than 0/false/no/off
      - a file at `kill_file` exists (default: KILL_SWITCH next to bot.py), so an operator
        can halt a scheduled job with `touch KILL_SWITCH` without editing the schedule.

    The kill switch deliberately does NOT flatten positions. It is pulled when something
    looks wrong (bad data, a bug, a broker incident), and in exactly those conditions an
    automatic market sell is the action most likely to do damage: it would trade on the
    same state we have just declared untrustworthy, possibly into a gap or a halt, and
    for a long-only daily strategy the position held overnight is the intended risk, not
    a runaway one. Flattening is a separate, deliberate human action (the broker
    dashboard, or DELETE /v2/positions).
    """
    raw = env.get("BOT_KILL_SWITCH")
    if raw is not None and raw.strip().lower() not in _KILL_SWITCH_OFF_VALUES:
        return f"env BOT_KILL_SWITCH={raw.strip()!r}"
    if kill_file is not None and kill_file.exists():
        return f"kill switch file present: {kill_file}"
    return None


def is_risk_increasing(current_qty: float, side: str, qty: float) -> bool:
    """True if the order makes the absolute position larger."""
    signed = qty if side == "buy" else -qty
    return abs(current_qty + signed) > abs(current_qty)


def evaluate_order(order: OrderIntent, current_qty: float, account: AccountSnapshot,
                   limits: RiskLimits, kill_switch_reason: str | None = None) -> RiskDecision:
    """Run every pre-trade check against a proposed order and return the verdict.

    Checks, in order:
      kill_switch       - blocks everything, including sells
      order_sanity      - side is buy/sell, qty > 0 and finite, price > 0
      no_short          - a sell may not exceed the current long position
      daily_loss        - (risk-increasing only) equity vs last_equity
      position_pct      - (risk-increasing only) post-trade position / equity
      position_notional - (risk-increasing only) post-trade position in dollars
      buying_power      - (buys only) order notional vs buying power
    """
    checks: list[CheckResult] = []

    def add(name: str, status: CheckStatus, reason: str) -> None:
        checks.append(CheckResult(name, status, reason))

    # 1. Kill switch
    if kill_switch_reason:
        add("kill_switch", "fail", f"kill switch engaged ({kill_switch_reason})")
    else:
        add("kill_switch", "pass", "not engaged")

    # 2. Sanity of the order itself
    sane = (order.side in ("buy", "sell") and math.isfinite(order.qty) and order.qty > 0
            and math.isfinite(order.price) and order.price > 0)
    if sane:
        add("order_sanity", "pass", f"{order.side} {order.qty:g} @ ~{order.price:.2f}")
    else:
        add("order_sanity", "fail",
            f"invalid order: side={order.side!r} qty={order.qty!r} price={order.price!r}")
        # Nothing below is meaningful without a sane order.
        return RiskDecision(approved=False, risk_increasing=True, checks=tuple(checks))

    risk_increasing = is_risk_increasing(current_qty, order.side, order.qty)
    projected_qty = current_qty + (order.qty if order.side == "buy" else -order.qty)
    projected_notional = abs(projected_qty) * order.price
    order_notional = order.qty * order.price

    # 3. Long-only: never sell more than we hold
    if order.side == "sell" and order.qty > max(current_qty, 0.0):
        add("no_short", "fail",
            f"sell {order.qty:g} exceeds current position {current_qty:g}; would open a short")
    else:
        add("no_short", "pass", f"post-trade qty {projected_qty:g}")

    if not risk_increasing:
        why = "risk-reducing order"
        add("daily_loss", "skip", why)
        add("position_pct", "skip", why)
        add("position_notional", "skip", why)
    else:
        # 4. Daily loss circuit breaker (fail closed)
        if account.last_equity is None or not (account.last_equity > 0):
            add("daily_loss", "fail",
                f"last_equity unavailable ({account.last_equity!r}); cannot evaluate daily loss")
        else:
            loss_pct = (account.last_equity - account.equity) / account.last_equity
            detail = (f"equity {account.equity:,.2f} vs last_equity {account.last_equity:,.2f} "
                      f"({-loss_pct:+.2%}), limit -{limits.max_daily_loss_pct:.2%}")
            if loss_pct >= limits.max_daily_loss_pct:
                add("daily_loss", "fail", f"daily loss limit breached: {detail}")
            else:
                add("daily_loss", "pass", detail)

        # 5. Position size as % of equity
        if account.equity <= 0:
            add("position_pct", "fail", f"non-positive equity {account.equity:,.2f}")
        else:
            pct = projected_notional / account.equity
            detail = f"post-trade {pct:.2%} of equity, limit {limits.max_position_pct:.2%}"
            add("position_pct", "fail" if pct > limits.max_position_pct + 1e-12 else "pass", detail)

        # 6. Position size in dollars
        detail = (f"post-trade ${projected_notional:,.2f}, "
                  f"limit ${limits.max_position_notional:,.2f}")
        add("position_notional",
            "fail" if projected_notional > limits.max_position_notional else "pass", detail)

    # 7. Buying power (buys only; the broker also checks, this just avoids a known reject)
    if order.side == "buy":
        detail = f"order ${order_notional:,.2f} vs buying power ${account.buying_power:,.2f}"
        if order_notional > account.buying_power:
            add("buying_power", "fail", f"insufficient buying power: {detail}")
        else:
            add("buying_power", "pass", detail)
    else:
        add("buying_power", "skip", "sell order")

    approved = all(c.status != "fail" for c in checks)
    return RiskDecision(approved=approved, risk_increasing=risk_increasing, checks=tuple(checks))
