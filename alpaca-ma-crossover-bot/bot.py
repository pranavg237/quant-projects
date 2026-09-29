"""
Runs one decision cycle of the MA crossover strategy against Alpaca's PAPER
trading API: pulls recent daily bars, computes the signal, compares to the
current position and any open orders, runs pre-trade risk checks, and submits
a market order if a change is needed and every check passes.

This is meant to be run once a day (e.g. via a scheduled task shortly after
market close, using daily bars) - it is NOT a continuously-running intraday
loop. Re-running it is safe: see "Idempotency" in the README.

Usage:
  cp .env.example .env        # fill in your paper keys; bot.py loads it automatically
  python3 bot.py --symbol SPY --dry-run
  python3 bot.py --symbol SPY               # actually submits orders (paper)

Logs are JSON lines on stderr; redirect them to a file to keep a history.

Exit codes: 0 ran cleanly (including "nothing to do" and "order pending");
1 something failed or the broker rejected the order; 2 an order was blocked by a
risk check or the kill switch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from alpaca_client import TERMINAL_ORDER_STATUSES, AlpacaClient, AlpacaError
from risk import AccountSnapshot, OrderIntent, RiskLimits, evaluate_order, read_kill_switch
from runlog import RunLogger, configure_logger, new_run_id
from strategy import completed_bars, compute_signal, decide_order, position_size, validate_bars

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_KILL_FILE = PROJECT_DIR / "KILL_SWITCH"

EXIT_CODES = {
    "no_action": 0,
    "dry_run": 0,
    "submitted": 0,
    "pending_order": 0,
    "already_submitted": 0,
    "rejected": 1,
    "submit_failed": 1,
    "blocked": 2,
}


class RunRefused(RuntimeError):
    """The bot refused to make a decision (account not tradable, unusable market data)."""


@dataclass
class RunResult:
    """What one decision cycle did. `outcome` is a key of EXIT_CODES."""

    outcome: str
    signal: str | None = None
    side: str | None = None
    qty: float | None = None
    client_order_id: str | None = None
    order_id: str | None = None
    order_status: str | None = None
    filled_qty: float | None = None
    reasons: list[str] = field(default_factory=list)


def make_client_order_id(symbol: str, bar_date: dt.date, side: str) -> str:
    """Deterministic id: at most one order per symbol, side and signal bar.

    A re-run on the same data produces the same id, so the bot can look up whether it
    already acted, and the broker rejects a second order with the same id.
    """
    return f"macx-{symbol}-{bar_date:%Y%m%d}-{side}"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _order_summary(order: dict[str, Any]) -> dict[str, Any]:
    return {
        k: order.get(k)
        for k in ("id", "client_order_id", "side", "qty", "filled_qty", "status", "submitted_at")
    }


def run_once(
    client: AlpacaClient,
    symbol: str,
    short_window: int,
    long_window: int,
    risk_fraction: float,
    max_position_fraction: float,
    dry_run: bool,
    *,
    limits: RiskLimits | None = None,
    kill_switch_reason: str | None = None,
    now: dt.datetime | None = None,
    log: RunLogger | None = None,
    max_bar_age_days: int = 5,
) -> RunResult:
    """Run one decision cycle and return what happened.

    Raises RunRefused if the account is not tradable or the market data is unusable, and
    lets other AlpacaErrors from read-only calls propagate (the caller logs them).
    """
    now = now or dt.datetime.now(dt.UTC)
    log = log or RunLogger(new_run_id(now))
    limits = limits or RiskLimits(max_position_pct=max_position_fraction)
    log.event(
        "run_started",
        symbol=symbol,
        dry_run=dry_run,
        short_window=short_window,
        long_window=long_window,
        risk_fraction=risk_fraction,
        limits=asdict(limits),
        kill_switch=kill_switch_reason,
    )

    def refuse(reason: str) -> RunRefused:
        log.event("run_refused", logging.ERROR, symbol=symbol, reason=reason)
        return RunRefused(reason)

    # 1. Account
    account = client.get_account()
    if account.get("status") != "ACTIVE" or account.get("trading_blocked"):
        raise refuse(
            f"Account not tradable: status={account.get('status')}, "
            f"trading_blocked={account.get('trading_blocked')}"
        )
    equity = float(account["equity"])
    snapshot = AccountSnapshot(
        equity=equity,
        last_equity=_float_or_none(account.get("last_equity")),
        buying_power=_float_or_none(account.get("buying_power")) or 0.0,
    )
    log.event(
        "account",
        equity=snapshot.equity,
        last_equity=snapshot.last_equity,
        buying_power=snapshot.buying_power,
    )

    # 2. Market clock (informational: after-close orders queue for the next open)
    clock = client.get_clock()
    log.event(
        "market_clock",
        is_open=clock.get("is_open"),
        next_open=clock.get("next_open"),
        next_close=clock.get("next_close"),
    )

    # 3. Market data
    lookback_start = (now.date() - dt.timedelta(days=long_window * 3)).isoformat()
    bars = completed_bars(client.get_daily_bars(symbol, start=lookback_start), now)
    problem = validate_bars(bars, now, long_window, max_age_days=max_bar_age_days)
    if problem:
        raise refuse(f"{symbol}: {problem}")
    bar_date = pd.Timestamp(bars[-1]["t"]).tz_convert("America/New_York").date()
    last_price = float(bars[-1]["c"])
    signal = compute_signal(pd.Series([float(b["c"]) for b in bars]), short_window, long_window)

    # 4. Current state: position and anything still working at the broker
    position = client.get_position(symbol)
    current_qty = position.qty if position else 0.0
    target_qty = position_size(
        equity, last_price, risk_fraction, max_position_fraction, buying_power=snapshot.buying_power
    )
    order = decide_order(current_qty, signal, target_qty)
    log.event(
        "decision",
        symbol=symbol,
        signal=signal.value,
        bar_date=str(bar_date),
        last_price=last_price,
        current_qty=current_qty,
        target_qty=target_qty,
        decision=f"{order[0]} {order[1]:g}" if order else "hold",
    )
    result = RunResult(outcome="no_action", signal=signal.value)

    open_orders = client.list_open_orders(symbol)
    if open_orders:
        # An order is still working (e.g. yesterday's after-close order queued for the
        # open, or a partial fill). Position is in flux; never stack a second order on it.
        log.event(
            "pending_order",
            logging.WARNING,
            symbol=symbol,
            open_orders=[_order_summary(o) for o in open_orders],
        )
        result.outcome = "pending_order"
        result.reasons = [f"{len(open_orders)} open order(s) for {symbol}"]
        return result
    if order is None:
        log.event("no_action", symbol=symbol, reason="already in the target state")
        return result

    side, qty = order
    result.side, result.qty = side, qty
    cid = make_client_order_id(symbol, bar_date, side)
    result.client_order_id = cid

    # 5. Idempotency: have we already acted on this bar?
    existing = client.get_order_by_client_order_id(cid)
    if existing is not None:
        log.event("already_submitted", logging.WARNING, symbol=symbol, **_order_summary(existing))
        result.outcome = "already_submitted"
        result.order_id, result.order_status = existing.get("id"), existing.get("status")
        result.reasons = [f"order {cid} already exists with status {existing.get('status')}"]
        return result

    # 6. Pre-trade risk checks
    decision = evaluate_order(
        OrderIntent(symbol, side, qty, last_price),
        current_qty,
        snapshot,
        limits,
        kill_switch_reason,
    )
    log.event(
        "risk_check",
        symbol=symbol,
        side=side,
        qty=qty,
        approved=decision.approved,
        risk_increasing=decision.risk_increasing,
        checks=decision.as_log(),
    )
    if not decision.approved:
        result.outcome = "blocked"
        result.reasons = [c.reason for c in decision.rejections]
        log.event(
            "order_blocked",
            logging.WARNING,
            symbol=symbol,
            side=side,
            qty=qty,
            reasons=result.reasons,
        )
        return result

    if dry_run:
        log.event("dry_run_order", symbol=symbol, side=side, qty=qty, client_order_id=cid)
        result.outcome = "dry_run"
        return result

    # 7. Submit exactly once
    try:
        placed = client.submit_market_order(symbol, qty, side, client_order_id=cid)
    except AlpacaError as e:
        log.event(
            "order_submit_error",
            logging.ERROR,
            symbol=symbol,
            client_order_id=cid,
            status_code=e.status_code,
            api_code=e.api_code,
            ambiguous=e.ambiguous,
            error=str(e),
        )
        if not e.ambiguous:
            result.outcome = "rejected"
            result.reasons = [str(e)]
            return result
        found = _reconcile(client, cid, log, symbol)
        if found is None:
            result.outcome = "submit_failed"
            result.reasons = [str(e)]
            return result
        placed = found

    result.outcome = "submitted"
    result.order_id = placed.get("id")
    log.event("order_submitted", symbol=symbol, **_order_summary(placed))
    _report_final_state(client, placed, clock, log, symbol, result)
    return result


def _reconcile(
    client: AlpacaClient, cid: str, log: RunLogger, symbol: str
) -> dict[str, Any] | None:
    """After an ambiguous submit error, find out whether the order exists. Never resubmits:
    if it did not land, the next scheduled run will try again with the same id."""
    try:
        found = client.get_order_by_client_order_id(cid)
    except AlpacaError as e:
        log.event(
            "reconcile_failed",
            logging.ERROR,
            symbol=symbol,
            client_order_id=cid,
            status_code=e.status_code,
            error=str(e),
            note="order state unknown; next run's open-order and client_order_id "
            "checks prevent a duplicate",
        )
        return None
    if found is None:
        log.event(
            "order_not_placed",
            logging.ERROR,
            symbol=symbol,
            client_order_id=cid,
            note="not retried in this run",
        )
        return None
    log.event("order_reconciled", logging.WARNING, symbol=symbol, **_order_summary(found))
    return found


def _report_final_state(
    client: AlpacaClient,
    placed: dict[str, Any],
    clock: dict[str, Any],
    log: RunLogger,
    symbol: str,
    result: RunResult,
) -> None:
    """Re-read the order once and log where it ended up for this run."""
    order = placed
    if placed.get("id"):
        try:
            order = client.get_order(placed["id"])
        except AlpacaError as e:
            log.event(
                "order_status_unavailable",
                logging.WARNING,
                symbol=symbol,
                order_id=placed.get("id"),
                error=str(e),
            )
    status = str(order.get("status"))
    qty = _float_or_none(order.get("qty")) or 0.0
    filled = _float_or_none(order.get("filled_qty")) or 0.0
    result.order_status, result.filled_qty = status, filled

    if status == "filled":
        log.event("order_filled", symbol=symbol, **_order_summary(order))
    elif status == "rejected":
        result.outcome = "rejected"
        result.reasons = [f"order {order.get('id')} rejected by broker"]
        log.event("order_rejected", logging.ERROR, symbol=symbol, **_order_summary(order))
    elif status in TERMINAL_ORDER_STATUSES:
        # canceled / expired / replaced, possibly after a partial fill
        log.event(
            "order_incomplete",
            logging.WARNING,
            symbol=symbol,
            unfilled_qty=qty - filled,
            **_order_summary(order),
            note="partial or no fill; next run tops up if position < half of target",
        )
    else:
        log.event(
            "order_open_at_end_of_run",
            logging.WARNING,
            symbol=symbol,
            market_open=clock.get("is_open"),
            next_open=clock.get("next_open"),
            **_order_summary(order),
            note="next run will see it as pending and will not add a second order",
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a number") from None


def build_parser() -> argparse.ArgumentParser:
    """CLI flags. Risk limits default to env vars (BOT_*) if set, else safe constants."""
    p = argparse.ArgumentParser(description="MA crossover paper trading bot (one decision cycle)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--short", type=int, default=50, dest="short_window")
    p.add_argument("--long", type=int, default=200, dest="long_window")
    p.add_argument(
        "--risk-fraction",
        type=float,
        default=0.20,
        help="Fraction of equity to allocate when entering a position (default 0.20)",
    )
    p.add_argument(
        "--max-position-fraction",
        type=float,
        default=_env_float("BOT_MAX_POSITION_PCT", 0.25),
        help="Hard cap on post-trade position as a fraction of equity; used for "
        "sizing and re-checked pre-trade (env BOT_MAX_POSITION_PCT, default 0.25)",
    )
    p.add_argument(
        "--max-position-notional",
        type=float,
        default=_env_float("BOT_MAX_POSITION_NOTIONAL", 50_000.0),
        help="Hard cap on post-trade position in dollars "
        "(env BOT_MAX_POSITION_NOTIONAL, default 50000)",
    )
    p.add_argument(
        "--max-daily-loss",
        type=float,
        default=_env_float("BOT_MAX_DAILY_LOSS_PCT", 0.02),
        help="Block risk-increasing orders once equity is down this fraction vs "
        "the previous close (env BOT_MAX_DAILY_LOSS_PCT, default 0.02)",
    )
    p.add_argument(
        "--kill-switch-file",
        type=Path,
        default=Path(os.environ.get("BOT_KILL_SWITCH_FILE", DEFAULT_KILL_FILE)),
        help="If this file exists, no order is submitted (default ./KILL_SWITCH; "
        "env BOT_KILL_SWITCH=1 does the same)",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Compute the decision but don't submit any order"
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run one decision cycle, and return a process exit code."""
    # Credentials live in .env (gitignored), never in code. Variables already set in the
    # shell take precedence over the file. Loaded first so BOT_* defaults can come from it.
    load_dotenv(PROJECT_DIR / ".env", override=False)
    args = build_parser().parse_args(argv)

    run_id = new_run_id()
    logger = configure_logger(
        sys.stderr,
        secrets=[os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")],
    )
    log = RunLogger(run_id, logger)
    try:
        limits = RiskLimits(
            max_position_pct=args.max_position_fraction,
            max_position_notional=args.max_position_notional,
            max_daily_loss_pct=args.max_daily_loss,
        )
        kill = read_kill_switch(os.environ, args.kill_switch_file)
        client = AlpacaClient()
        result = run_once(
            client,
            args.symbol,
            args.short_window,
            args.long_window,
            args.risk_fraction,
            args.max_position_fraction,
            args.dry_run,
            limits=limits,
            kill_switch_reason=kill,
            log=log,
        )
    except Exception as e:  # noqa: BLE001 - top level: log it structurally, exit non-zero
        log.event(
            "run_failed",
            logging.ERROR,
            symbol=args.symbol,
            error_type=type(e).__name__,
            error=str(e),
            status_code=getattr(e, "status_code", None),
        )
        return 1
    log.event("run_finished", **asdict(result))
    return EXIT_CODES[result.outcome]


if __name__ == "__main__":
    sys.exit(main())
