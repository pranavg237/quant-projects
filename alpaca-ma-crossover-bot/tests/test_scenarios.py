"""
End-to-end failure scenarios: the real run_once and the real AlpacaClient, talking to
FakeAlpacaServer at the HTTP boundary. No network, no keys.

Default setup: SPY in a steady uptrend (5/20-day MAs so the fixtures stay small), a $100k
account, flat, run on Friday 2026-09-25 at 16:15 ET after the close. The strategy wants to
buy floor(20% x 100k / 139) = 143 shares.
"""

import datetime as dt
import io
import json
import os
import sys
import uuid

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fake_alpaca import FakeAlpacaServer

from bot import EXIT_CODES, RunRefused, run_once
from risk import RiskLimits
from runlog import RunLogger, configure_logger

UTC = dt.UTC
FRI_AFTER_CLOSE = dt.datetime(2026, 9, 25, 20, 15, tzinfo=UTC)  # 16:15 ET
MON_AFTER_CLOSE = dt.datetime(2026, 9, 28, 20, 15, tzinfo=UTC)
UP = list(range(100, 140))
DOWN = list(range(140, 100, -1))
CID_BUY = "macx-SPY-20260925-buy"


class Harness:
    """Runs one cycle and keeps the parsed JSON log lines."""

    def __init__(self, server: FakeAlpacaServer) -> None:
        self.server = server
        self.client = server.client()
        self.events: list[dict] = []

    def run(self, now=FRI_AFTER_CLOSE, dry_run=False, limits=None, kill=None):
        stream = io.StringIO()
        log = RunLogger("test-run", configure_logger(stream, name=f"test-{uuid.uuid4().hex}"))
        try:
            return run_once(
                self.client,
                "SPY",
                short_window=5,
                long_window=20,
                risk_fraction=0.2,
                max_position_fraction=0.25,
                dry_run=dry_run,
                limits=limits,
                kill_switch_reason=kill,
                now=now,
                log=log,
            )
        finally:
            self.events = [json.loads(line) for line in stream.getvalue().splitlines()]

    def event(self, name):
        matches = [e for e in self.events if e["event"] == name]
        assert matches, f"no {name!r} event in {[e['event'] for e in self.events]}"
        return matches[-1]


def harness(closes=UP, **kwargs) -> Harness:
    return Harness(FakeAlpacaServer(closes, **kwargs))


# -- happy path, market closed, order open at end of run ------------------------------


def test_after_close_buy_is_queued_and_reported_as_open_at_end_of_run():
    h = harness(market_open=False)
    result = h.run()
    assert result.outcome == "submitted" and EXIT_CODES[result.outcome] == 0
    assert (result.side, result.qty, result.client_order_id) == ("buy", 143, CID_BUY)
    assert h.server.posts == 1
    # Market is closed, so the day order queues for the next open and is still open now.
    assert result.order_status == "accepted"
    open_evt = h.event("order_open_at_end_of_run")
    assert open_evt["market_open"] is False
    assert open_evt["next_open"].startswith("2026-09-28T09:30")
    assert h.event("risk_check")["approved"] is True
    assert all(e["run_id"] == "test-run" for e in h.events)


def test_order_filled_during_run_is_reported_as_filled():
    h = harness(market_open=True)
    h.server.on_submit = lambda o: h.server.fill(o, 143, "filled")
    result = h.run()
    assert (result.order_status, result.filled_qty) == ("filled", 143)
    assert h.event("order_filled")["id"] == result.order_id


# -- idempotency: re-running the same day ---------------------------------------------


def test_rerun_while_order_is_still_open_does_not_duplicate():
    h = harness()
    h.run()
    second = h.run()
    assert second.outcome == "pending_order"
    assert h.server.posts == 1
    assert h.event("pending_order")["open_orders"][0]["client_order_id"] == CID_BUY


def test_rerun_on_a_weekend_does_not_duplicate():
    h = harness()
    h.run(now=FRI_AFTER_CLOSE)
    saturday = FRI_AFTER_CLOSE + dt.timedelta(days=1)
    assert h.run(now=saturday).outcome == "pending_order"
    assert h.server.posts == 1


def test_rerun_after_order_was_cancelled_does_not_resubmit_for_the_same_bar():
    h = harness()
    h.run()
    order = next(iter(h.server.orders.values()))
    order["status"] = "canceled"  # e.g. cancelled by hand, nothing filled
    second = h.run()
    assert second.outcome == "already_submitted"
    assert second.order_status == "canceled"
    assert h.server.posts == 1


def test_rerun_after_fill_is_a_no_op():
    h = harness()
    h.server.on_submit = lambda o: h.server.fill(o, 143, "filled")
    h.run()
    assert h.run().outcome == "no_action"
    assert h.server.posts == 1


# -- partial fills ----------------------------------------------------------------------


def test_partial_fill_then_cancel_is_reported_and_topped_up_on_the_next_bar():
    h = harness()
    h.server.on_submit = lambda o: h.server.fill(o, 50, "canceled")
    result = h.run()
    assert (result.outcome, result.order_status, result.filled_qty) == ("submitted", "canceled", 50)
    assert h.event("order_incomplete")["unfilled_qty"] == 93

    # Same bar again: the buy for this bar already exists, so no second order.
    assert h.run().outcome == "already_submitted"
    assert h.server.posts == 1

    # Next session: 50 held is below half the 143 target, so the bot buys the shortfall.
    h.server.on_submit = None
    h.server.add_bar(140)
    h.server.account["buying_power"] = str(100_000 - 50 * 139)
    topped = h.run(now=MON_AFTER_CLOSE)
    assert topped.outcome == "submitted"
    assert topped.side == "buy" and topped.client_order_id == "macx-SPY-20260928-buy"
    assert 0 < topped.qty < 143
    assert h.server.posts == 2


def test_partial_fill_that_expires_is_reported_as_incomplete():
    h = harness()
    h.server.on_submit = lambda o: h.server.fill(o, 100, "expired")
    result = h.run()
    assert result.order_status == "expired"
    assert h.event("order_incomplete")["filled_qty"] == "100.0"


def test_partially_filled_order_still_working_blocks_the_next_run():
    h = harness()
    h.server.on_submit = lambda o: h.server.fill(o, 60, "partially_filled")
    result = h.run()
    assert result.order_status == "partially_filled"
    h.event("order_open_at_end_of_run")
    assert h.run().outcome == "pending_order"
    assert h.server.posts == 1


# -- broker rejections and ambiguous failures ---------------------------------------------


def test_insufficient_buying_power_rejection_is_not_retried():
    # The account snapshot said $100k, but by the time the order arrives the broker
    # disagrees (e.g. another order consumed it). The broker's word is final.
    h = harness()
    h.server.fail(
        "POST", "/v2/orders", (403, {"code": 40310000, "message": "insufficient buying power"})
    )
    result = h.run()
    assert result.outcome == "rejected" and EXIT_CODES["rejected"] == 1
    assert h.server.posts == 1
    err = h.event("order_submit_error")
    assert (err["status_code"], err["api_code"], err["ambiguous"]) == (403, 40310000, False)
    assert h.server.count("GET", "/v2/orders:by_client_order_id") == 1  # pre-submit check only


def test_timeout_after_order_reached_broker_is_reconciled_not_resubmitted():
    h = harness()
    h.server.fail("POST", "/v2/orders", requests.ReadTimeout("read timed out"), process_first=True)
    result = h.run()
    assert result.outcome == "submitted"
    assert h.server.posts == 1 and len(h.server.orders) == 1
    assert h.event("order_reconciled")["client_order_id"] == CID_BUY


def test_timeout_before_order_reached_broker_fails_without_retry_then_next_run_submits():
    h = harness()
    h.server.fail("POST", "/v2/orders", requests.ConnectTimeout("connect timed out"))
    result = h.run()
    assert result.outcome == "submit_failed" and EXIT_CODES["submit_failed"] == 1
    assert h.server.posts == 1 and not h.server.orders
    h.event("order_not_placed")

    again = h.run()
    assert again.outcome == "submitted" and again.client_order_id == CID_BUY
    assert h.server.posts == 2 and len(h.server.orders) == 1


def test_5xx_on_submit_is_treated_as_ambiguous():
    h = harness()
    h.server.fail("POST", "/v2/orders", (500, {"message": "internal error"}), process_first=True)
    assert h.run().outcome == "submitted"
    assert h.server.posts == 1


def test_reconcile_lookup_failing_leaves_state_unknown_but_next_run_is_safe():
    h = harness()
    h.server.fail("POST", "/v2/orders", requests.ReadTimeout("read timed out"), process_first=True)
    # The pre-submit lookup succeeds (404 = none); every later lookup (the reconcile,
    # including its retries) hits a connection error.
    original = h.server._route
    lookups = {"n": 0}

    def flaky_route(method, path, params, body):
        if path == "/v2/orders:by_client_order_id":
            lookups["n"] += 1
            if lookups["n"] > 1:
                raise requests.ConnectionError("down")
        return original(method, path, params, body)

    h.server._route = flaky_route
    result = h.run()
    assert result.outcome == "submit_failed"
    h.event("reconcile_failed")
    h.server._route = original
    # The order did land. The next run sees it as open and does not add another.
    assert h.run().outcome == "pending_order"
    assert h.server.posts == 1


def test_transient_5xx_on_reads_is_retried_and_the_run_completes():
    h = harness()
    h.server.fail("GET", "/v2/account", (503, {"message": "unavailable"}))
    h.server.fail("GET", "/v2/positions/SPY", requests.Timeout("slow"))
    assert h.run().outcome == "submitted"
    assert h.server.sleeps == [0.5, 0.5]


def test_persistent_read_failure_aborts_before_any_order():
    h = harness()
    h.server.fail("GET", "/v2/stocks/SPY/bars", *[(504, {"message": "gateway timeout"})] * 3)
    with pytest.raises(Exception) as exc:
        h.run()
    assert getattr(exc.value, "status_code", None) == 504
    assert h.server.posts == 0


# -- market data problems -------------------------------------------------------------------


def test_stale_bars_refuse_to_trade():
    h = harness()
    two_weeks_later = FRI_AFTER_CLOSE + dt.timedelta(days=14)
    with pytest.raises(RunRefused, match="stale bars"):
        h.run(now=two_weeks_later)
    assert h.server.posts == 0
    assert "stale bars" in h.event("run_refused")["reason"]


def test_holiday_weekend_is_not_stale():
    h = harness()
    tuesday = FRI_AFTER_CLOSE + dt.timedelta(days=4)  # e.g. Monday was a holiday, feed lagging
    assert h.run(now=tuesday).outcome == "submitted"


def test_missing_bars_refuse_to_trade():
    h = harness()
    h.server.bars = []
    with pytest.raises(RunRefused, match="only 0 completed bars"):
        h.run()
    assert h.server.posts == 0


def test_bar_with_missing_close_refuses_to_trade():
    h = harness()
    h.server.bars[-3]["c"] = None
    with pytest.raises(RunRefused, match="invalid close"):
        h.run()
    assert h.server.posts == 0


def test_intraday_run_ignores_todays_partial_bar():
    h = harness()
    midday = dt.datetime(2026, 9, 25, 15, 30, tzinfo=UTC)  # 11:30 ET, market open
    result = h.run(now=midday)
    # Signal from Thursday's close; the order id carries Thursday's bar date.
    assert result.client_order_id == "macx-SPY-20260924-buy"


# -- risk controls end to end -----------------------------------------------------------------


def test_kill_switch_blocks_the_buy_and_logs_why():
    h = harness()
    result = h.run(kill="kill switch file present: KILL_SWITCH")
    assert result.outcome == "blocked" and EXIT_CODES["blocked"] == 2
    assert h.server.posts == 0
    assert "kill switch engaged" in h.event("order_blocked")["reasons"][0]


def test_kill_switch_also_blocks_sells_and_does_not_flatten():
    h = harness(DOWN, position_qty=37)
    result = h.run(kill="env BOT_KILL_SWITCH='1'")
    assert result.outcome == "blocked"
    assert h.server.posts == 0 and h.server.position_qty == 37


def test_daily_loss_breach_blocks_buys_but_not_sells():
    h = harness(equity=96_000, last_equity=100_000)
    blocked = h.run()
    assert blocked.outcome == "blocked"
    assert "daily loss limit breached" in blocked.reasons[0]
    checks = {c["check"]: c["status"] for c in h.event("risk_check")["checks"]}
    assert checks["daily_loss"] == "fail" and checks["position_pct"] == "pass"

    seller = harness(DOWN, equity=96_000, last_equity=100_000, position_qty=37)
    assert seller.run().outcome == "submitted"
    assert seller.server.posts == 1


def test_notional_cap_blocks_an_oversized_order():
    h = harness()
    result = h.run(limits=RiskLimits(max_position_notional=10_000))
    assert result.outcome == "blocked"
    assert any("post-trade $19,877.00" in r for r in result.reasons)
    assert h.server.posts == 0


def test_dry_run_runs_risk_checks_but_never_submits():
    h = harness()
    assert h.run(dry_run=True).outcome == "dry_run"
    h.event("risk_check")
    assert h.server.posts == 0
