"""Unit tests for the pure pre-trade risk checks in risk.py."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk import (  # noqa: E402
    AccountSnapshot,
    OrderIntent,
    RiskLimits,
    evaluate_order,
    is_risk_increasing,
    read_kill_switch,
)

LIMITS = RiskLimits(max_position_pct=0.25, max_position_notional=50_000, max_daily_loss_pct=0.02)
HEALTHY = AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=100_000)


def status(decision, name):
    return next(c.status for c in decision.checks if c.name == name)


def buy(qty, price=100.0):
    return OrderIntent("SPY", "buy", qty, price)


def sell(qty, price=100.0):
    return OrderIntent("SPY", "sell", qty, price)


def test_normal_buy_is_approved_and_every_check_is_reported():
    d = evaluate_order(buy(200), 0, HEALTHY, LIMITS)
    assert d.approved and d.risk_increasing
    assert [c.name for c in d.checks] == [
        "kill_switch", "order_sanity", "no_short", "daily_loss",
        "position_pct", "position_notional", "buying_power",
    ]
    assert d.rejections == []


def test_position_pct_limit_uses_post_trade_position():
    # Already hold 200 shares ($20k); buying 60 more makes $26k = 26% > 25%.
    d = evaluate_order(buy(60), 200, HEALTHY, LIMITS)
    assert not d.approved
    assert status(d, "position_pct") == "fail"
    assert "26.00% of equity" in d.rejections[0].reason


def test_position_pct_exactly_at_limit_passes():
    d = evaluate_order(buy(250), 0, HEALTHY, LIMITS)  # exactly 25%
    assert status(d, "position_pct") == "pass"


def test_notional_limit():
    big = AccountSnapshot(equity=1_000_000, last_equity=1_000_000, buying_power=1_000_000)
    d = evaluate_order(buy(600), 0, big, LIMITS)  # $60k: 6% of equity but > $50k
    assert not d.approved
    assert status(d, "position_pct") == "pass"
    assert status(d, "position_notional") == "fail"


def test_daily_loss_breach_blocks_buys():
    down = AccountSnapshot(equity=97_000, last_equity=100_000, buying_power=97_000)
    d = evaluate_order(buy(10), 0, down, LIMITS)
    assert not d.approved
    assert status(d, "daily_loss") == "fail"
    assert "daily loss limit breached" in d.rejections[0].reason


def test_daily_loss_just_inside_limit_passes():
    down = AccountSnapshot(equity=98_100, last_equity=100_000, buying_power=98_100)
    assert evaluate_order(buy(10), 0, down, LIMITS).approved


def test_daily_loss_breach_never_blocks_a_sell():
    # Being unable to exit a losing position would be worse than the breach itself.
    down = AccountSnapshot(equity=90_000, last_equity=100_000, buying_power=0)
    d = evaluate_order(sell(100), 100, down, LIMITS)
    assert d.approved and not d.risk_increasing
    assert status(d, "daily_loss") == "skip"
    assert status(d, "position_pct") == "skip"
    assert status(d, "buying_power") == "skip"


def test_missing_last_equity_fails_closed_for_buys_only():
    unknown = AccountSnapshot(equity=100_000, last_equity=None, buying_power=100_000)
    d = evaluate_order(buy(10), 0, unknown, LIMITS)
    assert not d.approved
    assert "last_equity unavailable" in d.rejections[0].reason
    assert evaluate_order(sell(10), 10, unknown, LIMITS).approved


def test_insufficient_buying_power():
    poor = AccountSnapshot(equity=100_000, last_equity=100_000, buying_power=500)
    d = evaluate_order(buy(10), 0, poor, LIMITS)
    assert not d.approved
    assert status(d, "buying_power") == "fail"


def test_kill_switch_blocks_sells_too():
    d = evaluate_order(sell(100), 100, HEALTHY, LIMITS, kill_switch_reason="file present")
    assert not d.approved
    assert [c.name for c in d.rejections] == ["kill_switch"]


def test_sell_larger_than_position_is_rejected():
    d = evaluate_order(sell(150), 100, HEALTHY, LIMITS)
    assert not d.approved
    assert status(d, "no_short") == "fail"


@pytest.mark.parametrize("order", [
    OrderIntent("SPY", "buy", 0, 100.0),
    OrderIntent("SPY", "buy", -5, 100.0),
    OrderIntent("SPY", "buy", float("nan"), 100.0),
    OrderIntent("SPY", "buy", 10, 0.0),
    OrderIntent("SPY", "short", 10, 100.0),
])
def test_insane_orders_are_rejected(order):
    d = evaluate_order(order, 0, HEALTHY, LIMITS)
    assert not d.approved
    assert status(d, "order_sanity") == "fail"


def test_is_risk_increasing():
    assert is_risk_increasing(0, "buy", 10)
    assert is_risk_increasing(10, "buy", 5)
    assert not is_risk_increasing(10, "sell", 10)
    assert not is_risk_increasing(10, "sell", 4)


@pytest.mark.parametrize("kwargs", [
    {"max_position_pct": 0}, {"max_position_pct": 1.5},
    {"max_position_notional": 0}, {"max_position_notional": float("inf")},
    {"max_daily_loss_pct": 0}, {"max_daily_loss_pct": 1},
])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RiskLimits(**kwargs)


class TestKillSwitch:
    def test_off_by_default(self, tmp_path: Path):
        assert read_kill_switch({}, tmp_path / "KILL_SWITCH") is None

    @pytest.mark.parametrize("value", ["0", "false", "No", "off", ""])
    def test_explicit_off_values(self, tmp_path: Path, value):
        assert read_kill_switch({"BOT_KILL_SWITCH": value}, tmp_path / "KILL_SWITCH") is None

    @pytest.mark.parametrize("value", ["1", "true", "yes", "ture"])
    def test_env_engages_including_typos(self, tmp_path: Path, value):
        reason = read_kill_switch({"BOT_KILL_SWITCH": value}, tmp_path / "KILL_SWITCH")
        assert reason is not None and "BOT_KILL_SWITCH" in reason

    def test_file_engages(self, tmp_path: Path):
        flag = tmp_path / "KILL_SWITCH"
        flag.touch()
        assert "kill switch file present" in read_kill_switch({}, flag)
