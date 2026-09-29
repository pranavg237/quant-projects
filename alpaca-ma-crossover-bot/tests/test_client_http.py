"""
Retry / no-retry behaviour of AlpacaClient at the HTTP boundary (no network).

Policy under test: reads are retried on timeouts, connection errors, 429 and 5xx; other
4xx are not; order submission is never retried and reports whether the failure is
ambiguous (the order may have reached the broker).
"""

import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fake_alpaca import FakeAlpacaServer

from alpaca_client import AlpacaClient, AlpacaError


@pytest.fixture
def server():
    return FakeAlpacaServer(list(range(100, 140)))


def test_get_retries_5xx_then_succeeds_with_exponential_backoff(server):
    server.fail(
        "GET",
        "/v2/account",
        (503, {"message": "service unavailable"}),
        (502, "<html>bad gateway</html>"),
    )
    client = server.client(backoff_seconds=0.5)
    assert client.get_account()["status"] == "ACTIVE"
    assert server.count("GET", "/v2/account") == 3
    assert server.sleeps == [0.5, 1.0]


def test_get_retries_timeouts_and_429(server):
    server.fail(
        "GET",
        "/v2/clock",
        requests.Timeout("read timed out"),
        (429, {"message": "rate limit exceeded"}),
    )
    assert server.client().get_clock()["is_open"] is False
    assert server.count("GET", "/v2/clock") == 3


def test_get_gives_up_after_max_attempts(server):
    server.fail("GET", "/v2/account", *[requests.ConnectionError("connection refused")] * 3)
    with pytest.raises(AlpacaError) as exc:
        server.client(max_attempts=3).get_account()
    assert server.count("GET", "/v2/account") == 3
    assert exc.value.status_code is None and exc.value.ambiguous


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_get_does_not_retry_client_errors(server, status):
    server.fail("GET", "/v2/account", (status, {"code": 40000000 + status, "message": "nope"}))
    with pytest.raises(AlpacaError) as exc:
        server.client().get_account()
    assert server.count("GET", "/v2/account") == 1
    assert exc.value.status_code == status
    assert exc.value.api_code == 40000000 + status
    assert not exc.value.ambiguous


def test_wrong_credentials_are_a_401_not_retried(server):
    client = AlpacaClient("wrong-key-id", "wrong-secret", session=server, sleep=lambda s: None)
    with pytest.raises(AlpacaError) as exc:
        client.get_account()
    assert exc.value.status_code == 401
    assert server.count("GET", "/v2/account") == 1


def test_missing_position_is_none_via_status_code(server):
    assert server.client().get_position("SPY") is None


def test_order_lookup_by_client_order_id_404_is_none(server):
    assert server.client().get_order_by_client_order_id("macx-SPY-20260925-buy") is None


@pytest.mark.parametrize(
    "error, ambiguous",
    [
        (requests.Timeout("read timed out"), True),
        (requests.ConnectionError("connection reset"), True),
        ((500, {"message": "internal error"}), True),
        ((503, {"message": "unavailable"}), True),
        ((429, {"message": "rate limit"}), False),  # rate-limited: definitely not accepted
    ],
)
def test_order_submission_is_never_retried(server, error, ambiguous):
    server.fail("POST", "/v2/orders", error)
    client = server.client()
    with pytest.raises(AlpacaError) as exc:
        client.submit_market_order("SPY", 10, "buy", client_order_id="cid-1")
    assert server.posts == 1
    assert server.sleeps == []
    assert exc.value.ambiguous is ambiguous


def test_insufficient_buying_power_is_a_definitive_rejection(server):
    server.account["buying_power"] = "100"
    with pytest.raises(AlpacaError) as exc:
        server.client().submit_market_order("SPY", 10, "buy", client_order_id="cid-1")
    assert exc.value.status_code == 403
    assert exc.value.api_code == 40310000
    assert "insufficient buying power" in str(exc.value)
    assert not exc.value.ambiguous
    assert server.posts == 1


def test_duplicate_client_order_id_is_rejected_by_broker(server):
    client = server.client()
    client.submit_market_order("SPY", 10, "buy", client_order_id="cid-1")
    with pytest.raises(AlpacaError) as exc:
        client.submit_market_order("SPY", 10, "buy", client_order_id="cid-1")
    assert exc.value.status_code == 422 and not exc.value.ambiguous
    assert len(server.orders) == 1


def test_client_order_id_is_sent(server):
    server.client().submit_market_order("SPY", 7, "buy", client_order_id="macx-SPY-20260925-buy")
    (order,) = server.orders.values()
    assert order["client_order_id"] == "macx-SPY-20260925-buy"
    assert order["qty"] == "7"


def test_repr_hides_credentials(server):
    text = repr(server.client())
    assert FakeAlpacaServer.KEY_ID not in text and FakeAlpacaServer.SECRET not in text
