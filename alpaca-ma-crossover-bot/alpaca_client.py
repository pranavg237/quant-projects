"""
Minimal Alpaca REST client - just the handful of endpoints this bot needs
(account, clock, historical bars, positions, orders). No `alpaca-py` SDK
dependency, built directly on `requests`.

Reads credentials from environment variables:
  ALPACA_API_KEY_ID
  ALPACA_API_SECRET_KEY
  ALPACA_BASE_URL       (defaults to the PAPER endpoint - see below)

Get paper keys at https://app.alpaca.markets/paper/dashboard/overview
(free, no funding required). Never hardcode keys in this file - use a
.env file (see .env.example) or export them in your shell.

Retry policy
------------
- **Reads (GET) are retried** on timeouts, connection errors, HTTP 429 and 5xx, with
  exponential backoff (3 attempts by default). Repeating a read has no side effects.
- **Other 4xx are never retried.** A 403 or 422 means the request itself is wrong
  (bad credentials, insufficient buying power, invalid order); repeating it cannot help.
- **Order submission (POST /v2/orders) is never retried.** If a POST times out or gets a
  5xx, the order may or may not have reached the broker. Retrying blindly could buy
  twice. Instead the error is raised with `ambiguous=True`, and the caller looks the order
  up by its `client_order_id` (a GET, which is safe to retry) to find out what happened.
  Alpaca also rejects a second order with the same `client_order_id`, so even a mistaken
  resubmission with the same id cannot create a duplicate.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import requests

DEFAULT_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
# Order statuses after which nothing more will happen to the order.
TERMINAL_ORDER_STATUSES = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})


class AlpacaError(RuntimeError):
    """An Alpaca API call failed or credentials are missing.

    Attributes:
        status_code: HTTP status, or None for a timeout / connection error.
        api_code: Alpaca's numeric error code from the response body, if any.
        ambiguous: True when the request may or may not have taken effect on the server
            (timeout, connection error or 5xx). Only meaningful for order submission.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        api_code: int | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.api_code = api_code
        self.ambiguous = ambiguous


@dataclass
class Position:
    """An open position as reported by Alpaca."""

    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float


class AlpacaClient:
    """Thin wrapper over the Alpaca REST endpoints the bot uses. Defaults to paper trading.

    `session` is any object with a `requests.Session`-compatible `request()` method; tests
    pass a fake one so the whole client runs without a network.
    """

    def __init__(
        self,
        api_key_id: str | None = None,
        api_secret_key: str | None = None,
        trading_base_url: str | None = None,
        data_base_url: str | None = None,
        *,
        session: Any = None,
        timeout: float = 15.0,
        max_attempts: int = 3,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key_id = api_key_id or os.environ.get("ALPACA_API_KEY_ID")
        secret = api_secret_key or os.environ.get("ALPACA_API_SECRET_KEY")
        if not key_id or not secret:
            raise AlpacaError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY (see .env.example)."
            )
        self.trading_base_url = trading_base_url or os.environ.get(
            "ALPACA_BASE_URL", DEFAULT_TRADING_BASE_URL
        )
        self.data_base_url = data_base_url or DEFAULT_DATA_BASE_URL
        self._secrets = (key_id, secret)
        self._headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
        self._session = session if session is not None else requests.Session()
        self._timeout = timeout
        self._max_attempts = max(1, max_attempts)
        self._backoff = backoff_seconds
        self._sleep = sleep

    def __repr__(self) -> str:  # never show credentials, e.g. in a traceback's locals
        return f"AlpacaClient(trading_base_url={self.trading_base_url!r}, credentials=[REDACTED])"

    # -- internal ----------------------------------------------------
    def _scrub(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "[REDACTED]")
        return text

    def _error_from_response(self, method: str, path: str, resp: Any) -> AlpacaError:
        api_code: int | None = None
        detail = resp.text or ""
        try:
            body = resp.json()
            if isinstance(body, dict):
                api_code = body.get("code")
                detail = str(body.get("message", detail))
        except ValueError:
            pass
        status = int(resp.status_code)
        return AlpacaError(
            self._scrub(f"{method} {path} -> {status}: {detail[:500]}"),
            status_code=status,
            api_code=api_code,
            ambiguous=status >= 500,
        )

    def _request(
        self,
        method: str,
        base_url: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        retry: bool,
    ) -> Any:
        attempts = self._max_attempts if retry else 1
        last_error: AlpacaError | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._session.request(
                    method,
                    f"{base_url}{path}",
                    headers=self._headers,
                    params=params,
                    json=json_body,
                    timeout=self._timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = AlpacaError(
                    self._scrub(f"{method} {path} -> {type(e).__name__}: {e}"), ambiguous=True
                )
            else:
                if resp.ok:
                    return resp.json()
                last_error = self._error_from_response(method, path, resp)
                if resp.status_code not in RETRYABLE_STATUS:
                    raise last_error
            if attempt < attempts:
                self._sleep(self._backoff * 2 ** (attempt - 1))
        assert last_error is not None
        raise last_error

    def _get(self, base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", base_url, path, params=params, retry=True)

    # -- account / clock / positions ------------------------------------
    def get_account(self) -> dict[str, Any]:
        """Account status, equity, last_equity (previous close) and buying power."""
        return cast(dict[str, Any], self._get(self.trading_base_url, "/v2/account"))

    def get_clock(self) -> dict[str, Any]:
        """Market clock: is_open, next_open, next_close, timestamp."""
        return cast(dict[str, Any], self._get(self.trading_base_url, "/v2/clock"))

    def get_position(self, symbol: str) -> Position | None:
        """The open position in ``symbol``, or None if flat."""
        try:
            data = self._get(self.trading_base_url, f"/v2/positions/{symbol}")
        except AlpacaError as e:
            if e.status_code == 404:
                return None
            raise
        return Position(
            symbol=data["symbol"],
            qty=float(data["qty"]),
            market_value=float(data["market_value"]),
            avg_entry_price=float(data["avg_entry_price"]),
        )

    # -- market data ---------------------------------------------------
    def get_daily_bars(
        self, symbol: str, start: str, end: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Daily OHLCV bars, oldest first. `start`/`end` are YYYY-MM-DD strings.

        Requests split- and dividend-adjusted prices (`adjustment=all`). Alpaca's default
        is raw prices, where a 4-for-1 split looks like a 75% crash and can flip a
        moving-average signal on a corporate action rather than a trend.
        """
        params = {"timeframe": "1Day", "start": start, "limit": limit, "adjustment": "all"}
        if end:
            params["end"] = end
        data = self._get(self.data_base_url, f"/v2/stocks/{symbol}/bars", params=params)
        return cast(list[dict[str, Any]], data.get("bars") or [])

    # -- orders ----------------------------------------------------------
    def list_open_orders(self, symbol: str) -> list[dict[str, Any]]:
        """Orders for ``symbol`` that are not yet in a terminal state."""
        return cast(
            list[dict[str, Any]],
            self._get(
                self.trading_base_url,
                "/v2/orders",
                params={"status": "open", "symbols": symbol, "limit": 100},
            ),
        )

    def get_order(self, order_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], self._get(self.trading_base_url, f"/v2/orders/{order_id}"))

    def get_order_by_client_order_id(self, client_order_id: str) -> dict[str, Any] | None:
        """The order with this client_order_id (any status), or None if none exists."""
        try:
            return cast(
                dict[str, Any],
                self._get(
                    self.trading_base_url,
                    "/v2/orders:by_client_order_id",
                    params={"client_order_id": client_order_id},
                ),
            )
        except AlpacaError as e:
            if e.status_code == 404:
                return None
            raise

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Submit a market order and return Alpaca's order record. Never retried."""
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        body: dict[str, str] = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        if client_order_id:
            body["client_order_id"] = client_order_id
        return cast(
            dict[str, Any],
            self._request("POST", self.trading_base_url, "/v2/orders", json_body=body, retry=False),
        )


if __name__ == "__main__":
    client = AlpacaClient()
    account = client.get_account()
    print(
        f"Account status: {account['status']}, equity: ${float(account['equity']):,.2f}, "
        f"buying power: ${float(account['buying_power']):,.2f}"
    )
