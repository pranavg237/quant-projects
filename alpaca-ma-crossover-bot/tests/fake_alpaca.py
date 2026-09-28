"""
An in-memory fake of the Alpaca REST API, plugged in at the HTTP boundary.

`FakeAlpacaServer` implements `request()` like `requests.Session`, so the real
`AlpacaClient` (URL building, headers, retries, error parsing) runs unchanged against it.
No network, no credentials.

Response shapes and error codes follow Alpaca's public API docs (e.g. 403 with code
40310000 for insufficient buying power, 422 for a duplicate client_order_id, 404 for "no
position"). They have NOT been checked against the live API from this repository.
"""
from __future__ import annotations

import copy
import datetime as dt
import itertools
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

TERMINAL = {"filled", "canceled", "expired", "rejected", "replaced"}


class FakeResponse:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400

    def json(self) -> Any:
        if isinstance(self._body, str):
            raise ValueError("not JSON")
        return copy.deepcopy(self._body)


class Fault:
    """A scripted failure for the next matching request.

    `error` is either an exception to raise or a (status, body) response to return.
    With `process_first=True` the request is handled normally *and then* the error is
    returned instead of the real response: the "order reached the broker but the reply
    was lost" case.
    """

    def __init__(self, error: BaseException | tuple[int, Any], process_first: bool = False) -> None:
        self.error = error
        self.process_first = process_first


class FakeAlpacaServer:
    KEY_ID = "PKFAKEKEYID00001"
    SECRET = "fakeSecretValue-9f8e7d6c5b4a"

    def __init__(self, closes: list[float], *, last_bar: dt.date = dt.date(2026, 9, 25),
                 equity: float = 100_000.0, last_equity: float | None = None,
                 buying_power: float | None = None, status: str = "ACTIVE",
                 position_qty: float = 0.0, market_open: bool = False) -> None:
        dates = pd.bdate_range(end=pd.Timestamp(last_bar), periods=len(closes))
        # Alpaca stamps daily bars at midnight New York time (04:00 UTC in summer).
        self.bars: list[dict] = [{"t": f"{d.date()}T04:00:00Z", "c": float(c)}
                                 for d, c in zip(dates, closes, strict=True)]
        self.account: dict[str, Any] = {
            "status": status, "trading_blocked": False, "equity": str(equity),
            "last_equity": str(equity if last_equity is None else last_equity),
            "buying_power": str(equity if buying_power is None else buying_power),
        }
        self.position_qty = position_qty
        self.clock = {"is_open": market_open, "timestamp": f"{last_bar}T20:15:00-04:00",
                      "next_open": f"{last_bar + dt.timedelta(days=3)}T09:30:00-04:00",
                      "next_close": f"{last_bar + dt.timedelta(days=3)}T16:00:00-04:00"}
        self.orders: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.faults: dict[tuple[str, str], list[Fault]] = {}
        self.echo_headers_in_errors = False
        # Called with each accepted order right after submission, to simulate fills.
        self.on_submit: Callable[[dict], None] | None = None
        self._ids = itertools.count(1)

    # -- test helpers --------------------------------------------------
    def fail(self, method: str, path: str, *errors: BaseException | tuple[int, Any],
             process_first: bool = False) -> None:
        self.faults.setdefault((method, path), []).extend(
            Fault(e, process_first) for e in errors)

    def count(self, method: str, path: str) -> int:
        return sum(1 for c in self.calls if c == (method, path))

    @property
    def posts(self) -> int:
        return self.count("POST", "/v2/orders")

    def fill(self, order: dict, qty: float, status: str) -> None:
        """Fill `qty` shares of `order` and set its status."""
        order["filled_qty"] = str(float(order["filled_qty"]) + qty)
        order["status"] = status
        self.position_qty += qty if order["side"] == "buy" else -qty

    def add_bar(self, close: float) -> None:
        nxt = (pd.Timestamp(self.bars[-1]["t"][:10]) + pd.offsets.BDay(1)).date()
        self.bars.append({"t": f"{nxt}T04:00:00Z", "c": float(close)})

    # -- the HTTP boundary ------------------------------------------------
    def request(self, method: str, url: str, headers: dict | None = None,
                params: dict | None = None, json: dict | None = None,
                timeout: float | None = None) -> FakeResponse:
        path = urlsplit(url).path
        self.calls.append((method, path))
        headers = headers or {}
        if (headers.get("APCA-API-KEY-ID") != self.KEY_ID
                or headers.get("APCA-API-SECRET-KEY") != self.SECRET):
            return FakeResponse(401, {"code": 40110000, "message": "request is not authorized"})

        queue = self.faults.get((method, path))
        fault = queue.pop(0) if queue else None
        if fault and not fault.process_first:
            return self._raise_or_return(fault, headers)
        response = self._route(method, path, params or {}, json)
        if fault:
            return self._raise_or_return(fault, headers)
        return response

    def _raise_or_return(self, fault: Fault, headers: dict) -> FakeResponse:
        if isinstance(fault.error, BaseException):
            raise fault.error
        status, body = fault.error
        if self.echo_headers_in_errors:
            # A badly behaved proxy/error page that reflects the request headers.
            body = f"upstream error; request headers were {headers}"
        return FakeResponse(status, body)

    def _route(self, method: str, path: str, params: dict, body: dict | None) -> FakeResponse:
        if method == "GET" and path == "/v2/account":
            return FakeResponse(200, self.account)
        if method == "GET" and path == "/v2/clock":
            return FakeResponse(200, self.clock)
        if method == "GET" and path.startswith("/v2/stocks/") and path.endswith("/bars"):
            assert params.get("adjustment") == "all"
            return FakeResponse(200, {"bars": self.bars, "next_page_token": None})
        if method == "GET" and path.startswith("/v2/positions/"):
            if not self.position_qty:
                return FakeResponse(404, {"code": 40410000, "message": "position does not exist"})
            price = self.bars[-1]["c"]
            return FakeResponse(200, {"symbol": path.rsplit("/", 1)[1], "qty": str(self.position_qty),
                                      "market_value": str(self.position_qty * price),
                                      "avg_entry_price": str(price)})
        if method == "GET" and path == "/v2/orders":
            assert params.get("status") == "open"
            return FakeResponse(200, [o for o in self.orders.values()
                                      if o["symbol"] == params.get("symbols") and o["status"] not in TERMINAL])
        if method == "GET" and path == "/v2/orders:by_client_order_id":
            for o in self.orders.values():
                if o["client_order_id"] == params.get("client_order_id"):
                    return FakeResponse(200, o)
            return FakeResponse(404, {"code": 40410000, "message": "order not found"})
        if method == "GET" and path.startswith("/v2/orders/"):
            order = self.orders.get(path.rsplit("/", 1)[1])
            if order is None:
                return FakeResponse(404, {"code": 40410000, "message": "order not found"})
            return FakeResponse(200, order)
        if method == "POST" and path == "/v2/orders":
            return self._submit(body or {})
        return FakeResponse(404, {"message": f"no route {method} {path}"})

    def _submit(self, body: dict) -> FakeResponse:
        cid = body.get("client_order_id")
        if cid and any(o["client_order_id"] == cid for o in self.orders.values()):
            return FakeResponse(422, {"code": 40010001, "message": "client_order_id must be unique"})
        qty = float(body["qty"])
        if body["side"] == "buy" and qty * self.bars[-1]["c"] > float(self.account["buying_power"]):
            return FakeResponse(403, {"code": 40310000, "message": "insufficient buying power"})
        order_id = f"ord-{next(self._ids):04d}"
        order = {"id": order_id, "client_order_id": cid or order_id, "symbol": body["symbol"],
                 "side": body["side"], "qty": body["qty"], "filled_qty": "0",
                 "type": body["type"], "time_in_force": body["time_in_force"],
                 "status": "accepted", "submitted_at": "2026-09-25T20:15:01Z"}
        self.orders[order_id] = order
        if self.on_submit:
            self.on_submit(order)
        return FakeResponse(200, order)

    def client(self, **kwargs: Any):
        """A real AlpacaClient wired to this fake, with no-op sleeps (recorded)."""
        from alpaca_client import AlpacaClient

        self.sleeps: list[float] = []
        return AlpacaClient(self.KEY_ID, self.SECRET, session=self,
                            sleep=self.sleeps.append, **kwargs)
