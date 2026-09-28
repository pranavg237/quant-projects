"""
Minimal Alpaca REST client - just the handful of endpoints this bot needs
(account info, historical bars, current position, market orders). No
`alpaca-py` SDK dependency, built directly on `requests`.

Reads credentials from environment variables:
  ALPACA_API_KEY_ID
  ALPACA_API_SECRET_KEY
  ALPACA_BASE_URL       (defaults to the PAPER endpoint - see below)

Get paper keys at https://app.alpaca.markets/paper/dashboard/overview
(free, no funding required). Never hardcode keys in this file - use a
.env file (see .env.example) or export them in your shell.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import requests

DEFAULT_TRADING_BASE_URL = "https://paper-api.alpaca.markets"
DEFAULT_DATA_BASE_URL = "https://data.alpaca.markets"


class AlpacaError(RuntimeError):
    pass


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    avg_entry_price: float


class AlpacaClient:
    def __init__(self, api_key_id: str | None = None, api_secret_key: str | None = None,
                 trading_base_url: str | None = None, data_base_url: str | None = None):
        self.api_key_id = api_key_id or os.environ.get("ALPACA_API_KEY_ID")
        self.api_secret_key = api_secret_key or os.environ.get("ALPACA_API_SECRET_KEY")
        if not self.api_key_id or not self.api_secret_key:
            raise AlpacaError(
                "Missing Alpaca credentials. Set ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY (see .env.example)."
            )
        self.trading_base_url = trading_base_url or os.environ.get("ALPACA_BASE_URL", DEFAULT_TRADING_BASE_URL)
        self.data_base_url = data_base_url or DEFAULT_DATA_BASE_URL

        self._headers = {
            "APCA-API-KEY-ID": self.api_key_id,
            "APCA-API-SECRET-KEY": self.api_secret_key,
        }

    # -- internal ----------------------------------------------------
    def _get(self, base_url: str, path: str, params: dict | None = None) -> dict:
        resp = requests.get(f"{base_url}{path}", headers=self._headers, params=params, timeout=15)
        if not resp.ok:
            raise AlpacaError(f"GET {path} -> {resp.status_code}: {resp.text}")
        return resp.json()

    def _post(self, base_url: str, path: str, json_body: dict) -> dict:
        resp = requests.post(f"{base_url}{path}", headers=self._headers, json=json_body, timeout=15)
        if not resp.ok:
            raise AlpacaError(f"POST {path} -> {resp.status_code}: {resp.text}")
        return resp.json()

    # -- account / positions ------------------------------------------
    def get_account(self) -> dict:
        return self._get(self.trading_base_url, "/v2/account")

    def get_position(self, symbol: str) -> Position | None:
        try:
            data = self._get(self.trading_base_url, f"/v2/positions/{symbol}")
        except AlpacaError as e:
            if "404" in str(e):
                return None
            raise
        return Position(
            symbol=data["symbol"],
            qty=float(data["qty"]),
            market_value=float(data["market_value"]),
            avg_entry_price=float(data["avg_entry_price"]),
        )

    # -- market data ---------------------------------------------------
    def get_daily_bars(self, symbol: str, start: str, end: str | None = None, limit: int = 1000) -> list[dict]:
        """Daily OHLCV bars, oldest first. `start`/`end` are YYYY-MM-DD strings.

        Requests split- and dividend-adjusted prices (`adjustment=all`). Alpaca's default
        is raw prices, where a 4-for-1 split looks like a 75% crash and can flip a
        moving-average signal on a corporate action rather than a trend.
        """
        params = {"timeframe": "1Day", "start": start, "limit": limit, "adjustment": "all"}
        if end:
            params["end"] = end
        data = self._get(self.data_base_url, f"/v2/stocks/{symbol}/bars", params=params)
        return data.get("bars", [])

    # -- orders ----------------------------------------------------------
    def submit_market_order(self, symbol: str, qty: float, side: str, time_in_force: str = "day") -> dict:
        if side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        body = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        return self._post(self.trading_base_url, "/v2/orders", body)


if __name__ == "__main__":
    client = AlpacaClient()
    account = client.get_account()
    print(f"Account status: {account['status']}, equity: ${float(account['equity']):,.2f}, "
          f"buying power: ${float(account['buying_power']):,.2f}")
