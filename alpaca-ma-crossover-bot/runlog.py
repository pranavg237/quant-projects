"""
Structured JSON-lines logging for the bot.

Every log line is one JSON object:

    {"ts": "2026-09-28T20:15:02.113Z", "level": "info", "run_id": "20260928T201502Z-3f9a1c2e",
     "event": "risk_check", "symbol": "SPY", "approved": true, "checks": [...]}

so a day's runs can be grepped by run_id or filtered with `jq 'select(.event=="order_blocked")'`.

Secrets are kept out in two layers:
  1. The code never passes credentials to the logger, and the HTTP client scrubs its own
     key values out of any error text it raises (a broker error body could echo them).
  2. As a safety net, the formatter redacts any field whose *name* looks sensitive, and
     replaces any occurrence of the configured secret *values* in the final line.
tests/test_logging.py checks both layers.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import uuid
from collections.abc import Iterable
from typing import IO, Any

REDACTED = "[REDACTED]"
_SENSITIVE_NAME = re.compile(
    r"secret|password|token|authorization|api[_-]?key|key[_-]?id", re.IGNORECASE
)


def new_run_id(now: dt.datetime | None = None) -> str:
    """A sortable, unique id for one bot invocation, e.g. 20260928T201502Z-3f9a1c2e."""
    now = now or dt.datetime.now(dt.UTC)
    return f"{now.astimezone(dt.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def _sanitize(value: Any) -> Any:
    """Recursively redact values stored under sensitive-looking keys."""
    if isinstance(value, dict):
        return {
            k: (REDACTED if _SENSITIVE_NAME.search(str(k)) else _sanitize(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


class JsonLinesFormatter(logging.Formatter):
    """Formats records produced by `RunLogger.event()` as single-line JSON."""

    def __init__(self, secrets: Iterable[str | None] = ()) -> None:
        super().__init__()
        # Ignore empty/very short values so we never redact ordinary text by accident.
        self._secrets = sorted({s for s in secrets if s and len(s) >= 6}, key=len, reverse=True)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, dt.UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "run_id": getattr(record, "run_id", None),
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(_sanitize(fields))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        line = json.dumps(payload, default=str)
        for secret in self._secrets:
            line = line.replace(secret, REDACTED)
        return line


def configure_logger(
    stream: IO[str], secrets: Iterable[str | None] = (), name: str = "bot"
) -> logging.Logger:
    """Return a logger named `name` that writes JSON lines to `stream` (replacing any
    handlers it had), with secret-value redaction for `secrets`."""
    logger = logging.getLogger(name)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonLinesFormatter(secrets))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class RunLogger:
    """Attaches a run id to every event. Usage: `log.event("decision", symbol="SPY", ...)`."""

    def __init__(self, run_id: str, logger: logging.Logger | None = None) -> None:
        self.run_id = run_id
        self._logger = logger or logging.getLogger("bot")

    def event(self, event: str, level: int = logging.INFO, **fields: Any) -> None:
        self._logger.log(level, event, extra={"run_id": self.run_id, "fields": fields})
