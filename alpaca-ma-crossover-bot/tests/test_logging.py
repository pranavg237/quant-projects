"""
Structured logging: format, run id, and that API keys never reach the logs.

The secret tests run full bot invocations through main(), including error paths where a
misbehaving server echoes the request headers (with the keys) back in its error body.
"""
import datetime as dt
import io
import json
import logging
import os
import sys
import uuid

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import bot  # noqa: E402
from fake_alpaca import FakeAlpacaServer  # noqa: E402
from runlog import REDACTED, JsonLinesFormatter, RunLogger, configure_logger, new_run_id  # noqa: E402

KEY_ID, SECRET = FakeAlpacaServer.KEY_ID, FakeAlpacaServer.SECRET
REAL_RUN_ONCE = bot.run_once
REAL_CONFIGURE = bot.configure_logger
FRI_AFTER_CLOSE = dt.datetime(2026, 9, 25, 20, 15, tzinfo=dt.timezone.utc)


def test_events_are_json_lines_with_run_id_and_timestamp():
    stream = io.StringIO()
    log = RunLogger("run-123", configure_logger(stream, name=f"t-{uuid.uuid4().hex}"))
    log.event("decision", symbol="SPY", decision="buy 10")
    log.event("order_blocked", logging.WARNING, reasons=["x"])
    first, second = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert first["run_id"] == "run-123" and first["event"] == "decision"
    assert first["symbol"] == "SPY" and first["level"] == "info"
    assert first["ts"].endswith("Z")
    assert second["level"] == "warning" and second["reasons"] == ["x"]


def test_run_ids_are_unique():
    assert new_run_id() != new_run_id()


def test_formatter_redacts_sensitive_field_names_even_nested():
    fmt = JsonLinesFormatter()
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "evt", None, None)
    record.fields = {"api_secret_key": "abc", "headers": {"APCA-API-KEY-ID": "k1"},
                     "Authorization": "Bearer x", "client_order_id": "keep-me"}
    out = json.loads(fmt.format(record))
    assert out["api_secret_key"] == REDACTED
    assert out["headers"]["APCA-API-KEY-ID"] == REDACTED
    assert out["Authorization"] == REDACTED
    assert out["client_order_id"] == "keep-me"


def test_formatter_redacts_known_secret_values_anywhere():
    fmt = JsonLinesFormatter(secrets=[SECRET, None, ""])
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "evt", None, None)
    record.fields = {"error": f"boom {SECRET} boom"}
    line = fmt.format(record)
    assert SECRET not in line and REDACTED in line


def _run_main(monkeypatch, capsys, server, *args, redact=True):
    """Run bot.main() against `server`, returning (exit code, stderr text)."""
    monkeypatch.setenv("ALPACA_API_KEY_ID", KEY_ID)
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", SECRET)
    monkeypatch.delenv("BOT_KILL_SWITCH", raising=False)
    monkeypatch.setattr(bot, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(bot, "AlpacaClient", lambda: server.client())
    # Without redact, disable the value-redaction safety net to prove the code itself
    # never puts keys into a log line.
    monkeypatch.setattr(bot, "configure_logger",
                        lambda stream, secrets=(): REAL_CONFIGURE(stream, secrets if redact else ()))
    # Freeze "now" to the fake's last bar (Friday after the close).
    monkeypatch.setattr(bot, "run_once",
                        lambda *a, **k: REAL_RUN_ONCE(*a, now=FRI_AFTER_CLOSE, **k))
    code = bot.main(["--symbol", "SPY", "--short", "5", "--long", "20", *args])
    return code, capsys.readouterr().err


@pytest.mark.parametrize("redact", [True, False])
def test_keys_never_logged_on_success_or_failure(monkeypatch, capsys, tmp_path, redact):
    kill = str(tmp_path / "KILL")
    outputs = []

    ok = FakeAlpacaServer(list(range(100, 140)))
    outputs.append(_run_main(monkeypatch, capsys, ok, "--kill-switch-file", kill, redact=redact))

    rejected = FakeAlpacaServer(list(range(100, 140)))
    rejected.echo_headers_in_errors = True
    rejected.fail("POST", "/v2/orders", (403, {"message": "x"}))
    outputs.append(_run_main(monkeypatch, capsys, rejected, "--kill-switch-file", kill, redact=redact))

    broken = FakeAlpacaServer(list(range(100, 140)))
    broken.echo_headers_in_errors = True
    broken.fail("GET", "/v2/account", *[(500, "x")] * 3)
    outputs.append(_run_main(monkeypatch, capsys, broken, "--kill-switch-file", kill, redact=redact))

    timeout = FakeAlpacaServer(list(range(100, 140)))
    timeout.fail("POST", "/v2/orders", requests.ReadTimeout(f"timed out; headers={SECRET}"))
    outputs.append(_run_main(monkeypatch, capsys, timeout, "--kill-switch-file", kill, redact=redact))

    codes = [c for c, _ in outputs]
    assert codes == [0, 1, 1, 1]
    for _, err in outputs:
        lines = [json.loads(line) for line in err.splitlines()]
        assert lines, "expected log output"
        assert len({line["run_id"] for line in lines}) == 1
        assert KEY_ID not in err and SECRET not in err
    # The failures were logged, not swallowed.
    assert '"event": "order_submit_error"' in outputs[1][1]
    assert '"event": "run_failed"' in outputs[2][1]
    assert "[REDACTED]" in outputs[2][1]   # the echoed headers were scrubbed


def test_main_exit_code_2_when_kill_switch_file_present(monkeypatch, capsys, tmp_path):
    flag = tmp_path / "KILL_SWITCH"
    flag.touch()
    server = FakeAlpacaServer(list(range(100, 140)))
    code, err = _run_main(monkeypatch, capsys, server, "--kill-switch-file", str(flag))
    assert code == 2 and server.posts == 0
    finished = [json.loads(line) for line in err.splitlines()][-1]
    assert finished["event"] == "run_finished" and finished["outcome"] == "blocked"


def test_main_reads_risk_limits_from_env(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BOT_MAX_POSITION_NOTIONAL", "1000")
    server = FakeAlpacaServer(list(range(100, 140)))
    code, err = _run_main(monkeypatch, capsys, server, "--kill-switch-file", str(tmp_path / "k"))
    assert code == 2 and server.posts == 0
    started = json.loads(err.splitlines()[0])
    assert started["limits"]["max_position_notional"] == 1000.0


def test_main_cli_flag_overrides_env(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("BOT_MAX_POSITION_NOTIONAL", "1000")
    server = FakeAlpacaServer(list(range(100, 140)))
    code, _ = _run_main(monkeypatch, capsys, server, "--kill-switch-file", str(tmp_path / "k"),
                        "--max-position-notional", "30000")
    assert code == 0 and server.posts == 1


def test_bad_env_limit_fails_loudly(monkeypatch):
    monkeypatch.setenv("BOT_MAX_DAILY_LOSS_PCT", "two percent")
    with pytest.raises(SystemExit, match="BOT_MAX_DAILY_LOSS_PCT"):
        bot.build_parser()
