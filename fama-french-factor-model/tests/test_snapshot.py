"""Snapshots: committed files match their manifests, and commands run from a data dir offline."""
import hashlib
import io
import json
import socket
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_factors

from ffmodel import cli, data, snapshot

PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOTS = sorted((PROJECT / "data").glob("snapshot-*"))
PORTFOLIO_NAMES = [
    ("SMALL " if i == 0 else "BIG " if i == 4 else f"ME{i + 1} ")
    + ("LoBM" if j == 0 and i in (0, 4) else "HiBM" if j == 4 and i in (0, 4) else f"BM{j + 1}")
    for i in range(5) for j in range(5)
]


@pytest.fixture
def no_network(monkeypatch):
    """Fail any attempt to download or open a socket."""
    def refuse(*args, **kwargs):
        raise AssertionError("tried to use the network")

    monkeypatch.setattr(data, "_download", refuse)
    monkeypatch.setattr(data.requests.sessions.Session, "request", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


# ---------------------------------------------------------------- committed snapshots

@pytest.mark.parametrize("directory", SNAPSHOTS, ids=[d.name for d in SNAPSHOTS])
def test_committed_snapshot_files_match_manifest(directory):
    manifest = snapshot.read_manifest(directory)
    listed = set(manifest["french_library"])
    assert listed == {p.name for p in directory.glob("*.zip")}, "every zip is listed, and every listed zip exists"
    assert all(snapshot.verify_snapshot(directory).values())
    for name, entry in manifest["french_library"].items():
        body = (directory / name).read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        assert entry["url"] == data.BASE_URL + name
        if "bytes" in entry:
            assert len(body) == entry["bytes"]
    if "returns" in manifest:
        assert (directory / manifest["returns"]["file"]).exists()


@pytest.mark.parametrize("directory", SNAPSHOTS, ids=[d.name for d in SNAPSHOTS])
def test_committed_snapshot_sample_periods(directory):
    manifest = snapshot.read_manifest(directory)
    for name, entry in manifest["french_library"].items():
        title = ("value weight", "return") if "period_of" in entry else ()
        table = data._select(data._read_zip(directory / name), manifest["frequency"], title)
        assert f"{table.data.index[0]:%Y-%m}" == entry["first"]
        assert f"{table.data.index[-1]:%Y-%m}" == entry["last"]
        if "period_of" in entry:
            assert table.title == entry["period_of"]


def test_example_reports_reproduce_from_snapshot(no_network, tmp_path):
    """The committed example-factors and example-25-portfolios reports come out of the snapshot unchanged."""
    snap = str(PROJECT / "data" / "snapshot-2026-09-29")
    runs = {
        "example-factors": ["factors", "--model", "ff6", "--end", "2026-07"],
        "example-25-portfolios": ["test-portfolios", "--dataset", "25_Portfolios_5x5", "--start", "1963-07",
                                  "--end", "2026-07", "--compare"],
    }
    for name, argv in runs.items():
        out = tmp_path / name
        assert cli.main(argv + ["--data-dir", snap, "--out", str(out)]) == 0
        expected = (PROJECT / "reports" / name / "report.md").read_text()
        assert (out / "report.md").read_text() == expected, name


# ---------------------------------------------------------------- a small synthetic snapshot

def _french_text(frame: pd.DataFrame, title: str = "") -> str:
    """``frame`` (decimals, month-end index) in the French library's CSV layout (percent)."""
    lines = [title, ""] if title else []
    lines.append("," + ",".join(frame.columns))
    for date, row in frame.iterrows():
        lines.append(f"{date:%Y%m}," + ",".join(f"{100 * v:7.2f}" for v in row))
    return "\r\n".join(lines) + "\r\n\r\n"


def _zip(path: Path, text: str) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr(path.name.replace("_CSV.zip", ".csv"), text)
    path.write_bytes(buffer.getvalue())


def _fixture_texts(T: int = 120) -> dict:
    f = make_factors(T=T, start="2000-01-31")
    rng = np.random.default_rng(7)
    betas = rng.uniform(0.3, 1.4, size=(3, 25))
    value = f["RF"].to_numpy()[:, None] + f[["Mkt-RF", "SMB", "HML"]].to_numpy() @ betas
    value = pd.DataFrame(value + rng.normal(0.001, 0.01, (T, 25)), index=f.index, columns=PORTFOLIO_NAMES)
    equal = value + 0.002
    return {
        "F-F_Research_Data_Factors": _french_text(f[["Mkt-RF", "SMB", "HML", "RF"]]),
        "F-F_Research_Data_5_Factors_2x3": _french_text(f[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]]),
        "F-F_Momentum_Factor": _french_text(f[["MOM"]].rename(columns={"MOM": "Mom"})),
        "25_Portfolios_5x5": _french_text(value, "  Average Value Weighted Returns -- Monthly")
        + _french_text(equal, "  Average Equal Weighted Returns -- Monthly"),
    }


@pytest.fixture
def fixture_snapshot(monkeypatch, tmp_path):
    """A French-only snapshot of synthetic data, made by ``ffmodel snapshot`` with a fake downloader."""
    texts = _fixture_texts()
    monkeypatch.setattr(data, "_download", lambda name, path: _zip(path, texts[name]))
    snap = tmp_path / "snapshot-2000-01-01"
    assert cli.main(["snapshot", "--portfolios", "25_Portfolios_5x5", "--out", str(snap)]) == 0
    return snap


def test_snapshot_command_saves_portfolios_without_tickers(fixture_snapshot):
    manifest = json.loads((fixture_snapshot / "manifest.json").read_text())
    assert len(manifest["french_library"]) == 4
    assert "returns" not in manifest and not (fixture_snapshot / "returns.csv").exists()
    entry = manifest["french_library"]["25_Portfolios_5x5_CSV.zip"]
    assert (entry["first"], entry["last"]) == ("2000-01", "2009-12")
    assert entry["period_of"] == "Average Value Weighted Returns -- Monthly"
    assert entry["bytes"] == (fixture_snapshot / "25_Portfolios_5x5_CSV.zip").stat().st_size
    assert all(snapshot.verify_snapshot(fixture_snapshot).values())


def test_commands_load_from_data_dir_without_network(fixture_snapshot, no_network, tmp_path):
    day = snapshot.read_manifest(fixture_snapshot)["downloaded_utc"][:10]
    source = f"saved snapshot `{fixture_snapshot.name}`, downloaded {day}"

    out = tmp_path / "tp"
    argv = ["test-portfolios", "--compare", "--data-dir", str(fixture_snapshot), "--out", str(out)]
    assert cli.main(argv) == 0
    text = (out / "report.md").read_text()
    assert "Jan 2000 to Dec 2009 (120 periods)" in text and source in text
    assert "Fama-MacBeth: Fama-French 6-factor" in text and (out / "alpha-heatmap.png").exists()

    out = tmp_path / "factors"
    assert cli.main(["factors", "--model", "ff6", "--data-dir", str(fixture_snapshot), "--out", str(out)]) == 0
    text = (out / "report.md").read_text()
    assert source in text and "| MOM |" in text and (out / "spanning.csv").exists()

    # The data dir must be complete: a missing file is an error, never a download.
    assert cli.main(["test-portfolios", "--dataset", "6_Portfolios_2x3", "--data-dir", str(fixture_snapshot),
                     "--no-report"]) == 1


def test_changed_snapshot_file_is_refused(fixture_snapshot, no_network, capsys):
    path = fixture_snapshot / "F-F_Momentum_Factor_CSV.zip"
    texts = _fixture_texts()
    _zip(path, texts["F-F_Momentum_Factor"].replace("Mom", "MOM"))  # a valid zip, but not the one downloaded
    assert snapshot.verify_snapshot(fixture_snapshot)[path.name] is False
    with pytest.raises(RuntimeError, match="does not match the SHA-256"):
        data.load_factors("carhart", data_dir=fixture_snapshot)
    assert cli.main(["factors", "--model", "ff6", "--data-dir", str(fixture_snapshot), "--no-report"]) == 1
    assert "SHA-256" in capsys.readouterr().err
    data.load_factors("ff3", data_dir=fixture_snapshot)  # files that still match load as before
