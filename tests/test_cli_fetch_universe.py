"""Tests for `python -m stockpool fetch-universe` source handling."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_cfg(tmp_path: Path, source: str) -> Path:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(tmp_path / "data")
    raw["data"]["source"] = source
    raw["data"]["history_days"] = 30
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return cfg_file


def _stub_listing(monkeypatch):
    """list_universe always must use mootdx (only impl), so stub the listing API."""
    df = pd.DataFrame({
        "code": ["605589", "603986"],
        "name": ["A", "B"],
        "market": ["sh", "sh"],
    })

    def _fake_list(source="mootdx"):
        return df

    monkeypatch.setattr("stockpool.cli.list_universe", _fake_list)


def test_fetch_universe_uses_cfg_source_when_no_cli_flag(tmp_path, monkeypatch):
    """When --source is omitted, fetch-universe must honor cfg.data.source."""
    cfg_file = _write_cfg(tmp_path, source="baostock")
    _stub_listing(monkeypatch)

    calls: list[str] = []

    def _fake_fetch_universe(codes, *, history_days, cache_dir, source,
                             force_refresh=False, max_workers=8):
        calls.append(source)
        return {c: pd.DataFrame() for c in codes}

    monkeypatch.setattr("stockpool.cli.fetch_universe", _fake_fetch_universe)

    rc = main(["fetch-universe", "--config", str(cfg_file)])
    assert rc == 0
    assert calls == ["baostock"], (
        f"expected baostock (from cfg.data.source), got {calls!r}"
    )


def test_fetch_universe_cli_source_overrides_cfg(tmp_path, monkeypatch):
    """--source on the CLI must win over cfg.data.source."""
    cfg_file = _write_cfg(tmp_path, source="baostock")
    _stub_listing(monkeypatch)

    calls: list[str] = []

    def _fake_fetch_universe(codes, *, history_days, cache_dir, source,
                             force_refresh=False, max_workers=8):
        calls.append(source)
        return {c: pd.DataFrame() for c in codes}

    monkeypatch.setattr("stockpool.cli.fetch_universe", _fake_fetch_universe)

    rc = main(["fetch-universe", "--config", str(cfg_file), "--source", "akshare"])
    assert rc == 0
    assert calls == ["akshare"]


def test_fetch_universe_force_refresh_on_source_change(tmp_path, monkeypatch):
    """If a prior fetch used a different source, this run must auto-force-refresh."""
    cfg_file = _write_cfg(tmp_path, source="baostock")
    _stub_listing(monkeypatch)

    # Seed marker so it looks like the prior run was mootdx.
    cache_dir = Path(yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
                     ["data"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / ".data_source").write_text("mootdx", encoding="utf-8")

    seen: dict = {}

    def _fake_fetch_universe(codes, *, history_days, cache_dir, source,
                             force_refresh=False, max_workers=8):
        seen["source"] = source
        seen["force_refresh"] = force_refresh
        return {c: pd.DataFrame() for c in codes}

    monkeypatch.setattr("stockpool.cli.fetch_universe", _fake_fetch_universe)

    rc = main(["fetch-universe", "--config", str(cfg_file)])
    assert rc == 0
    assert seen["source"] == "baostock"
    assert seen["force_refresh"] is True, (
        "marker said mootdx, cfg says baostock → CLI must force_refresh"
    )
