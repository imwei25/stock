"""Smoke test for `python -m stockpool ab`."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from stockpool.cli import main


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def isolated_cache_two_stocks(tmp_path, monkeypatch):
    cache_dir = tmp_path / "data"
    cache_dir.mkdir()
    for code, seed in [("605589", 7), ("300750", 19)]:
        rng = np.random.default_rng(seed)
        n = 220
        returns = rng.normal(0.0005, 0.02, n)
        close = 100.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=n, freq="B"),
            "open":  close * 0.998, "high": close * 1.005,
            "low":   close * 0.995, "close": close,
            "volume": rng.integers(500_000, 5_000_000, n).astype(float),
        })
        df.to_parquet(cache_dir / f"{code}_daily.parquet", index=False)
    cache_last = pd.date_range("2024-01-02", periods=220, freq="B")[-1]
    fresh_today = pd.Timestamp(cache_last) + pd.Timedelta(days=1)
    monkeypatch.setattr("stockpool.fetcher._today", lambda: fresh_today)
    return cache_dir


def _write_configs(tmp_path: Path, cache_dir: Path) -> tuple[Path, Path]:
    raw = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["data"]["cache_dir"] = str(cache_dir)
    raw["data"]["history_days"] = 200
    raw["report"]["output_dir"] = str(tmp_path / "reports")
    raw["stocks"] = [
        {"code": "605589", "name": "Alpha", "sector": ""},
        {"code": "300750", "name": "Bravo", "sector": ""},
    ]
    base_path = tmp_path / "config.yaml"
    base_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # factors_file 相对配置目录解析(P1-9)→ selection.json 一并复制
    sel_src = PROJECT_ROOT / "reports" / "selection.json"
    if sel_src.exists():
        (tmp_path / "reports").mkdir(exist_ok=True)
        (tmp_path / "reports" / "selection.json").write_bytes(sel_src.read_bytes())

    ab_raw = {
        "base_config": "config.yaml",
        "arms": {
            "single": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "single"},
            },
            "multi": {
                "strategy": {"name": "composite_verdict"},
                "backtest": {"equity_curve_holding_days": [10], "engine": "multi_lot"},
            },
        },
    }
    ab_path = tmp_path / "ab.yaml"
    # sort_keys=False so arm insertion order is preserved
    ab_path.write_text(yaml.safe_dump(ab_raw, sort_keys=False), encoding="utf-8")
    return ab_path, base_path


def test_cmd_ab_smoke_produces_html(tmp_path, isolated_cache_two_stocks):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    rc = main(["ab", "--config", str(ab_path)])
    assert rc == 0
    latest = tmp_path / "reports" / "ab" / "latest.html"
    assert latest.exists()
    assert latest.stat().st_size > 2048
    html = latest.read_text(encoding="utf-8")
    assert "single" in html and "multi" in html
    assert "605589" in html and "300750" in html


def test_cmd_ab_arm_flag_runs_only_one_arm(tmp_path, isolated_cache_two_stocks, capsys):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    rc = main(["ab", "--config", str(ab_path), "--arm", "single"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "single" in captured.out
    # No HTML written when --arm is used
    assert not (tmp_path / "reports" / "ab" / "latest.html").exists()


def test_cmd_ab_arm_unknown_returns_2(tmp_path, isolated_cache_two_stocks):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    rc = main(["ab", "--config", str(ab_path), "--arm", "typo"])
    assert rc == 2


def test_cmd_ab_no_share_pool_propagates(tmp_path, isolated_cache_two_stocks, monkeypatch):
    ab_path, _ = _write_configs(tmp_path, isolated_cache_two_stocks)
    calls = {"count": 0}
    from stockpool.ab import runner as ab_runner
    real = ab_runner._decide_pool_sharing
    def _counted(*a, **kw):
        calls["count"] += 1
        return real(*a, **kw)
    monkeypatch.setattr(ab_runner, "_decide_pool_sharing", _counted)
    rc = main(["ab", "--config", str(ab_path), "--no-share-pool"])
    assert rc == 0
    assert calls["count"] == 0  # short-circuited by --no-share-pool
