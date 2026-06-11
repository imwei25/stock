"""B8 / P2-15:factor panel 缓存 sig/manifest 覆盖补全。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def _daily(n, start="2025-01-06"):
    dates = pd.bdate_range(start, periods=n)
    close = 10 + np.arange(n) * 0.01
    return pd.DataFrame({
        "date": dates, "open": close, "high": close + 0.1,
        "low": close - 0.1, "close": close, "volume": 1e6,
    })


def test_sig_changes_with_history_start():
    """history_days 改变(first_date 移动)而 last_date 不变 → sig 必须变。"""
    from stockpool.strategy_factory import _factor_panel_sig
    long_pool = {"000001": _daily(100)}
    short_pool = {"000001": _daily(100).tail(50).reset_index(drop=True)}
    sig_long, _ = _factor_panel_sig(["momentum_5"], long_pool)
    sig_short, _ = _factor_panel_sig(["momentum_5"], short_pool)
    assert sig_long != sig_short, "warmup 段不同的两个窗口不能共用缓存"


def test_industry_map_refresh_invalidates_cache(tmp_path):
    """industry_map parquet mtime 变化 → 缓存 manifest 判 stale 重建。"""
    from stockpool.strategy_factory import load_or_build_factor_panel
    pool = {"000001": _daily(60), "000002": _daily(60)}

    fp1, _ = load_or_build_factor_panel(["momentum_5"], pool, tmp_path)
    # 模拟 industry map 刷新(建缓存时不存在 → 出现 = 变化)
    time.sleep(0.05)
    (tmp_path / "stock_industry_map.parquet").write_bytes(b"")
    pd.DataFrame({"code": ["000001"], "industry": ["X"]}).to_parquet(
        tmp_path / "stock_industry_map.parquet")

    import logging
    logging.getLogger("stockpool.strategy_factory").setLevel(logging.INFO)
    # 第二次调用应检测到 stale 并重建(不抛错即可;断言 manifest 记录了新快照)
    fp2, _ = load_or_build_factor_panel(["momentum_5"], pool, tmp_path)
    sig_dirs = list((tmp_path / "factor_panels").iterdir())
    manifest = json.loads((sig_dirs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["aux_snapshots"]["industry_map"] is not None, (
        "重建后的 manifest 应记录 industry_map 快照"
    )
