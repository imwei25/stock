"""Force-refresh baostock fundamentals + verify all 9 fundamental factors compute.

Steps:
  1. force_refresh 5 张 baostock 表(profit / growth / balance / cash_flow / dupont)
     —— 会绕开 30 天缓存,从 baostock API 全量重抓 4000+ 票 × ~16 季报
  2. 装 universe + OHLCV panel(用 load_universe_cache,不进 ml_factor)
  3. 对 9 个 fundamental 因子各 compute 一次,报告 shape / 非 NaN 覆盖率 / value 分位数

Usage:
    .venv/Scripts/python.exe scripts/refresh_fundamentals_and_check.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from stockpool._instrumentation import checkpoint, panel_size_mb
from stockpool.config import load_config
from stockpool.fetcher import load_universe_cache
from stockpool.fundamentals_loader import load_or_build_fundamentals, _VALID_TABLES


def step1_refresh_fundamentals(cache_dir: str) -> dict[str, pd.DataFrame]:
    """force_refresh 全部 5 张表,逐表打 checkpoint 看时间 + RSS。"""
    tables: dict[str, pd.DataFrame] = {}
    for name in _VALID_TABLES:
        checkpoint(f"fundamentals fetch start: {name}")
        t0 = time.perf_counter()
        df = load_or_build_fundamentals(
            name,
            codes=None,
            cache_dir=cache_dir,
            max_age_days=30,
            force_refresh=True,
        )
        elapsed = time.perf_counter() - t0
        tables[name] = df
        checkpoint(f"fundamentals fetch done:  {name}", extra={
            "rows": len(df),
            "codes": df["code"].nunique() if not df.empty else 0,
            "elapsed_s": elapsed,
        })
    return tables


def step2_build_ohlcv_panel(cfg):
    """装 universe + OHLCV panel(只为后续 compute 用)。"""
    checkpoint("load_universe_cache: start")
    pool_data = load_universe_cache(
        cfg.data.cache_dir,
        cfg.data.history_days,
        warmup_days=cfg.data.warmup_days,
    )
    checkpoint("load_universe_cache: done", extra={"n_stocks": len(pool_data)})

    # 拼 5 个 wide OHLCV 宽表,与 build_factor_panel 第一步一致。
    per_stock = {}
    for code, df in pool_data.items():
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"])
        per_stock[code] = d.set_index("date").sort_index()
    all_dates = sorted(set().union(*(d.index for d in per_stock.values())))
    idx = pd.DatetimeIndex(all_dates, name="date")
    panel = {}
    for field in ("open", "high", "low", "close", "volume"):
        panel[field] = pd.DataFrame(
            {code: d[field].reindex(idx) for code, d in per_stock.items()},
            index=idx,
        )
    checkpoint("OHLCV wide panel built", extra={
        "T": len(idx), "N": len(per_stock),
        "ohlcv_mb": panel_size_mb(panel),
    })
    return panel


def step3_compute_fundamental_factors(panel):
    """逐个 fundamental 因子 compute,报告每个的覆盖率 / 范围。"""
    from stockpool.factors.registry import list_specs, make_factor

    fund_specs = sorted(
        [s for s in list_specs() if "fundamental" in s.types],
        key=lambda s: s.base_name,
    )

    T = len(panel["close"])
    N = len(panel["close"].columns)
    total_cells = T * N

    print()
    print("=" * 90)
    print(f"{'factor':<28} {'T×N':>14} {'cov%':>7} {'min':>14} {'p50':>14} {'max':>14}")
    print("-" * 90)

    results = []
    for spec in fund_specs:
        try:
            t0 = time.perf_counter()
            f = make_factor(spec.base_name)
            out = f.compute(panel)
            elapsed = time.perf_counter() - t0

            shape = out.shape
            n_valid = int(out.notna().values.sum())
            cov = 100.0 * n_valid / max(1, shape[0] * shape[1])
            vals = out.values[np.isfinite(out.values)]
            if len(vals) > 0:
                vmin, vmed, vmax = (
                    float(np.percentile(vals, 1)),
                    float(np.percentile(vals, 50)),
                    float(np.percentile(vals, 99)),
                )
            else:
                vmin = vmed = vmax = float("nan")
            print(f"{spec.base_name:<28} {f'{shape[0]}x{shape[1]}':>14} {cov:6.1f}% "
                  f"{vmin:14.4g} {vmed:14.4g} {vmax:14.4g}")
            results.append((spec.base_name, "OK", elapsed, cov))
        except Exception as e:  # noqa: BLE001
            print(f"{spec.base_name:<28} ERROR: {type(e).__name__}: {e}")
            results.append((spec.base_name, "ERROR", 0.0, 0.0))
    print("=" * 90)
    print()

    failed = [r for r in results if r[1] != "OK"]
    low_cov = [r for r in results if r[1] == "OK" and r[3] < 5.0]
    print(f"Total fundamental factors: {len(fund_specs)}")
    print(f"  OK: {len(results) - len(failed)}")
    print(f"  failed: {len(failed)}  {[r[0] for r in failed]}")
    print(f"  low-coverage (<5%): {len(low_cov)}  {[r[0] for r in low_cov]}")


def main():
    checkpoint("start")
    cfg = load_config("config.yaml")
    print(f"cache_dir={cfg.data.cache_dir}, history_days={cfg.data.history_days}, "
          f"warmup_days={cfg.data.warmup_days}, source={cfg.data.source}")

    # Step 1: refresh fundamentals.
    tables = step1_refresh_fundamentals(cfg.data.cache_dir)
    print()
    print("Fundamentals refresh summary:")
    for name, df in tables.items():
        n_codes = df["code"].nunique() if not df.empty else 0
        print(f"  {name:<14} rows={len(df):>10}  codes={n_codes:>6}")
    print()

    # Step 2: build OHLCV panel.
    panel = step2_build_ohlcv_panel(cfg)

    # Step 3: compute fundamental factors.
    step3_compute_fundamental_factors(panel)

    checkpoint("done")


if __name__ == "__main__":
    main()
