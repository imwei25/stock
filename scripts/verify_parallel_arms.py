"""Minimal example 校验:portfolio-ab 两 arm 串行 vs 并行结果在 rtol=1e-12 内一致.

Usage:
    .venv/Scripts/python.exe scripts/verify_parallel_arms.py --config <ab_yaml_path>

Prereq:
    需要事先有一份小型 portfolio_ab.yaml(2 arm, ~30 stocks, 1 年历史) +
    对应 base config + cache_dir 已经 warm 起来。

跑完看到 'OK: serial == parallel for all arms (rtol=1e-12, atol=0)' 即通过。
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from stockpool.config import load_config
from stockpool.fetcher import load_universe_cache
from stockpool.portfolio_ab.config import load_portfolio_ab_config
from stockpool.portfolio_ab.runner import run_portfolio_ab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True,
        help="path to portfolio_ab.yaml",
    )
    args = parser.parse_args()

    ab_cfg = load_portfolio_ab_config(args.config)
    base_cfg = load_config(ab_cfg.base_config)

    # Load pool_data and derive sector_map / name_map from universe.parquet.
    pool_data = load_universe_cache(base_cfg.data.cache_dir)
    universe = pd.read_parquet(f"{base_cfg.data.cache_dir}/universe.parquet")
    sector_map = dict(zip(
        universe["code"],
        universe.get("industry", [""] * len(universe)),
    ))
    name_map = dict(zip(universe["code"], universe["name"]))

    res_serial = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data, sector_map, name_map,
        parallel_arms=False,
    )
    res_par = run_portfolio_ab(
        ab_cfg, base_cfg, pool_data, sector_map, name_map,
        parallel_arms=True, refresh_scores=False,
    )

    for arm in ab_cfg.arms:
        m_s = res_serial.arms[arm].primary_metrics
        m_p = res_par.arms[arm].primary_metrics
        for k in m_s:
            v_s, v_p = m_s[k], m_p[k]
            if isinstance(v_s, float) and np.isnan(v_s):
                assert isinstance(v_p, float) and np.isnan(v_p), (
                    f"{arm}.{k}: NaN mismatch"
                )
            else:
                assert np.isclose(v_s, v_p, rtol=1e-12, atol=0), (
                    f"{arm}.{k}: serial={v_s} parallel={v_p}"
                )
        np.testing.assert_allclose(
            res_serial.arms[arm].primary_curve["equity"].values,
            res_par.arms[arm].primary_curve["equity"].values,
            rtol=1e-12, atol=0,
            equal_nan=True,
        )
        print(f"OK: arm={arm} serial == parallel (rtol=1e-12, atol=0)")
    print("OK: serial == parallel for all arms (rtol=1e-12, atol=0)")


if __name__ == "__main__":
    main()
