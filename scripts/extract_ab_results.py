"""F1 收尾:重放各 A/B 对照(全缓存,快),导出结构化 diff + 每组独立 HTML。

第一轮编排把 11 组报告都写进了同一个 reports/ab/<date>.html(互相覆盖);
本脚本直接调 run_ab + compute_diff_table,把每组的 sharpe/return/max_dd
diff 写进 reports/ab_rerun_results.json,并把每组 HTML 存到
reports/ab_rerun/<group>/。
"""
import json
import sys
from pathlib import Path

from stockpool.ab.config import load_ab_config
from stockpool.ab.report import compute_diff_table, render_ab_report, shared_holding_days
from stockpool.ab.runner import run_ab
from stockpool.config import load_config

GROUPS = [
    ("P2-1_embargo", "configs/ab/runbook_P2_1.yaml"),
    ("P0-1_composite_vs_lgblgb", "configs/ab/runbook_P0_1.yaml"),
    ("P0-2_lassoic_vs_lgblgb", "configs/ab/runbook_P0_2.yaml"),
    ("P1-1_lassoic_vs_lgbic", "configs/ab/runbook_P1_1.yaml"),
    ("P1-2_lgbic_vs_lgblgb", "configs/ab/runbook_P1_2.yaml"),
    ("P3-1_perstock_vs_pooled", "configs/ab/runbook_P3_1.yaml"),
    ("P3-2_pool_vs_all", "configs/ab/runbook_P3_2.yaml"),
    ("P4-1_preprocess", "configs/ab/ab_preprocess.yaml"),
    ("P4-23_neutralize", "configs/ab/ab_neutralize.yaml"),
    ("P4-4_orthogonalize", "configs/ab/ab_orthogonalize.yaml"),
    ("sizing_fixed_vs_voltarget", "configs/ab/ab_sizing.yaml"),
]

KEEP = {"sharpe", "total_return", "max_drawdown", "win_rate"}

out: dict = {}
for name, cfg_path in GROUPS:
    print(f"=== {name} ({cfg_path})", flush=True)
    try:
        ab_cfg = load_ab_config(cfg_path)
        base = load_config((Path(cfg_path).parent / ab_cfg.base_config).resolve())
        stocks = [
            s for s in base.stocks
            if not ab_cfg.stocks_filter or s.code in ab_cfg.stocks_filter
        ]
        res = run_ab(ab_cfg, base, stocks, refresh=False)
        render_ab_report(res, f"reports/ab_rerun/{name}")
        ns = shared_holding_days(res.arm_a, res.arm_b) or [None]
        group: dict = {
            "arm_a": res.arm_a.name, "arm_b": res.arm_b.name,
            "per_n": {},
        }
        for n in ns:
            tbl = compute_diff_table(res.arm_a, res.arm_b, N=n)
            rows = {}
            for r in tbl["rows"]:
                key = r["label"].lower().replace(" ", "_").replace("(%)", "").strip("_")
                rows[key] = {
                    "a_mean": r["a_mean"], "b_mean": r["b_mean"],
                    "diff_mean": r["diff_mean"],
                    "a_wins": r["a_wins"], "b_wins": r["b_wins"],
                }
            group["per_n"][str(n)] = {
                "common_stocks": tbl["common_stocks_count"], "rows": rows,
            }
        out[name] = group
        print(f"    ok: arms={res.arm_a.name} vs {res.arm_b.name}", flush=True)
    except Exception as e:  # noqa: BLE001
        out[name] = {"error": str(e)}
        print(f"    FAILED: {e}", flush=True)

Path("reports/ab_rerun_results.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8",
)
print("written reports/ab_rerun_results.json", flush=True)
