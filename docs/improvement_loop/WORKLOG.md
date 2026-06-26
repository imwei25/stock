# 自驱改进循环 — Worklog (逐方向结果)

> 每个方向跑完一条记录:方向 / 假设 / AB 配置 / 指标对比 / 判定 / commit。
> 指标取 `portfolio-ab` 聚合表(total_return / annualized_return / sharpe / max_drawdown / trade_count / win_rate)。

---

## 预备:历史已知 AB 结果(本循环开始前)

- **selection_v2 (20) vs selection_with_gtja (30)** — portfolio-ab @ 238 ab_pool, 2026-06-25
  - Sharpe 0.63 → **1.33** (+0.70);ann_return 0.129 → 0.317;maxDD 0.195 → 0.181
  - 结论:gtja 集大幅胜 v2。但 v2 **不是**当前 prod 基线(prod=selection.json),
    故本循环 A1 重新以 prod 为基线复核 gtja。
  - 报告:`reports/portfolio_ab/2026-06-25.html`

---
<!-- 新记录追加到下方 -->
