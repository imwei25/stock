#!/usr/bin/env bash
# Re-run every prior AB direction on the FULL MARKET (no sub-pool) with the
# hardened significance test (bootstrap ΔSharpe CI + sub-period + validity).
# Reuses cached score panels: the gtja-default baseline is scored once (first
# direction) and every later baseline arm cache-hits it. Resilient: a failed
# direction is logged and the batch continues.
#
# Usage: bash docs/improvement_loop/analysis/run_full_market_batch.sh [n_workers]
set -u
cd "C:/Users/wei gu/Desktop/claude/stock" || exit 1
export PYTHONIOENCODING=utf-8
NW="${1:-3}"
PY=".venv/Scripts/python.exe"
HARN="docs/improvement_loop/analysis/ab_significance.py"
CFG="docs/improvement_loop/configs"
RES="docs/improvement_loop/FULL_MARKET_RESULTS.md"
LOGDIR="/tmp/fm_batch"
mkdir -p "$LOGDIR"

# label:config  (GTJA_check first -> seeds the shared gtja baseline cache;
# G* are engine-only and cache-hit; the rest score one variant arm each)
DIRS=(
  "factorset_gtja:GTJA_check.yaml"
  "G1_topk_20v10:G1.yaml"
  "G2_rebal_5v10:G2.yaml"
  "G3_cap_5v3:G3.yaml"
  "B1_industry_neut:B1.yaml"
  "B2_mcap_neut:B2.yaml"
  "B3_winsor_off:B3.yaml"
  "C1_horizon_3v5:C1.yaml"
  "C2_trainwin_250v500:C2.yaml"
  "C3_alpha_1e3v5e4:C3.yaml"
  "C3b_alpha_1e3v5e3:C3b.yaml"
  "D1_weighter_ic_v_equal:D1.yaml"
  "D2_selector_lasso_v_lgbm:D2.yaml"
  "E1_embargo_auto_v0:E1.yaml"
  "F1_mask_off_v_on:F1.yaml"
)

echo "# 全市场 AB 复核结果 (full-market re-validation, n_workers=$NW)" > "$RES"
echo "" >> "$RES"
echo "判定器: bootstrap ΔSharpe 95% CI 排除 0 + 各子段符号一致 + arm valid。对照原 238-池结论。" >> "$RES"
echo "" >> "$RES"
echo "| 方向 | Sharpe A | Sharpe B | ΔSharpe | 95% CI | 子段一致 | VERDICT |" >> "$RES"
echo "|---|---|---|---|---|---|---|" >> "$RES"

for entry in "${DIRS[@]}"; do
  label="${entry%%:*}"; cfg="${entry##*:}"
  log="$LOGDIR/${label}.log"
  echo "[$(date +%H:%M:%S)] === $label ($cfg) ==="
  $PY "$HARN" --config "$CFG/$cfg" --full-market --workers "$NW" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ] || ! grep -q "VERDICT" "$log"; then
    echo "| $label | — | — | — | — | — | **ERROR rc=$rc** |" >> "$RES"
    echo "[$(date +%H:%M:%S)] $label FAILED rc=$rc (see $log)"
    continue
  fi
  shA=$(grep -oE "Sharpe A=[-0-9.]+" "$log" | head -1 | cut -d= -f2)
  shB=$(grep -oE "B=[-0-9.]+" "$log" | head -1 | cut -d= -f2)
  dpt=$(grep -oE "point = [-+0-9.]+" "$log" | head -1 | sed 's/point = //')
  ci=$(grep -oE "95% CI for ΔSharpe: \[[^]]+\]" "$log" | head -1 | sed 's/.*\[/[/')
  sub=$(grep -oE "sign holds in all sub-periods: (True|False)" "$log" | head -1 | grep -oE "True|False")
  verd=$(grep -oE "VERDICT: [A-Z ]+" "$log" | head -1 | sed 's/VERDICT: //')
  echo "| $label | $shA | $shB | $dpt | $ci | $sub | $verd |" >> "$RES"
  echo "[$(date +%H:%M:%S)] $label -> $verd (Δ$dpt CI$ci)"
done

echo "" >> "$RES"
echo "完成于 $(date +%Y-%m-%d_%H:%M:%S)" >> "$RES"
echo "[$(date +%H:%M:%S)] BATCH DONE -> $RES"
