# F1: 串行重跑全部 A/B 验证(新口径),结果汇总到 reports/ab_rerun_<date>.log
$ErrorActionPreference = "Continue"
$log = "reports/ab_rerun_2026-06-12.log"
"=== A/B rerun started $(Get-Date -Format s) ===" | Out-File $log -Encoding utf8

$perStock = @(
    "configs/ab/runbook_P2_1.yaml",   # embargo 0 vs auto(最便宜,先验证管线)
    "configs/ab/runbook_P0_1.yaml",   # composite vs lgb+lgb
    "configs/ab/runbook_P0_2.yaml",   # lasso+ic vs lgb+lgb
    "configs/ab/runbook_P1_1.yaml",   # lasso+ic vs lgb+ic
    "configs/ab/runbook_P1_2.yaml",   # lgb+ic vs lgb+lgb
    "configs/ab/runbook_P3_1.yaml",   # per_stock vs pooled
    "configs/ab/runbook_P3_2.yaml",   # training pool vs all
    "configs/ab/ab_preprocess.yaml",  # P4-1 preprocess on/off
    "configs/ab/ab_neutralize.yaml",  # P4-2/3 中性化
    "configs/ab/ab_orthogonalize.yaml", # P4-4 对称正交
    "configs/ab/ab_sizing.yaml"       # sizing fixed vs vol_target
)
foreach ($cfg in $perStock) {
    if (-not (Test-Path $cfg)) { "SKIP (missing): $cfg" | Tee-Object -Append $log; continue }
    "`n##### RUN ab: $cfg  $(Get-Date -Format s)" | Tee-Object -Append $log
    & .venv\Scripts\python.exe -m stockpool ab --config $cfg 2>&1 | Tee-Object -Append $log
    "##### EXIT $cfg -> $LASTEXITCODE" | Tee-Object -Append $log
}

$portfolio = @(
    "configs/ab/portfolio_ab_simple.yaml",
    "configs/ab/portfolio_ab_mask_medium.yaml"
)
foreach ($cfg in $portfolio) {
    if (-not (Test-Path $cfg)) { "SKIP (missing): $cfg" | Tee-Object -Append $log; continue }
    "`n##### RUN portfolio-ab: $cfg  $(Get-Date -Format s)" | Tee-Object -Append $log
    & .venv\Scripts\python.exe -m stockpool portfolio-ab --config $cfg 2>&1 | Tee-Object -Append $log
    "##### EXIT $cfg -> $LASTEXITCODE" | Tee-Object -Append $log
}
"`n=== A/B rerun finished $(Get-Date -Format s) ===" | Tee-Object -Append $log
