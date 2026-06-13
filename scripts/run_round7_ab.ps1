# 改进轮 7:lasso alpha 扫描 + min_abs_ic 过滤,三组串行
Set-Location "C:\Users\wei gu\Desktop\claude\stock"
$groups = @("alpha_low", "alpha_high", "min_abs_ic")
foreach ($g in $groups) {
    "##### GROUP $g start $(Get-Date -Format s)"
    & .venv\Scripts\python.exe -X utf8 -m stockpool ab --config "configs/ab/ab_eval48_$g.yaml"
    "##### GROUP $g exit $LASTEXITCODE"
    Copy-Item "reports\ab\2026-06-13.html" "reports\ab_round7_$g.html" -Force
}
"##### ALL GROUPS DONE $(Get-Date -Format s)"
