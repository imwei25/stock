# 改进轮 3:三组单旋钮 A/B 串行(horizon5 / thresholds_tight / window500)
Set-Location "C:\Users\wei gu\Desktop\claude\stock"
$groups = @("horizon5", "thresholds_tight", "window500")
foreach ($g in $groups) {
    "##### GROUP $g start $(Get-Date -Format s)"
    & .venv\Scripts\python.exe -X utf8 -m stockpool ab --config "configs/ab/ab_eval48_$g.yaml"
    "##### GROUP $g exit $LASTEXITCODE"
    Copy-Item "reports\ab\2026-06-13.html" "reports\ab_round3\$g.html" -Force
}
"##### ALL GROUPS DONE $(Get-Date -Format s)"
