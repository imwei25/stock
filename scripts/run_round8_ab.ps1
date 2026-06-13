# 改进轮 8(收敛确认):combo / winsor_tight / horizon2 三组串行
Set-Location "C:\Users\wei gu\Desktop\claude\stock"
$groups = @("combo", "winsor_tight", "horizon2")
foreach ($g in $groups) {
    "##### GROUP $g start $(Get-Date -Format s)"
    & .venv\Scripts\python.exe -X utf8 -m stockpool ab --config "configs/ab/ab_eval48_$g.yaml"
    "##### GROUP $g exit $LASTEXITCODE"
    Copy-Item "reports\ab\2026-06-13.html" "reports\ab_round8_$g.html" -Force
}
"##### ALL GROUPS DONE $(Get-Date -Format s)"
