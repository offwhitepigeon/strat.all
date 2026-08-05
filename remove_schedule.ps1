# ============================================================
# ETF信号系统 - 定时任务卸载脚本
# ============================================================
# 功能：移除 Windows 任务计划 "ETF信号系统"
#
# 使用方法：右键 → 以管理员身份运行 PowerShell → 执行此脚本
#   powershell -ExecutionPolicy Bypass -File remove_schedule.ps1
# ============================================================

$taskName = "ETF信号系统"

Write-Host "=" * 60
Write-Host "ETF信号系统 - 定时任务卸载" -ForegroundColor Cyan
Write-Host "=" * 60

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "任务 '$taskName' 已移除" -ForegroundColor Green
} else {
    Write-Host "任务 '$taskName' 不存在" -ForegroundColor Yellow
}

Write-Host "=" * 60
