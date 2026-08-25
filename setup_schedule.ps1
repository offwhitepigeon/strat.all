# ============================================================
# ETF信号系统 - 定时任务安装脚本
# ============================================================
# 功能：注册 Windows 任务计划，每天 14:45 运行一次 daily_run.py
#   同时清理旧的任务计划和启动项
#
# 使用方法：右键 → 以管理员身份运行 PowerShell → 执行此脚本
#   powershell -ExecutionPolicy Bypass -File setup_schedule.ps1
# ============================================================

$taskName = "ETF信号系统"
$python = "C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
$script = "D:\workspace\strat.all\daily_run.py"
$workDir = "D:\workspace\strat.all"

Write-Host "=" * 60
Write-Host "ETF信号系统 - 定时任务安装" -ForegroundColor Cyan
Write-Host "=" * 60

# ---- 1. 移除旧任务 ----
Write-Host "`n[1/3] 清理旧任务..." -ForegroundColor Yellow
$oldTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($oldTask) {
    Write-Host "  移除旧任务计划: $taskName"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
Write-Host "  清理完成" -ForegroundColor Green

# ---- 2. 创建新的定时任务（每天14:45单次运行）----
Write-Host "`n[2/3] 创建定时任务（每天 14:45 单次运行）..." -ForegroundColor Yellow

$trigger = New-ScheduledTaskTrigger -Daily -At 2:45PM

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "-X utf8 `"$script`"" `
    -WorkingDirectory $workDir

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)

Register-ScheduledTask `
    -TaskName $taskName `
    -Trigger $trigger `
    -Action $action `
    -Settings $settings `
    -Description "ETF多策略信号系统 - 每天14:45生成信号并发送邮件" `
    -Force | Out-Null

Write-Host "  任务创建成功" -ForegroundColor Green

# ---- 3. 验证 ----
Write-Host "`n[3/3] 验证任务..." -ForegroundColor Yellow

$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    $info = $task | Get-ScheduledTaskInfo
    Write-Host "  任务名称: $($task.TaskName)"
    Write-Host "  状态: $($task.State)"
    Write-Host "  触发器: 每天 $($task.Triggers[0].StartBoundary.ToString().Substring(11,5))"
    Write-Host "  下次运行: $($info.NextRunTime)"
    Write-Host "  执行: $python -X utf8 `"$script`""
    Write-Host "`n安装完成!" -ForegroundColor Green
} else {
    Write-Host "  [错误] 任务创建失败!" -ForegroundColor Red
}

Write-Host "`n提示:"
Write-Host "  - 任务每天 14:45 自动运行 daily_run.py（生成信号+发送邮件）"
Write-Host "  - 非交易日由 daily_run.py 内部逻辑判断后跳过"
Write-Host "  - 卸载请运行: powershell -ExecutionPolicy Bypass -File remove_schedule.ps1"
Write-Host "=" * 60
