# ============================================================
# ETF信号系统 - 定时任务安装脚本
# ============================================================
# 功能：注册 Windows 任务计划，每天 9:30 自动启动 run_intraday.py
#   同时清理旧的开机启动项（启动文件夹/注册表Run/旧任务计划）
#
# 使用方法：右键 → 以管理员身份运行 PowerShell → 执行此脚本
#   powershell -ExecutionPolicy Bypass -File setup_schedule.ps1
# ============================================================

$taskName = "ETF信号系统"
$python = "C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
$script = "D:\workspace\strat.all\run_intraday.py"
$workDir = "D:\workspace\strat.all"

Write-Host "=" * 60
Write-Host "ETF信号系统 - 定时任务安装" -ForegroundColor Cyan
Write-Host "=" * 60

# ---- 1. 清理旧的开机启动项 ----
Write-Host "`n[1/3] 清理旧启动项..." -ForegroundColor Yellow

# 1a. 清理启动文件夹
$startupPath = [Environment]::GetFolderPath('Startup')
$commonStartup = [Environment]::GetFolderPath('CommonStartup')
foreach ($dir in @($startupPath, $commonStartup)) {
    if ($dir -and (Test-Path $dir)) {
        Get-ChildItem $dir -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -match 'strat\.all|intraday|ETF信号'
        } | ForEach-Object {
            Write-Host "  删除启动项: $($_.Name)"
            Remove-Item $_.FullName -Force
        }
    }
}

# 1b. 清理注册表 Run 项（仅删除与本项目相关的启动项）
foreach ($regPath in @(
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run',
    'HKLM:\Software\Microsoft\Windows\CurrentVersion\Run'
)) {
    if (Test-Path $regPath) {
        $props = Get-ItemProperty $regPath -ErrorAction SilentlyContinue
        if ($props) {
            $props.PSObject.Properties | Where-Object {
                $_.Value -match 'strat\.all|run_intraday|daily_run\.py'
            } | ForEach-Object {
                Write-Host "  删除注册表启动项: $($_.Name)"
                Remove-ItemProperty -Path $regPath -Name $_.Name -Force
            }
        }
    }
}

# 1c. 移除旧的任务计划（如果存在）
$oldTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($oldTask) {
    Write-Host "  移除旧任务计划: $taskName"
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Write-Host "  清理完成" -ForegroundColor Green

# ---- 2. 创建新的定时任务 ----
Write-Host "`n[2/3] 创建定时任务（每天 9:30 启动）..." -ForegroundColor Yellow

$trigger = New-ScheduledTaskTrigger -Daily -At 9:30AM

$action = New-ScheduledTaskAction `
    -Execute $python `
    -Argument "-X utf8 `"$script`"" `
    -WorkingDirectory $workDir

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6)

Register-ScheduledTask `
    -TaskName $taskName `
    -Trigger $trigger `
    -Action $action `
    -Settings $settings `
    -Description "ETF多策略信号系统 - 每日9:30自动启动盘中信号运行" `
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
Write-Host "  - 任务每天 9:30 自动启动 run_intraday.py"
Write-Host "  - run_intraday.py 会判断是否交易日，非交易日自动退出"
Write-Host "  - 卸载请运行: powershell -ExecutionPolicy Bypass -File remove_schedule.ps1"
Write-Host "=" * 60
