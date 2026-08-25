@echo off
REM ============================================================
REM ETF信号系统 - 单次运行 + 邮件推送
REM ============================================================
REM 此脚本用于手动运行或被 Windows 任务计划(14:45)调用
REM
REM 定时任务管理:
REM   安装: powershell -ExecutionPolicy Bypass -File setup_schedule.ps1
REM   卸载: powershell -ExecutionPolicy Bypass -File remove_schedule.ps1
REM ============================================================

cd /d D:\workspace\strat.all

set PYTHON=C:\Users\Administrator\.workbuddy\binaries\python\envs\default\Scripts\python.exe
set SCRIPT=D:\workspace\strat.all\daily_run.py

echo ============================================
echo ETF信号系统 启动 %date% %time%
echo ============================================

"%PYTHON%" -X utf8 "%SCRIPT%"

echo.
echo ============================================
echo 运行完成 %date% %time%
echo ============================================

timeout /t 3 /nobreak >nul
