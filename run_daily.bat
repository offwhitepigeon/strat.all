@echo off
REM ============================================================
REM ETF信号系统 - 每日14:45自动运行 + 邮件推送
REM ============================================================
REM 可通过Windows任务计划程序调用此批处理文件
REM
REM 创建任务计划:
REM   schtasks /create /tn "ETF信号系统" /tr "D:\workspace\strat.all\run_daily.bat" /sc daily /st 14:45
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

REM 如果通过任务计划运行,保持窗口3秒后关闭
timeout /t 3 /nobreak >nul
