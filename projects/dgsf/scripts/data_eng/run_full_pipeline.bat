@echo off
REM ================================================================
REM DGSF Full Data Engineering Pipeline - Unattended Execution
REM ================================================================
REM 自动顺序执行 DE3 -> DE5 -> DE7
REM 完成或失败时弹窗+声音提醒
REM 
REM 用法: 双击此文件或在 CMD 中运行
REM 预计运行时间: 6-12 小时
REM ================================================================

echo ================================================================
echo DGSF FULL DATA ENGINEERING PIPELINE
echo ================================================================
echo.
echo 此脚本将自动执行:
echo   1. DE3 Financial Indicators (~2-4 小时)
echo   2. DE5 Microstructure       (~3-6 小时)
echo   3. DE7 Factor Panel         (~30 分钟)
echo.
echo 总预计时间: 6-12 小时
echo 完成/失败时会弹窗+声音提醒
echo.
echo 按任意键开始，或 Ctrl+C 取消...
pause >nul

REM 切换到项目目录
cd /d "E:\AI Tools\AI Workflow OS\projects\dgsf"

REM 检查 TUSHARE_TOKEN
if "%TUSHARE_TOKEN%"=="" (
    echo.
    echo ERROR: TUSHARE_TOKEN 未设置!
    echo.
    echo 请在 CMD 中执行:
    echo   set TUSHARE_TOKEN=your_token_here
    echo.
    echo 或在 PowerShell 中执行:
    echo   $env:TUSHARE_TOKEN = "your_token_here"
    echo.
    pause
    exit /b 1
)

echo.
echo TUSHARE_TOKEN 已设置，开始执行...
echo 日志文件: data\pipeline_run.log
echo.

REM 执行 Python 管道
python scripts\run_full_pipeline.py

echo.
echo ================================================================
echo Pipeline 执行结束
echo ================================================================
pause
