@echo off
REM DE3/DE4 Full Financial Loader - Run in standalone CMD
REM ======================================================
REM This script runs the full DE3/DE4 financial data pipeline
REM Run from Windows CMD (not VS Code terminal)
REM Expected runtime: 2-4 hours

echo ============================================================
echo DE3/DE4 FULL FINANCIAL LOADER - DGSF Data Engineering
echo ============================================================
echo.
echo IMPORTANT: Run this from a standalone CMD window, NOT VS Code
echo Expected runtime: 2-4 hours for 5186 stocks
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

cd /d "E:\AI Tools\AI Workflow OS\projects\dgsf"
python scripts\de3_financial_loader.py

echo.
echo ============================================================
echo DE3/DE4 COMPLETE - Check data\raw\ for output files
echo ============================================================
pause
