@echo off
REM DE1 Full Data Loader - Run in standalone CMD
REM =============================================
REM This script runs the full DE1 data pipeline
REM Run from Windows CMD (not VS Code terminal)
REM Expected runtime: 2-4 hours

echo ============================================================
echo DE1 FULL DATA LOADER - DGSF Data Engineering
echo ============================================================
echo.
echo IMPORTANT: Run this from a standalone CMD window, NOT VS Code
echo Expected runtime: 2-4 hours for 5186 stocks
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

cd /d "E:\AI Tools\AI Workflow OS\projects\dgsf"
python scripts\de1_batch_runner.py

echo.
echo ============================================================
echo DE1 COMPLETE - Check data\raw\ for output files
echo ============================================================
pause
