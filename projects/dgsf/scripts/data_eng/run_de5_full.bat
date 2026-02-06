@echo off
REM DE5/DE6 Full Microstructure Loader - Run in standalone CMD
REM ===========================================================
REM This script runs the full DE5/DE6 microstructure data pipeline
REM Run from Windows CMD (not VS Code terminal)
REM Expected runtime: 3-6 hours

echo ============================================================
echo DE5/DE6 FULL MICROSTRUCTURE LOADER - DGSF Data Engineering
echo ============================================================
echo.
echo IMPORTANT: Run this from a standalone CMD window, NOT VS Code
echo Expected runtime: 3-6 hours for ~2700 trading days
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

cd /d "E:\AI Tools\AI Workflow OS\projects\dgsf"
python scripts\de5_microstructure_loader.py

echo.
echo ============================================================
echo DE5/DE6 COMPLETE - Check data\raw\ and data\full\ for output files
echo ============================================================
pause
