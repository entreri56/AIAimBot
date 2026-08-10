@echo off
title YoloAimBot - Capture Card Edition
echo ============================================
echo   YoloAimBot - Capture Card Edition
echo ============================================
echo.
echo Starting GUI...
echo.

cd /d "%~dp0"
python gui.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Failed to start. Make sure you have installed:
    echo   pip install ultralytics opencv-python numpy mss makcu
    echo.
    pause
)