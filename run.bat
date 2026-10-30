@echo off
title AI Drowsiness Alert System
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo Could not find the venv folder here.
    echo Put run.bat in the same folder as drowsiness_alert.py and venv.
    pause
    exit /b 1
)
"venv\Scripts\python.exe" drowsiness_alert.py
pause