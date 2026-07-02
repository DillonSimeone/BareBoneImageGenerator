@echo off
echo ===================================================
echo   Antigravity Image Studio Launcher
echo ===================================================
echo.
echo [1/2] Launching backend server (FastAPI + SDXL) ...
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\venv\Scripts\python.exe main.py
pause
