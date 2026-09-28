@echo off
setlocal enabledelayedexpansion
echo ===================================================
echo   Antigravity Image Studio Launcher (GPU Optimized)
echo ===================================================
echo.
echo Cleaning up any hung background Python processes...
taskkill /F /IM python.exe 2>nul
echo.

set /p LOCKDOWN="Enable GPU Lockdown Mode? (Kills Discord, Slack, Firefox, Signal, Steam, Ollama, etc. to maximize VRAM) [Y/N] (Default: Y): "
if "!LOCKDOWN!"=="" set LOCKDOWN=Y

if /I "!LOCKDOWN!"=="Y" (
    echo.
    echo [LOCKDOWN] Suspending background GPU-accelerated applications...
    taskkill /F /IM Discord.exe 2>nul
    taskkill /F /IM Signal.exe 2>nul
    taskkill /F /IM Slack.exe 2>nul
    taskkill /F /IM firefox.exe 2>nul
    taskkill /F /IM steamwebhelper.exe 2>nul
    taskkill /F /IM "NVIDIA Overlay.exe" 2>nul
    taskkill /F /IM "ollama app.exe" 2>nul
    taskkill /F /IM ollama.exe 2>nul
    taskkill /F /IM qbittorrent.exe 2>nul
    taskkill /F /IM LLMTokenTracker.exe 2>nul
    echo GPU Lockdown active.
)

echo.
echo Launching backend server (FastAPI + SDXL) ...
set HF_HOME=%~dp0hf_cache
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
.\venv\Scripts\python.exe main.py

if /I "!LOCKDOWN!"=="Y" (
    echo.
    echo ===================================================
    echo Server stopped. Restoring closed applications...
    echo ===================================================
    echo.
    set /p RESTART="Restart closed applications now? [Y/N] (Default: Y): "
    if "!RESTART!"=="" set RESTART=Y
    if /I "!RESTART!"=="Y" (
        echo Restoring applications...
        
        if exist "C:\Program Files\Mozilla Firefox\firefox.exe" (
            start "" "C:\Program Files\Mozilla Firefox\firefox.exe"
        )
        
        if exist "%LocalAppData%\Discord\Update.exe" (
            start "" "%LocalAppData%\Discord\Update.exe" --processStart Discord.exe
        )
        
        if exist "%LocalAppData%\Programs\signal-desktop\Signal.exe" (
            start "" "%LocalAppData%\Programs\signal-desktop\Signal.exe"
        )
        
        if exist "%LocalAppData%\Programs\Ollama\ollama app.exe" (
            start "" "%LocalAppData%\Programs\Ollama\ollama app.exe"
        )
        
        if exist "C:\Program Files (x86)\Steam\Steam.exe" (
            start "" "C:\Program Files (x86)\Steam\Steam.exe"
        )
        
        if exist "X:\Torrenting\qBittorrent\qbittorrent.exe" (
            start "" "X:\Torrenting\qBittorrent\qbittorrent.exe"
        )
        
        echo Apps restored successfully.
    )
)
pause
