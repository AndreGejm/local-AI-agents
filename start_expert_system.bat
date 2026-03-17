@echo off
SETLOCAL

:: Global Antigravity Toolset Location
SET TOOLS_DIR=C:\Users\vikel\.gemini\antigravity\mcp-servers\local-expert
SET OLLAMA_EXE=F:\Programs\Deepseek\ollama.exe
SET PYTHON_EXE=C:\Users\vikel\AppData\Local\Programs\Python\Python314\python.exe

echo ===================================================
echo   Global Local Expert System Startup
echo ===================================================

:: 1. Ensure Ollama Service is running
echo [1/2] Checking Ollama Service...
tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo --- Ollama already running
) else (
    echo --- Starting Ollama...
    start /B "" "%OLLAMA_EXE%" serve
)

:: Wait for Ollama API
echo --- Waiting for Ollama API...
:wait_ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    timeout /t 2 >nul
    goto wait_ollama
)
echo --- Ollama ready

:: 2. Verify System Configuration
echo [2/2] Verifying Global Expert Toolset...
if exist "%TOOLS_DIR%\local_orchestrator.py" (
    echo --- Orchestrator found: %TOOLS_DIR%\local_orchestrator.py
) else (
    echo [ERROR] Expert scripts missing from %TOOLS_DIR%
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Global Startup Verification Complete.
echo Ollama is running and Expert tools are globalized.
echo ===================================================
timeout /t 5
