@echo off
SETLOCAL

:: ===================================================
:: Global Local Expert System Startup
:: Sets up Ollama + verifies the expert tool suite.
:: Compatible with any Python 3.12+ installation.
:: ===================================================

SET TOOLS_DIR=C:\Users\vikel\.gemini\antigravity\mcp-servers\local-expert
SET PYTHON_EXE=C:\Users\vikel\AppData\Local\Programs\Python\Python314\python.exe

:: -- Try to locate Ollama from PATH first, fall back to known install locations
SET OLLAMA_EXE=
WHERE ollama >nul 2>&1
IF %ERRORLEVEL%==0 (
    FOR /F "tokens=*" %%i IN ('WHERE ollama') DO SET OLLAMA_EXE=%%i
) ELSE IF EXIST "F:\Programs\Deepseek\ollama.exe" (
    SET OLLAMA_EXE=F:\Programs\Deepseek\ollama.exe
) ELSE IF EXIST "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    SET OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
)

echo ===================================================
echo   Global Local Expert System Startup
echo ===================================================

:: 1. Ensure Ollama is running
echo [1/3] Checking Ollama service...
IF "%OLLAMA_EXE%"=="" (
    echo [WARNING] Could not locate ollama.exe — manual startup may be required.
    echo           If Ollama is running as a service, this is fine.
) ELSE (
    tasklist /FI "IMAGENAME eq ollama.exe" 2>NUL | find /I "ollama.exe" >NUL
    IF "%ERRORLEVEL%"=="0" (
        echo --- Ollama already running
    ) ELSE (
        echo --- Starting Ollama from: %OLLAMA_EXE%
        start /B "" "%OLLAMA_EXE%" serve
    )
)

:: 2. Wait for Ollama API to respond (up to 30 seconds)
echo [2/3] Waiting for Ollama API...
SET /A WAIT_TRIES=0
:wait_ollama
curl -s http://localhost:11434/api/tags >nul 2>&1
IF %ERRORLEVEL%==0 GOTO ollama_ready
SET /A WAIT_TRIES+=1
IF %WAIT_TRIES% GEQ 15 (
    echo [WARNING] Ollama API not responding after 30s — continuing anyway.
    GOTO ollama_done
)
timeout /t 2 >nul
goto wait_ollama
:ollama_ready
echo --- Ollama API ready
:ollama_done

:: 3. Verify expert tool suite
echo [3/3] Verifying expert toolset...
IF NOT EXIST "%TOOLS_DIR%\local_orchestrator.py" (
    echo [ERROR] Expert scripts missing from: %TOOLS_DIR%
    pause
    exit /b 1
)
IF NOT EXIST "%PYTHON_EXE%" (
    echo [ERROR] Python not found at: %PYTHON_EXE%
    pause
    exit /b 1
)

:: Quick syntax check
"%PYTHON_EXE%" -c "import sys; sys.path.insert(0,'%TOOLS_DIR%'); import config, server, escalation_controller" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Import check failed — dependencies may be missing.
    echo Try: %PYTHON_EXE% -m pip install fastmcp httpx pytest pytest-asyncio
    pause
    exit /b 1
)
echo --- All imports OK

echo.
echo ===================================================
echo   Startup Complete. MCP server ready at:
echo   %TOOLS_DIR%\local_orchestrator.py
echo ===================================================
timeout /t 3 >nul
