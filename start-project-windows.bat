@echo off
REM Document-MCP Start Script for Windows
REM This script opens two terminal windows - one for backend and one for frontend

echo Starting Document-MCP...
echo.

REM Get the project root directory
set PROJECT_ROOT=%~dp0

REM Start Backend in a new terminal window
echo Starting Backend (FastAPI on port 8000)...
start "Document-MCP Backend" cmd /k "cd /d "%PROJECT_ROOT%backend" && .venv\Scripts\activate && uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000"

REM Wait a moment before starting frontend
timeout /t 3 /nobreak

REM Start Frontend in a new terminal window
echo Starting Frontend (Vite on port 5173)...
start "Document-MCP Frontend" cmd /k "cd /d "%PROJECT_ROOT%frontend" && npm run dev"

echo.
echo ============================================
echo Document-MCP is starting!
echo ============================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:5173
echo.
echo Both services should open in separate terminal windows.
echo Press Ctrl+C in each window to stop the services.
echo.
pause
