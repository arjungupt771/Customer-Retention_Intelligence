@echo off
REM Telecom Churn Analysis Dashboard Launcher (Windows)
cd /d "%~dp0"

echo 🚀 Starting Telecom Churn Analysis Dashboard...
echo.

REM Check if Python dependencies are installed
echo 📦 Checking Python dependencies...
python -c "import fastapi, uvicorn, pandas, numpy, sklearn" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Python dependencies missing. Installing...
python -m pip install -r requirements.txt
)

REM Check if Node.js dependencies are installed
echo 📦 Checking Node.js dependencies...
if not exist "churn-dashboard\node_modules" (
    echo ❌ Node.js dependencies missing. Installing...
    cd churn-dashboard
    npm ci
    cd ..
)

echo.
echo 🎯 Starting services...
echo.

REM Start backend in background
echo 🔧 Starting FastAPI backend on http://localhost:8000
start /B python api.py

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend
echo 🌐 Starting React dashboard on http://localhost:5173
cd churn-dashboard
start /B npm run dev
cd ..

echo.
echo ✅ Dashboard is running!
echo    Backend API: http://localhost:8000
echo    Frontend UI: http://localhost:5173
echo.
echo Press Ctrl+C in the terminal windows to stop services
echo.

pause
