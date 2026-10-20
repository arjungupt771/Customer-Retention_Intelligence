#!/bin/bash
# Telecom Churn Analysis Dashboard Launcher
cd "$(dirname "$0")"

echo "🚀 Starting Telecom Churn Analysis Dashboard..."
echo ""

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
python -c "import fastapi, uvicorn, pandas, numpy, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Python dependencies missing. Installing..."
    python -m pip install -r requirements.txt
fi

# Check if Node.js dependencies are installed
echo "📦 Checking Node.js dependencies..."
if [ ! -d "churn-dashboard/node_modules" ]; then
    echo "❌ Node.js dependencies missing. Installing..."
    cd churn-dashboard
    npm ci
    cd ..
fi

echo ""
echo "🎯 Starting services..."
echo ""

# Start backend in background
echo "🔧 Starting FastAPI backend on http://localhost:8000"
python api.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting React dashboard on http://localhost:5173"
cd churn-dashboard
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Dashboard is running!"
echo "   Backend API: http://localhost:8000"
echo "   Frontend UI: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo ''; echo '🛑 Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
