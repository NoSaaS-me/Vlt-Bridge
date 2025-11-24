#!/bin/bash
# Stop development servers for Document Viewer

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"

echo -e "${RED}Stopping Document Viewer Development Environment${NC}"
echo "=================================================="

# Stop Backend
if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo "Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID
        rm "$BACKEND_PID_FILE"
        echo -e "${GREEN}✓ Backend stopped${NC}"
    else
        echo "Backend process not found (PID: $BACKEND_PID)"
        rm "$BACKEND_PID_FILE"
    fi
else
    echo "No backend PID file found"
fi

# Stop Frontend
if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID
        rm "$FRONTEND_PID_FILE"
        echo -e "${GREEN}✓ Frontend stopped${NC}"
    else
        echo "Frontend process not found (PID: $FRONTEND_PID)"
        rm "$FRONTEND_PID_FILE"
    fi
else
    echo "No frontend PID file found"
fi

echo ""
echo -e "${GREEN}Development servers stopped${NC}"

