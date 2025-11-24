#!/bin/bash
# Check status of development servers for Document Viewer

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"

echo -e "${BLUE}Document Viewer Development Status${NC}"
echo "=================================================="

# Check Backend
if [ -f "$BACKEND_PID_FILE" ]; then
    BACKEND_PID=$(cat "$BACKEND_PID_FILE")
    if ps -p $BACKEND_PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend: RUNNING${NC} (PID: $BACKEND_PID)"
        echo "  URL: http://localhost:8000"
    else
        echo -e "${RED}✗ Backend: NOT RUNNING${NC} (stale PID: $BACKEND_PID)"
    fi
else
    echo -e "${RED}✗ Backend: NOT RUNNING${NC}"
fi

# Check Frontend
if [ -f "$FRONTEND_PID_FILE" ]; then
    FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend: RUNNING${NC} (PID: $FRONTEND_PID)"
        echo "  URL: http://localhost:5173"
    else
        echo -e "${RED}✗ Frontend: NOT RUNNING${NC} (stale PID: $FRONTEND_PID)"
    fi
else
    echo -e "${RED}✗ Frontend: NOT RUNNING${NC}"
fi

echo ""
echo "Logs:"
echo "  Backend:  tail -f $PROJECT_ROOT/backend.log"
echo "  Frontend: tail -f $PROJECT_ROOT/frontend.log"
echo ""

