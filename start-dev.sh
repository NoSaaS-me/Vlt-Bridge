#!/bin/bash
# Development startup script for Document Viewer

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# PID files
BACKEND_PID_FILE="$PROJECT_ROOT/.backend.pid"
FRONTEND_PID_FILE="$PROJECT_ROOT/.frontend.pid"

echo -e "${BLUE}Starting Document Viewer Development Environment${NC}"
echo "=================================================="

# Check if backend venv exists
if [ ! -d "$BACKEND_DIR/.venv" ]; then
    echo -e "${RED}Error: Backend virtual environment not found${NC}"
    echo "Run: cd backend && uv venv && uv pip install -e ."
    exit 1
fi

# Check if frontend node_modules exists
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${RED}Error: Frontend dependencies not installed${NC}"
    echo "Run: cd frontend && npm install"
    exit 1
fi

# Start Backend
echo -e "${GREEN}Starting backend server...${NC}"
cd "$BACKEND_DIR"
JWT_SECRET_KEY="local-dev-secret-key-123" \
VAULT_BASE_PATH="$PROJECT_ROOT/data/vaults" \
.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload > "$PROJECT_ROOT/backend.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$BACKEND_PID_FILE"
echo -e "${GREEN}✓ Backend started (PID: $BACKEND_PID)${NC}"
echo "  Logs: $PROJECT_ROOT/backend.log"
echo "  URL: http://localhost:8000"

# Wait a moment for backend to start
sleep 2

# Start Frontend
echo -e "${GREEN}Starting frontend dev server...${NC}"
cd "$FRONTEND_DIR"
npm run dev > "$PROJECT_ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$FRONTEND_PID_FILE"
echo -e "${GREEN}✓ Frontend started (PID: $FRONTEND_PID)${NC}"
echo "  Logs: $PROJECT_ROOT/frontend.log"
echo "  URL: http://localhost:5173"

echo ""
echo -e "${BLUE}=================================================="
echo "Development servers are running!"
echo "=================================================="
echo -e "${NC}"
echo "Frontend: http://localhost:5173"
echo "Backend:  http://localhost:8000"
echo ""
echo "To stop servers, run: ./stop-dev.sh"
echo "To view logs, run: tail -f backend.log frontend.log"
echo ""

