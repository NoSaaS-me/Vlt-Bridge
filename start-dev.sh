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

# --- Guard: kill any stale instances before starting ---
_kill_stale_pid() {
    local pid_file="$1"
    local label="$2"
    if [ -f "$pid_file" ]; then
        local old_pid
        old_pid=$(cat "$pid_file" 2>/dev/null)
        if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
            echo -e "${RED}⚠ $label already running (PID $old_pid) — stopping it first${NC}"
            kill "$old_pid" 2>/dev/null
            sleep 1
            # Force-kill if still alive
            kill -0 "$old_pid" 2>/dev/null && kill -9 "$old_pid" 2>/dev/null
        fi
        rm -f "$pid_file"
    fi
}

_kill_stale_pid "$BACKEND_PID_FILE"  "Backend"
_kill_stale_pid "$FRONTEND_PID_FILE" "Frontend"

# Also kill anything holding port 8000 or 5173
for port in 8000 5173; do
    pid=$(lsof -ti tcp:"$port" 2>/dev/null)
    if [ -n "$pid" ]; then
        echo -e "${RED}⚠ Port $port in use (PID $pid) — killing${NC}"
        kill "$pid" 2>/dev/null
        sleep 0.5
    fi
done

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

