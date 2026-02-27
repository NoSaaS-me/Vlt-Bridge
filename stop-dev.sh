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

_stop_service() {
    local pid_file="$1"
    local label="$2"
    local port="$3"

    if [ -f "$pid_file" ]; then
        local pid
        pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo "Stopping $label (PID: $pid and children)..."
            # Kill the process group to catch uvicorn reload child workers
            kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null
            sleep 1
            ps -p "$pid" > /dev/null 2>&1 && kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$pid_file"
    fi

    # Also kill anything still holding the port (catches orphaned children)
    if [ -n "$port" ]; then
        local port_pid
        port_pid=$(lsof -ti tcp:"$port" 2>/dev/null)
        if [ -n "$port_pid" ]; then
            echo "Killing process on port $port (PID: $port_pid)..."
            kill "$port_pid" 2>/dev/null
            sleep 0.5
            kill -0 "$port_pid" 2>/dev/null && kill -9 "$port_pid" 2>/dev/null
        fi
    fi

    echo -e "${GREEN}✓ $label stopped${NC}"
}

_stop_service "$BACKEND_PID_FILE"  "Backend"  8000
_stop_service "$FRONTEND_PID_FILE" "Frontend" 5173

echo ""
echo -e "${GREEN}Development servers stopped${NC}"

