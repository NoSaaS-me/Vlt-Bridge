# MCP HTTP Server Testing Guide

This guide shows how to test your MCP HTTP server on both Windows and Linux systems.

## Prerequisites

- MCP server running on `http://localhost:8001/mcp`
- Bearer token: `local-dev-token` (for local development)

## 🪟 Windows Testing (PowerShell)

### Test 1: Initialize Connection

```powershell
# Set up headers
$headers = @{
    "Authorization" = "Bearer local-dev-token"
    "Content-Type" = "application/json"
    "Accept" = "application/json, text/event-stream"
}

# Create initialize request body
$body = @{
    jsonrpc = "2.0"
    id = 1
    method = "initialize"
    params = @{
        protocolVersion = "2024-11-05"
        capabilities = @{}
        clientInfo = @{
            name = "test-client"
            version = "1.0.0"
        }
    }
} | ConvertTo-Json -Depth 10

# Send request
Invoke-RestMethod -Uri "http://localhost:8001/mcp" -Method POST -Headers $headers -Body $body
```

### Test 2: List Available Tools

```powershell
# Reuse headers from above
$toolsBody = @{
    jsonrpc = "2.0"
    id = 2
    method = "tools/list"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8001/mcp" -Method POST -Headers $headers -Body $toolsBody
```

### Test 3: Call a Tool (List Notes)

```powershell
$toolCallBody = @{
    jsonrpc = "2.0"
    id = 3
    method = "tools/call"
    params = @{
        name = "list_notes"
        arguments = @{}
    }
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Uri "http://localhost:8001/mcp" -Method POST -Headers $headers -Body $toolCallBody
```

### Test 4: Server Status Check

```powershell
# Check if MCP server is running (should return connection info or error)
try {
    Invoke-RestMethod -Uri "http://localhost:8001/mcp" -Method GET
    Write-Host "✅ MCP server is running on port 8001"
} catch {
    Write-Host "❌ MCP server not responding on port 8001"
}

# Alternative: Check FastAPI health endpoint (different server on port 8000)
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method GET
```

## 🐧 Linux Testing (curl + bash)

### Test 1: Initialize Connection

```bash
# Initialize connection
curl -X POST http://localhost:8001/mcp \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {
        "name": "test-client",
        "version": "1.0.0"
      }
    }
  }'
```

### Test 2: List Available Tools

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/list"
  }'
```

### Test 3: Call a Tool (List Notes)

```bash
curl -X POST http://localhost:8001/mcp \
  -H "Authorization: Bearer local-dev-token" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "list_notes",
      "arguments": {}
    }
  }'
```

### Test 4: Server Status Check

```bash
# Check if MCP server is running
curl -f http://localhost:8001/mcp && echo "✅ MCP server running" || echo "❌ MCP server not responding"

# Alternative: Check FastAPI health endpoint (different server on port 8000)
curl http://localhost:8000/health
```

## 🐍 Python Testing Script

### Cross-Platform Python Test

```python
#!/usr/bin/env python3
"""Cross-platform MCP HTTP server test."""

import json
import requests

def test_mcp_server(base_url="http://localhost:8001"):
    """Test MCP server functionality."""
    
    headers = {
        "Authorization": "Bearer local-dev-token",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    mcp_url = f"{base_url}/mcp"
    
    # Test 1: Initialize
    print("🔌 Testing initialize...")
    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "test-client",
                "version": "1.0.0"
            }
        }
    }
    
    response = requests.post(mcp_url, json=init_request, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    # Test 2: List tools
    print("\n🛠️ Testing tools/list...")
    tools_request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list"
    }
    
    response = requests.post(mcp_url, json=tools_request, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        if "result" in result and "tools" in result["result"]:
            tools = result["result"]["tools"]
            print(f"Found {len(tools)} tools:")
            for tool in tools:
                print(f"  - {tool['name']}: {tool.get('description', 'No description')}")

if __name__ == "__main__":
    test_mcp_server()
```

## 📋 Expected Responses

### Successful Initialize Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "experimental": {},
      "prompts": {"listChanged": true},
      "resources": {"subscribe": false, "listChanged": true},
      "tools": {"listChanged": true}
    },
    "serverInfo": {
      "name": "obsidian-docs-viewer",
      "version": "2.13.1"
    }
  }
}
```

### Available Tools Response

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "list_notes",
        "description": "List notes in the vault (optionally scoped to a folder)."
      },
      {
        "name": "read_note",
        "description": "Read a Markdown note with metadata and body."
      },
      {
        "name": "write_note",
        "description": "Create or update a note."
      },
      {
        "name": "delete_note",
        "description": "Delete a note and remove it from the index."
      },
      {
        "name": "search_notes",
        "description": "Search notes using full-text search with BM25 ranking."
      },
      {
        "name": "get_backlinks",
        "description": "List notes that reference the target note."
      },
      {
        "name": "get_tags",
        "description": "List tags and associated note counts."
      }
    ]
  }
}
```

## 🚨 Common Errors and Solutions

### Error: "Missing session ID"

```json
{"jsonrpc":"2.0","id":"server-error","error":{"code":-32600,"message":"Bad Request: Missing session ID"}}
```

**Solution**: This is expected for tools/list and tools/call without proper session management. The initialize method should work.

### Error: "Authorization header required"

```json
{"jsonrpc":"2.0","id":"server-error","error":{"code":-32600,"message":"Authorization header required"}}
```

**Solution**: Make sure you include the Authorization header with Bearer token.

### Error: Connection refused

**Solution**: 
1. Check if MCP server is running: `netstat -ano | findstr :8001` (Windows) or `lsof -i :8001` (Linux)
2. Start the server: `python -m src.mcp.server` with `MCP_TRANSPORT=http` and `MCP_PORT=8001`

## 🔧 Troubleshooting

### Check Server Status

**Windows:**
```powershell
netstat -ano | findstr :8001
```

**Linux:**
```bash
lsof -i :8001
# or
netstat -tlnp | grep :8001
```

### Start MCP Server

**Windows:**
```powershell
cd backend
$env:MCP_TRANSPORT="http"
$env:MCP_PORT="8001"
$env:LOCAL_USER_ID="local-dev"
python -m src.mcp.server
```

**Linux:**
```bash
cd backend
export MCP_TRANSPORT=http
export MCP_PORT=8001
export LOCAL_USER_ID=local-dev
python -m src.mcp.server
```

## 🎯 Testing Checklist

- [ ] Server starts without errors
- [ ] Health endpoint responds: `GET /health`
- [ ] Initialize method works: Returns server info
- [ ] Tools list method works: Returns available tools
- [ ] Bearer token authentication works
- [ ] User isolation works (different tokens = different vaults)

## 📚 Next Steps

1. **For Cursor Integration**: Use STDIO transport in `mcp.json`
2. **For HF Spaces**: Deploy with HTTP transport and JWT authentication
3. **For Production**: Set proper `JWT_SECRET_KEY` and use real JWT tokens
