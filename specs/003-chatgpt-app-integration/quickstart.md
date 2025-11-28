# Quickstart: ChatGPT App Integration

## Prerequisites
-   Existing backend running.
-   `CHATGPT_SERVICE_TOKEN` set in environment.

## Testing the Integration

### 1. Authentication
```bash
# Test service token access
curl -H "Authorization: Bearer <your-service-token>" http://localhost:8000/api/notes
```

### 2. Widget Serving
```bash
# Test widget endpoint
curl -v http://localhost:8000/widget/note
# Expect: Content-Type: text/html+skybridge (or similar)
```

### 3. Tool Metadata
```bash
# Test tool call via /mcp endpoint
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
      "name": "read_note",
      "arguments": {"note_path": "Welcome.md"}
    },
    "id": 1
  }'
# Expect response to contain _meta.openai.outputTemplate
```

