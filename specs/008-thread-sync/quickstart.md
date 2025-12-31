# Quickstart: Thread Sync

**Feature**: 008-thread-sync
**Date**: 2025-12-31

## Prerequisites

1. vlt-cli installed and configured with a project
2. Document-MCP backend running at `http://localhost:8000`
3. JWT token obtained from web UI Settings page

## Setup

### 1. Configure sync in vlt.toml

```toml
[oracle]
vault_url = "http://localhost:8000"

[sync]
token = "your-jwt-token-here"
```

Or set via CLI:
```bash
vlt config set sync.token "your-jwt-token"
```

### 2. Verify connection

```bash
vlt sync status
# Expected: Connected to http://localhost:8000, authenticated as <user_id>
```

## Usage

### Automatic sync on push

After configuration, `vlt thread push` automatically syncs:

```bash
# Create a new thread (syncs automatically)
vlt thread new auth-design "Exploring JWT vs session auth" --project myapp

# Push an insight (syncs automatically)
vlt thread push auth-design "Decided on JWT with 90-day expiry for mobile clients"

# Check sync status
vlt thread read auth-design --sync-status
```

### Manual sync

If automatic sync failed (offline, network error):

```bash
# Retry pending syncs
vlt sync retry

# Force full sync for a thread
vlt sync thread auth-design --force

# View sync queue
vlt sync queue
```

### View in Web UI

1. Open `http://localhost:5173`
2. Navigate to **Threads** in sidebar
3. Filter by project or search content
4. Click thread to view full history

### Oracle queries with thread context

Threads are now included in Oracle context:

```bash
# CLI
vlt oracle "What authentication approach did we decide on?"

# Web UI - Chat panel
# Ask: "What did we discuss about auth?"
```

## Troubleshooting

### Sync fails with 401

```bash
# Check token validity
vlt config get sync.token

# Re-authenticate via web UI and update token
vlt config set sync.token "new-token"
```

### Entries not appearing in web UI

```bash
# Check sync status
vlt sync status

# View pending queue
vlt sync queue

# Force retry
vlt sync retry --verbose
```

### Network errors

Entries are queued locally and retried automatically:

```bash
# View queue
cat ~/.vlt/sync_queue.json

# Manual retry
vlt sync retry
```

## API Examples

### Sync entries (CLI → Backend)

```bash
curl -X POST http://localhost:8000/api/threads/sync \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "thread_id": "auth-design",
    "project_id": "myapp",
    "name": "Auth Design Discussion",
    "entries": [
      {
        "entry_id": "550e8400-e29b-41d4-a716-446655440000",
        "sequence_id": 0,
        "content": "Exploring JWT vs session auth",
        "author": "user",
        "timestamp": "2025-12-31T10:00:00Z"
      }
    ]
  }'
```

### List threads

```bash
curl http://localhost:8000/api/threads \
  -H "Authorization: Bearer $TOKEN"
```

### Search threads

```bash
curl "http://localhost:8000/api/threads/search?q=authentication" \
  -H "Authorization: Bearer $TOKEN"
```
