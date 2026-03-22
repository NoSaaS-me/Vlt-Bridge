# API Contracts: Artifact Sandbox

## Daemon REST Endpoints

### Artifact CRUD

#### `POST /api/artifacts`
Create a new artifact.

**Request**:
```json
{
  "name": "My Dashboard",
  "description": "Analytics dashboard artifact",
  "type": "ephemeral",
  "project_id": "default"
}
```

**Response** (201):
```json
{
  "id": "a1b2c3d4",
  "name": "My Dashboard",
  "state": "draft",
  "disk_path": "/data/artifacts/user-1/a1b2c3d4",
  "created_at": "2026-03-15T10:00:00Z"
}
```

#### `GET /api/artifacts`
List all artifacts for the current user/project.

**Query params**: `project_id` (optional), `state` (optional filter)

**Response** (200):
```json
[
  {
    "id": "a1b2c3d4",
    "name": "My Dashboard",
    "type": "ephemeral",
    "state": "building",
    "updated_at": "2026-03-15T11:00:00Z"
  }
]
```

#### `GET /api/artifacts/{artifact_id}`
Get full artifact details including manifest.

**Response** (200):
```json
{
  "id": "a1b2c3d4",
  "name": "My Dashboard",
  "state": "building",
  "state_history": [
    {"state": "draft", "at": "2026-03-15T10:00:00Z", "by": "user"},
    {"state": "building", "at": "2026-03-15T10:05:00Z", "by": "agent:session-xyz"}
  ],
  "manifest": { ... },
  "thread_id": "artifacts-system",
  "disk_path": "/data/artifacts/user-1/a1b2c3d4"
}
```

#### `PUT /api/artifacts/{artifact_id}`
Update artifact metadata or manifest.

**Request**:
```json
{
  "name": "Updated Name",
  "manifest": { ... }
}
```

**Response** (200): Updated artifact object.

#### `DELETE /api/artifacts/{artifact_id}`
Delete an artifact. Stops backend process if running, removes disk directory.

**Response** (204): No content.

### Artifact State Machine

#### `POST /api/artifacts/{artifact_id}/state`
Transition artifact state.

**Request**:
```json
{
  "target_state": "building",
  "actor": "user"
}
```

**Response** (200):
```json
{
  "previous_state": "draft",
  "current_state": "building",
  "transitioned_at": "2026-03-15T10:05:00Z"
}
```

**Error** (409): Invalid state transition.

### Artifact Backend

#### `POST /api/artifacts/{artifact_id}/backend/start`
Start the artifact's backend process.

**Response** (200):
```json
{
  "status": "running",
  "pid": 12345
}
```

#### `POST /api/artifacts/{artifact_id}/backend/stop`
Stop the artifact's backend process.

**Response** (200):
```json
{
  "status": "stopped"
}
```

#### `POST /api/artifacts/{artifact_id}/backend/call`
Proxy a call to the artifact backend's `handle()` function.

**Request**:
```json
{
  "action": "get_status",
  "params": {"detailed": true}
}
```

**Response** (200):
```json
{
  "result": { ... }
}
```

### Artifact Frontend Serving

#### `GET /api/artifacts/{artifact_id}/frontend/{path:path}`
Serve static files from the artifact's frontend directory. The daemon injects the VltBridge and HMR scripts into `index.html` before serving.

**Response**: Raw file content with appropriate MIME type.

### Artifact Testing

#### `POST /api/artifacts/{artifact_id}/test`
Run the artifact's test command.

**Response** (200):
```json
{
  "passed": false,
  "exit_code": 1,
  "stdout": "...",
  "stderr": "...",
  "duration_ms": 1234
}
```

### Artifact Screenshots

#### `POST /api/artifacts/{artifact_id}/screenshot`
Capture a screenshot of the artifact's frontend via headless browser.

**Response** (200):
```json
{
  "path": ".vlt/screenshots/2026-03-15T11-30-00.png",
  "width": 1280,
  "height": 720
}
```

#### `GET /api/artifacts/{artifact_id}/screenshot/{filename}`
Retrieve a previously captured screenshot.

**Response**: PNG image.

### Artifact Import/Export

#### `GET /api/artifacts/{artifact_id}/export`
Export artifact as a zip file.

**Response**: `application/zip` file download.

#### `POST /api/artifacts/import`
Import artifact from zip file.

**Request**: `multipart/form-data` with `file` field.

**Response** (201): Created artifact object.

### Artifact Events

#### `POST /api/artifacts/{artifact_id}/events/emit`
Emit an event from an artifact (called by VltBridge).

**Request**:
```json
{
  "event_type": "post_created",
  "payload": {"url": "...", "subreddit": "..."}
}
```

**Response** (200):
```json
{
  "delivered_to": ["artifact-b", "artifact-c"]
}
```

## Daemon WebSocket Endpoints

### `WS /ws/artifact/{artifact_id}/hmr`
Hot Module Replacement signals for the artifact iframe.

**Server → Client messages**:
```json
{"type": "css_update", "files": ["style.css"]}
{"type": "will_reload"}
{"type": "reload"}
{"type": "state_restore", "state": {...}}
```

**Client → Server messages**:
```json
{"type": "state_saved", "state": {...}}
```

### `WS /ws/artifact/{artifact_id}/state`
State machine updates for the sidebar.

**Server → Client messages**:
```json
{"type": "state_change", "from": "building", "to": "testing", "by": "harness"}
{"type": "test_result", "passed": true, "summary": "3/3 passing"}
{"type": "error", "message": "Backend process crashed"}
```

### `WS /ws/artifact/{artifact_id}/logs`
Stdout/stderr stream from the artifact backend.

**Server → Client messages**:
```json
{"type": "stdout", "data": "Processing request..."}
{"type": "stderr", "data": "Warning: deprecated API"}
```

### `WS /ws/artifact/{artifact_id}/events`
Artifact-to-artifact IPC events.

**Server → Client messages**:
```json
{"type": "event", "source": "artifact-a", "event_type": "post_created", "payload": {...}}
```

## Backend REST Endpoints (for connector multi-instance)

### `GET /api/connectors/{name}/instances`
List all instances of a connector.

**Response** (200):
```json
[
  {"instance_id": "default", "configured": true, "enabled": true},
  {"instance_id": "bot-1", "configured": true, "enabled": true},
  {"instance_id": "bot-2", "configured": false, "enabled": false}
]
```

### `GET /api/connectors/{name}/instances/{instance_id}/config`
Get config for a specific connector instance (secrets masked).

### `PUT /api/connectors/{name}/instances/{instance_id}/config`
Set config for a specific connector instance.

### `DELETE /api/connectors/{name}/instances/{instance_id}`
Delete a connector instance and its config.

## Backend REST Endpoints (for proxy profiles)

### `GET /api/proxy-profiles`
List all proxy profiles for the current user.

### `POST /api/proxy-profiles`
Create a new proxy profile.

**Request**:
```json
{
  "name": "us-residential",
  "proxy_url": "socks5://proxy.example.com:1080",
  "proxy_username": "user",
  "proxy_password": "pass"
}
```

### `PUT /api/proxy-profiles/{name}`
Update a proxy profile.

### `DELETE /api/proxy-profiles/{name}`
Delete a proxy profile.

## Backend REST Endpoints (for vision model settings)

### `GET /api/settings/models`
Extended to include `vision_model` and `vision_provider` fields.

### `PUT /api/settings/models`
Extended to accept `vision_model` and `vision_provider` fields.

### `GET /api/models`
Extended: `ModelInfo` now includes `supports_vision: boolean` field.

## MCP Tools

### `vlt_artifact_create(name, description, project_id, type?)`
Create a new artifact.

### `vlt_artifact_list(project_id?)`
List artifacts.

### `vlt_artifact_update(artifact_id, files)`
Write source files to an artifact. `files` is a dict of `{path: content}`.

### `vlt_artifact_state(artifact_id, target_state)`
Transition artifact state.

### `vlt_artifact_test(artifact_id)`
Run tests and return results.

### `vlt_artifact_screenshot(artifact_id)`
Capture and return screenshot.

## VltBridge postMessage Protocol

All messages between iframe and parent follow:

**Iframe → Parent (request)**:
```json
{
  "type": "vlt_request",
  "id": "req-123",
  "method": "storage.get",
  "params": {"key": "my-key"}
}
```

**Parent → Iframe (response)**:
```json
{
  "type": "vlt_response",
  "id": "req-123",
  "result": {"value": "stored-data"}
}
```

**Parent → Iframe (error)**:
```json
{
  "type": "vlt_response",
  "id": "req-123",
  "error": {"code": "NOT_FOUND", "message": "Key not found"}
}
```

**Parent → Iframe (event push)**:
```json
{
  "type": "vlt_event",
  "event_type": "post_created",
  "source": "artifact-a",
  "payload": {...}
}
```
