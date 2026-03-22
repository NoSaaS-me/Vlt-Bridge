# Data Model: Artifact Sandbox

## Entities

### Artifact

The core entity representing an executable plugin bundle.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| id | TEXT (UUID) | PK | Unique artifact identifier |
| user_id | TEXT | NOT NULL, FK→user_settings | Owner |
| project_id | TEXT | NOT NULL | Associated vlt project |
| name | TEXT | NOT NULL, max 128 chars | Human-readable name |
| description | TEXT | nullable, max 1024 chars | What the artifact does |
| type | TEXT | NOT NULL, DEFAULT 'ephemeral' | 'ephemeral' or 'persistent' |
| state | TEXT | NOT NULL, DEFAULT 'draft' | Current state machine position |
| state_history_json | TEXT | NOT NULL, DEFAULT '[]' | JSON array of {state, at, by} |
| manifest_json | TEXT | NOT NULL | Full manifest (frontend, backend, connectors, mcp_tools, events, quotas, tests) |
| thread_id | TEXT | nullable | vlt thread for artifact logs |
| disk_path | TEXT | NOT NULL | Absolute path to artifact directory |
| version | INTEGER | NOT NULL, DEFAULT 1 | Optimistic concurrency counter |
| created_at | TEXT | NOT NULL | ISO 8601 |
| updated_at | TEXT | NOT NULL | ISO 8601 |

**State Machine**:
```
draft → building → testing → reviewing → approved → deployed
                     ↑                       ↓
                     └── revision_requested ─┘

Any state → error (on crash/quota exceeded)
approved → testing (on code change)
deployed → update_pending (on code change)
```

**Validation Rules**:
- `name` must be non-empty, max 128 characters, no path separators
- `type` must be one of: `ephemeral`, `persistent`
- `state` must be one of the defined states above
- `manifest_json` must be valid JSON conforming to the manifest schema
- `disk_path` must exist and be within the artifacts data directory
- State transitions must follow the state machine graph

### Connector Instance

Extension of existing `connector_configs` to support multiple named instances per connector.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| user_id | TEXT | PK (composite) | Owner |
| connector_name | TEXT | PK (composite) | Connector identifier |
| instance_id | TEXT | PK (composite), DEFAULT 'default' | Named instance |
| config_key | TEXT | PK (composite) | Configuration key |
| config_value | TEXT | nullable | Configuration value (encrypted if secret) |
| proxy_profile | TEXT | nullable | Reference to proxy_profiles.name |
| updated_at | TEXT | nullable | Last modification |

**Migration**: Existing rows get `instance_id='default'`. Primary key changes from `(user_id, connector_name, config_key)` to `(user_id, connector_name, instance_id, config_key)`.

### Proxy Profile

Named proxy configurations that connector instances can reference.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| user_id | TEXT | PK (composite) | Owner |
| name | TEXT | PK (composite), max 64 chars | Profile name |
| proxy_url | TEXT | NOT NULL | Proxy URL (http://, socks5://) |
| proxy_username | TEXT | nullable, encrypted | Auth username |
| proxy_password | TEXT | nullable, encrypted | Auth password |
| created_at | TEXT | NOT NULL | ISO 8601 |
| updated_at | TEXT | NOT NULL | ISO 8601 |

### Vision Model Setting

Extension of existing `user_settings` table.

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| vision_model | TEXT | nullable | Model ID for vision review |
| vision_provider | TEXT | nullable | 'openrouter', 'google', or 'glm' |

**Migration**: Two `ALTER TABLE user_settings ADD COLUMN` statements.

## Relationships

```
User ─1:N─ Artifact (user_id)
User ─1:N─ ConnectorInstance (user_id, connector_name, instance_id)
User ─1:N─ ProxyProfile (user_id, name)
ConnectorInstance ─N:1─ ProxyProfile (proxy_profile → name)
Artifact ──── references ConnectorInstances via manifest_json.connectors
Artifact ──── references vlt thread via thread_id
Artifact ──── publishes MCP tools via manifest_json.mcp_tools
Artifact ←→ Artifact via event bus (manifest_json.events.emits/subscribes)
```

## Manifest Schema

The `manifest_json` field stores:

```
{
  frontend: {
    entry: string          // relative path to index.html
    deps?: string[]        // CDN URLs for external scripts
  }
  backend?: {
    entry: string          // relative path to main.py
    runtime: "python"      // only python in Phase 1
    deps?: string          // relative path to requirements.txt
    stateful?: boolean     // supports save_state/load_state
  }
  connectors?: [{
    name: string           // connector name
    instances: string[]    // instance IDs needed
  }]
  mcp_tools?: [{
    name: string           // tool name (prefixed with artifact_ on registration)
    description: string
    parameters: object     // JSON Schema for parameters
  }]
  events?: {
    emits?: string[]       // event types this artifact produces
    subscribes?: string[]  // event types this artifact consumes
  }
  quotas?: {
    max_cpu_seconds?: number    // default 60
    max_memory_mb?: number      // default 512
    max_storage_mb?: number     // default 50
  }
  tests?: {
    command?: string       // e.g., "pytest tests/"
    timeout?: number       // seconds, default 30
  }
}
```

## Disk Layout

```
data/artifacts/{user_id}/{artifact_id}/
├── manifest.json
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── backend/
│   ├── main.py
│   └── requirements.txt
├── tests/
│   └── test_backend.py
├── .vlt/
│   ├── hot_state.json         # backend state for hot reload
│   └── screenshots/           # vision review captures
└── .git/                      # auto-initialized
```
