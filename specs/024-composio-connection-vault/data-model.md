# Data Model: Composio Connection Vault

---

## Entities

### ComposioConnection (new — local SQLite)

Tracks individual Composio connections with user-provided labels. Composio stores the tokens; we store the metadata for routing and UI.

| Field | Type | Constraints | Description |
|-------|------|------------|-------------|
| `id` | TEXT | PK | Composio's `connectedAccountId` (UUID) |
| `user_id` | TEXT | NOT NULL, indexed | Maps to `entity_id` / `auth.user_id` |
| `app_name` | TEXT | NOT NULL | Lowercase app identifier (e.g. `gmail`) |
| `label` | TEXT | NOT NULL, default `''` | User-given name (e.g. "Work Gmail") |
| `auth_mode` | TEXT | NOT NULL, default `''` | Auth scheme used: `OAUTH2`, `API_KEY`, etc. |
| `status` | TEXT | NOT NULL, default `'active'` | `active`, `initiated`, `failed`, `revoked` |
| `created_at` | TEXT | NOT NULL | ISO 8601 datetime |
| `updated_at` | TEXT | NOT NULL | ISO 8601 datetime |

**Indexes**:
- `idx_composio_conn_user_app ON (user_id, app_name)` — fast lookup per user per app

**DDL**:
```sql
CREATE TABLE IF NOT EXISTS composio_connections (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    label TEXT NOT NULL DEFAULT '',
    auth_mode TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_composio_conn_user_app
    ON composio_connections(user_id, app_name);
```

**State transitions**:
```
initiated  ──(OAuth callback success)──→  active
initiated  ──(OAuth callback failure)──→  failed
active     ──(user disconnects)────────→  revoked
active     ──(reconcile: not in Composio)→ revoked
```

---

### AppAuthInfo (API response — not persisted)

Returned by `GET /api/composio/{app_name}/auth-info`. Describes what credentials an app requires for connection.

```python
class AuthFieldInfo(BaseModel):
    name: str               # "client_id", "api_key", etc.
    display_name: str       # "Client ID", "API Key"
    description: str        # Help text
    type: str               # "string" (always, per Composio SDK)
    required: bool          # Whether field is mandatory
    expected_from_customer: bool  # True = user supplies, False = operator/Composio supplies

class AuthSchemeInfo(BaseModel):
    auth_mode: str                      # "OAUTH2", "API_KEY", "BASIC", etc.
    integration_fields: list[AuthFieldInfo]  # Operator-level (client_id, client_secret)
    user_fields: list[AuthFieldInfo]         # User-level (api_key, token)

class AppAuthInfo(BaseModel):
    has_managed_auth: bool              # True if Composio provides OAuth creds
    primary_auth_mode: str              # First/preferred auth mode
    auth_schemes: list[AuthSchemeInfo]  # All available schemes
```

---

### ConnectRequest (API request body)

```python
class ConnectRequest(BaseModel):
    label: str = ""                                      # User-given connection name
    auth_mode: str | None = None                         # Override scheme selection
    auth_config: dict[str, str] | None = None            # OAuth: {client_id, client_secret}
    connected_account_params: dict[str, str] | None = None  # API_KEY: {api_key}
    redirect_url: str | None = None                      # Override OAuth redirect
```

---

### ConnectResponse (API response)

```python
class ConnectResponse(BaseModel):
    app: str
    connection_id: str          # Composio's connectedAccountId
    label: str
    redirect_url: str | None    # null for non-OAuth (API_KEY immediate activation)
    status: str                 # "initiated" (OAuth) or "active" (API key)
```

---

### ConnectionInfo (API response — list endpoint)

```python
class ConnectionInfo(BaseModel):
    connection_id: str
    app_name: str
    label: str
    auth_mode: str
    status: str
    created_at: str
```

---

### InvokeRequest (updated)

```python
class InvokeRequest(BaseModel):
    action: str
    params: dict[str, Any] = {}
    connection_id: str | None = None   # NEW: select specific connection
```

---

## Existing Entities (unchanged)

### connector_configs (existing table)

Used for per-action permissions (`__action_*` keys). No changes needed. Keyed by `(user_id, connector_name, config_key)`.

### ConnectorInfo (frontend type — unchanged)

Native connectors continue to use the existing type. Composio connectors are a separate flow.

---

## Relationships

```
User (user_id)
  │
  ├── 1:N ──→ ComposioConnection
  │              │
  │              ├── app_name ──→ Composio App (external)
  │              └── id ───────→ Composio ConnectedAccount (external)
  │
  └── 1:N ──→ connector_configs (existing, for __action_* permissions)
                 └── connector_name = "composio:{app_name}"
```

---

## Validation Rules

1. `label` must be <=100 chars, stripped of leading/trailing whitespace
2. `app_name` must be lowercase, alphanumeric + hyphens only
3. `connection_id` must be a valid UUID (from Composio)
4. A user can have at most 10 active connections per app (prevent abuse)
5. `auth_config` keys must match the fields from `GET /auth-info` integration_fields
6. `connected_account_params` keys must match user_fields from `GET /auth-info`
