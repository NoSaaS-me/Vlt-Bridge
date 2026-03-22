# Feature Specification: Composio Connection Vault

**Feature Branch**: `024-composio-connection-vault`
**Created**: 2026-03-14
**Status**: Draft

---

## Problem Statement

Our Composio integration is a thin wrapper that assumes one connection per app per user, uses Composio's managed OAuth for everything, and has a runtime bug where disconnect calls a nonexistent SDK method. In reality:

1. **Many apps lack managed auth.** Twitter, LinkedIn, and others require the operator or user to supply their own OAuth client credentials. These fail with `Auth_Config_DefaultAuthConfigNotFound`.
2. **Users need multiple connections per app.** A user with 3 Reddit accounts or personal + work Gmail needs N connections per app, each selectable at invocation time.
3. **The SDK's default connection selection is broken for our use case.** Without passing `connected_account_id`, `execute_action()` silently picks the **oldest** connection. There is no disambiguation.
4. **Labels are write-only.** Composio's SDK accepts labels at creation time but never surfaces them back — they cannot be read or filtered by. We must store connection metadata ourselves.
5. **Disconnect is broken.** `connected_accounts.delete()` does not exist on the SDK's `ConnectedAccounts` class. Every disconnect call raises `AttributeError`.

This spec designs a **Connection Vault** — a local registry that tracks connections, supports multi-account, routes invocations to the correct connection, and handles both managed and custom auth flows.

---

## Composio SDK Internals (Verified from Source)

All findings verified by reading SDK source at `backend/.venv/lib/python3.11/site-packages/composio/`.

### Entity Model

- An **Entity** is a string ID wrapper scoping all operations to a logical user.
- `entity_id` maps to `userUuid` in Composio's API — this is our `auth.user_id`.
- We already pass entity_id correctly per-call. Multi-tenancy is handled.

### Connection Model (`ConnectedAccountModel`)

```
Fields:
  id: str                    # UUID — the primary key for connection routing
  status: str                # "ACTIVE", "INITIATED", "FAILED"
  createdAt: str             # ISO datetime
  updatedAt: str
  appUniqueId: str           # lowercase app name (e.g. "gmail")
  appName: str               # display name
  integrationId: str         # which integration config was used
  connectionParams: object   # contains scope, tokens, client_id, etc.
  clientUniqueUserId: str    # same as entity_id
  entityId: str              # defaults to "default"
```

**No `labels` field.** Labels sent at creation time are not returned by the API.

### Multiple Connections Per App

- `Entity.initiate_connection()` can be called multiple times for the same `app_name`. Each call creates a new connection with a unique `id`.
- `Entity.get_connections()` returns ALL connections (active, initiated, failed) for the entity.
- `Entity.get_connection(app=name)` returns the **oldest** active connection, not newest. (Misleading variable name `latest_account` in SDK — verified it selects `min(createdAt)`.)

### Auth Schemes

```python
# 10 auth types in the SDK
ALL_AUTH_SCHEMES = (
    "OAUTH2", "OAUTH1",
    "API_KEY", "BASIC", "BEARER_TOKEN", "BASIC_WITH_JWT",
    "GOOGLE_SERVICE_ACCOUNT", "GOOGLEADS_AUTH",
    "NO_AUTH", "CALCOM_AUTH",
)
```

**Managed vs Custom detection:**

```python
app = toolset.client.apps.get(name="twitter")
has_managed_auth = bool(app.testConnectors)  # non-empty = Composio has OAuth creds
```

This is the field the SDK itself uses to decide `use_composio_auth`. Verified at `toolset.py:1374` and `cli/add.py:262`.

**`AppAuthScheme.fields`** — each field has:
- `name`, `display_name`, `description`, `type`, `default`, `required`
- `expected_from_customer: bool` — `True` = user supplies (API key, token), `False` = operator/integration supplies (client_id, client_secret)

**Multiple schemes per app:** Yes. An app can support both `OAUTH2` and `API_KEY`. The SDK picks the first from priority order if `auth_mode` is not specified.

### Connection Initiation

```python
entity.initiate_connection(
    app_name: str,
    auth_mode: Optional[str] = None,          # "OAUTH2", "API_KEY", etc.
    auth_config: Optional[Dict] = None,       # {client_id, client_secret} for custom OAuth
    redirect_url: Optional[str] = None,
    use_composio_auth: bool = True,           # MUST be False for custom auth
    force_new_integration: bool = False,
    connected_account_params: Optional[Dict] = None,  # {api_key} for API_KEY apps
    labels: Optional[List] = None,
) -> ConnectionRequestModel
```

**Return:** `ConnectionRequestModel` with `redirectUrl` (for OAuth) and `connectedAccountId`.

For non-OAuth schemes (`API_KEY`, `BASIC`, `BEARER_TOKEN`), the connection may be immediately `ACTIVE` — no redirect needed.

### Deletion

The SDK's `ConnectedAccounts` class has NO `delete()` method. The only way to delete is raw HTTP:

```python
from composio.client.endpoints import v1
toolset.client.http.delete(url=str(v1 / "connectedAccounts" / connection_id))
```

---

## Current Implementation (What Exists)

### Service Layer (`vlt-connectors/service/composio.py`)

| Method | Status | Issues |
|--------|--------|--------|
| `catalog()` | Working | Filters `no_auth` apps correctly |
| `connected(entity_id)` | Working | Returns flat list, no multi-account awareness |
| `initiate_connection(app_name, entity_id)` | Broken | Always uses `use_composio_auth=True`, no custom auth support |
| `disconnect(app_name, entity_id)` | Broken | Calls nonexistent `connected_accounts.delete()` |
| `get_actions(app_name)` | Working | Returns action schemas correctly |
| `execute(app_name, action, params, entity_id)` | Partial | Never passes `connected_account_id`, oldest connection wins |

### Backend Routes (`backend/api/routes/composio_hub.py`)

| Route | Issues |
|-------|--------|
| `POST /connect/{app_name}` | No body — can't pass custom auth credentials |
| `DELETE /{app_name}` | By app_name only — can't target specific connection |
| `POST /{app_name}/invoke` | No `connection_id` param — can't select which account |
| `GET /connected` | Returns `{app_name, status, connection_id}` but frontend ignores `connection_id` |

### Frontend (`ConnectorsPage.tsx`)

- `ComposioApp` type has no `connections` array — assumes 0 or 1 per app.
- Connect/Disconnect buttons operate on app name, not connection ID.
- No UI for labeling connections or choosing between multiples.

### MCP Tools (`connector_tools.py`)

- `connector_call` has no `connection_id` parameter.
- `connector_list` shows Composio connectors but not individual connections.

---

## Design

### Principles

1. **Composio stores tokens, we store metadata.** Composio manages the OAuth tokens and API keys. We store `connection_id`, user-given label, app_name, and auth_mode locally in SQLite. This avoids duplicating credential storage.
2. **Multi-account is a first-class concept.** Every connection has a unique ID and a user-provided label. One app can have N connections.
3. **Connection selection must be explicit.** When a user has >1 connection for an app, the agent must specify which one. When there's exactly 1, it's auto-selected.
4. **Auth flow adapts to the app.** Managed OAuth apps go straight to redirect. Custom OAuth apps collect `client_id`/`client_secret` first. API key apps collect the key inline and activate immediately.
5. **No breaking changes to existing working flows.** Gmail (managed OAuth, single connection) should continue to work exactly as it does today.

### Connection Registry (Local SQLite)

New table in the backend database:

```sql
CREATE TABLE composio_connections (
    id TEXT PRIMARY KEY,                    -- Composio's connected_account_id (UUID)
    user_id TEXT NOT NULL,                  -- maps to entity_id
    app_name TEXT NOT NULL,                 -- lowercase app identifier
    label TEXT NOT NULL DEFAULT '',         -- user-given name ("personal gmail", "work gmail")
    auth_mode TEXT NOT NULL DEFAULT '',     -- "OAUTH2", "API_KEY", etc.
    status TEXT NOT NULL DEFAULT 'active',  -- "active", "failed", "revoked"
    created_at TEXT NOT NULL,               -- ISO datetime
    updated_at TEXT NOT NULL,
    UNIQUE(id)
);
CREATE INDEX idx_composio_conn_user_app ON composio_connections(user_id, app_name);
```

**Sync strategy:** On `GET /connected`, fetch from Composio API, reconcile with local registry (add new, mark removed as revoked). Local table is the source of truth for labels and is the lookup table for `connection_id` routing.

### Auth Info Endpoint

New endpoint to query what credentials an app needs before connecting:

```
GET /api/composio/{app_name}/auth-info
→ {
    has_managed_auth: bool,
    auth_schemes: [
      {
        auth_mode: "OAUTH2",
        integration_fields: [{name, display_name, required, type}],  -- operator provides (client_id, client_secret)
        user_fields: [{name, display_name, required, type}],         -- end-user provides (api_key, token)
      },
      ...
    ],
    primary_auth_mode: "OAUTH2"
  }
```

The frontend calls this when "Connect" is clicked. If `has_managed_auth` is true and primary mode is OAuth, proceed directly. Otherwise, show a credentials form.

### Updated Connection Flow

```
[User clicks Connect]
       │
       ▼
  GET /auth-info
       │
       ├─ has_managed_auth=true + OAUTH  ──→  POST /connect/{app} (empty body) → redirect
       │
       ├─ has_managed_auth=false + OAUTH ──→  Show form: client_id, client_secret
       │                                       POST /connect/{app} {auth_config: {...}}  → redirect
       │
       └─ API_KEY / BASIC / BEARER       ──→  Show form: api_key / username+password
                                               POST /connect/{app} {connected_account_params: {...}}
                                               → immediate activation (no redirect)
```

**On successful connection:**
- Backend receives `connectedAccountId` from Composio
- Inserts row into `composio_connections` with user-provided label
- Returns `{connection_id, redirect_url?, status}`

### Updated Connect Endpoint

```
POST /api/composio/connect/{app_name}
Body (all optional):
{
    "label": "Work Gmail",                                    -- user-given name
    "auth_mode": "OAUTH2",                                    -- override scheme selection
    "auth_config": {"client_id": "...", "client_secret": "..."}, -- for custom OAuth
    "connected_account_params": {"api_key": "..."},           -- for API_KEY/BASIC
    "redirect_url": "https://..."                             -- override redirect
}
→ {
    app: str,
    connection_id: str,         -- Composio's connected_account_id
    redirect_url: str | null,   -- null for non-OAuth flows
    status: str                 -- "INITIATED" (OAuth) or "ACTIVE" (API key)
  }
```

### Updated Disconnect Endpoint

```
DELETE /api/composio/connections/{connection_id}
→ { connection_id, disconnected: true }
```

No longer by app_name — always by specific connection_id. Uses raw HTTP to Composio since SDK lacks delete method.

Keep the old `DELETE /{app_name}` as a convenience that disconnects ALL connections for that app.

### Updated Connected Endpoint

```
GET /api/composio/connected
→ {
    connections: [
      {
        connection_id: str,
        app_name: str,
        label: str,             -- from local registry
        auth_mode: str,
        status: str,
        created_at: str
      },
      ...
    ],
    total: int
  }
```

Groups by app_name in the frontend: shows each app once, with N connections listed underneath.

### Updated Invoke Endpoint

```
POST /api/composio/{app_name}/invoke
Body:
{
    "action": "GMAIL_SEND_EMAIL",
    "params": {...},
    "connection_id": "uuid"      -- NEW: optional, required if >1 connection exists
}
```

If `connection_id` is provided, pass it as `connected_account_id` to `toolset.execute_action()`.

If omitted AND user has exactly 1 active connection for the app, auto-select it.

If omitted AND user has >1 active connection, return 400 with the list of connections so the caller can pick.

### MCP Tool Changes

**`connector_list`** — add `connections` array to each Composio connector entry:

```python
{
    "name": "composio:gmail",
    "display_name": "Gmail (via Composio)",
    "connections": [
        {"connection_id": "uuid-1", "label": "Personal Gmail"},
        {"connection_id": "uuid-2", "label": "Work Gmail"},
    ],
    "actions": [...]
}
```

**`connector_call`** — add optional `connection_id` parameter:

```python
def connector_call(
    connector: str,
    action: str,
    params: str = "{}",
    connection_id: Optional[str] = None,    # NEW
) -> dict:
```

If the user has multiple connections and `connection_id` is not provided, return an error listing the available connections with their labels so the agent can ask the user or pick one.

### Frontend Changes

**AppCard** — for connected apps, show connection list:
```
┌─────────────────────────────────┐
│ [G] Gmail                    ✓  │
│     Send, receive, manage email │
│                                 │
│  Connections:                   │
│    Personal (abc...def)   [×]   │
│    Work     (ghi...jkl)   [×]   │
│                                 │
│  [+ Add Connection]            │
│                                 │
│  Actions (12)                   │
│  ▸ GMAIL_SEND_EMAIL      Allow  │
│  ▸ GMAIL_LIST_EMAILS     Allow  │
│  ...                            │
└─────────────────────────────────┘
```

**Connect dialog** — adapts to auth type:
1. Always asks for a **label** first ("What do you want to call this connection?")
2. If custom OAuth: shows `client_id` + `client_secret` fields
3. If API key: shows API key field
4. If managed OAuth: proceeds directly to redirect

---

## User Scenarios

### Scenario 1: Connect a managed OAuth app (Gmail) — P0

User clicks Connect on Gmail. No credentials needed. Redirected to Google OAuth. On return, connection is registered with auto-label "Gmail" (or user-provided label). Actions are immediately available via MCP/CLI.

**Acceptance:** Works exactly as today, but now stores `connection_id` locally.

### Scenario 2: Connect a custom OAuth app (Twitter) — P0

User clicks Connect on Twitter. Frontend detects `has_managed_auth=false`. Shows a form asking for Twitter OAuth App `client_id` and `client_secret`. User fills them in. Redirected to Twitter OAuth. On return, connection is active.

**Acceptance:** Twitter connection works without 502. The credentials form shows the correct fields based on the auth scheme.

### Scenario 3: Connect an API key app — P1

User clicks Connect on an API-key-based app. Frontend shows an API key input field. User pastes their key. Connection is immediately active (no redirect).

**Acceptance:** Connection status is "ACTIVE" without any redirect flow.

### Scenario 4: Multiple connections for same app — P1

User has a personal Gmail connected. Clicks "Add Connection" on Gmail. Goes through OAuth with a different Google account. Now has 2 Gmail connections with distinct labels.

When an agent calls `connector_call("composio:gmail", "GMAIL_SEND_EMAIL", ...)` without specifying `connection_id`, it gets an error listing both connections with labels so it can ask the user which to use.

**Acceptance:** Agent can disambiguate between connections via labels. Single-connection apps auto-select without requiring `connection_id`.

### Scenario 5: Disconnect a specific connection — P1

User has 2 Gmail connections. Clicks the [x] button on "Work Gmail". Only that specific connection is disconnected; "Personal Gmail" remains active.

**Acceptance:** Disconnect targets a specific `connection_id`, not the entire app.

### Scenario 6: Agent sends email via specific connection — P2

Agent calls `connector_call("composio:gmail", "GMAIL_SEND_EMAIL", params, connection_id="uuid-of-work-gmail")`. The email is sent from the work Gmail account.

**Acceptance:** `connection_id` is passed through to Composio's `connected_account_id` parameter and the correct account is used.

---

## Implementation Phases

### Phase 1: Fix What's Broken (P0)

1. Fix `disconnect()` — use raw HTTP instead of nonexistent SDK method
2. Add `app_auth_info()` to service — detect managed vs custom auth
3. Update `initiate_connection()` to accept `auth_config`, `auth_mode`, `use_composio_auth`
4. Add `GET /auth-info` backend endpoint
5. Update `POST /connect/{app_name}` to accept body with custom credentials
6. Frontend: show credentials form when `has_managed_auth=false`

### Phase 2: Connection Registry (P1)

1. Add `composio_connections` table to SQLite schema
2. Store `connection_id` + label on successful connection
3. Reconcile local registry with Composio API on `GET /connected`
4. Update disconnect to target by `connection_id`
5. Update invoke to accept optional `connection_id`
6. Frontend: show connection list per app, add "Add Connection" button

### Phase 3: MCP/CLI Routing (P2)

1. Add `connection_id` param to `connector_call` MCP tool
2. Add `connections` array to `connector_list` output
3. Auto-select when single connection, error with options when multiple
4. Update CLI `vlt connectors call` to accept `--connection-id`
5. Update tool docstrings and server instructions

---

## Out of Scope

- Credential encryption at rest (Composio holds the tokens, not us)
- Connection sharing between users
- Automatic connection health checks / token refresh monitoring
- Composio webhook event handling
- Per-connection permission overrides (permissions stay per-app for now)
