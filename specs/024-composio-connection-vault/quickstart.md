# Quickstart: Composio Connection Vault

## Phase 1: Fix What's Broken (P0)

### 1. Fix disconnect bug

**File**: `packages/vlt-connectors/src/vlt_connectors/service/composio.py`

Replace line 143 (`toolset.client.connected_accounts.delete(connection_id=conn_id)`) with:
```python
from composio.client.endpoints import v1
toolset.client.http.delete(url=str(v1 / "connectedAccounts" / conn_id))
```

### 2. Add `app_auth_info()` method

**File**: `packages/vlt-connectors/src/vlt_connectors/service/composio.py`

```python
def app_auth_info(self, app_name: str) -> dict:
    toolset = self._toolset()
    app = toolset.client.apps.get(name=app_name.lower())
    has_managed = bool(app.testConnectors)
    schemes = []
    for scheme in (app.auth_schemes or []):
        integration_fields = []
        user_fields = []
        for f in scheme.fields:
            entry = {
                "name": f.name,
                "display_name": getattr(f, "display_name", None) or f.name,
                "description": getattr(f, "description", ""),
                "type": getattr(f, "type", "string"),
                "required": f.required,
                "expected_from_customer": f.expected_from_customer,
            }
            if f.expected_from_customer:
                user_fields.append(entry)
            else:
                integration_fields.append(entry)
        schemes.append({
            "auth_mode": scheme.auth_mode,
            "integration_fields": integration_fields,
            "user_fields": user_fields,
        })
    primary = schemes[0]["auth_mode"] if schemes else "OAUTH2"
    return {
        "has_managed_auth": has_managed,
        "primary_auth_mode": primary,
        "auth_schemes": schemes,
    }
```

### 3. Update `initiate_connection()` signature

**File**: `packages/vlt-connectors/src/vlt_connectors/service/composio.py`

```python
def initiate_connection(
    self,
    app_name: str,
    entity_id: str,
    label: str = "",
    auth_mode: str | None = None,
    auth_config: dict[str, str] | None = None,
    connected_account_params: dict[str, str] | None = None,
    redirect_url: str | None = None,
) -> dict:
    toolset = self._toolset()
    app = toolset.client.apps.get(name=app_name.lower())
    has_managed = bool(app.testConnectors)

    # Determine auth mode from app if not explicitly provided
    if not auth_mode:
        for scheme in (app.auth_schemes or []):
            auth_mode = scheme.auth_mode
            break

    entity = toolset.get_entity(id=entity_id)
    use_composio = has_managed and not auth_config

    request = entity.initiate_connection(
        app_name=app_name,
        auth_mode=auth_mode,
        auth_config=auth_config or {},
        use_composio_auth=use_composio,
        force_new_integration=bool(auth_config),
        connected_account_params=connected_account_params or {},
        redirect_url=redirect_url,
        labels=[label] if label else None,
    )

    return {
        "connection_id": getattr(request, "connectedAccountId", ""),
        "redirect_url": getattr(request, "redirectUrl", None) or "",
        "status": getattr(request, "connectionStatus", "initiated"),
    }
```

### 4. Add backend route + frontend form

See `contracts/api.yaml` for the `GET /auth-info` endpoint contract.
See `data-model.md` for the `ConnectRequest` body schema.
Follow the existing `ConnectorSettingsDialog` pattern for the credential form.

---

## Phase 2: Connection Registry (P1)

### 1. Add `composio_connections` table

**File**: `backend/src/services/database.py`

Add to both `DDL_STATEMENTS` and `MIGRATION_STATEMENTS`:
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

### 2. Add `ComposioConnectionService`

**File**: `backend/src/services/composio_connections.py` (new)

Follow `ConnectorService` pattern:
- Constructor: `def __init__(self, db_service=None)`
- Methods: `upsert(id, user_id, app_name, label, auth_mode, status)`, `list_for_user(user_id)`, `list_for_user_app(user_id, app_name)`, `get(connection_id)`, `mark_revoked(connection_id)`, `reconcile(user_id, composio_connections)`

### 3. Update routes to use connection registry

Update `POST /connect/{app_name}` to insert into local table on success.
Update `GET /connected` to read from local table (with Composio reconciliation).
Update `DELETE` to use connection_id.
Update `POST /invoke` to accept + route by connection_id.

---

## Phase 3: MCP/CLI Routing (P2)

### 1. Add `connection_id` to MCP tool

**File**: `packages/vlt-cli/src/vlt/mcp/connector_tools.py`

Add `connection_id: Optional[str] = None` param to `connector_call`.
Pass through to invoke endpoint.

### 2. Update `connector_list` response

Include `connections` array per Composio connector entry.

### 3. Update CLI

Add `--connection-id` flag to `vlt connectors call`.

---

## Testing Checklist

- [ ] `disconnect()` no longer raises AttributeError
- [ ] `GET /auth-info` returns correct fields for gmail (managed) vs twitter (custom)
- [ ] `POST /connect/gmail` works with empty body (managed auth)
- [ ] `POST /connect/twitter` works with `auth_config` body (custom auth)
- [ ] API_KEY app connects immediately (no redirect)
- [ ] Multiple connections stored in `composio_connections` table
- [ ] `POST /invoke` with `connection_id` routes to correct connection
- [ ] `POST /invoke` without `connection_id` auto-selects when 1 connection
- [ ] `POST /invoke` without `connection_id` returns 400 when >1 connection
- [ ] `connector_call` MCP tool passes `connection_id` through
- [ ] `connector_list` MCP tool includes `connections` array
