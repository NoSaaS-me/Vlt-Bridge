# Research: Composio Connection Vault

**Date**: 2026-03-14
**Method**: SDK source code reading + live API verification + codebase pattern analysis

---

## 1. Composio SDK Auth Detection — Verified

**Decision**: Use `app.testConnectors` to detect managed vs custom auth.
**Rationale**: This is the exact field the SDK itself checks at `toolset.py:1374` and `cli/add.py:262`. Live API confirms:

| App | `testConnectors` | Primary Auth | User must supply |
|-----|-----------------|-------------|-----------------|
| gmail | True (1 entry) | OAUTH2 | Nothing (Composio manages) |
| reddit | True (1 entry) | OAUTH2 | Nothing |
| slack | True (1 entry) | OAUTH2 | Nothing |
| twitter | False (0) | OAUTH2 | client_id, client_secret |
| openai | False (0) | API_KEY | api_key, organization_id |
| notion | True (1) | OAUTH2 + API_KEY | OAUTH2: nothing; API_KEY: api_key |

**Key insight**: `expected_from_customer=False` on a field means Composio fills it server-side. For managed OAuth apps, ALL fields have `expected_from_customer=False`. For custom OAuth apps, the fields still show `expected_from_customer=False` but there are no `testConnectors` to actually fill them — the operator must supply `client_id`/`client_secret` via `auth_config`.

**Alternatives considered**:
- Check `no_auth` field — only filters toolkit-level entries (already done in catalog)
- Check field count — unreliable, varies per app
- Hard-code a list of managed apps — fragile, Composio changes their catalog

---

## 2. Multi-Account Connection Model — Verified

**Decision**: Store connection metadata locally in SQLite; use Composio's `connected_account_id` for routing.
**Rationale**:
- Composio allows N connections per (entity_id, app) pair — no uniqueness constraint
- `Entity.get_connection()` silently picks the **oldest** connection (variable named `latest_account` but uses `<` comparison on `createdAt`)
- Labels sent during creation are NOT returned by the API — `ConnectedAccountModel` has no `labels` field
- The only stable identifier is `ConnectedAccountModel.id` (UUID)

We must store labels and connection metadata ourselves because Composio doesn't surface them.

**Alternatives considered**:
- Rely on Composio labels — write-only, can't read back or filter
- Use `integrationId` to distinguish — different concept (integration = OAuth app config, connection = user session)
- Store in `connector_configs` table — wrong shape, it's a KV store not a connection registry

---

## 3. Database Pattern — Existing Codebase

**Decision**: Add `composio_connections` table using existing DDL + MIGRATION pattern.
**Rationale**: The project uses raw `sqlite3` with:
- `DDL_STATEMENTS` tuple for fresh installs (`CREATE TABLE IF NOT EXISTS`)
- `MIGRATION_STATEMENTS` tuple for upgrades (same DDL, applied with try/except to skip already-existing tables)
- `ConnectorService` pattern: constructor takes optional `DatabaseService`, all access via `conn = self.db.connect()` / `finally: conn.close()`, upsert via `INSERT ... ON CONFLICT ... DO UPDATE`

No ORM. No migration framework. Simple and effective.

**Alternatives considered**:
- SQLAlchemy — overkill for one table, mismatches existing pattern
- Store in `connector_configs` KV table — wrong shape for structured connection data
- Separate database file — unnecessary complexity

---

## 4. Connection Deletion — SDK Bug Workaround

**Decision**: Use raw HTTP `DELETE /v1/connectedAccounts/{id}` via `toolset.client.http.delete()`.
**Rationale**: `ConnectedAccounts` class has NO `delete()` method. Only methods: `get`, `info`, `initiate`. Our current `disconnect()` calls `connected_accounts.delete()` which will `AttributeError` at runtime.

The raw HTTP path is available via `toolset.client.http` which is a standard `HttpClient` instance.

**Alternatives considered**:
- Patch the SDK locally — fragile, breaks on upgrade
- Only "soft delete" locally — leaves orphaned connections in Composio
- Wait for SDK fix — unknown timeline, blocks shipping

---

## 5. Frontend Credential Form Pattern

**Decision**: Extend the existing `ConnectorSettingsDialog` pattern for Composio auth forms.
**Rationale**: The codebase already has:
- Dynamic field rendering from a schema (`credential_fields` array)
- Secret field masking (`••••••••` pattern)
- Loading/saving state management
- Error display via `Alert variant="destructive"`
- Native HTML `<select>` for dropdowns (simpler than shadcn Select)

The Composio auth form follows the same pattern: receive field schema from `GET /auth-info`, render appropriate inputs, POST credentials on submit.

**Alternatives considered**:
- shadcn Select component — more complex than needed, existing code uses native select
- Multi-step wizard — over-engineered for 2-3 fields
- External OAuth app registration page — poor UX, forces user out of flow

---

## 6. Action Execution Routing

**Decision**: Add optional `connected_account_id` to `execute_action()` call, auto-select when single connection.
**Rationale**: `toolset.execute_action()` accepts `connected_account_id: Optional[str]` at line 1833 of `toolset.py`. When omitted, `Entity.get_connection()` picks the oldest. When provided, it uses that specific connection.

Auto-select logic: if user has exactly 1 active connection for the app, use it. If >1 and no `connection_id` specified, return error listing available connections with labels.

**Alternatives considered**:
- Always require `connection_id` — breaks single-connection UX, backwards incompatible
- Default to newest — still arbitrary, doesn't solve disambiguation
- Let Composio pick — unpredictable (oldest wins), no user control
