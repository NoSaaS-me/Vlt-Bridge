# Data Model: Unified Agent Platform

## Entities

### Daemon
Represents a running instance of the Vlt Daemon.
- `id`: UUID (Primary Key)
- `hostname`: String (e.g., "gpu-server-01")
- `status`: ONLINE | OFFLINE | BUSY
- `last_seen`: Timestamp
- `capabilities`: JSON (e.g., {"gpu": true, "ram": "64GB"})
- `owner_id`: User ID (Owner of this daemon instance)

### AgentSession
Represents a persistent execution context for an agent.
- `id`: UUID (Primary Key)
- `project_id`: String (Foreign Key to Project)
- `daemon_id`: UUID (Foreign Key to Daemon)
- `status`: CREATED | RUNNING | PAUSED | COMPLETED | FAILED
- `created_at`: Timestamp
- `updated_at`: Timestamp
- `metadata`: JSON (Task description, model config)

### SessionLease
Represents the temporary authority granting a Daemon exclusive control over a Session.
- `id`: UUID
- `session_id`: UUID
- `daemon_id`: UUID
- `granted_at`: Timestamp
- `expires_at`: Timestamp
- `active`: Boolean

### TerminalEvent
Represents a chunk of output or state change from a session.
- `id`: UUID
- `session_id`: UUID
- `sequence`: Integer (Ordered index)
- `type`: STDOUT | STDERR | STATE_CHANGE
- `payload`: Text/JSON
- `timestamp`: Timestamp

## Relationships

- **Daemon** `1:N` **AgentSession**: One daemon can manage multiple sessions.
- **AgentSession** `1:N` **TerminalEvent**: A session produces a stream of events.
- **AgentSession** `1:1` **SessionLease**: A active session has one active lease.

## Storage Strategy

- **PostgreSQL/SQLite**: Primary store for `Daemon`, `AgentSession`, and `SessionLease`.
- **Redis (Optional)**: Pub/Sub for `TerminalEvent` broadcasting if scale requires, otherwise WebSocket direct streaming.
