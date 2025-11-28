# Data Model: Gemini Vault Chat Agent

**Feature**: 004-gemini-vault-chat  
**Date**: 2025-11-28

## Entities

### ChatMessage

Represents a single message in a conversation.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| role | enum | Message author | `user` or `assistant` |
| content | string | Message text | Max 10,000 characters |
| timestamp | datetime | When message was created | ISO 8601 format |
| sources | SourceReference[] | Referenced notes (assistant only) | Optional, empty for user messages |
| notes_written | NoteWritten[] | Notes created by agent | Optional, Phase 2 only |

### SourceReference

Metadata about a note used to generate a response.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| path | string | Relative path in vault | Valid vault path, ends in `.md` |
| title | string | Note title | Derived from frontmatter/H1/filename |
| snippet | string | Relevant text excerpt | Max 500 characters |
| score | float | Relevance score | 0.0 to 1.0, optional |

### NoteWritten (Phase 2)

Metadata about a note created or updated by the agent.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| path | string | Path to created/updated note | Must be in `agent-notes/` folder |
| title | string | Note title | Required |
| action | enum | What the agent did | `created` or `updated` |

### ChatRequest

Request payload for the RAG chat endpoint.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| messages | ChatMessage[] | Conversation history | At least 1 message, last must be `user` |

### ChatResponse

Response payload from the RAG chat endpoint.

| Field | Type | Description | Constraints |
|-------|------|-------------|-------------|
| answer | string | AI-generated response | Required |
| sources | SourceReference[] | Notes used in response | May be empty |
| notes_written | NoteWritten[] | Notes created (Phase 2) | May be empty |

## State Transitions

### Conversation Session

```
[No Session] ---(user opens chat panel)---> [Active Session]
[Active Session] ---(user sends message)---> [Waiting for Response]
[Waiting for Response] ---(response received)---> [Active Session]
[Active Session] ---(page refresh/close)---> [No Session]
```

### Index Lifecycle

```
[No Index] ---(startup, no persist dir)---> [Building Index]
[Building Index] ---(indexing complete)---> [Index Ready]
[No Index] ---(startup, persist dir exists)---> [Loading Index]
[Loading Index] ---(load successful)---> [Index Ready]
[Loading Index] ---(load failed)---> [Building Index]
[Index Ready] ---(query received)---> [Index Ready]
```

## Validation Rules

### ChatMessage Validation

1. `role` must be exactly `user` or `assistant`
2. `content` must not be empty (whitespace-only is invalid)
3. `content` must be ≤10,000 characters
4. `sources` must be empty for `user` role messages

### SourceReference Validation

1. `path` must be a valid vault path (see `validate_note_path` in vault.py)
2. `title` must not be empty
3. `snippet` must be ≤500 characters
4. `score` if present must be between 0.0 and 1.0

### NoteWritten Validation (Phase 2)

1. `path` must start with `agent-notes/`
2. `path` must be a valid vault path
3. `action` must be `created` or `updated`

## Relationships

```
ChatSession (frontend state)
  └── contains 0..* ChatMessage
                      └── assistant messages contain 0..* SourceReference
                                                    └── references 1 VaultNote (existing)
                      └── assistant messages may contain 0..* NoteWritten
                                                    └── creates/updates 1 VaultNote
```

## Persistence

| Entity | Storage | Lifetime |
|--------|---------|----------|
| ChatMessage | Frontend memory | Session (cleared on refresh) |
| SourceReference | Derived from query | Per response |
| NoteWritten | VaultService (filesystem) | Permanent |
| Vector Index | LlamaIndex persist dir | Until rebuild |

