# Data Model: AI Chat Window

## Entities

### ChatMessage
Represents a single message in the conversation history.

| Field | Type | Description |
|-------|------|-------------|
| `role` | `enum` | `user`, `assistant`, `system` |
| `content` | `string` | The text content of the message |
| `timestamp` | `datetime` | ISO 8601 timestamp of creation |

### ChatRequest
The payload sent from Frontend to Backend to initiate/continue a chat.

| Field | Type | Description |
|-------|------|-------------|
| `message` | `string` | The new user message |
| `history` | `List[ChatMessage]` | Previous conversation context |
| `persona` | `string` | ID of the selected persona (e.g., "default", "auto-linker") |
| `model` | `string` | Optional: Specific OpenRouter model ID |

### ChatResponseChunk (SSE)
The streaming data format received by the frontend.

| Field | Type | Description |
|-------|------|-------------|
| `type` | `enum` | `token` (text chunk) or `tool_call` (tool execution status) |
| `content` | `string` | The text fragment or status message |
| `done` | `boolean` | True if generation is complete |

## Persistence (Markdown Format)
Saved in `data/vaults/{user_id}/Chat Logs/{timestamp}.md`

```markdown
---
title: Chat Session - {timestamp}
date: {date}
tags: [chat-log, {persona}]
model: {model_id}
---

# Chat Session

## User ({time})
What is the summary of...

## Assistant ({time})
Based on your notes...
```
