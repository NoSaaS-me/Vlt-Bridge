# Oracle SSE Stream Contract

**Feature**: 012-oracle-turn-control
**Date**: 2026-01-02

## Overview

The Oracle agent streams responses via Server-Sent Events (SSE). This document defines the new `system` chunk type for limit notifications.

## Existing Chunk Types

| Type | Purpose | Key Fields |
|------|---------|------------|
| `thinking` | Reasoning traces | `content` |
| `content` | Response text | `content` |
| `source` | Citation | `source: SourceReference` |
| `tool_call` | Tool invocation | `tool_call: {id, name, arguments, status}` |
| `tool_result` | Tool result | `tool_call_id`, `tool_result` |
| `done` | Query complete | `tokens_used`, `model_used`, `context_id` |
| `error` | Failure | `error` |

## New Chunk Type: `system`

### Purpose

Notify the user and frontend of system-level events:
- Approaching iteration/token limits (soft warnings)
- Hard limits reached
- No-progress detected
- Error limits reached

### Schema

```typescript
interface SystemChunk {
  type: "system";
  system_type: "limit_warning" | "limit_reached" | "no_progress" | "error_limit";
  system_message: string;
  metadata?: {
    current_value?: number;
    limit_value?: number;
    percent?: number;
    limit_type?: "iteration" | "token" | "timeout";
  };
}
```

### Examples

#### Iteration Warning (70% of limit)

```json
{
  "type": "system",
  "system_type": "limit_warning",
  "system_message": "Approaching iteration limit (7/10). Consider wrapping up your response.",
  "metadata": {
    "current_value": 7,
    "limit_value": 10,
    "percent": 70,
    "limit_type": "iteration"
  }
}
```

#### Token Warning (80% of budget)

```json
{
  "type": "system",
  "system_type": "limit_warning",
  "system_message": "Approaching token budget (40,000/50,000 tokens). Consider being more concise.",
  "metadata": {
    "current_value": 40000,
    "limit_value": 50000,
    "percent": 80,
    "limit_type": "token"
  }
}
```

#### Iteration Limit Reached

```json
{
  "type": "system",
  "system_type": "limit_reached",
  "system_message": "Maximum iterations reached (10/10). Saving partial response.",
  "metadata": {
    "current_value": 10,
    "limit_value": 10,
    "percent": 100,
    "limit_type": "iteration"
  }
}
```

#### No Progress Detected

```json
{
  "type": "system",
  "system_type": "no_progress",
  "system_message": "No progress detected - the same action was attempted 3 times. Terminating to prevent infinite loop.",
  "metadata": {
    "repeated_action": "search_code({\"query\": \"authentication\"})"
  }
}
```

#### Error Limit Reached

```json
{
  "type": "system",
  "system_type": "error_limit",
  "system_message": "Multiple consecutive errors (3/3). Terminating with partial results.",
  "metadata": {
    "error_count": 3,
    "last_error": "Tool execution failed: connection timeout"
  }
}
```

## Frontend Handling

### ChatPanel.tsx

```typescript
case 'system':
  // Create new system message in conversation
  setMessages(prev => [
    ...prev,
    {
      _id: generateMessageId(),
      role: 'system',
      content: chunk.system_message,
      timestamp: new Date().toISOString(),
      system_type: chunk.system_type,
      metadata: chunk.metadata,
    }
  ]);
  break;
```

### ChatMessage.tsx

```tsx
if (message.role === 'system') {
  return (
    <div className="bg-amber-50 border-l-4 border-amber-400 p-3 my-2 rounded">
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-amber-600 flex-shrink-0" />
        <span className="text-sm text-amber-800">{message.content}</span>
      </div>
      {message.metadata && (
        <div className="text-xs text-amber-600 mt-1">
          {message.metadata.limit_type}: {message.metadata.current_value}/{message.metadata.limit_value}
        </div>
      )}
    </div>
  );
}
```

## Emission Points

| Condition | When Checked | Chunk Emitted |
|-----------|--------------|---------------|
| Iteration warning | Start of turn | `limit_warning` |
| Token warning | After token count update | `limit_warning` |
| Max iterations | End of turn | `limit_reached` |
| Token budget exceeded | After token count update | `limit_reached` |
| Timeout exceeded | Start of turn | `limit_reached` |
| No progress (3x same action) | After tool result | `no_progress` |
| Error limit (3x consecutive) | After tool error | `error_limit` |

## Backward Compatibility

- New `system` type added; existing types unchanged
- Clients that don't handle `system` can ignore it
- No breaking changes to existing consumers
