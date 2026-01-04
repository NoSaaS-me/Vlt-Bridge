# Quickstart: Agent Notification System

**Feature Branch**: `013-agent-notification-system`

This guide helps developers get started with implementing and extending the Agent Notification System (ANS).

## Prerequisites

1. Backend running (`uv run uvicorn src.api.main:app --reload`)
2. Frontend running (`npm run dev`)
3. Familiarity with:
   - FastAPI services pattern
   - React component patterns
   - TOON format basics

## Quick Test: Verify Integration Points

### 1. Check Tool Failure Event Point

```python
# backend/src/services/tool_executor.py
# Look for timeout handling around line 244-270

# Add temporary debug log:
logger.info(f"[ANS-DEBUG] Tool timeout: {tool_name}")
```

### 2. Check Budget Warning Point

```python
# backend/src/services/oracle_agent.py
# Look for budget checks around line 350-380 (_check_token_budget method)

# Add temporary debug log:
logger.info(f"[ANS-DEBUG] Budget check: {percentage}%")
```

## Implementing a New Subscriber

### Step 1: Create TOML Config

```toml
# backend/src/services/ans/subscribers/my_subscriber.toml

[subscriber]
id = "my_subscriber"
name = "My Custom Notifications"
description = "Notifies agent when X happens"
version = "1.0.0"

[events]
types = ["my.custom.event"]
severity_filter = "info"

[batching]
window_ms = 1000
max_size = 5

[output]
priority = "normal"
inject_at = "after_tool"
template = "my_subscriber.toon.j2"
core = false  # User can disable this one
```

### Step 2: Create TOON Template

```jinja
{# backend/src/services/ans/templates/my_subscriber.toon.j2 #}
{% if events|length == 1 %}
my_event: {{ events[0].payload.message }}
{% else %}
my_events[{{ events|length }}]{ts,message}:
{% for e in events %}
  {{ e.timestamp | format_time }},{{ e.payload.message }}
{% endfor %}
{% endif %}
```

### Step 3: Emit Events

```python
# In your service code (within backend/src/services/)
# Use relative imports if inside the services package:
from .ans.bus import get_event_bus
from .ans.event import Event, Severity

# Or use absolute imports from outside:
from src.services.ans.bus import get_event_bus
from src.services.ans.event import Event, Severity

bus = get_event_bus()
bus.emit(Event(
    type="my.custom.event",
    source="my_service",
    severity=Severity.INFO,
    payload={"message": "Something happened!"}
))
```

## Testing TOON Output

### Quick TOON Test

```python
from toon import encode, decode

# Test encoding
data = [
    {"tool": "search", "status": "failed"},
    {"tool": "write", "status": "success"},
]
print(encode(data))
# [2,]{tool,status}:
#   search,failed
#   write,success

# Test decoding
decoded = decode(encoded_str)
assert decoded == data
```

### Test Template Rendering

```python
from jinja2 import Template
from toon import encode

template = Template("""
{% if events|length == 1 %}
single: {{ events[0].name }}
{% else %}
batch[{{ events|length }}]{name}:
{% for e in events %}
  {{ e.name }}
{% endfor %}
{% endif %}
""")

events = [{"name": "foo"}, {"name": "bar"}]
print(template.render(events=events))
```

## Frontend: System Message Support (Already Implemented)

The frontend already has full system message support. This section documents the implementation.

### 1. Types (Already Updated)

```typescript
// frontend/src/types/oracle.ts
export interface OracleMessage {
  role: 'user' | 'assistant' | 'system';  // 'system' included
  // ...existing fields
}
```

### 2. SSE Chunk Handling (ChatPanel.tsx line ~693)

```typescript
// frontend/src/components/ChatPanel.tsx
// In the streamOracle callback:

} else if (chunk.type === 'system') {
  const systemMsg: OracleMessageWithId = {
    _id: `sys_${Date.now()}_${Math.random().toString(36).substring(7)}`,
    role: 'system',
    content: chunk.content || '',
    timestamp: new Date().toISOString(),
  };
  // Insert system message before the current assistant message
  return [...prev.slice(0, lastIndex), systemMsg, lastMsg];
}
```

### 3. System Message Styling (ChatMessage.tsx)

```typescript
// frontend/src/components/ChatMessage.tsx
const isSystem = message.role === 'system';

// Features implemented:
// - Yellow/amber background and left border
// - AlertCircle icon with "System" label
// - Collapsible content for verbose messages (>200 chars or >3 lines)
// - TOON parsing with graceful fallback on error
```

## API Endpoints

### List Subscribers

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/notifications/subscribers
```

### Toggle Subscriber

```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}' \
  http://localhost:8000/api/notifications/subscribers/tool_success/toggle
```

### Get Notification Settings

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/settings/notifications
```

## Running Tests

```bash
# Backend unit tests
cd backend
uv run pytest tests/unit/test_ans_*.py -v

# Specific test
uv run pytest tests/unit/test_ans_bus.py::test_event_emission -v
```

## Debugging Tips

### Enable Debug Logging

```python
# In oracle_agent.py or tool_executor.py
import logging
logging.getLogger("ans").setLevel(logging.DEBUG)
```

### Check Event Flow

1. Add log in event emission point
2. Add log in subscriber filter
3. Add log in accumulator
4. Add log in SSE stream yield

### Common Issues

| Issue | Solution |
|-------|----------|
| Notifications not appearing | Check subscriber is enabled, event type matches |
| TOON parsing error | Verify template syntax, check for unescaped commas |
| Batching not working | Check window_ms > 0 and events fire within window |
| System messages not styled | Verify role='system' in message object |

## File Locations

| Component | Path |
|-----------|------|
| Event Bus | `backend/src/services/ans/bus.py` |
| Subscribers | `backend/src/services/ans/subscribers/*.toml` |
| Templates | `backend/src/services/ans/templates/*.toon.j2` |
| API Routes | `backend/src/api/routes/notifications.py` |
| Frontend Types | `frontend/src/types/notifications.ts` |
| Settings UI | `frontend/src/components/NotificationSettings.tsx` |

## Next Steps

1. Run `/speckit.tasks` to generate implementation tasks
2. Start with P1 user stories (tool failure, system messages)
3. Add core subscribers one at a time
4. Add Settings UI after backend is working
