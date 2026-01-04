# Research Document: Agent Notification System

**Feature Branch**: `013-agent-notification-system`
**Date**: 2026-01-03
**Status**: Complete

## Table of Contents

1. [TOON Format Implementation](#1-toon-format-implementation)
2. [Oracle Agent Integration Points](#2-oracle-agent-integration-points)
3. [Frontend Chat UI Architecture](#3-frontend-chat-ui-architecture)
4. [Settings Page Structure](#4-settings-page-structure)
5. [Conversation Persistence](#5-conversation-persistence)
6. [Integration Summary](#6-integration-summary)

---

## 1. TOON Format Implementation

### Decision: Use `python-toon` from PyPI

**Rationale**: Official Python implementation with active maintenance, zero external dependencies, and complete spec compliance.

**Installation**:
```bash
pip install python-toon
# Or from GitHub for latest:
pip install git+https://github.com/toon-format/toon-python.git
```

### Basic API

```python
from toon import encode, decode

# Encode Python dict/list to TOON
data = {"name": "Alice", "age": 30}
toon_str = encode(data)
# Output:
# name: Alice
# age: 30

# Decode TOON back to Python
decoded = decode(toon_str)
```

### Tabular Format (Key Efficiency Feature)

For arrays of uniform objects (our notification use case), TOON uses a highly efficient tabular format:

```python
notifications = [
    {"tool": "vault_search", "error": "timeout", "count": 3},
    {"tool": "coderag_search", "error": "index_missing", "count": 1},
]

print(encode(notifications))
# [2,]{tool,error,count}:
#   vault_search,timeout,3
#   coderag_search,index_missing,1
```

**Token Savings**: 40-60% reduction compared to equivalent JSON for tabular data.

### Templating Strategy

TOON has no built-in Jinja2 integration. Use Jinja2 for template structure, TOON for data encoding:

```python
from jinja2 import Template
from toon import encode

template = Template("""
System Notifications
====================
{{ encoded_events }}
""")

rendered = template.render(encoded_events=encode(events))
```

### Limitations

- **Uniform arrays required**: All objects must have identical keys in identical order for tabular format
- **No comments**: TOON format does not support inline comments
- **Beta status**: Library is v0.9.x, still maturing

### Alternatives Considered

| Package | Status | Notes |
|---------|--------|-------|
| `python-toon` | ✓ Selected | Official, most complete |
| `toon-py` | Evaluated | Requires Python ≥3.10 |
| `py-toon-format` | Evaluated | Compatible but less active |
| Custom parser | Rejected | Unnecessary complexity |

---

## 2. Oracle Agent Integration Points

### Overview

The Oracle Agent (`backend/src/services/oracle_agent.py`) has **33 identified integration points** for event emission and notification injection.

### Primary Integration Points (High Priority)

#### Turn Lifecycle Events

| Event Type | Location | Line | Function |
|------------|----------|------|----------|
| `agent.turn.start` | oracle_agent.py | 423 | `run_turn_loop()` start |
| `agent.turn.end` | oracle_agent.py | 704 | `run_turn_loop()` completion |

#### Tool Execution Events

| Event Type | Location | Line | Function |
|------------|----------|------|----------|
| `tool.call.pending` | oracle_agent.py | 928-936 | Before tool execution |
| `tool.call.success` | oracle_agent.py | 964 | After successful execution |
| `tool.call.failure` | oracle_agent.py | 1043-1051 | On execution error |
| `tool.call.timeout` | tool_executor.py | 240-261 | Timeout handling |

#### Budget Events

| Event Type | Location | Line | Function |
|------------|----------|------|----------|
| `budget.token.warning` | oracle_agent.py | 491-499 | Token threshold check |
| `budget.iteration.warning` | oracle_agent.py | 491-499 | Iteration threshold check |
| `budget.exceeded` | oracle_agent.py | 491-499 | Hard limit reached |

### Notification Injection Points

```python
# In oracle_agent.py run_turn_loop():

# 1. TURN_START injection point (line 423)
async def run_turn_loop(...):
    # INJECT: notifications.drain("turn_start")

    # 2. AFTER_TOOL injection point (line 964)
    for tool_result in tool_results:
        # After processing each tool result
        # INJECT: notifications.drain("after_tool")

    # 3. TURN_END injection point (line 704)
    # INJECT: notifications.drain("turn_end")
```

### SSE Stream Chunk Types

Current chunk types in `OracleStreamChunk`:
- `status`, `thinking`, `content`, `tool_call`, `tool_result`, `sources`, `error`, `done`

**Required Addition**: New chunk type `system` for notification messages.

### Key Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `backend/src/services/oracle_agent.py` | Agent loop | `run_turn_loop()`, `_execute_tool_calls()` |
| `backend/src/services/tool_executor.py` | Tool execution | `execute()`, `_execute_single_tool()` |
| `backend/src/models/oracle.py` | Stream models | `OracleStreamChunk`, `StreamEventType` |

---

## 3. Frontend Chat UI Architecture

### Current Message Types

**File**: `frontend/src/types/oracle.ts` (Lines 77-86)

```typescript
export interface OracleMessage {
  role: 'user' | 'assistant';  // CHANGE: Add 'system'
  content: string;
  timestamp: string;
  thinking?: string;
  sources?: RetrievalResult[];
  tool_calls?: ToolCallInfo[];
  model?: string;
  is_error?: boolean;
}
```

### Required Type Changes

1. **oracle.ts**: Add `'system'` to role union
2. **rag.ts**: Update `Role` type to include `'system'`

### ChatPanel.tsx Message Flow

**File**: `frontend/src/components/ChatPanel.tsx`

```
SSE Stream → streamOracle callback (lines 549-701)
    → Process chunk types (status, content, tool_call, etc.)
    → Update message state
    → Render via ChatMessage component
```

**New Chunk Handling Required** (around line 549):
```typescript
case 'system':
  // Create new system message or append to existing
  const systemMsg: OracleMessageWithId = {
    _id: generateId(),
    role: 'system',
    content: chunk.content,
    timestamp: new Date().toISOString(),
  };
  setMessages(prev => [...prev, systemMsg]);
  break;
```

### ChatMessage.tsx Styling

**File**: `frontend/src/components/ChatMessage.tsx`

Current role detection (line 41):
```typescript
const isUser = message.role === 'user';
```

**Proposed Change**:
```typescript
const isUser = message.role === 'user';
const isSystem = message.role === 'system';
```

**System Message Styling**:
```typescript
// Avatar
<div className={cn(
  "h-8 w-8 rounded-full flex items-center justify-center",
  isUser ? "bg-primary text-primary-foreground" :
  isSystem ? "bg-yellow-500/20 text-yellow-700 dark:text-yellow-300 border border-yellow-500/30" :
  "bg-secondary text-secondary-foreground"
)}>
  {isUser ? <User /> : isSystem ? <AlertCircle /> : <Bot />}
</div>

// Container
<div className={cn(
  "flex gap-3 p-4",
  isUser ? "bg-transparent" :
  isSystem ? "bg-yellow-500/5 border-l-2 border-yellow-500/30" :
  "bg-muted/30"
)}>
```

### Key Files

| File | Purpose | Changes Required |
|------|---------|------------------|
| `frontend/src/types/oracle.ts` | Message types | Add 'system' role |
| `frontend/src/types/rag.ts` | Role type | Add 'system' to union |
| `frontend/src/components/ChatPanel.tsx` | Message handling | Handle 'system' chunks |
| `frontend/src/components/ChatMessage.tsx` | Message rendering | Add system styling |

---

## 4. Settings Page Structure

### Current Architecture

**File**: `frontend/src/pages/Settings.tsx` (1,025 lines)

**Layout Pattern**: Vertical card-based layout with `space-y-6` spacing

Current sections:
1. Profile Section (user info)
2. API Token Section
3. Index Health Section
4. Code Index Section (CodeRAG)
5. AI Models Section
6. Context Tree Settings
7. System Logs

### State Management Pattern

```typescript
// Per-section independent state
const [modelSettings, setModelSettings] = useState<ModelSettings | null>(null);
const [isSavingModels, setIsSavingModels] = useState(false);
const [modelsSaved, setModelsSaved] = useState(false);

// Save pattern
const handleSave = async () => {
  setIsSaving(true);
  try {
    await saveSettings(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);  // 2s success toast
  } finally {
    setIsSaving(false);
  }
};
```

### Tabbed Interface Approach

**Decision**: Use shadcn/ui Tabs component

**Rationale**:
- Settings page is already 1,000+ lines
- Notifications section adds more complexity
- Tabs provide cleaner UX for related settings

**Required Setup**:
```bash
# Add tabs component to shadcn/ui
npx shadcn@latest add tabs
```

**Proposed Tab Structure**:
```typescript
<Tabs defaultValue="account">
  <TabsList>
    <TabsTrigger value="account">Account</TabsTrigger>
    <TabsTrigger value="models">AI Models</TabsTrigger>
    <TabsTrigger value="notifications">Notifications</TabsTrigger>
    <TabsTrigger value="advanced">Advanced</TabsTrigger>
  </TabsList>

  <TabsContent value="notifications">
    {/* Subscriber toggle list */}
  </TabsContent>
</Tabs>
```

### Notification Settings Component

```typescript
// frontend/src/components/NotificationSettings.tsx
interface SubscriberInfo {
  id: string;
  name: string;
  description: string;
  event_types: string[];
  core: boolean;  // Cannot be disabled
  enabled: boolean;
}

// Render toggles
{subscribers.map(sub => (
  <div key={sub.id} className="flex items-center justify-between">
    <div>
      <label className="text-sm font-medium">{sub.name}</label>
      <p className="text-xs text-muted-foreground">{sub.description}</p>
    </div>
    <Switch
      checked={sub.enabled}
      disabled={sub.core}
      onCheckedChange={(checked) => toggleSubscriber(sub.id, checked)}
    />
  </div>
))}
```

---

## 5. Conversation Persistence

### Database Schema

**File**: `backend/src/services/database.py`

#### Tree-Based Context (Primary)

```sql
CREATE TABLE context_nodes (
    id TEXT PRIMARY KEY,
    root_id TEXT NOT NULL,
    parent_id TEXT,
    user_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    tool_calls_json TEXT DEFAULT '[]',
    tokens_used INTEGER DEFAULT 0,
    model_used TEXT,
    -- ... more fields
);
```

### Current Role Support

**File**: `backend/src/models/oracle_context.py`

```python
class ExchangeRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    # MISSING: SYSTEM
```

**File**: `backend/src/models/oracle.py` (Line 63)

```python
class ConversationMessage(BaseModel):
    role: Literal["user", "assistant", "system"]  # Already supports system!
```

**Key Finding**: API model already supports "system" role, but persistence enum does not.

### Required Changes

1. **Add SYSTEM to ExchangeRole**:
```python
class ExchangeRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"  # NEW
```

2. **Add system_messages column to context_nodes** (optional, for rich storage):
```sql
ALTER TABLE context_nodes ADD COLUMN system_messages_json TEXT DEFAULT '[]';
```

3. **Update history building** in oracle_agent.py to include system messages.

### Migration Strategy

**Decision**: Backward-compatible extension

**Rationale**:
- Adding new enum value doesn't break existing data
- Optional column addition with default value is safe
- No data migration required for existing conversations

---

## 6. Integration Summary

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         EVENT SOURCES                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ToolExecutor  │  │ OracleAgent  │  │ BudgetMonitor│                   │
│  │ (tool_executor│  │ (oracle_agent│  │ (oracle_agent│                   │
│  │  .py:240-261) │  │  .py:423-704)│  │  .py:491-499)│                   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                 │                 │                            │
│         └────────────────┴─────────────────┘                            │
│                           │                                              │
│                           ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                      EVENT BUS                                    │   │
│  │  backend/src/services/ans/bus.py                                 │   │
│  │  - Receives events from all sources                              │   │
│  │  - Dispatches to matching subscribers                            │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   SUBSCRIBER SYSTEM                               │   │
│  │  backend/src/services/ans/subscribers/                           │   │
│  │  ├── tool_failure.toml     (event filter + template)             │   │
│  │  ├── budget_warning.toml                                         │   │
│  │  └── loop_detected.toml                                          │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                  NOTIFICATION ACCUMULATOR                         │   │
│  │  - Batches events by subscriber                                  │   │
│  │  - Deduplicates within window                                    │   │
│  │  - Formats to TOON via Jinja2 templates                          │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                   CONTEXT INJECTOR                                │   │
│  │  oracle_agent.py injection points:                               │   │
│  │  - turn_start (line 423)                                         │   │
│  │  - after_tool (line 964)                                         │   │
│  │  - turn_end (line 704)                                           │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
└─────────────────────────────────────┼────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          SSE STREAM                                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  OracleStreamChunk(type="system", content=toon_notification)     │   │
│  └──────────────────────────────────┬───────────────────────────────┘   │
│                                     │                                    │
│                                     ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     FRONTEND                                      │   │
│  │  ChatPanel.tsx → handles "system" chunks                         │   │
│  │  ChatMessage.tsx → renders with system styling                   │   │
│  │  Settings.tsx → Notifications tab for subscriber management     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### File Changes Summary

#### Backend (New Files)

| File | Purpose |
|------|---------|
| `backend/src/services/ans/__init__.py` | ANS package init |
| `backend/src/services/ans/bus.py` | Event bus implementation |
| `backend/src/services/ans/event.py` | Event dataclass and types |
| `backend/src/services/ans/subscriber.py` | Subscriber protocol and loader |
| `backend/src/services/ans/accumulator.py` | Batching, deduplication |
| `backend/src/services/ans/toon_formatter.py` | TOON + Jinja2 integration |
| `backend/src/services/ans/subscribers/*.toml` | Subscriber configs |
| `backend/src/services/ans/templates/*.toon.j2` | Notification templates |

#### Backend (Modified Files)

| File | Changes |
|------|---------|
| `backend/src/services/oracle_agent.py` | Emit events, inject notifications |
| `backend/src/services/tool_executor.py` | Emit tool events |
| `backend/src/models/oracle.py` | Add StreamEventType.SYSTEM |
| `backend/src/models/oracle_context.py` | Add ExchangeRole.SYSTEM |
| `backend/src/api/routes/models.py` | Add subscriber settings endpoints |
| `backend/src/services/database.py` | Add notification settings columns |

#### Frontend (New Files)

| File | Purpose |
|------|---------|
| `frontend/src/components/ui/tabs.tsx` | shadcn/ui Tabs component |
| `frontend/src/components/NotificationSettings.tsx` | Subscriber management |
| `frontend/src/services/notifications.ts` | API client for settings |
| `frontend/src/types/notifications.ts` | TypeScript types |

#### Frontend (Modified Files)

| File | Changes |
|------|---------|
| `frontend/src/types/oracle.ts` | Add 'system' to role |
| `frontend/src/types/rag.ts` | Add 'system' to Role type |
| `frontend/src/components/ChatPanel.tsx` | Handle 'system' chunks |
| `frontend/src/components/ChatMessage.tsx` | System message styling |
| `frontend/src/pages/Settings.tsx` | Add Notifications tab |

### Dependencies to Add

**Backend** (`backend/pyproject.toml`):
```toml
dependencies = [
    # ... existing
    "python-toon>=0.9.0",
]
```

**Frontend** (`frontend/package.json`):
```json
{
  "dependencies": {
    "@radix-ui/react-tabs": "^1.0.4"  # Already installed
  }
}
```

---

## Confidence Assessment

**Overall Confidence: 9/10**

| Area | Confidence | Notes |
|------|------------|-------|
| TOON Integration | 8/10 | Library is beta but well-documented |
| Oracle Agent Integration | 9/10 | Exact line numbers identified |
| Chat UI Changes | 9/10 | Backend already supports system role |
| Settings Page | 9/10 | Clear existing patterns to follow |
| Persistence | 9/10 | Backward-compatible approach |

**Remaining Uncertainties**:
1. TOON library edge cases in production
2. Exact debounce/batch timing may need tuning
3. UI color scheme for system messages needs design review
