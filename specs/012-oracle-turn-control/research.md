# Research Findings: Oracle Agent Turn Control

**Feature**: 012-oracle-turn-control
**Date**: 2026-01-02
**Status**: Complete

## 1. Current Oracle Agent Implementation

### Decision: Keep existing architecture, refactor loop internals
**Rationale**: The current architecture is sound - parallel tool execution, tree-based context, SSE streaming. Only the loop control logic needs refactoring.
**Alternatives Considered**: Complete rewrite rejected - brownfield integration per constitution

### Current State Analysis

| Aspect | Current Value | Location |
|--------|--------------|----------|
| MAX_TURNS | 30 (hardcoded) | `oracle_agent.py:126` |
| Tool execution | Parallel via `asyncio.gather()` | `oracle_agent.py:1006-1009` |
| Termination | finish_reason="stop" or max turns | `oracle_agent.py:449, 457` |
| Settings integration | Per-agent init, not per-query | `oracle_agent.py:162` |

### SSE Chunk Types (Existing)

```python
type: Literal["thinking", "content", "source", "tool_call", "tool_result", "done", "error"]
```

**Gap**: No "system" type for notifications. Must add.

### State Currently Tracked

| Variable | Purpose |
|----------|---------|
| `_context` | Legacy context object |
| `_current_tree_root_id` | Active conversation tree |
| `_current_node_id` | HEAD position in tree |
| `_collected_sources` | Citations gathered |
| `_collected_tool_calls` | All tool executions |
| `_cancelled` | Cancellation flag |
| `turn` (local) | Current iteration counter |

**Gap**: No token tracking, no elapsed time, no action history for no-progress detection.

### DecisionTree Hook Points Identified

| Location | Current Code | Hook Opportunity |
|----------|--------------|------------------|
| Line 416 | `for turn in range(MAX_TURNS)` | `should_continue()` |
| Line 418 | Cancellation check | `on_turn_start()` |
| Line 1061 | Post-tool result | `on_tool_result()` |
| Line 743 | Pre-finalization | `should_finalize()` |

---

## 2. Frontend Implementation

### Decision: Add "system" message role with distinct styling
**Rationale**: Clean separation of concerns - system notifications are not user or assistant content.
**Alternatives Considered**: Inline in assistant messages rejected - less clear UX.

### Settings Page Patterns

| Pattern | Component | Usage |
|---------|-----------|-------|
| Numeric input | `<Input type="number">` | Timeout, max nodes |
| Slider | Custom slider | Could use for iterations |
| Toggle | `<Switch>` | Boolean settings |
| Select | `<Select>` with groups | Model selection |
| Auto-save | `useState` + API call | No save button needed |

### ChatPanel SSE Handling

```typescript
// Current chunk handling pattern
switch (chunk.type) {
  case 'thinking': // Append to msg.thinking
  case 'content':  // Append to msg.content
  case 'source':   // Add to msg.sources[]
  case 'tool_call': // Add to msg.tool_calls[]
  case 'tool_result': // Match by ID, update status
  case 'done':     // Mark complete, save context_id
  case 'error':    // Set is_error flag
}
```

**Implementation for system**: Add `case 'system'` that creates new message with `role: 'system'`.

### Message Type Extension

```typescript
interface OracleMessage {
  role: 'user' | 'assistant' | 'system';  // Add 'system'
  // ... existing fields
  system_type?: 'limit_warning' | 'limit_reached' | 'no_progress';
}
```

---

## 3. Protocol Patterns for DecisionTree

### Decision: Use `typing.Protocol` with `@runtime_checkable`
**Rationale**: Structural subtyping allows skill trees to implement interface without inheritance. More flexible than ABC.
**Alternatives Considered**: ABC (existing pattern in tool_parsers) rejected for new code - Protocol is more Pythonic.

### Recommended Protocol Interface

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class DecisionTree(Protocol):
    def should_continue(self, state: "AgentState") -> tuple[bool, str]:
        """Return (continue?, reason)"""
        ...

    def on_turn_start(self, state: "AgentState") -> "AgentState":
        """Hook before each turn, can inject system messages"""
        ...

    def on_tool_result(self, state: "AgentState", result: dict) -> "AgentState":
        """Process tool result, update state"""
        ...

    def get_config(self) -> "AgentConfig":
        """Return configuration for this tree"""
        ...
```

### Decorator-Based Skill Registration

```python
from functools import wraps

_decision_trees: dict[str, type[DecisionTree]] = {}

def decision_tree(name: str):
    """Decorator to register a decision tree implementation."""
    def decorator(cls):
        _decision_trees[name] = cls
        return cls
    return decorator

# Usage:
@decision_tree("default")
class DefaultDecisionTree:
    ...

@decision_tree("deep_researcher")
class DeepResearcherTree:
    ...
```

---

## 4. AgentState Extension Pattern

### Decision: Use `@dataclass(frozen=True, kw_only=True)`
**Rationale**: Immutable state prevents bugs; kw_only solves inheritance ordering problem.
**Alternatives Considered**: Mutable dataclass rejected - harder to reason about state changes.

### Base AgentState

```python
@dataclass(frozen=True, kw_only=True)
class AgentState:
    # Core
    user_id: str
    project_id: str

    # Tracking (new)
    turn: int = 0
    tokens_used: int = 0
    start_time: float = field(default_factory=time.time)
    recent_actions: tuple[str, ...] = field(default_factory=tuple)

    # Termination
    termination_reason: Optional[str] = None

    # For extensions
    extensions: dict[str, Any] = field(default_factory=dict)
```

### Extension for DeepResearcher (Future)

```python
@dataclass(frozen=True, kw_only=True)
class DeepResearcherState(AgentState):
    research_depth: int = 0
    reflection_count: int = 0
    hypotheses: tuple[str, ...] = field(default_factory=tuple)
```

---

## 5. AgentConfig Persistence

### Decision: Extend existing `ModelSettings` in user_settings
**Rationale**: User settings service already handles persistence; add new fields.
**Alternatives Considered**: Separate table rejected - unnecessary complexity.

### New Fields for ModelSettings

```python
class ModelSettings(BaseModel):
    # Existing
    oracle_model: str
    subagent_model: str
    thinking_enabled: bool
    librarian_timeout: int
    max_context_nodes: int

    # New AgentConfig fields
    max_iterations: int = 15
    soft_warning_percent: int = 70
    token_budget: int = 50000
    token_warning_percent: int = 80
    timeout_seconds: int = 120
    max_tool_calls_per_turn: int = 5
    max_parallel_tools: int = 3
```

### Validation Bounds

```python
class AgentConfigBounds:
    max_iterations: tuple[int, int] = (1, 50)
    token_budget: tuple[int, int] = (1000, 200000)
    timeout_seconds: tuple[int, int] = (10, 600)
    max_tool_calls_per_turn: tuple[int, int] = (1, 20)
    max_parallel_tools: tuple[int, int] = (1, 10)
```

---

## 6. System Message Implementation

### Decision: New SSE chunk type + frontend message role
**Rationale**: Clean separation, consistent with existing chunk pattern.

### Backend: New Chunk Type

```python
# In OracleStreamChunk
type: Literal[..., "system"]
system_type: Optional[Literal["limit_warning", "limit_reached", "no_progress", "error_limit"]]
system_message: Optional[str]
```

### Emission Points

1. **Pre-turn check** (70% iterations): Yield system chunk with warning
2. **Token budget check** (80% tokens): Yield system chunk with token info
3. **On termination** (limit reached): Yield system chunk with explanation
4. **No-progress detection**: Yield system chunk with action history

### Frontend: System Message Styling

```tsx
// ChatMessage.tsx
if (message.role === 'system') {
  return (
    <div className="bg-amber-50 border-l-4 border-amber-400 p-3 my-2">
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-4 w-4 text-amber-600" />
        <span className="text-sm text-amber-800">{message.content}</span>
      </div>
    </div>
  );
}
```

---

## 7. No-Progress Detection Algorithm

### Decision: Track last 3 actions, compare stringified tool calls
**Rationale**: Simple, effective; avoids false positives from result variation.

### Algorithm

```python
def detect_no_progress(recent_actions: tuple[str, ...]) -> bool:
    if len(recent_actions) < 3:
        return False

    # Same action 3 times
    if recent_actions[-1] == recent_actions[-2] == recent_actions[-3]:
        return True

    return False

def action_signature(tool_name: str, arguments: dict) -> str:
    """Create comparable signature from tool call."""
    import json
    return f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
```

---

## Summary of Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Architecture | Refactor internals, keep structure | Brownfield integration |
| Protocol | `typing.Protocol` + `@runtime_checkable` | Structural subtyping |
| State | `@dataclass(frozen=True, kw_only=True)` | Immutable, extensible |
| Registration | Decorator-based registry | Self-registering skills |
| Settings | Extend ModelSettings | Reuse existing persistence |
| System messages | New chunk type + role | Clean separation |
| No-progress | 3-action comparison | Simple, effective |
