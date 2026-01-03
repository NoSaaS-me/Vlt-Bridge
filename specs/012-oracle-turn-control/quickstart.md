# Quickstart: Oracle Agent Turn Control

**Feature**: 012-oracle-turn-control

## What This Feature Does

1. **Configurable Limits**: Users can set max iterations, token budgets, and timeouts
2. **Smart Termination**: Agent stops when goal achieved or limits reached
3. **System Notifications**: Warnings appear in chat before limits are hit
4. **Pluggable Control Flow**: DecisionTree protocol for future skill extensions

## Key Files to Modify

### Backend

| File | Changes |
|------|---------|
| `backend/src/models/settings.py` | Add AgentConfig fields to ModelSettings |
| `backend/src/models/oracle.py` | Add "system" chunk type |
| `backend/src/models/agent_state.py` | NEW: AgentState dataclass |
| `backend/src/services/decision_tree/` | NEW: DecisionTree module |
| `backend/src/services/oracle_agent.py` | Integrate DecisionTree, emit system chunks |
| `backend/src/api/routes/models.py` | Add AgentConfig endpoints |

### Frontend

| File | Changes |
|------|---------|
| `frontend/src/types/oracle.ts` | Add 'system' role, AgentConfig interface |
| `frontend/src/components/ChatPanel.tsx` | Handle 'system' chunk type |
| `frontend/src/components/ChatMessage.tsx` | System message styling |
| `frontend/src/pages/Settings.tsx` | Add AgentConfig UI section |

## Quick Reference

### AgentConfig Defaults

```python
max_iterations = 15        # Max agent turns
soft_warning_percent = 70  # Warn at 70% of max
token_budget = 50000       # Max tokens per query
token_warning_percent = 80 # Warn at 80% of budget
timeout_seconds = 120      # 2 minute timeout
max_tool_calls_per_turn = 5
max_parallel_tools = 3
```

### DecisionTree Protocol

```python
class DecisionTree(Protocol):
    def should_continue(self, state: AgentState) -> tuple[bool, str]: ...
    def on_turn_start(self, state: AgentState) -> AgentState: ...
    def on_tool_result(self, state: AgentState, result: dict) -> AgentState: ...
    def get_config(self) -> AgentConfig: ...
```

### Termination Priority

1. User cancellation
2. Model finish_reason="stop" (no tool calls)
3. Max iterations reached
4. Token budget exceeded
5. Timeout exceeded
6. No-progress (3x identical actions)
7. Error limit (3x consecutive errors)

### System Chunk Types

| Type | When |
|------|------|
| `limit_warning` | 70% iterations or 80% tokens |
| `limit_reached` | Hard limit hit |
| `no_progress` | Same action 3x |
| `error_limit` | 3 consecutive errors |

## Testing Checklist

- [ ] AgentState is immutable (frozen dataclass)
- [ ] Settings persist across sessions
- [ ] System messages appear in chat
- [ ] Warnings at 70%/80% thresholds
- [ ] No-progress detection triggers on 3x same action
- [ ] Partial content saved on termination
- [ ] Frontend handles all system chunk types

## Common Pitfalls

1. **Token counting is approximate** - Don't expect exact enforcement
2. **No-progress compares stringified args** - JSON key order matters
3. **System messages are ephemeral** - Not persisted to conversation tree
4. **Subagent work not saved** - Only main chain content preserved on termination
