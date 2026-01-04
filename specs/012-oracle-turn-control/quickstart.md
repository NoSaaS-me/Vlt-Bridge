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

## Testing Scenarios

### 1. Max Iterations Limit

Test that the agent stops at the configured max_iterations limit.

**Setup:**
1. Go to Settings > Agent Configuration
2. Set `max_iterations` to 3 (minimum for testing)
3. Set `soft_warning_percent` to 70

**Test:**
1. Ask a complex question that requires multiple tool calls: "Find all Python files in the vault and summarize their contents"
2. Observe warning system message at ~2 turns (70% of 3)
3. Agent should terminate after 3 turns with "limit_reached" system message
4. Check backend logs for: `[TERMINATION:max_iterations]`

### 2. Token Budget Warning

Test that token budget warnings appear correctly.

**Setup:**
1. Go to Settings > Agent Configuration
2. Set `token_budget` to 5000 (low value for testing)
3. Set `token_warning_percent` to 80

**Test:**
1. Ask a question that generates verbose output: "List all vault notes with their full content"
2. Observe warning system message when approaching 4000 tokens (80%)
3. Check backend logs for token usage estimates
4. Note: Token counting is approximate (4 chars per token estimate)

### 3. Timeout

Test that queries timeout correctly.

**Setup:**
1. Go to Settings > Agent Configuration
2. Set `timeout_seconds` to 30 (short for testing)

**Test:**
1. Ask a question that might take time: "Search the entire codebase for security issues"
2. If query runs longer than 30s, observe timeout system message
3. Check backend logs for: `[TERMINATION:timeout]`

### 4. System Messages in Chat

Verify system messages appear correctly in the UI.

**Test:**
1. Configure low limits as above
2. Trigger any limit (iterations, tokens, timeout)
3. Verify system message appears in chat panel with correct styling
4. Message should include the limit type and current value

### 5. No-Progress Detection

Test detection of repeated identical actions (harder to trigger manually).

**Programmatic Test:**
```python
# Run via pytest
pytest tests/unit/test_decision_tree.py::test_no_progress_detection -v
```

The no-progress detector triggers when:
- Same tool with same arguments is called 3 consecutive times
- Compares stringified JSON of arguments (key order matters)

### 6. Settings Persistence

Verify AgentConfig settings persist across sessions.

**Test:**
1. Change any AgentConfig setting in Settings page
2. Refresh the browser
3. Go back to Settings page
4. Verify setting persisted (check SQLite user_settings table)

### Log Inspection

All termination events are logged with a consistent format for debugging:
- `[TERMINATION:max_iterations]` - Hit iteration limit
- `[TERMINATION:token_budget]` - Exceeded token budget
- `[TERMINATION:timeout]` - Query timed out
- `[TERMINATION:no_progress]` - 3 identical consecutive actions
- `[TERMINATION:error_limit]` - 3 consecutive errors
- `[TERMINATION:cancelled]` - User cancelled query

View logs with:
```bash
cd backend
tail -f /tmp/vlt-oracle.log | grep TERMINATION
```

## Common Pitfalls

1. **Token counting is approximate** - Don't expect exact enforcement
2. **No-progress compares stringified args** - JSON key order matters
3. **System messages are ephemeral** - Not persisted to conversation tree
4. **Subagent work not saved** - Only main chain content preserved on termination
