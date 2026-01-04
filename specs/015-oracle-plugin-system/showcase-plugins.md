# Plugin Showcase Specification

**Date**: 2026-01-04
**Branch**: `015-oracle-plugin-system`
**Purpose**: API surface coverage testing and documentation through practical plugins

## Overview

The Plugin Showcase is a suite of 10 plugins designed to:
1. **Test every documented API surface** of the Oracle Plugin System
2. **Provide practical examples** for plugin developers
3. **Validate the implementation** before release
4. **Self-document** the API through working code

Each plugin addresses a real user pain point while exercising specific API features.

---

## Plugin Inventory

| # | Plugin ID | Type | Pain Point | Primary API Coverage |
|---|-----------|------|------------|---------------------|
| 1 | query-logger | TOML | No query visibility | on_query_start, log action |
| 2 | session-stats-logger | TOML | No session metrics | on_session_end, log action, state |
| 3 | tool-call-counter | TOML+State | Tool overuse blindness | on_tool_call, set_state, history.tool_count |
| 4 | context-monitor | TOML+Func | Context overflow surprise | on_turn_end, context_above_threshold() |
| 5 | conversation-length-warning | TOML+Func | Long conversation drift | on_turn_start, message_count_above() |
| 6 | research-tracker | Lua Script | Research without synthesis | on_tool_complete, Lua, state persistence |
| 7 | failure-escalation | TOML+Event | No failure monitoring | on_tool_failure, emit_event |
| 8 | turn-checkpoint | TOML+Multi | No progress checkpoints | on_turn_end, set_state + emit_event |
| 9 | vault-research-plugin | Multi-Rule | Need bundled workflows | manifest.toml, plugin settings |
| 10 | github-rate-monitor | TOML+Tool | Rate limit surprises | on_tool_complete, tool_completed() |

---

## Plugin Specifications

### 1. query-logger

**Pain Point**: Developers have no visibility into what queries the agent receives for debugging.

**User Story**: "As a developer, I want to log every incoming query so I can debug agent behavior."

**API Coverage**:
- Hook: `on_query_start`
- Action: `log`
- Context: `context.turn.number`, `context.user.id`

**TOML Definition**:
```toml
[rule]
id = "query-logger"
name = "Query Logger"
description = "Log every incoming query for debugging"
trigger = "on_query_start"
priority = 50
core = false

[condition]
expression = "true"

[action]
type = "log"
message = "Query #{{ context.turn.number }} received from user {{ context.user.id }}"
level = "info"
```

**Test Criteria**:
- [ ] Rule fires on every new query
- [ ] Log message contains correct turn number
- [ ] Log message contains correct user ID
- [ ] Log appears in system logs at INFO level

---

### 2. session-stats-logger

**Pain Point**: No metrics captured when sessions end for operational analysis.

**User Story**: "As an operator, I want session statistics logged when sessions close for analytics."

**API Coverage**:
- Hook: `on_session_end`
- Action: `log`
- Context: `context.turn.number`, `context.history.tool_count`, `context.state`

**TOML Definition**:
```toml
[rule]
id = "session-stats-logger"
name = "Session Stats Logger"
description = "Log session statistics when session closes"
trigger = "on_session_end"
priority = 10
core = false

[condition]
expression = "true"

[action]
type = "log"
message = "Session ended: {{ context.turn.number }} turns, {{ context.history.tool_count }} tool calls"
level = "info"
```

**Test Criteria**:
- [ ] Rule fires when session ends
- [ ] Log contains accurate turn count
- [ ] Log contains accurate tool count
- [ ] Works even if user disconnected

---

### 3. tool-call-counter

**Pain Point**: Agent may use excessive tools without awareness, wasting resources.

**User Story**: "As a user, I want warning when the agent uses too many tools in one session."

**API Coverage**:
- Hook: `on_tool_call`
- Action: `set_state`, `notify_self`
- Context: `context.history.tool_count`
- Expression: Numeric comparison

**TOML Definition**:
```toml
[rule]
id = "tool-call-counter"
name = "Tool Call Counter"
description = "Warn when tool usage exceeds threshold"
trigger = "on_tool_call"
priority = 70
core = false

[condition]
expression = "context.history.tool_count > 20"

[action]
type = "notify_self"
message = "High tool usage: {{ context.history.tool_count }} calls. Consider whether all are necessary."
category = "warning"
priority = "normal"
```

**Test Criteria**:
- [ ] Rule fires on 21st tool call
- [ ] Rule does not fire on calls 1-20
- [ ] Notification contains accurate count
- [ ] Works across multiple turn boundaries

---

### 4. context-monitor

**Pain Point**: Context window fills silently until agent starts forgetting.

**User Story**: "As a user, I want warning before context window overflows."

**API Coverage**:
- Hook: `on_turn_end`
- Action: `notify_self` (priority: high)
- Context: `context.turn.context_usage`
- Function: `context_above_threshold()`

**TOML Definition**:
```toml
[rule]
id = "context-monitor"
name = "Context Monitor"
description = "Warn when context window usage is high"
trigger = "on_turn_end"
priority = 90
core = false

[condition]
expression = "context_above_threshold(0.85)"

[action]
type = "notify_self"
message = "Context window at {{ (context.turn.context_usage * 100) | int }}%. Consider summarizing or wrapping up."
category = "warning"
priority = "high"
deliver_at = "turn_start"
```

**Test Criteria**:
- [ ] Rule fires when context > 85%
- [ ] Rule does not fire when context < 85%
- [ ] `context_above_threshold()` function works correctly
- [ ] Notification delivered at next turn start

---

### 5. conversation-length-warning

**Pain Point**: Very long conversations lose coherence and context quality.

**User Story**: "As a user, I want warning when conversation gets excessively long."

**API Coverage**:
- Hook: `on_turn_start`
- Action: `notify_self`
- Context: `context.history.messages`
- Function: `message_count_above()`

**TOML Definition**:
```toml
[rule]
id = "conversation-length-warning"
name = "Conversation Length Warning"
description = "Warn when conversation exceeds message threshold"
trigger = "on_turn_start"
priority = 60
core = false

[condition]
expression = "message_count_above(30)"

[action]
type = "notify_self"
message = "Long conversation detected. Consider starting fresh for better coherence."
category = "reminder"
priority = "normal"
```

**Test Criteria**:
- [ ] Rule fires when messages > 30
- [ ] Rule does not fire when messages <= 30
- [ ] `message_count_above()` function works correctly
- [ ] Count includes both user and assistant messages

---

### 6. research-tracker

**Pain Point**: Agent searches endlessly without stopping to synthesize findings.

**User Story**: "As a user, I want the agent reminded to synthesize after multiple searches."

**API Coverage**:
- Hook: `on_tool_complete`
- Action: `notify_self`, `set_state`
- Context: `context.history.tools`, `context.state`
- Advanced: Lua scripting

**TOML Definition**:
```toml
[rule]
id = "research-tracker"
name = "Research Tracker"
description = "Remind agent to synthesize after multiple searches"
trigger = "on_tool_complete"
priority = 75
core = false

[condition]
script = "scripts/research_tracker.lua"

[action]
type = "notify_self"
message = "{{ state.search_count }} searches completed. Consider synthesizing findings before continuing."
category = "reminder"
priority = "normal"
```

**Lua Script** (`scripts/research_tracker.lua`):
```lua
-- Track search tool usage and fire when threshold reached
local SEARCH_TOOLS = {
    vault_search = true,
    web_search = true,
    github_search = true,
    thread_seek = true
}

-- Get current count from state
local search_count = context.state.get("search_count") or 0

-- Check if last completed tool was a search
local tools = context.history.tools
if #tools > 0 then
    local last_tool = tools[#tools]
    if SEARCH_TOOLS[last_tool.name] then
        search_count = search_count + 1
        context.state.set("search_count", search_count)
    end
end

-- Fire every 5 searches
if search_count > 0 and search_count % 5 == 0 then
    -- Make count available to action template
    state.search_count = search_count
    return true
end

return false
```

**Test Criteria**:
- [ ] Lua script executes without error
- [ ] State persists across tool calls
- [ ] Fires on 5th, 10th, 15th search
- [ ] Only counts search tools, not other tools
- [ ] Script timeout (5s) enforced

---

### 7. failure-escalation

**Pain Point**: Tool failures aren't visible to external monitoring systems.

**User Story**: "As an operator, I want tool failures to emit events for monitoring integration."

**API Coverage**:
- Hook: `on_tool_failure`
- Action: `emit_event`
- Context: `context.event`, `context.history.failures`

**TOML Definition**:
```toml
[rule]
id = "failure-escalation"
name = "Failure Escalation"
description = "Emit monitoring events on tool failures"
trigger = "on_tool_failure"
priority = 95
core = false

[condition]
expression = "true"

[action]
type = "emit_event"
event_type = "monitoring.tool.failure"
payload = { tool_name = "{{ context.event.payload.tool_name }}", error = "{{ context.event.payload.error_message }}", failure_count = "{{ context.history.failures.get(context.event.payload.tool_name, 0) }}" }
```

**Test Criteria**:
- [ ] Rule fires on every tool failure
- [ ] Event emitted to ANS bus
- [ ] Payload contains tool name
- [ ] Payload contains error message
- [ ] Failure count accurate from history

---

### 8. turn-checkpoint

**Pain Point**: Long agent runs have no progress checkpoints.

**User Story**: "As a developer, I want automatic checkpoints every N turns for debugging."

**API Coverage**:
- Hook: `on_turn_end`
- Action: `set_state` + `emit_event`
- Context: `context.turn.number`, `context.state`
- Expression: Arithmetic (modulo)

**TOML Definition**:
```toml
[rule]
id = "turn-checkpoint"
name = "Turn Checkpoint"
description = "Emit checkpoint events every 5 turns"
trigger = "on_turn_end"
priority = 40
core = false

[condition]
expression = "context.turn.number % 5 == 0"

[action]
type = "emit_event"
event_type = "agent.checkpoint"
payload = { turn = "{{ context.turn.number }}", timestamp = "{{ now() }}" }
```

**Test Criteria**:
- [ ] Fires on turns 5, 10, 15, etc.
- [ ] Does not fire on other turns
- [ ] Modulo expression works correctly
- [ ] Event payload contains turn number

---

### 9. vault-research-plugin (Multi-Rule)

**Pain Point**: Need to bundle related rules as a distributable package.

**User Story**: "As a plugin developer, I want to package multiple rules with shared settings."

**API Coverage**:
- Plugin manifest format
- Plugin settings (configurable thresholds)
- Multi-rule discovery
- Capability requirements

**Directory Structure**:
```
plugins/vault-research-plugin/
├── manifest.toml
├── rules/
│   ├── vault-heavy-use.toml
│   ├── vault-save-reminder.toml
│   └── vault-cleanup-hint.toml
└── scripts/
    └── vault_usage.lua
```

**Manifest** (`manifest.toml`):
```toml
[plugin]
id = "vault-research-plugin"
name = "Vault Research Plugin"
version = "1.0.0"
description = "Enhanced vault usage tracking and workflow hints"

[capabilities]
requires = ["vault_search", "vault_read", "vault_write"]

[rules]
include = ["rules/*.toml"]

[settings.heavy_use_threshold]
type = "integer"
default = 10
min = 5
max = 50
description = "Number of vault operations before heavy use warning"

[settings.auto_save_reminder]
type = "boolean"
default = true
description = "Remind agent to save important findings to vault"
```

**Test Criteria**:
- [ ] Manifest parses correctly
- [ ] All rules in rules/*.toml discovered
- [ ] Settings accessible in rule conditions
- [ ] Capability check validates required tools
- [ ] Plugin appears in Settings UI

---

### 10. github-rate-monitor

**Pain Point**: GitHub rate limits cause unexpected failures.

**User Story**: "As a user, I want awareness of GitHub API usage to avoid rate limits."

**API Coverage**:
- Hook: `on_tool_complete`
- Action: `notify_self`
- Context: `context.result`
- Function: `tool_completed()`

**TOML Definition**:
```toml
[rule]
id = "github-rate-monitor"
name = "GitHub Rate Monitor"
description = "Track GitHub API usage for rate limit awareness"
trigger = "on_tool_complete"
priority = 65
core = false

[condition]
expression = "tool_completed('github_read') or tool_completed('github_search')"

[action]
type = "notify_self"
message = "GitHub API used. Rate limit status: {{ context.result.rate_limit_remaining | default('unknown') }} remaining."
category = "context"
priority = "low"
```

**Test Criteria**:
- [ ] Fires on github_read completion
- [ ] Fires on github_search completion
- [ ] Does not fire on other tools
- [ ] `tool_completed()` function works correctly
- [ ] Result data accessible if present

---

## API Coverage Matrix

| API Element | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 |
|-------------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|
| **HOOKS** |
| on_query_start | ✅ | | | | | | | | | |
| on_turn_start | | | | | ✅ | | | | | |
| on_turn_end | | | | ✅ | | | | ✅ | ✅ | |
| on_tool_call | | | ✅ | | | | | | | |
| on_tool_complete | | | | | | ✅ | | | | ✅ |
| on_tool_failure | | | | | | | ✅ | | | |
| on_session_end | | ✅ | | | | | | | | |
| **CONTEXT** |
| turn.number | ✅ | ✅ | | | | | | ✅ | | |
| turn.context_usage | | | | ✅ | | | | | | |
| history.messages | | | | | ✅ | | | | | |
| history.tools | | | | | | ✅ | | | | |
| history.tool_count | | ✅ | ✅ | | | | | | | |
| history.failures | | | | | | | ✅ | | | |
| user.id | ✅ | | | | | | | | | |
| state (get/set) | | | ✅ | | | ✅ | | ✅ | | |
| event | | | | | | | ✅ | | | |
| result | | | | | | | | | | ✅ |
| **ACTIONS** |
| notify_self | | | ✅ | ✅ | ✅ | ✅ | | | ✅ | ✅ |
| log | ✅ | ✅ | | | | | | | | |
| set_state | | | ✅ | | | ✅ | | | | |
| emit_event | | | | | | | ✅ | ✅ | | |
| **FUNCTIONS** |
| tool_completed() | | | | | | | | | | ✅ |
| context_above_threshold() | | | | ✅ | | | | | | |
| message_count_above() | | | | | ✅ | | | | | |
| **ADVANCED** |
| Lua scripting | | | | | | ✅ | | | ✅ | |
| Plugin manifest | | | | | | | | | ✅ | |
| Plugin settings | | | | | | | | | ✅ | |
| Arithmetic expressions | | | | | | | | ✅ | | |
| Boolean composition | | | | | | | | | | ✅ |

---

## Implementation Order

**Phase 1: Simple TOML Rules** (validates core engine)
1. query-logger
2. session-stats-logger
3. tool-call-counter
4. conversation-length-warning

**Phase 2: Functions & Context** (validates expression evaluator)
5. context-monitor
6. github-rate-monitor
7. turn-checkpoint

**Phase 3: Advanced Features** (validates Lua + events)
8. failure-escalation
9. research-tracker

**Phase 4: Plugin System** (validates manifest/discovery)
10. vault-research-plugin

---

## Success Criteria

1. **All 10 plugins load without errors**
2. **Each plugin fires correctly** per its trigger/condition
3. **All API surfaces exercised** per coverage matrix
4. **No gaps in coverage** - every documented feature tested
5. **Plugins serve as documentation** - developers can copy/adapt

---

## Dependencies

- Rule engine implementation (T020-T045 from tasks.md)
- Expression evaluator (simpleeval)
- Lua sandbox (lupa)
- ANS EventBus integration
- Plugin loader and manifest parser
