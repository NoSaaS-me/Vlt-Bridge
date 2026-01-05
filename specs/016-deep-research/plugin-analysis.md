# Deep Research Plugin Analysis: Feasibility and Architecture

**Date**: 2026-01-04
**Analyst**: Claude Opus 4.5
**Scope**: Determine if Deep Research can be implemented as an Oracle Plugin System (015) plugin, or must remain core functionality

---

## Executive Summary

**Verdict: Deep Research CANNOT be a pure plugin. It must be core functionality with plugin enhancements.**

Deep Research fundamentally requires **nested agent loops** - multiple researchers iterating with tools 5-10 times each, coordinated by a supervisor that dynamically decides research strategy. The Oracle Plugin System (015) is designed for **reactive event handlers** that fire notifications based on conditions. These are architecturally incompatible goals.

**Recommended Architecture**: Hybrid approach where:
1. **Core**: Research orchestration remains a core tool (`deep_research`) with proper agentic iteration
2. **Plugin Enhancement**: Rules detect research-intent queries, inject research prompts, track cross-session research state

---

## 1. Implementation Comparison (Code-Level)

### 1.1 LangChain's Researcher Loop

LangChain's researchers **iterate** with tools. Here's the actual flow from `deep_researcher.py`:

```python
# deep_researcher.py lines 470-543
async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    """Executes tools and decides whether to continue looping."""

    # Check if we've exceeded tool call budget
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls

    # Check for explicit "done" signal
    research_complete_called = any(
        tc.get("name") == "ResearchComplete"
        for tc in state.get("researcher_messages", [])
    )

    if exceeded_iterations or research_complete_called:
        # EXIT: Go to compression phase
        return Command(goto="compress_research", ...)

    # Execute all pending tool calls in parallel
    observations = await asyncio.gather(*tool_execution_tasks)

    # LOOP BACK: Continue the ReAct cycle
    return Command(
        goto="researcher",  # Back to LLM decision
        update={"researcher_messages": tool_outputs}
    )
```

**Key insight**: This is a **state machine** that loops until an exit condition. The LLM decides on each iteration what to do next.

### 1.2 Our Current Single-Pass Implementation

Our researchers make **exactly one batch call**:

```python
# backend/src/services/research/behaviors.py lines 243-327
async def run_single(self, state: ResearchState, researcher: ResearcherState):
    """Run research for a single subtopic."""

    # Generate queries upfront (NO LLM LOOP)
    queries = await self._generate_search_queries(state, researcher)

    # Execute searches - ONE TIME
    search_responses = await self.tavily.search_parallel(
        queries=queries,
        max_results_per_query=3,
        deduplicate=True,
    )

    # Convert to sources and extract findings - ONCE
    for response in search_responses:
        for result in response.results:
            source = ResearchSource(...)
            researcher.sources.append(source)

    # DONE - no iteration, no LLM decision loop
    researcher.completed = True
    return researcher
```

**Key insight**: This is **batch processing**, not agent behavior. The researcher cannot:
- Decide to search more based on what it found
- Use `think_tool` to reflect on progress
- Call `ResearchComplete` when satisfied
- Iterate until it has sufficient sources

### 1.3 State Tracking Comparison

| Aspect | LangChain | Ours | Gap |
|--------|-----------|------|-----|
| `tool_call_iterations` | Tracked, checked each loop | Field exists but not enforced | Critical |
| `researcher_messages` | Full LLM conversation history | Not implemented | Critical |
| `raw_notes` | Accumulated across iterations | Not implemented | High |
| `compressed_research` | Per-researcher compression | Not implemented | High |
| `supervisor_messages` | Supervisor decision history | Not implemented (no supervisor) | Critical |
| `research_iterations` | Supervisor loop counter | Not implemented | Critical |

---

## 2. Plugin System Architecture Analysis

### 2.1 What the Plugin System CAN Do

From `specs/015-oracle-plugin-system/spec.md` and `backend/src/services/plugins/engine.py`:

**Hook Points Available**:
```python
# backend/src/services/plugins/engine.py lines 48-57
EVENT_TO_HOOK: Dict[str, HookPoint] = {
    EventType.QUERY_START: HookPoint.ON_QUERY_START,
    EventType.AGENT_TURN_START: HookPoint.ON_TURN_START,
    EventType.AGENT_TURN_END: HookPoint.ON_TURN_END,
    EventType.TOOL_CALL_PENDING: HookPoint.ON_TOOL_CALL,
    EventType.TOOL_CALL_SUCCESS: HookPoint.ON_TOOL_COMPLETE,
    EventType.TOOL_CALL_FAILURE: HookPoint.ON_TOOL_FAILURE,
    EventType.SESSION_END: HookPoint.ON_SESSION_END,
}
```

**Context API Available**:
```python
# backend/src/services/plugins/context.py
@dataclass
class RuleContext:
    turn: TurnState        # number, token_usage, context_usage, iteration_count
    history: HistoryState  # messages, tools, failures
    user: UserState        # id, settings
    project: ProjectState  # id, settings
    state: PluginState     # get(), set() for persistence
    event: EventData       # type, source, severity, payload, timestamp
    result: Optional[Dict] # Tool result (for ON_TOOL_COMPLETE)
```

**Actions Available**:
```python
# From spec.md FR-5
class ActionType(Enum):
    NOTIFY_SELF = "notify_self"  # Inject notification into agent context
    LOG = "log"                   # Write to system log
    SET_STATE = "set_state"       # Store plugin-scoped persistent state
    EMIT_EVENT = "emit_event"     # Emit an ANS event
```

### 2.2 What the Plugin System CANNOT Do

1. **Cannot spawn new agent loops**: Plugins fire on events and return. They cannot take over and run their own multi-turn LLM conversations.

2. **Cannot call tools directly**: Plugins can notify the agent to use tools, but cannot execute tools themselves.

3. **Cannot create parallel execution**: Plugins are synchronous event handlers. They cannot spawn 5 parallel researchers.

4. **Cannot manage complex state machines**: The plugin system evaluates rules on each event. It doesn't maintain its own execution graph.

5. **Cannot intercept and replace tool responses**: Plugins fire after tools complete, not instead of them.

### 2.3 The Architectural Mismatch

**Plugin System Design Pattern**:
```
Event Occurs -> Rule Evaluates -> Action Fires -> Continue Normal Flow
    ^                                                      |
    +------------------------------------------------------+
                    (Oracle's main loop continues)
```

**Deep Research Required Pattern**:
```
User Query
    |
    v
[Research Mode Activated]
    |
    +---[Supervisor Loop]--+
    |       |              |
    |       v              |
    |   [Spawn Researchers]|
    |       |              |
    |   +---+---+---+      |
    |   |   |   |   |      |
    |   R1  R2  R3  R4     |  (Parallel)
    |   |   |   |   |      |
    |   +---+---+---+      |
    |       |              |
    |       v              |
    |   [Each Researcher   |
    |    Loops 5-10x]      |
    |       |              |
    +-------+--------------+
            |
            v
    [Compression + Report]
            |
            v
    [Return to Oracle]
```

The research workflow is an **alternative execution mode**, not an event-driven enhancement.

---

## 3. Plugin Architecture Design (What IS Possible)

While Deep Research cannot BE a plugin, plugins CAN enhance the research experience.

### 3.1 Research Intent Detection Rule

**File**: `backend/src/services/plugins/rules/research_intent.toml`

```toml
[rule]
id = "research-intent"
name = "Research Intent Detection"
description = "Suggests deep research for queries that appear to need comprehensive investigation"
enabled = true
core = false

[trigger]
hook = "on_query_start"

[condition]
# Expression evaluated against context.event.payload.query
expression = """
any([
    'research' in lower(event.payload.query),
    'investigate' in lower(event.payload.query),
    'comprehensive' in lower(event.payload.query),
    'analyze' in lower(event.payload.query),
    'best practices' in lower(event.payload.query),
    'state of the art' in lower(event.payload.query),
])
"""

[action]
type = "notify_self"
message = """
This query appears to need comprehensive research. Consider using the `deep_research` tool:

Example:
deep_research(
    query="{event.payload.query}",
    depth="standard",
    save_to_vault=true
)

This will spawn parallel researchers to gather information from multiple sources and synthesize a comprehensive report.
"""
priority = "normal"
category = "suggestion"
deliver_at = "immediate"
```

### 3.2 Research Progress Tracking Rule

**File**: `backend/src/services/plugins/rules/research_progress.toml`

```toml
[rule]
id = "research-progress"
name = "Research Progress Tracker"
description = "Tracks ongoing research and suggests continuation"
enabled = true
core = false

[trigger]
hook = "on_tool_complete"
tool = "deep_research"

[condition]
# Check if research completed successfully
expression = "result.success == true"

[action]
type = "set_state"
key = "last_research"
value = """
{
    "research_id": result.research_id,
    "query": result.query,
    "sources_count": result.sources_count,
    "timestamp": now()
}
"""
```

### 3.3 Research Continuation Script (Lua)

**File**: `backend/src/services/plugins/scripts/research_continuation.lua`

```lua
-- Research continuation check
-- Runs on session_resumed to suggest continuing incomplete research

local state = context.state
local last_research = state:get("last_research")

if not last_research then
    return nil  -- No prior research, don't match
end

-- Check if research is recent (within 24 hours)
local age_hours = (now() - last_research.timestamp) / 3600
if age_hours > 24 then
    return nil  -- Too old, don't suggest continuation
end

-- Return action to suggest continuation
return {
    type = "notify_self",
    message = string.format(
        "You have a recent research session on '%s' (%d sources found). " ..
        "Use `vault_read('%s')` to review findings or continue research.",
        last_research.query,
        last_research.sources_count,
        last_research.vault_path
    ),
    priority = "normal",
    category = "reminder",
    deliver_at = "turn_start"
}
```

### 3.4 What These Plugins Enable

1. **Proactive Suggestions**: Detect when user queries would benefit from research
2. **State Tracking**: Remember research across sessions
3. **Context Injection**: Remind agent about prior research
4. **UX Enhancement**: Surface research capabilities at appropriate moments

**What They Don't Do**: Actually execute the research workflow. That remains the job of the core `deep_research` tool and `ResearchOrchestrator`.

---

## 4. Tool Exposure Requirements

### 4.1 Existing Tool (`deep_research`)

The `deep_research` tool already exists in `backend/prompts/tools.json`:

```json
{
  "type": "function",
  "function": {
    "name": "deep_research",
    "description": "Trigger comprehensive web research on a topic...",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string"},
        "depth": {"type": "string", "enum": ["quick", "standard", "thorough"]},
        "save_to_vault": {"type": "boolean"},
        "output_folder": {"type": "string"}
      },
      "required": ["query"]
    }
  },
  "agent_scope": ["oracle"]
}
```

### 4.2 New Tools Needed for Agentic Research

To implement true iterative research, we need internal tools for researchers:

| Tool | Purpose | Scope |
|------|---------|-------|
| `research_search` | Execute web search (wraps Tavily/OpenRouter) | Internal (researcher only) |
| `think` | Strategic reflection on progress | Internal (researcher only) |
| `research_complete` | Signal researcher is done | Internal (researcher only) |
| `conduct_research` | Supervisor delegates to researcher | Internal (supervisor only) |

**Note**: These are NOT Oracle tools. They're internal tools for the research workflow. The Oracle only calls `deep_research`, which orchestrates everything internally.

### 4.3 Plugin-Accessible Tools

Plugins should NOT have direct tool access (security boundary). However, they can:

1. **Read tool results** via `context.result` in `on_tool_complete` hooks
2. **Influence tool use** via `notify_self` suggestions to the agent
3. **Track tool history** via `context.history.tools`

---

## 5. Implementation Recommendation

### 5.1 Recommended Architecture: Hybrid Core + Plugin

```
                         +---------------------------+
                         |     Oracle Plugin System  |
                         |---------------------------|
                         | - research_intent rule    |
                         | - research_progress rule  |
                         | - continuation script     |
                         +------------+--------------+
                                      |
                                      | Suggestions/State
                                      v
+------------+        +---------------+---------------+
|   User     | -----> |         Oracle Agent         |
+------------+        |-------------------------------|
                      | Decides to call deep_research |
                      +---------------+---------------+
                                      |
                                      | Tool Call
                                      v
                      +---------------+---------------+
                      |    ResearchOrchestrator      |
                      |      (CORE - NOT PLUGIN)     |
                      |-------------------------------|
                      | - GenerateBriefBehavior      |
                      | - SupervisorBehavior [NEW]   |
                      | - ResearcherBehavior [FIX]   |
                      | - CompressFindingsBehavior   |
                      | - GenerateReportBehavior     |
                      +---------------+---------------+
                                      |
                                      | Results
                                      v
                      +-------------------------------+
                      |   Oracle receives report     |
                      |   Continues conversation     |
                      +-------------------------------+
```

### 5.2 What Needs to Change in Core

**Priority 0 - Critical (Must Fix)**:

1. **Add Researcher Iteration Loop** (`behaviors.py`)
   - Replace `run_single()` with ReAct loop
   - Track `tool_call_iterations`
   - Implement exit conditions (max calls, research complete, no new info)
   - Estimated effort: 3-4 days

2. **Add Supervisor Behavior** (NEW: `supervisor.py`)
   - LLM-driven decision making with `ConductResearch`, `think_tool`, `ResearchComplete`
   - Can iterate and spawn additional researchers dynamically
   - Estimated effort: 2-3 days

3. **Implement think_tool** (`behaviors.py`)
   - Strategic reflection tool for researchers
   - Records reflection to message history
   - Estimated effort: 0.5 days

**Priority 1 - High**:

4. **Add Token Overflow Handling** (`behaviors.py`, `llm_service.py`)
   - Detect provider-specific token limit errors
   - Progressive truncation (10% per retry)
   - Estimated effort: 1 day

5. **Message-Based State** (`models/research.py`)
   - Add `researcher_messages`, `supervisor_messages`
   - Track full conversation history for compression
   - Estimated effort: 1 day

**Priority 2 - Plugin Integration**:

6. **Add Research Intent Rule** (plugin)
   - Detect research-appropriate queries
   - Suggest deep_research tool
   - Estimated effort: 0.5 days

7. **Add Research State Tracking** (plugin)
   - Track completed research in plugin state
   - Suggest continuation on session resume
   - Estimated effort: 0.5 days

### 5.3 What Should NOT Change

1. **Don't try to make research a plugin** - It's fundamentally an alternative workflow
2. **Don't extend plugin system for agent loops** - Keep plugins simple and reactive
3. **Don't expose internal research tools to Oracle** - Maintain clean interface

---

## 6. Alternative Considered: Skill Plugin Type

The spec mentions "Skill Plugins" as a stretch goal:

> ### Skill Plugins
> Higher-level plugins that define callable capabilities with structured inputs/outputs:
> - Deep Researcher: Multi-step web research with source synthesis

This would require extending the Plugin System to support:
- Plugins that can "take over" the agent turn
- Plugins with their own tool access
- Plugins with their own LLM calls

**Analysis**: This is architecturally sound but represents a significant increase in Plugin System complexity. Benefits:
- Clean separation of concerns
- Research is truly pluggable/replaceable
- Other "skills" could use same pattern

Costs:
- Substantial new infrastructure
- Security concerns with plugin tool access
- More complex plugin lifecycle management

**Recommendation**: Defer Skill Plugin type to a future iteration. Current approach (core + enhancement plugins) achieves 90% of benefit with 20% of complexity.

---

## 7. Edge Cases and Error Handling

### 7.1 What Happens When Research Fails?

**Current behavior**: Single failure point, state set to `FAILED`

**Needed behavior**:
- Retry individual researchers (1-2 attempts)
- Continue with partial results if some researchers succeed
- Surface partial results even on overall failure
- Plugin can track failure state for cross-session recovery

### 7.2 Token Limits During Research

**Current behavior**: Single attempt, crash on overflow

**Needed behavior**:
```python
# From LangChain reference (deep_researcher.py lines 662-697)
while current_retry <= max_retries:
    try:
        final_report = await configurable_model.ainvoke(...)
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.final_report_model):
            current_retry += 1
            # Reduce by 10% each retry
            findings_token_limit = int(findings_token_limit * 0.9)
            findings = findings[:findings_token_limit]
            continue
```

### 7.3 Plugin Interaction with Research

**Question**: How do plugins see research in progress?

**Answer**: Plugins see research as a single tool call:
- `on_tool_call`: Sees `deep_research` starting
- (research executes internally - plugins don't see intermediate steps)
- `on_tool_complete`: Sees `deep_research` result with full report

This is intentional - research is atomic from Oracle's perspective.

---

## 8. Conclusion

### Summary

| Component | Implementation Location | Rationale |
|-----------|------------------------|-----------|
| Research Orchestration | Core (`services/research/`) | Requires nested agent loops |
| Researcher Iteration | Core (`behaviors.py`) | Needs tool execution and LLM calls |
| Supervisor Logic | Core (`supervisor.py` NEW) | Needs LLM-driven delegation |
| Token Management | Core (`llm_service.py`) | Deep integration with LLM calls |
| Intent Detection | Plugin (rule) | Simple pattern matching |
| State Tracking | Plugin (rule + lua) | Cross-session persistence |
| Continuation Suggestions | Plugin (lua script) | UX enhancement |

### Next Steps

1. **Immediate**: Fix researcher iteration in `behaviors.py` (P0)
2. **Week 1**: Add supervisor behavior and think_tool (P0)
3. **Week 2**: Add token overflow handling (P1)
4. **Week 3**: Add plugin enhancement rules (P2)

### Confidence Assessment

**Confidence: 9/10**

I have high confidence in this analysis because:
1. Read complete LangChain reference implementation
2. Read complete Vlt-Bridge research implementation
3. Read complete Plugin System spec and implementation
4. Code-level comparison of execution patterns
5. Architectural analysis of hook points and capabilities

The 1/10 uncertainty is in:
- Exact effort estimates (depends on implementation details)
- Whether "Skill Plugin" type should be prioritized differently
- Specific edge cases in token overflow handling per provider

---

*Document generated 2026-01-04 by Claude Opus 4.5*
