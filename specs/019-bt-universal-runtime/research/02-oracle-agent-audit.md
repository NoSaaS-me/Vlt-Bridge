# Oracle Agent Deep Audit for BT Migration

**File**: `backend/src/services/oracle_agent.py`
**Lines**: 2,765
**Purpose**: AI agent with tool calling via OpenRouter, with context persistence, budget tracking, ANS integration, and plugin rules.

---

## 1. Main Execution Loop: `query()`

**Location**: Lines 900-1334

### Step-by-Step Flow

1. **Reset State** (lines 924-932)
   - Reset cancellation via `reset_cancellation()`
   - Reset loop detection via `_reset_loop_detection()`
   - Reset budget tracking via `_reset_budget_tracking(max_tokens)`
   - Reset deferred queue via `reset_deferred_queue()`
   - Clear collected sources, tool calls, system messages

2. **Emit QUERY_START Event** (lines 939-950)
   - Plugin system hook for query lifecycle

3. **Load Tree Context** (lines 957-1001)
   - If `context_id` provided, load from existing tree node
   - Otherwise get/create active tree for user/project
   - Sets `_current_tree_root_id` and `_current_node_id`

4. **Load Legacy Context** (lines 1002-1011)
   - Backwards compatibility with `OracleContextService`

5. **Load Cross-Session Notifications** (lines 1013-1042)
   - ANS persistence feature for notifications across sessions

6. **Build Initial Messages** (lines 1043-1183)
   - Load vault files and threads for system prompt
   - Render system prompt from template `oracle/system.md`
   - Add tree history (walk root to current node)
   - Add legacy context history (fallback)
   - Inject cross-session notifications as system messages
   - Add current user question

7. **Get Tool Definitions** (line 1185)
   - `tool_executor.get_tool_schemas(agent="oracle")`

8. **Initialize Context Window Tracking** (lines 1187-1204)
   - Set `_max_context_tokens` from model lookup
   - Estimate initial `_context_tokens`
   - Yield initial `context_update` chunk

9. **Agent Loop** (lines 1222-1308)
   ```python
   for turn in range(self.MAX_TURNS):  # MAX_TURNS = 30
       # Check cancellation
       # Check iteration budget
       # Drain turn_start notifications
       # Execute _agent_turn()
       # Track accumulated content
       # Check for done/error chunks
   ```

10. **Max Turns Handling** (lines 1269-1307)
    - Emit iteration_exceeded event
    - Drain immediate notifications
    - Save partial exchange
    - Yield done chunk

11. **Finally Block** (lines 1309-1333)
    - Save partial exchange if connection dropped
    - Emit SESSION_END event

### BT Migration Mapping

| Oracle Function | BT Node Type | Blackboard Keys |
|-----------------|--------------|-----------------|
| `query()` entry | **Sequence** (tree root) | `query`, `user_id`, `project_id`, `context_id` |
| Reset state | **Action** leaf | - |
| Load context | **Action** leaf | `tree_root_id`, `current_node_id`, `context` |
| Build messages | **Action** leaf | `system_prompt`, `messages`, `tools` |
| Agent loop | **Repeater** decorator (until SUCCESS/FAILURE, max 30) | `turn`, `accumulated_content` |
| Turn execution | **Sequence** (subtree) | See `_agent_turn` |
| Save exchange | **Action** leaf | `question`, `answer`, `node_id` |

---

## 2. Tool Calling: `_execute_tools()`

**Location**: Lines 1843-2167

### Flow

1. **Loop Detection Check** (lines 1868-1895)
   - Call `_detect_loop(tool_calls)`
   - If loop detected, emit ANS event and inject system message

2. **Parse Tool Calls** (lines 1900-1923)
   - Extract `call_id`, `name`, `arguments` from each call
   - Handle malformed JSON arguments

3. **Yield Pending Status** (lines 1936-1946)
   - Yield `tool_call` chunk with `status="pending"` for each tool

4. **Inject Project Context** (lines 1949-1972)
   - Tools in `PROJECT_SCOPED_TOOLS` get `project_id` injected

5. **Parallel Execution** (lines 1996-2019)
   - Create async tasks for each tool
   - `asyncio.gather(*tasks, return_exceptions=True)`

6. **Process Results** (lines 2037-2167)
   - For each result (preserving order):
     - Success: yield result chunk, emit success event, extract sources, collect for persistence, add to messages
     - Failure: format error, emit failure event, yield error chunk, collect for persistence, add to messages
   - Update context tokens after each tool result
   - Call `_drain_and_yield_notifications()`

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys | Success/Failure |
|----------|--------------|-----------------|-----------------|
| `_execute_tools()` | **Parallel** composite | `tool_calls`, `tool_results` | Policy: REQUIRE_ALL or continue-on-fail |
| Loop detection | **Condition** leaf | `recent_tool_patterns` | True if no loop |
| Single tool exec | **Action** leaf | `tool_{id}_result` | SUCCESS if tool succeeds |
| Source extraction | **Action** leaf | `collected_sources` | Always SUCCESS |
| Notification drain | **Action** leaf | `pending_notifications` | Always SUCCESS |

---

## 3. LLM Interaction: `_agent_turn()` and `_process_stream()`

### `_agent_turn()` (Lines 1335-1444)

**Flow**:
1. Apply `:thinking` suffix if model supports it
2. Build request body with model, messages, tools, stream, max_tokens
3. Make HTTP POST to OpenRouter
4. If streaming: call `_process_stream()`
5. If not streaming: call `_process_response()`
6. Handle HTTP errors (status, timeout)

### `_process_stream()` (Lines 1446-1729)

**Flow**:
1. Initialize buffers: `content_buffer`, `reasoning_buffer`, `tool_calls_buffer`
2. Iterate SSE lines
3. For each delta:
   - Extract `reasoning` content (thinking traces)
   - Extract regular `content`
   - Detect tool call syntax in content/reasoning
   - Buffer tool calls
   - Yield safe content incrementally
4. On finish:
   - If `finish_reason == "tool_calls"`: execute tools
   - If XML tool calls detected: parse and execute
   - Otherwise: yield remaining content, save exchange, yield done

### Models Supported

```python
DEFAULT_MODEL_CONTEXT_SIZES = {
    "deepseek/deepseek-chat": 64000,
    "deepseek/deepseek-r1": 64000,
    "anthropic/claude-3-opus": 200000,
    "anthropic/claude-sonnet-4": 200000,
    "gemini-2.0-flash-exp": 1000000,
    "openai/gpt-4-turbo": 128000,
    "openai/o1": 200000,
    "meta-llama/llama-3.3-70b": 131072,
    # ... and more
}
```

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys | Success/Failure |
|----------|--------------|-----------------|-----------------|
| `_agent_turn()` | **LLMCallNode** (special) | `messages`, `model`, `max_tokens` | SUCCESS on response, FAILURE on error |
| Stream processing | Internal to LLMCallNode | `content_buffer`, `reasoning_buffer` | RUNNING while streaming |
| Tool call detection | **Condition** inside LLMCallNode | `tool_calls_buffer` | - |
| Content yield | Callback in LLMCallNode | `:stream-to [:partial-response]` | - |

---

## 4. Context Management

### Tree-Based Context (`ContextTreeService`)

**State Variables**:
- `_current_tree_root_id`: Root ID of active conversation tree
- `_current_node_id`: Current HEAD node

**Operations**:
- `get_node()`: Load existing node
- `create_tree()`: Create new conversation tree
- `get_nodes()`: Get all nodes in tree
- `create_node()`: Add new exchange to tree

**Path Building** (lines 1094-1127):
```python
# Walk from root to current node
path_nodes = []
current_id = self._current_node_id
while current_id and current_id in node_map:
    path_nodes.insert(0, node_map[current_id])
    current_id = node_map[current_id].parent_id
```

### Legacy Context (`OracleContextService`)

**State Variables**:
- `_context`: `OracleContext` object with `recent_exchanges`

**Operations**:
- `get_or_create_context()`
- `add_exchange()`

### `_save_exchange()` (Lines 2620-2730)

Saves to BOTH systems:
1. Tree: Create new node as child of current HEAD
2. Legacy: Add user exchange, then assistant exchange

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys |
|----------|--------------|-----------------|
| Load tree context | **Action** leaf | `tree_root_id`, `node_path`, `message_history` |
| Build message history | **Action** leaf | `messages` |
| Save exchange | **Action** leaf | `question`, `answer`, `tool_calls`, `system_messages` |

---

## 5. Budget Tracking

### State Variables

```python
self._iteration_warning_emitted = False
self._iteration_exceeded_emitted = False
self._token_warning_emitted = False
self._token_exceeded_emitted = False
self._total_tokens_used = 0
self._max_tokens_budget = 0

# Thresholds
self.ITERATION_WARNING_THRESHOLD = 0.70  # 70% of MAX_TURNS
self.TOKEN_WARNING_THRESHOLD = 0.80      # 80% of max_tokens

# Context window
self._context_tokens = 0
self._max_context_tokens = DEFAULT_CONTEXT_SIZE
self.CONTEXT_WARNING_THRESHOLD = 0.70    # 70% of model context
```

### Budget Checks

1. **`_check_iteration_budget(turn)`** (lines 762-787)
   - At 70% of MAX_TURNS, emit `BUDGET_ITERATION_WARNING`

2. **`_check_token_budget(tokens_used)`** (lines 789-818)
   - At 80% of max_tokens, emit `BUDGET_TOKEN_WARNING`

3. **`_emit_iteration_exceeded()`** (lines 820-839)
   - At MAX_TURNS, emit `BUDGET_ITERATION_EXCEEDED`

4. **`_emit_token_exceeded()`** (lines 841-860)
   - When token budget exceeded, emit `BUDGET_TOKEN_EXCEEDED`

5. **`_check_context_limit()`** (lines 862-898)
   - At 70% of model context window, emit `CONTEXT_APPROACHING_LIMIT`

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys | Success/Failure |
|----------|--------------|-----------------|-----------------|
| Budget tracking | **Decorator** around agent loop | `tokens_used`, `turn`, `context_tokens` | FAILURE if exceeded |
| Iteration check | **Condition** | `turn`, `max_turns` | FAILURE if turn >= MAX_TURNS |
| Token check | **Condition** | `tokens_used`, `token_budget` | FAILURE if exceeded |
| Context check | **Condition** | `context_tokens`, `max_context` | WARNING event only |

---

## 6. Loop Detection: `_detect_loop()`

**Location**: Lines 649-713

### Algorithm

1. Create pattern signature from tool names + key arguments
   ```python
   pattern_parts = []
   for call in tool_calls:
       name = function.get("name")
       key_args = ["path", "query", "thread_id", "file_path"]
       pattern_parts.append(f"{name}({','.join(key_args)})")
   current_pattern = "|".join(sorted(pattern_parts))
   ```

2. Add to `_recent_tool_patterns` (window size: 6)

3. Count occurrences of current pattern

4. If count >= `_loop_threshold` (3) and not already warned:
   - Set `_loop_already_warned = True`
   - Return pattern info dict

### Response to Loop Detection

In `_execute_tools()` (lines 1868-1895):
- Emit `AGENT_LOOP_DETECTED` event
- Inject system notification into messages
- Yield system chunk

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys | Success/Failure |
|----------|--------------|-----------------|-----------------|
| Loop detection | **Condition** leaf | `recent_tool_patterns`, `loop_threshold` | TRUE if no loop, FALSE if loop detected |
| Loop response | **Action** leaf (side effect) | `loop_warning` | Always SUCCESS |

---

## 7. ANS Integration

### Initialization (Lines 445-478)

```python
self._event_bus = get_event_bus()
self._accumulator = NotificationAccumulator()
self._toon_formatter = get_toon_formatter()
self._subscriber_loader = SubscriberLoader()

# Load and register subscribers
subscribers = self._subscriber_loader.load_all()
self._accumulator.register_subscribers(list(subscribers.values()))
```

### Events Emitted

| Event Type | Location | Severity | Trigger |
|------------|----------|----------|---------|
| `QUERY_START` | query() | INFO | Start of new query |
| `SESSION_RESUMED` | query() | INFO | Loaded context with history |
| `SESSION_END` | query() finally | INFO | Query complete |
| `BUDGET_ITERATION_WARNING` | _check_iteration_budget() | WARNING | 70% turns used |
| `BUDGET_ITERATION_EXCEEDED` | _emit_iteration_exceeded() | ERROR | MAX_TURNS reached |
| `BUDGET_TOKEN_WARNING` | _check_token_budget() | WARNING | 80% tokens used |
| `BUDGET_TOKEN_EXCEEDED` | _emit_token_exceeded() | ERROR | Token limit exceeded |
| `CONTEXT_APPROACHING_LIMIT` | _check_context_limit() | WARNING | 70% context used |
| `TOOL_CALL_SUCCESS` | _execute_tools() | INFO | Tool succeeded |
| `TOOL_CALL_FAILURE` | _execute_tools() | ERROR | Tool failed |
| `AGENT_LOOP_DETECTED` | _execute_tools() | WARNING | Loop pattern found |

### Notification Draining

Three drain methods:
1. **`_drain_and_yield_notifications()`** (lines 2169-2237)
   - Called after tool execution
   - Drains `after_tool` notifications
   - Drains deferred notifications for executed tools

2. **`_drain_and_yield_turn_start_notifications()`** (lines 2239-2303)
   - Called at start of each turn
   - Drains `turn_start` notifications (budget warnings)

3. **`_drain_and_yield_immediate_notifications()`** (lines 2305-2364)
   - Called on critical events
   - Drains `immediate` notifications (exceeded events)

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys |
|----------|--------------|-----------------|
| Event emission | Side effect of any node | - |
| Notification drain | **Action** leaf | `pending_events`, `notifications` |
| Inject into messages | Part of notification drain | `messages` |

---

## 8. Plugin Integration: RuleEngine

### Initialization (Lines 479-522)

```python
def _init_rule_engine(self) -> None:
    rules_dir = Path(__file__).parent / "plugins" / "rules"
    loader = RuleLoader(rules_dir)
    evaluator = ExpressionEvaluator()
    dispatcher = ActionDispatcher(
        event_bus=self._event_bus,
        state_setter=self._set_plugin_state,
    )
    self._rule_engine = RuleEngine(
        loader=loader,
        evaluator=evaluator,
        dispatcher=dispatcher,
        event_bus=self._event_bus,
        context_builder=self._build_rule_context,
        auto_subscribe=True,
    )
    self._rule_engine.start()
```

### Context Builder (Lines 538-625)

`_build_rule_context(event)` creates `RuleContext` with:
- `TurnState`: turn number, token/context usage, iteration count
- `HistoryState`: messages, tool call records, failure counts
- `UserState`: user ID, settings
- `ProjectState`: project ID, settings
- `PluginState`: empty for now
- `EventData`: type, source, severity, payload

### State Setter (Lines 524-536)

`_set_plugin_state(key, value)` - placeholder for plugin state persistence

### BT Migration Mapping

The RuleEngine could become:
- A **decorator** that wraps tree execution, evaluating rules on events
- Or a **parallel** subtree that runs alongside main execution
- Rules translate to **condition** nodes with **action** leaves

---

## 9. Cancellation: `cancel()`

**Location**: Lines 627-648

### Implementation

```python
def cancel(self) -> None:
    logger.info(f"Cancelling Oracle agent for user {self.user_id}")
    self._cancelled = True
    for task in self._active_tasks:
        if not task.done():
            task.cancel()
    self._active_tasks.clear()

def is_cancelled(self) -> bool:
    return self._cancelled

def reset_cancellation(self) -> None:
    self._cancelled = False
    self._active_tasks.clear()
```

### Cancellation Checkpoints

1. **query() start** (lines 952-955): Before any work
2. **Agent loop** (lines 1224-1227): Before each turn
3. **Streaming** (lines 1247-1250): During stream processing

### BT Migration Mapping

| Function | BT Node Type | Mechanism |
|----------|--------------|-----------|
| Cancellation flag | **Decorator** (interrupt) | Sets blackboard `cancelled=True` |
| Checkpoint checks | **Condition** before expensive nodes | Checks `blackboard.cancelled` |
| Task cancellation | Part of LLMCallNode | Cancels HTTP request on interrupt |

---

## 10. Subagent Delegation: Librarian

### Delegation Logic (Lines 2532-2613)

`_should_delegate_to_librarian(tool_name, tool_result)` checks:
- Token estimate > 4000
- vault_search/search_code: > 6 results with similar scores
- vault_list: > 10 files
- thread_read: > 20 entries

### Thresholds

```python
DELEGATION_THRESHOLDS = {
    "vault_search_results": 6,
    "search_code_results": 6,
    "vault_list_files": 10,
    "thread_read_entries": 20,
    "token_estimate": 4000,
    "score_similarity": 0.1,
}
```

### Note

This delegation logic exists but is NOT actively called in the current code. It appears to be preparation for future Librarian subagent integration.

### BT Migration Mapping

| Function | BT Node Type | Blackboard Keys |
|----------|--------------|-----------------|
| Delegation check | **Condition** leaf | `tool_result`, `delegation_thresholds` |
| Librarian subtree | **SubTree** reference | `summarization_request`, `summary` |

---

## Full Migration Mapping: Oracle Functions to BT Nodes

```lisp
(tree "oracle-agent"
  :description "Main Oracle chat agent"
  :blackboard-schema {
    :query nil
    :user_id nil
    :project_id nil
    :context_id nil
    :messages []
    :tools []
    :turn 0
    :accumulated_content ""
    :tool_calls []
    :collected_sources []
    :cancelled false
  }

  (sequence
    ;; Phase 1: Initialization
    (action reset-state :fn "oracle.reset_state")
    (action emit-query-start :fn "oracle.emit_query_start")

    ;; Phase 2: Context Loading
    (selector
      (sequence
        (condition has-context-id?)
        (action load-tree-node :fn "oracle.load_tree_node"))
      (action get-or-create-tree :fn "oracle.get_or_create_tree"))
    (action load-legacy-context :fn "oracle.load_legacy_context")
    (action load-cross-session-notifications :fn "oracle.load_cross_session")

    ;; Phase 3: Message Building
    (action build-system-prompt :fn "oracle.build_system_prompt")
    (action add-tree-history :fn "oracle.add_tree_history")
    (action inject-notifications :fn "oracle.inject_notifications")
    (action add-user-question :fn "oracle.add_user_question")
    (action get-tool-schemas :fn "oracle.get_tool_schemas")

    ;; Phase 4: Agent Loop
    (repeater :max-iterations 30 :until-success
      (decorator budget-check
        (sequence
          ;; Check cancellation
          (condition not-cancelled?)

          ;; Budget warnings
          (action check-iteration-budget :fn "oracle.check_iteration_budget")
          (action drain-turn-start-notifications :fn "oracle.drain_turn_start")

          ;; Agent Turn (subtree)
          (subtree agent-turn
            (sequence
              (llm-call :model (blackboard :model)
                        :messages (blackboard :messages)
                        :tools (blackboard :tools)
                        :stream-to [:partial-response]
                        :budget (blackboard :max_tokens)
                        :interruptible true
                        :timeout 60)

              ;; Handle response
              (selector
                ;; Tool calls detected
                (sequence
                  (condition has-tool-calls?)
                  (action detect-loop :fn "oracle.detect_loop")
                  (parallel :policy :continue-on-failure
                    (for-each (blackboard :tool_calls)
                      (action execute-tool :fn "tools.execute")))
                  (action drain-notifications :fn "oracle.drain_after_tool"))

                ;; Final response (no tool calls)
                (sequence
                  (action extract-xml-tools :fn "oracle.extract_xml_tools")
                  (selector
                    (sequence
                      (condition has-xml-tools?)
                      (parallel :policy :continue-on-failure
                        (for-each (blackboard :xml_tool_calls)
                          (action execute-tool :fn "tools.execute"))))
                    (sequence
                      (action save-exchange :fn "oracle.save_exchange")
                      (action yield-sources :fn "oracle.yield_sources")
                      (action emit-done :fn "oracle.emit_done")
                      (succeed))))))))))

    ;; Phase 5: Max Turns Exceeded
    (sequence
      (action emit-iteration-exceeded :fn "oracle.emit_iteration_exceeded")
      (action drain-immediate-notifications :fn "oracle.drain_immediate")
      (action save-partial-exchange :fn "oracle.save_partial")
      (action emit-done-with-warning :fn "oracle.emit_done_warning")))

  :finally
    (sequence
      (action save-partial-if-needed :fn "oracle.save_partial_cleanup")
      (action emit-session-end :fn "oracle.emit_session_end")))
```

---

## E2E Test Scenarios

### Current Test Coverage (Inferred)

Based on code structure, these scenarios should be tested:

| Scenario | Current Coverage | Priority |
|----------|------------------|----------|
| Basic query -> response | Unknown | P0 |
| Tool calling (single) | Unknown | P0 |
| Tool calling (parallel) | Unknown | P0 |
| Streaming to frontend | Unknown | P0 |
| XML tool call parsing | Unknown | P1 |
| Budget exceeded (turns) | Unknown | P1 |
| Budget exceeded (tokens) | Unknown | P1 |
| Loop detection trigger | Unknown | P1 |
| Cancellation mid-stream | Unknown | P2 |
| ANS event emission | Unknown | P2 |
| Context tree persistence | Unknown | P1 |
| Cross-session notifications | Unknown | P2 |

### Required E2E Tests for BT Migration

1. **Basic Query Flow**
   - Input: Simple question, no tools needed
   - Assert: Content streamed, exchange saved, done chunk emitted

2. **Single Tool Call**
   - Input: Question requiring one tool (e.g., vault_search)
   - Assert: Tool call chunk, tool result chunk, response incorporates result

3. **Multiple Parallel Tools**
   - Input: Question requiring multiple tools
   - Assert: All tools executed concurrently, results merged

4. **Streaming Verification**
   - Input: Long response expected
   - Assert: Multiple content chunks before done, context_update chunks emitted

5. **Iteration Budget Exceeded**
   - Setup: MAX_TURNS = 3, question that loops
   - Assert: Warning at turn 2, exceeded at turn 3, partial exchange saved

6. **Token Budget Exceeded**
   - Setup: max_tokens = 100
   - Assert: Warning at 80 tokens, response truncated or fails gracefully

7. **Loop Detection**
   - Input: Question that causes same tool call 3+ times
   - Assert: Loop detected event, system notification injected

8. **Mid-Stream Cancellation**
   - Setup: Long streaming response
   - Action: Call cancel() during stream
   - Assert: Cancelled error chunk, partial exchange saved

9. **ANS Event Verification**
   - Subscribe to event bus
   - Assert: QUERY_START, SESSION_END, TOOL_CALL_SUCCESS events emitted

10. **Context Tree Continuity**
    - Query 1: Save context_id from response
    - Query 2: Use context_id
    - Assert: Message history includes Query 1

---

## Key Async Behaviors

### Async Patterns in Current Code

1. **httpx.AsyncClient** for LLM calls (line 1406)
2. **asyncio.gather** for parallel tool execution (line 2017)
3. **response.aiter_lines()** for SSE streaming (line 1474)
4. **AsyncGenerator** yields throughout

### BT Considerations

- LLMCallNode must be async-aware (return RUNNING while awaiting)
- Parallel node must use asyncio.gather semantics
- Tick loop needs `await ctx.wait_for_async()` support
- Cancellation must cancel HTTP requests, not just set flag

---

## Summary: Migration Complexity

| Component | Lines | BT Node Type | Complexity |
|-----------|-------|--------------|------------|
| query() orchestration | ~400 | Sequence + Repeater | Medium |
| _agent_turn() | ~110 | LLMCallNode | High (streaming) |
| _process_stream() | ~280 | Internal to LLMCallNode | High |
| _execute_tools() | ~320 | Parallel + Action | Medium |
| Context management | ~200 | Action leaves | Low |
| Budget tracking | ~150 | Decorator + Conditions | Low |
| Loop detection | ~70 | Condition leaf | Low |
| ANS integration | ~200 | Side effects | Low |
| Error handling | ~100 | Spread throughout | Medium |

**Total estimated LISP**: 300-400 lines (vs 2,765 Python)

**Key Migration Risks**:
1. Streaming behavior in LLMCallNode is complex
2. XML tool parsing is model-specific edge case
3. Two context systems (tree + legacy) must both persist
4. ANS event timing must be preserved
