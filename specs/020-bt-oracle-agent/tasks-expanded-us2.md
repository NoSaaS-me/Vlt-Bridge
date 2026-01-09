# Expanded Tasks: User Story 2 - Agent Self-Reflection via Signals

**Feature**: 020-bt-oracle-agent
**User Story**: Agent emits XML signals that BT parses to manage conversation flow

---

## T020: Create signals.py Conditions

**File**: `backend/src/bt/conditions/signals.py`

### Function Signatures

```python
from ..core.context import TickContext
from ..state.base import RunStatus

def check_signal(ctx: TickContext) -> RunStatus:
    """Check if a signal was parsed in the current turn."""

def has_signal(ctx: TickContext, signal_type: str | None = None) -> RunStatus:
    """Check if any/specific signal exists in blackboard.last_signal."""

def signal_type_is(ctx: TickContext, expected_type: str) -> RunStatus:
    """Check if last_signal.type matches expected_type."""

def signal_confidence_above(ctx: TickContext, threshold: float = 0.5) -> RunStatus:
    """Check if last_signal.confidence >= threshold."""

def consecutive_same_reason_gte(ctx: TickContext, count: int = 3) -> RunStatus:
    """Check if consecutive_same_reason >= count (loop detection)."""
```

### Core Algorithm

1. Import `bb_get` helper from `oracle.py` for blackboard access
2. `check_signal`: Return SUCCESS if `bb.last_signal is not None`
3. `has_signal`: If `signal_type` provided, compare against `bb.last_signal.type`
4. `signal_type_is`: Strict equality check on signal type enum string
5. `signal_confidence_above`: Parse confidence float, compare with threshold
6. `consecutive_same_reason_gte`: Access `bb.consecutive_same_reason` counter
7. All return `RunStatus.SUCCESS` or `RunStatus.FAILURE` (no RUNNING)
8. Handle None/missing values gracefully - return FAILURE

### Key Code Snippet

```python
from typing import Optional
from ..state.base import RunStatus
from ..core.context import TickContext
from ..actions.oracle import bb_get

def has_signal(ctx: TickContext, signal_type: Optional[str] = None) -> RunStatus:
    """Check if signal exists, optionally matching specific type."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    last_signal = bb_get(bb, "last_signal")
    if last_signal is None:
        return RunStatus.FAILURE

    if signal_type is not None:
        sig_type = last_signal.get("type") if isinstance(last_signal, dict) else getattr(last_signal, "type", None)
        if sig_type != signal_type:
            return RunStatus.FAILURE

    return RunStatus.SUCCESS
```

### Acceptance Criteria Mapping

| AC | Implementation |
|----|----------------|
| AC-4a: BT parses signal | `check_signal()` returns SUCCESS after parsing |
| AC-4b: BT acts on signal | Conditions drive BT routing via selector nodes |
| US3-AC-3: 3x same reason = stuck | `consecutive_same_reason_gte(ctx, 3)` |

---

## T021: Create signal_actions.py

**File**: `backend/src/bt/actions/signal_actions.py`

### Function Signatures

```python
from ..core.context import TickContext
from ..state.base import RunStatus
from src.models.signals import Signal, SignalType

def parse_response_signal(ctx: TickContext) -> RunStatus:
    """Parse XML signal from accumulated_content, store in blackboard."""

def strip_signal_from_response(ctx: TickContext) -> RunStatus:
    """Remove signal XML from accumulated_content for user display."""

def log_signal(ctx: TickContext) -> RunStatus:
    """Log signal to ANS event bus and append to signals_emitted list."""

def update_signal_state(ctx: TickContext) -> RunStatus:
    """Update consecutive_same_reason counter based on signal pattern."""
```

### Core Algorithm

**parse_response_signal:**
1. Get `accumulated_content` from blackboard
2. Use regex: `r'<signal\s+type="([^"]+)">(.*?)</signal>'` with `re.DOTALL`
3. If no match, set `bb.last_signal = None`, increment `turns_without_signal`
4. Extract signal type and inner XML content
5. Parse inner fields: `reason`, `confidence`, `sources_found`, etc.
6. Construct `Signal` dataclass or dict with `type`, `confidence`, `fields`, `raw_xml`
7. Store in `bb.last_signal`
8. Reset `turns_without_signal` to 0

**strip_signal_from_response:**
1. Get `accumulated_content` from blackboard
2. Use `re.sub()` to remove signal block
3. Strip trailing whitespace
4. Update `accumulated_content` in blackboard

**update_signal_state:**
1. Get `last_signal` and previous `last_signal` (store as `prev_signal`)
2. If both are `need_turn` with same `reason`, increment `consecutive_same_reason`
3. Otherwise, reset `consecutive_same_reason` to 1
4. Append signal to `signals_emitted` list

### Key Code Snippet

```python
import re
from typing import Any, Dict, Optional
from datetime import datetime, timezone

SIGNAL_PATTERN = re.compile(
    r'<signal\s+type="([^"]+)">\s*(.*?)\s*</signal>',
    re.DOTALL
)

FIELD_PATTERN = re.compile(r'<(\w+)>([^<]*)</\1>')

def _parse_signal_xml(content: str) -> Optional[Dict[str, Any]]:
    """Extract signal from response content."""
    match = SIGNAL_PATTERN.search(content)
    if not match:
        return None

    signal_type = match.group(1)
    inner_xml = match.group(2)
    raw_xml = match.group(0)

    # Parse inner fields
    fields: Dict[str, Any] = {}
    for field_match in FIELD_PATTERN.finditer(inner_xml):
        key = field_match.group(1)
        value = field_match.group(2).strip()
        # Type coercion
        if key == "confidence":
            fields[key] = float(value) if value else 0.5
        elif key == "sources_found" or key == "expected_turns":
            fields[key] = int(value) if value.isdigit() else 0
        elif value.startswith("[") and value.endswith("]"):
            # JSON list
            import json
            try:
                fields[key] = json.loads(value)
            except json.JSONDecodeError:
                fields[key] = value
        else:
            fields[key] = value

    return {
        "type": signal_type,
        "confidence": fields.pop("confidence", 0.5),
        "fields": fields,
        "raw_xml": raw_xml,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
```

### Acceptance Criteria Mapping

| AC | Implementation |
|----|----------------|
| AC-1: Emits need_turn | Parsed by `parse_response_signal()` |
| AC-2: Emits context_sufficient | Same parsing, different type |
| AC-3: Emits stuck | Same parsing, fields include `attempted`, `blocker` |
| FR-005: Strip signal XML | `strip_signal_from_response()` |

---

## T022: Modify oracle_wrapper.py for Signal Parsing

**File**: `backend/src/bt/wrappers/oracle_wrapper.py`

### Function Signature Changes

```python
# Add to OracleBTWrapper class:
async def _run_tree(self) -> AsyncGenerator[OracleStreamChunk, None]:
    """Run tree with signal parsing after each LLM response."""

# Add new method:
def _parse_signal_from_response(self) -> None:
    """Call signal parser after LLM response in blackboard."""
```

### Core Algorithm

1. Locate the `_run_tree()` method (lines 344-406)
2. After tree tick, check if `bb.llm_response` was set this tick
3. If LLM response present, call signal actions in sequence:
   a. `parse_response_signal(ctx)`
   b. `update_signal_state(ctx)`
   c. `log_signal(ctx)`
   d. `strip_signal_from_response(ctx)`
4. Signal parsing happens BEFORE yielding done chunk
5. Store parsed signal in blackboard for condition nodes to read

### Key Code Snippet

```python
# In _run_tree(), after tree tick:
async def _run_tree(self) -> AsyncGenerator[OracleStreamChunk, None]:
    # ... existing code ...

    while status == RunStatus.RUNNING and tick_count < max_ticks:
        # ... existing tick logic ...

        # After tick, check for signal parsing
        if self._should_parse_signal():
            self._parse_signal_from_response()

        # ... rest of loop ...

def _should_parse_signal(self) -> bool:
    """Check if LLM response was received this tick."""
    if not self._blackboard:
        return False
    llm_response = self._blackboard._lookup("llm_response")
    already_parsed = self._blackboard._lookup("_signal_parsed_this_turn")
    return llm_response is not None and not already_parsed

def _parse_signal_from_response(self) -> None:
    """Parse signal from LLM response and update state."""
    from ..actions.signal_actions import (
        parse_response_signal,
        update_signal_state,
        log_signal,
        strip_signal_from_response
    )

    # Create minimal context for action calls
    parse_response_signal(self._ctx)
    update_signal_state(self._ctx)
    log_signal(self._ctx)
    strip_signal_from_response(self._ctx)

    # Mark as parsed to avoid double-parsing
    self._blackboard._data["_signal_parsed_this_turn"] = True
```

### Acceptance Criteria Mapping

| AC | Implementation |
|----|----------------|
| FR-002: Real-time signal parsing | Signal parsed immediately after LLM response |
| FR-009: Log all signals | `log_signal()` emits to ANS bus |
| AC-4: BT parses and acts | Signal in blackboard for conditions |

---

## T023: Add Signal State Tracking to Blackboard

**File**: `backend/src/bt/state/blackboard.py` (schema update)
**File**: `backend/src/bt/actions/oracle.py` (initialization)
**File**: `backend/src/bt/trees/oracle-agent.lua` (schema declaration)

### Field Definitions

```python
# Blackboard signal state fields
SIGNAL_STATE_FIELDS = {
    "last_signal": Optional[Dict[str, Any]],      # Most recent parsed signal
    "signals_emitted": List[Dict[str, Any]],      # All signals this session
    "consecutive_same_reason": int,                # Loop detection counter
    "turns_without_signal": int,                   # Fallback trigger counter
    "_signal_parsed_this_turn": bool,              # Prevent double-parsing
}
```

### Core Algorithm

**In reset_state() action (oracle.py):**
1. Add initialization for signal state fields
2. `last_signal = None`
3. `signals_emitted = []`
4. `consecutive_same_reason = 0`
5. `turns_without_signal = 0`

**In oracle-agent.lua blackboard schema:**
1. Add `last_signal = "Signal"` type
2. Add `signals_emitted = "list"`
3. Add `consecutive_same_reason = "int"`
4. Add `turns_without_signal = "int"`

### Key Code Snippet

```python
# In reset_state() function, add after line ~131:

def reset_state(ctx: "TickContext") -> RunStatus:
    bb = ctx.blackboard
    # ... existing reset code ...

    # Signal state tracking (T023)
    bb_set(bb, "last_signal", None)
    bb_set(bb, "signals_emitted", [])
    bb_set(bb, "consecutive_same_reason", 0)
    bb_set(bb, "turns_without_signal", 0)
    bb_set(bb, "_signal_parsed_this_turn", False)

    logger.debug("Oracle state reset for new query (with signal tracking)")
    ctx.mark_progress()
    return RunStatus.SUCCESS
```

```lua
-- In oracle-agent.lua, add to blackboard schema (line ~25):
blackboard = {
    -- ... existing fields ...

    -- Signal state tracking (T023)
    last_signal = "Signal",
    signals_emitted = "list",
    consecutive_same_reason = "int",
    turns_without_signal = "int",
},
```

### Acceptance Criteria Mapping

| AC | Implementation |
|----|----------------|
| FR-009: Log all signals | `signals_emitted` list tracks history |
| US3-AC-3: Same reason 3x | `consecutive_same_reason` counter |
| US5-AC-1: No signal 3+ turns | `turns_without_signal` counter |

---

## T024: Update oracle-agent.lua Agent Loop for Signal Checking

**File**: `backend/src/bt/trees/oracle-agent.lua`

### Structure Changes

Add signal checking phase after LLM call in agent-turn subtree:

```lua
-- After LLM call (line ~357), add signal check sequence
BT.sequence({
    -- Parse and process signal
    BT.action("parse-signal", {
        fn = "backend.src.bt.actions.signal_actions.parse_response_signal"
    }),
    BT.action("update-signal-state", {
        fn = "backend.src.bt.actions.signal_actions.update_signal_state"
    }),
    BT.action("log-signal", {
        fn = "backend.src.bt.actions.signal_actions.log_signal"
    }),
    BT.action("strip-signal", {
        fn = "backend.src.bt.actions.signal_actions.strip_signal_from_response"
    }),
})
```

### Core Algorithm

1. Locate agent-turn subtree (line 329)
2. After `BT.llm_call()` block (line 344-357)
3. Insert signal processing sequence before tool handling
4. Add signal-based routing in main loop:
   - Check `need_turn` signal with confidence >= 0.5 -> continue loop
   - Check `context_sufficient` signal -> exit loop with success
   - Check `stuck` signal -> trigger fallback/exit

### Key Code Snippet

```lua
--[[
    Agent Turn Subtree (modified for T024)
--]]
BT.subtree("agent-turn", {
    description = "Single agent turn with signal processing",

    root = BT.sequence({
        -- Build and execute LLM call (existing)
        BT.action("build-llm-request", {
            fn = "backend.src.bt.actions.oracle.build_llm_request"
        }),

        BT.llm_call({ --[[ existing config ]] }),

        -- NEW: Signal processing phase (T024)
        BT.always_succeed(
            BT.sequence({
                BT.action("parse-signal", {
                    fn = "backend.src.bt.actions.signal_actions.parse_response_signal",
                    description = "Parse XML signal from LLM response"
                }),
                BT.action("update-signal-state", {
                    fn = "backend.src.bt.actions.signal_actions.update_signal_state",
                    description = "Track consecutive reasons for loop detection"
                }),
                BT.always_succeed(
                    BT.action("log-signal", {
                        fn = "backend.src.bt.actions.signal_actions.log_signal"
                    })
                ),
                BT.action("strip-signal", {
                    fn = "backend.src.bt.actions.signal_actions.strip_signal_from_response",
                    description = "Remove signal XML from user-visible content"
                })
            })
        ),

        -- NEW: Signal-based routing (T024)
        BT.selector({
            -- If stuck signal, acknowledge and exit
            BT.sequence({
                BT.condition("is-stuck", {
                    fn = "backend.src.bt.conditions.signals.signal_type_is",
                    args = { expected_type = "stuck" }
                }),
                -- Let the response through - agent already acknowledged
                BT.action("noop", { fn = "backend.src.bt.actions.oracle.noop" })
            }),

            -- If context_sufficient, ready for final answer
            BT.sequence({
                BT.condition("context-sufficient", {
                    fn = "backend.src.bt.conditions.signals.signal_type_is",
                    args = { expected_type = "context_sufficient" }
                }),
                BT.action("noop", { fn = "backend.src.bt.actions.oracle.noop" })
            }),

            -- Default: continue with existing tool handling
            BT.action("noop", { fn = "backend.src.bt.actions.oracle.noop" })
        }),

        -- Existing tool handling...
        -- ... rest of agent-turn subtree ...
    })
})
```

### Acceptance Criteria Mapping

| AC | Implementation |
|----|----------------|
| AC-4: BT acts on signal | Signal routing via selector after parse |
| FR-006: BT controls loop | Signal types determine continue/exit |
| FR-005: Strip from response | `strip_signal_from_response` before output |

---

## T025: Tests for Signal Conditions

**File**: `backend/tests/unit/bt/test_signal_conditions.py`

### Test Functions

```python
import pytest
from unittest.mock import MagicMock
from backend.src.bt.conditions.signals import (
    check_signal,
    has_signal,
    signal_type_is,
    signal_confidence_above,
    consecutive_same_reason_gte
)
from backend.src.bt.state.base import RunStatus

class TestCheckSignal:
    def test_returns_success_when_signal_present(self)
    def test_returns_failure_when_no_signal(self)
    def test_returns_failure_when_blackboard_none(self)

class TestHasSignal:
    def test_returns_success_with_any_signal(self)
    def test_returns_success_with_matching_type(self)
    def test_returns_failure_with_wrong_type(self)
    def test_returns_failure_when_no_signal(self)

class TestSignalTypeIs:
    def test_need_turn_match(self)
    def test_context_sufficient_match(self)
    def test_stuck_match(self)
    def test_mismatch_returns_failure(self)

class TestSignalConfidenceAbove:
    def test_confidence_above_threshold(self)
    def test_confidence_at_threshold(self)
    def test_confidence_below_threshold(self)

class TestConsecutiveSameReasonGte:
    def test_at_threshold_returns_success(self)
    def test_above_threshold_returns_success(self)
    def test_below_threshold_returns_failure(self)
```

### Core Algorithm

1. Create mock `TickContext` with mock `TypedBlackboard`
2. Set up blackboard state using `_data` dict
3. Call condition function
4. Assert `RunStatus.SUCCESS` or `RunStatus.FAILURE`
5. Cover edge cases: None values, missing keys, type mismatches

### Key Code Snippet

```python
import pytest
from unittest.mock import MagicMock, patch
from backend.src.bt.state.base import RunStatus

@pytest.fixture
def mock_ctx():
    """Create mock TickContext with blackboard."""
    ctx = MagicMock()
    ctx.blackboard = MagicMock()
    ctx.blackboard._data = {}
    ctx.blackboard._lookup = lambda key: ctx.blackboard._data.get(key)
    return ctx

class TestHasSignal:
    def test_returns_success_with_matching_type(self, mock_ctx):
        # Arrange
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.85,
            "fields": {"reason": "Testing API"}
        }

        # Act
        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx, signal_type="need_turn")

        # Assert
        assert result == RunStatus.SUCCESS

    def test_returns_failure_with_wrong_type(self, mock_ctx):
        mock_ctx.blackboard._data["last_signal"] = {
            "type": "context_sufficient"
        }

        from backend.src.bt.conditions.signals import has_signal
        result = has_signal(mock_ctx, signal_type="need_turn")

        assert result == RunStatus.FAILURE

class TestConsecutiveSameReasonGte:
    def test_at_threshold_returns_success(self, mock_ctx):
        mock_ctx.blackboard._data["consecutive_same_reason"] = 3

        from backend.src.bt.conditions.signals import consecutive_same_reason_gte
        result = consecutive_same_reason_gte(mock_ctx, count=3)

        assert result == RunStatus.SUCCESS
```

### Acceptance Criteria Mapping

| AC | Test Coverage |
|----|---------------|
| Signal parsing | `TestCheckSignal`, `TestHasSignal` |
| Signal type routing | `TestSignalTypeIs` |
| Loop detection | `TestConsecutiveSameReasonGte` |
| Confidence check | `TestSignalConfidenceAbove` |

---

## T026: Integration Test - Tool Failure Triggers need_turn

**File**: `backend/tests/integration/test_oracle_bt_integration.py`

### Test Function

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_failure_triggers_need_turn_signal():
    """Verify that tool failure scenario causes agent to emit need_turn signal."""
```

### Core Algorithm

1. Set up OracleBTWrapper with mock tool executor
2. Configure mock to fail a specific tool (e.g., `search_code` returns error)
3. Craft query that would trigger that tool: "What does the auth function do?"
4. Run `process_query()` and collect all chunks
5. Inspect blackboard `last_signal` after first LLM turn with failed tool
6. Assert signal type is `need_turn`
7. Assert signal reason mentions retry or alternative approach
8. Verify signal was stripped from accumulated_content

### Key Code Snippet

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_failure_triggers_need_turn_signal():
    """
    Scenario: Tool failure causes agent to request another turn

    Given: Agent tries to search code
    When: search_code tool returns an error
    Then: Agent emits need_turn signal to try alternative approach
    """
    # Arrange
    from backend.src.bt.wrappers.oracle_wrapper import OracleBTWrapper

    wrapper = OracleBTWrapper(
        user_id="test-user",
        project_id="test-project",
        model="test-model"
    )

    # Mock LLM to emit tool call, then signal after failure
    mock_llm_responses = [
        # First response: tool call
        MagicMock(
            content="Let me search for that function.",
            tool_calls=[{
                "id": "call_1",
                "function": {
                    "name": "search_code",
                    "arguments": '{"query": "auth function"}'
                }
            }]
        ),
        # Second response: need_turn signal after tool failure
        MagicMock(
            content="""The search encountered an error. Let me try a different approach.

<signal type="need_turn">
  <reason>search_code failed, trying vault search instead</reason>
  <confidence>0.8</confidence>
</signal>""",
            tool_calls=[]
        )
    ]

    # Mock tool executor to fail
    mock_executor = MagicMock()
    mock_executor.execute = MagicMock(side_effect=Exception("Index not ready"))

    with patch("backend.src.services.tool_executor.ToolExecutor", return_value=mock_executor):
        with patch_llm_call(mock_llm_responses):
            # Act
            chunks = []
            async for chunk in wrapper.process_query("What does the auth function do?"):
                chunks.append(chunk)

            # Assert
            # 1. Check signal was parsed
            last_signal = wrapper._blackboard._lookup("last_signal")
            assert last_signal is not None
            assert last_signal["type"] == "need_turn"
            assert "search_code failed" in last_signal["fields"]["reason"]

            # 2. Check signal was logged
            signals_emitted = wrapper._blackboard._lookup("signals_emitted")
            assert len(signals_emitted) >= 1
            assert signals_emitted[-1]["type"] == "need_turn"

            # 3. Check signal stripped from content
            accumulated = wrapper._blackboard._lookup("accumulated_content")
            assert "<signal" not in accumulated

            # 4. Check tool failure was recorded
            tool_results = wrapper._blackboard._lookup("tool_results")
            assert any(r.get("success") == False for r in tool_results)
```

### Acceptance Criteria Mapping

| AC | Test Verification |
|----|-------------------|
| US2-AC-1: need_turn with reason | Assert signal type and reason field |
| US2-AC-4: BT parses signal | Assert `last_signal` in blackboard |
| FR-005: Strip signal | Assert `<signal` not in `accumulated_content` |
| FR-009: Log signal | Assert `signals_emitted` list updated |

---

## Summary

| Task | Files | LOC Est. | Dependencies |
|------|-------|----------|--------------|
| T020 | conditions/signals.py | ~80 | T006 (signal_parser) |
| T021 | actions/signal_actions.py | ~150 | T002 (Signal model), T006 |
| T022 | wrappers/oracle_wrapper.py | ~40 | T021 |
| T023 | oracle.py, oracle-agent.lua | ~30 | None |
| T024 | oracle-agent.lua | ~60 | T020, T021 |
| T025 | tests/unit/bt/test_signal_conditions.py | ~120 | T020 |
| T026 | tests/integration/test_oracle_bt_integration.py | ~80 | T020-T024 |

**Execution Order:**
1. T023 (schema) - no dependencies
2. T020 (conditions) - after T006 signal_parser from Phase 2
3. T021 (actions) - after T002, T006 from Phase 2
4. T022 (wrapper) - after T021
5. T024 (lua tree) - after T020, T021
6. T025 (unit tests) - parallel with T020-T024
7. T026 (integration) - after all above
