# Expanded Tasks: User Story 5 - BERT Fallback for Edge Cases

**Feature**: 020-bt-oracle-agent
**User Story**: US5 - BERT Fallback for Edge Cases
**Priority**: P3
**Tasks**: T042-T049

**Goal**: Heuristic fallback when agent signals are absent or weak (BERT placeholder)

**Independent Test**: 3 turns without signals triggers fallback classification

---

## T042: Create fallback_classifier.py with heuristic_classify()

**File**: `backend/src/services/fallback_classifier.py`

### Function Signature

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

class FallbackAction(Enum):
    CONTINUE = "continue"           # Let agent continue
    FORCE_RESPONSE = "force_response"  # Force agent to respond
    RETRY_WITH_HINT = "retry_with_hint"  # Retry with guidance
    ESCALATE = "escalate"           # Surface to user

@dataclass
class FallbackClassification:
    action: FallbackAction
    confidence: float  # 0.0-1.0
    hint: Optional[str] = None  # Guidance for retry
    reason: str = ""  # Why this classification

def heuristic_classify(
    query: str,
    accumulated_content: str,
    turns_without_signal: int,
    tool_results: List[dict],
    last_signal_confidence: Optional[float] = None,
) -> FallbackClassification:
    """
    Classify fallback action using heuristics (BERT placeholder).

    Args:
        query: Original user question
        accumulated_content: Response accumulated so far
        turns_without_signal: Count of turns without signal emission
        tool_results: List of tool execution results
        last_signal_confidence: Last signal confidence (if any)

    Returns:
        FallbackClassification with action and reasoning
    """
```

### Core Algorithm

1. **Check accumulated content length**: If substantial response exists (>500 chars), lean toward FORCE_RESPONSE
2. **Analyze tool results**: Count successes/failures; high failure rate suggests ESCALATE
3. **Check query complexity**: Simple queries (< 20 words) with no tools called -> FORCE_RESPONSE
4. **Detect stuck patterns**: Repeated failures or no progress -> RETRY_WITH_HINT
5. **Calculate base confidence**: Start at 0.6 (heuristic baseline)
6. **Adjust confidence**: Increase if clear signals (many tools succeeded), decrease if ambiguous
7. **Select action**: Map analysis to FallbackAction enum
8. **Generate hint**: If RETRY_WITH_HINT, provide specific guidance based on tool failures

### Key Code Snippet

```python
def heuristic_classify(
    query: str,
    accumulated_content: str,
    turns_without_signal: int,
    tool_results: List[dict],
    last_signal_confidence: Optional[float] = None,
) -> FallbackClassification:
    """Heuristic fallback classifier (BERT placeholder)."""

    # Analyze current state
    has_content = len(accumulated_content) > 500
    tool_failures = sum(1 for r in tool_results if not r.get("success", False))
    tool_successes = len(tool_results) - tool_failures
    query_words = len(query.split())

    # High failure rate -> escalate
    if len(tool_results) > 2 and tool_failures / len(tool_results) > 0.7:
        return FallbackClassification(
            action=FallbackAction.ESCALATE,
            confidence=0.7,
            reason=f"High tool failure rate: {tool_failures}/{len(tool_results)}"
        )

    # Substantial content accumulated -> force response
    if has_content and turns_without_signal >= 2:
        return FallbackClassification(
            action=FallbackAction.FORCE_RESPONSE,
            confidence=0.8,
            reason="Sufficient content accumulated without signal"
        )

    # Simple query, no tools needed
    if query_words < 20 and not tool_results:
        return FallbackClassification(
            action=FallbackAction.FORCE_RESPONSE,
            confidence=0.75,
            hint="This appears to be a simple query. Please provide a direct answer.",
            reason="Simple query without tool usage"
        )

    # No progress after multiple turns
    if turns_without_signal >= 3 and not has_content:
        hint = "Previous attempts did not yield results. "
        if tool_failures > 0:
            failed_tools = [r.get("name", "unknown") for r in tool_results if not r.get("success")]
            hint += f"Tools {failed_tools} failed. Try alternative approaches."
        return FallbackClassification(
            action=FallbackAction.RETRY_WITH_HINT,
            confidence=0.6,
            hint=hint,
            reason=f"No progress after {turns_without_signal} turns"
        )

    # Default: continue
    return FallbackClassification(
        action=FallbackAction.CONTINUE,
        confidence=0.5,
        reason="No clear fallback trigger"
    )
```

### Acceptance Criteria Mapping

- **FR-022**: System functions with heuristic defaults when BERT unavailable
- **SC-006**: System functions correctly with BERT disabled (heuristic-only mode)

---

## T043: Create fallback.py conditions - needs_fallback()

**File**: `backend/src/bt/conditions/fallback.py`

### Function Signature

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.context import TickContext

# Thresholds
TURNS_WITHOUT_SIGNAL_THRESHOLD = 3
LOW_CONFIDENCE_THRESHOLD = 0.3

def needs_fallback(ctx: "TickContext") -> bool:
    """
    Check if fallback classification should be triggered.

    Triggers when:
    - No signal for 3+ turns OR
    - Last signal confidence < 0.3 OR
    - Last signal was type "stuck"

    Returns:
        True if fallback should be activated
    """

def no_signal_for_n_turns(ctx: "TickContext", n: int = 3) -> bool:
    """Check if no signal has been emitted for n consecutive turns."""

def signal_confidence_below(ctx: "TickContext", threshold: float = 0.3) -> bool:
    """Check if last signal confidence is below threshold."""

def is_stuck_signal(ctx: "TickContext") -> bool:
    """Check if last signal was type 'stuck'."""
```

### Core Algorithm

1. **Get blackboard**: Access context blackboard for signal state
2. **Check turns_without_signal**: Read counter from blackboard (`bb.turns_without_signal`)
3. **Check last_signal**: Get last signal from `bb.last_signal`
4. **Evaluate stuck**: If `last_signal.type == "stuck"`, return True immediately
5. **Evaluate confidence**: If `last_signal.confidence < 0.3`, return True
6. **Evaluate turn count**: If `turns_without_signal >= 3`, return True
7. **Return composite**: Any condition True -> needs_fallback

### Key Code Snippet

```python
from ..state.base import RunStatus
from ..actions.oracle import bb_get

TURNS_WITHOUT_SIGNAL_THRESHOLD = 3
LOW_CONFIDENCE_THRESHOLD = 0.3


def needs_fallback(ctx: "TickContext") -> bool:
    """Check if fallback classification should be triggered."""
    bb = ctx.blackboard
    if bb is None:
        return False

    # Condition 1: No signal for 3+ turns
    turns_without_signal = bb_get(bb, "turns_without_signal") or 0
    if turns_without_signal >= TURNS_WITHOUT_SIGNAL_THRESHOLD:
        return True

    # Condition 2: Low confidence signal
    last_signal = bb_get(bb, "last_signal")
    if last_signal:
        confidence = last_signal.get("confidence", 1.0)
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            return True

        # Condition 3: Stuck signal
        if last_signal.get("type") == "stuck":
            return True

    return False


def no_signal_for_n_turns(ctx: "TickContext", n: int = 3) -> bool:
    """Check if no signal has been emitted for n consecutive turns."""
    bb = ctx.blackboard
    if bb is None:
        return False

    turns_without_signal = bb_get(bb, "turns_without_signal") or 0
    return turns_without_signal >= n


def signal_confidence_below(ctx: "TickContext", threshold: float = 0.3) -> bool:
    """Check if last signal confidence is below threshold."""
    bb = ctx.blackboard
    if bb is None:
        return False

    last_signal = bb_get(bb, "last_signal")
    if not last_signal:
        return False

    return last_signal.get("confidence", 1.0) < threshold


def is_stuck_signal(ctx: "TickContext") -> bool:
    """Check if last signal was type 'stuck'."""
    bb = ctx.blackboard
    if bb is None:
        return False

    last_signal = bb_get(bb, "last_signal")
    if not last_signal:
        return False

    return last_signal.get("type") == "stuck"
```

### Acceptance Criteria Mapping

- **FR-019**: BERT fallback activates when no signal for 3+ turns
- **FR-020**: BERT fallback activates when signal confidence < 0.3
- **FR-021**: BERT fallback activates on explicit `stuck` signal

---

## T044: Create fallback_actions.py

**File**: `backend/src/bt/actions/fallback_actions.py`

### Function Signatures

```python
from typing import TYPE_CHECKING
from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext

def trigger_fallback(ctx: "TickContext") -> RunStatus:
    """
    Trigger fallback classification and store result.

    Calls heuristic_classify() with current state and stores
    FallbackClassification in blackboard.

    Returns:
        RunStatus.SUCCESS if classification successful
    """

def apply_heuristic_classification(ctx: "TickContext") -> RunStatus:
    """
    Apply the fallback classification action.

    Based on FallbackClassification.action:
    - CONTINUE: Do nothing, return SUCCESS
    - FORCE_RESPONSE: Inject "respond now" message, return SUCCESS
    - RETRY_WITH_HINT: Inject hint message, return SUCCESS
    - ESCALATE: Emit system message to user, return SUCCESS

    Returns:
        RunStatus.SUCCESS after applying action
    """

def inject_fallback_hint(ctx: "TickContext") -> RunStatus:
    """
    Inject fallback hint into conversation messages.

    Adds a system message with the classification hint to guide
    the agent toward a response.

    Returns:
        RunStatus.SUCCESS after injection
    """
```

### Core Algorithm for trigger_fallback

1. **Get blackboard state**: Extract query, accumulated_content, turns_without_signal, tool_results
2. **Get last signal confidence**: From `bb.last_signal.confidence` if exists
3. **Call heuristic_classify**: Pass all state to classifier
4. **Store classification**: Set `bb.fallback_classification` with result
5. **Log trigger**: Log why fallback was triggered
6. **Mark progress**: Call `ctx.mark_progress()`
7. **Return SUCCESS**

### Core Algorithm for apply_heuristic_classification

1. **Get classification**: Read `bb.fallback_classification`
2. **Switch on action**:
   - CONTINUE: Log "continuing", return SUCCESS
   - FORCE_RESPONSE: Inject "Please provide your final answer now" system message
   - RETRY_WITH_HINT: Inject classification.hint as system message
   - ESCALATE: Yield system chunk to frontend, add user-facing message
3. **Clear fallback state**: Reset `bb.fallback_classification`
4. **Return SUCCESS**

### Key Code Snippet

```python
import logging
from ..state.base import RunStatus
from .oracle import bb_get, bb_set, _add_pending_chunk

logger = logging.getLogger(__name__)


def trigger_fallback(ctx: "TickContext") -> RunStatus:
    """Trigger fallback classification and store result."""
    from src.services.fallback_classifier import heuristic_classify

    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Gather state for classification
    query = bb_get(bb, "query") or ""
    if hasattr(query, "question"):
        query = query.question
    elif isinstance(query, dict):
        query = query.get("question", str(query))

    accumulated = bb_get(bb, "accumulated_content") or ""
    turns_without = bb_get(bb, "turns_without_signal") or 0
    tool_results = bb_get(bb, "tool_results") or []

    last_signal = bb_get(bb, "last_signal")
    last_confidence = last_signal.get("confidence") if last_signal else None

    # Run classification
    classification = heuristic_classify(
        query=str(query),
        accumulated_content=accumulated,
        turns_without_signal=turns_without,
        tool_results=tool_results,
        last_signal_confidence=last_confidence,
    )

    # Store result
    bb_set(bb, "fallback_classification", {
        "action": classification.action.value,
        "confidence": classification.confidence,
        "hint": classification.hint,
        "reason": classification.reason,
    })

    logger.info(f"Fallback triggered: {classification.action.value} ({classification.reason})")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def apply_heuristic_classification(ctx: "TickContext") -> RunStatus:
    """Apply the fallback classification action."""
    from src.services.fallback_classifier import FallbackAction

    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    classification = bb_get(bb, "fallback_classification")
    if not classification:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    action = classification.get("action", "continue")
    hint = classification.get("hint")
    messages = bb_get(bb, "messages") or []

    if action == FallbackAction.CONTINUE.value:
        logger.debug("Fallback action: continue")

    elif action == FallbackAction.FORCE_RESPONSE.value:
        messages.append({
            "role": "system",
            "content": "[System] You have gathered sufficient information. Please provide your final response now."
        })
        bb_set(bb, "messages", messages)
        logger.info("Fallback action: force_response")

    elif action == FallbackAction.RETRY_WITH_HINT.value:
        if hint:
            messages.append({
                "role": "system",
                "content": f"[System Guidance] {hint}"
            })
            bb_set(bb, "messages", messages)
        logger.info(f"Fallback action: retry_with_hint - {hint}")

    elif action == FallbackAction.ESCALATE.value:
        # Emit to frontend
        _add_pending_chunk(bb, {
            "type": "system",
            "content": "The agent is having difficulty completing this request. Consider rephrasing or breaking down your question.",
            "severity": "warning"
        })
        logger.warning("Fallback action: escalate")

    # Clear classification
    bb_set(bb, "fallback_classification", None)

    ctx.mark_progress()
    return RunStatus.SUCCESS


def inject_fallback_hint(ctx: "TickContext") -> RunStatus:
    """Inject fallback hint into conversation messages."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    classification = bb_get(bb, "fallback_classification")
    if not classification or not classification.get("hint"):
        ctx.mark_progress()
        return RunStatus.SUCCESS

    messages = bb_get(bb, "messages") or []
    messages.append({
        "role": "system",
        "content": f"[Fallback Hint] {classification['hint']}"
    })
    bb_set(bb, "messages", messages)

    ctx.mark_progress()
    return RunStatus.SUCCESS
```

### Acceptance Criteria Mapping

- **US5.3**: Given explicit `stuck` signal, BERT attempts to identify alternative strategy
- **US5.4**: System functions with heuristic defaults when BERT unavailable

---

## T045: Update oracle-agent.lua with fallback selector

**File**: `backend/src/bt/trees/oracle-agent.lua`

### Location

Insert fallback selector after signal check in the agent loop (Phase 4), specifically after the LLM response processing but before the final response check.

### Core Algorithm

1. **Add condition check**: After LLM call, before tool execution decision
2. **Check needs_fallback**: Use condition registered from fallback.py
3. **If fallback needed**: Execute trigger_fallback, then apply_heuristic_classification
4. **Continue normal flow**: Fallback is advisory, doesn't break the loop

### Key Code Snippet (Lua DSL)

```lua
-- In agent-turn subtree, after LLM call and before tool handling:

-- Check for fallback after each LLM response
BT.always_succeed(
    BT.sequence({
        BT.condition("needs-fallback", {
            fn = "backend.src.bt.conditions.fallback.needs_fallback",
            description = "Check if fallback should activate (no signal 3+ turns, low confidence, or stuck)"
        }),
        BT.action("trigger-fallback", {
            fn = "backend.src.bt.actions.fallback_actions.trigger_fallback",
            description = "Run heuristic classification"
        }),
        BT.action("apply-fallback", {
            fn = "backend.src.bt.actions.fallback_actions.apply_heuristic_classification",
            description = "Apply fallback action (inject hint, force response, or escalate)"
        })
    })
),
```

### Integration Point in oracle-agent.lua

Add after line ~394 (after `accumulate-content` action) in the `agent-turn` subtree:

```lua
-- No tool calls - just accumulate content
BT.action("accumulate-content", {
    fn = "backend.src.bt.actions.oracle.accumulate_content",
    description = "Add LLM response to accumulated_content"
}),

-- NEW: Fallback check after content accumulation
BT.always_succeed(
    BT.selector({
        BT.sequence({
            BT.condition("needs-fallback", {
                fn = "backend.src.bt.conditions.fallback.needs_fallback"
            }),
            BT.action("trigger-fallback", {
                fn = "backend.src.bt.actions.fallback_actions.trigger_fallback"
            }),
            BT.action("apply-fallback", {
                fn = "backend.src.bt.actions.fallback_actions.apply_heuristic_classification"
            })
        }),
        BT.action("noop", { fn = "backend.src.bt.actions.oracle.noop" })
    })
)
```

### Acceptance Criteria Mapping

- **US5.1**: Given agent response with no signal, when 3 turns pass, BERT fallback activates
- **US5.2**: Given signal with confidence < 0.3, BERT fallback is consulted

---

## T046: Add fallback logging

**File**: `backend/src/services/fallback_classifier.py` (extend) and `backend/src/bt/actions/fallback_actions.py`

### Function Signature

```python
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger("fallback")

def log_fallback_trigger(
    reason: str,
    turns_without_signal: int,
    last_signal_type: Optional[str],
    last_signal_confidence: Optional[float],
    classification_action: str,
    classification_confidence: float,
) -> None:
    """
    Log fallback activation for debugging and audit.

    Logs to both Python logger and ANS event bus.
    """
```

### Core Algorithm

1. **Format log entry**: Include timestamp, reason, signal state, classification
2. **Log to Python logger**: At INFO level for normal, WARNING for escalate
3. **Emit ANS event**: Type `fallback.triggered` with full payload
4. **Include metrics**: turns_without_signal, classification confidence

### Key Code Snippet

```python
import logging
from datetime import datetime, timezone

logger = logging.getLogger("fallback")


def log_fallback_trigger(
    reason: str,
    turns_without_signal: int,
    last_signal_type: Optional[str],
    last_signal_confidence: Optional[float],
    classification_action: str,
    classification_confidence: float,
) -> None:
    """Log fallback activation for debugging and audit."""

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "turns_without_signal": turns_without_signal,
        "last_signal_type": last_signal_type,
        "last_signal_confidence": last_signal_confidence,
        "classification_action": classification_action,
        "classification_confidence": classification_confidence,
    }

    # Python logging
    if classification_action == "escalate":
        logger.warning(f"Fallback ESCALATE: {log_entry}")
    else:
        logger.info(f"Fallback triggered: {log_entry}")

    # ANS event
    try:
        from src.services.ans.bus import get_event_bus
        from src.services.ans.event import Event, Severity

        severity = Severity.WARNING if classification_action == "escalate" else Severity.INFO

        bus = get_event_bus()
        bus.emit(Event(
            type="fallback.triggered",
            source="fallback_classifier",
            severity=severity,
            payload=log_entry
        ))
    except Exception as e:
        logger.debug(f"Failed to emit fallback event: {e}")
```

### Update trigger_fallback to use logging

```python
def trigger_fallback(ctx: "TickContext") -> RunStatus:
    """Trigger fallback classification and store result."""
    # ... existing code ...

    # After classification
    log_fallback_trigger(
        reason=classification.reason,
        turns_without_signal=turns_without,
        last_signal_type=last_signal.get("type") if last_signal else None,
        last_signal_confidence=last_confidence,
        classification_action=classification.action.value,
        classification_confidence=classification.confidence,
    )

    # ... rest of function ...
```

### Acceptance Criteria Mapping

- **FR-009**: BT logs all signals for debugging and audit
- **SC-009**: All signals are logged and auditable

---

## T047: Tests for fallback classifier

**File**: `backend/tests/unit/test_fallback_classifier.py`

### Test Cases

```python
import pytest
from src.services.fallback_classifier import (
    heuristic_classify,
    FallbackAction,
    FallbackClassification,
)


class TestHeuristicClassify:
    """Tests for heuristic fallback classification."""

    def test_high_tool_failure_rate_escalates(self):
        """High tool failure rate should trigger ESCALATE."""
        result = heuristic_classify(
            query="How do I fix this bug?",
            accumulated_content="",
            turns_without_signal=2,
            tool_results=[
                {"name": "search_code", "success": False},
                {"name": "read_file", "success": False},
                {"name": "get_repo_map", "success": False},
            ],
        )
        assert result.action == FallbackAction.ESCALATE
        assert result.confidence >= 0.6
        assert "failure" in result.reason.lower()

    def test_substantial_content_forces_response(self):
        """Substantial accumulated content should force response."""
        result = heuristic_classify(
            query="Explain the architecture",
            accumulated_content="A" * 600,  # > 500 chars
            turns_without_signal=2,
            tool_results=[{"name": "search_vault", "success": True}],
        )
        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.confidence >= 0.7

    def test_simple_query_no_tools_forces_response(self):
        """Simple query with no tool usage should force response."""
        result = heuristic_classify(
            query="What is Python?",
            accumulated_content="",
            turns_without_signal=1,
            tool_results=[],
        )
        assert result.action == FallbackAction.FORCE_RESPONSE
        assert result.hint is not None

    def test_no_progress_retries_with_hint(self):
        """No progress after many turns should retry with hint."""
        result = heuristic_classify(
            query="Find the authentication middleware implementation",
            accumulated_content="",
            turns_without_signal=4,
            tool_results=[
                {"name": "search_code", "success": False},
            ],
        )
        assert result.action == FallbackAction.RETRY_WITH_HINT
        assert result.hint is not None
        assert "search_code" in result.hint

    def test_low_confidence_signal_considered(self):
        """Low confidence signal should affect classification."""
        result = heuristic_classify(
            query="Complex multi-part question",
            accumulated_content="Partial answer...",
            turns_without_signal=1,
            tool_results=[],
            last_signal_confidence=0.2,
        )
        # Low confidence doesn't directly change action but is passed through
        assert result.confidence <= 0.7

    def test_default_continues(self):
        """Normal state should continue."""
        result = heuristic_classify(
            query="How does the database work?",
            accumulated_content="The database uses SQLite...",
            turns_without_signal=0,
            tool_results=[{"name": "search_code", "success": True}],
        )
        assert result.action == FallbackAction.CONTINUE

    def test_classification_has_required_fields(self):
        """Classification should have all required fields."""
        result = heuristic_classify(
            query="Test query",
            accumulated_content="",
            turns_without_signal=0,
            tool_results=[],
        )
        assert isinstance(result, FallbackClassification)
        assert isinstance(result.action, FallbackAction)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.reason, str)
```

### Acceptance Criteria Mapping

- **US5.1-5.4**: All fallback trigger scenarios tested
- **SC-006**: Heuristic-only mode validation

---

## T048: Tests for fallback conditions

**File**: `backend/tests/unit/bt/test_fallback_conditions.py`

### Test Cases

```python
import pytest
from unittest.mock import MagicMock, patch

from src.bt.conditions.fallback import (
    needs_fallback,
    no_signal_for_n_turns,
    signal_confidence_below,
    is_stuck_signal,
    TURNS_WITHOUT_SIGNAL_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
)


@pytest.fixture
def mock_context():
    """Create mock TickContext with blackboard."""
    ctx = MagicMock()
    ctx.blackboard = MagicMock()
    ctx.blackboard._data = {}
    return ctx


def set_bb(ctx, key, value):
    """Helper to set blackboard value."""
    ctx.blackboard._data[key] = value


class TestNeedsFallback:
    """Tests for the composite needs_fallback condition."""

    def test_triggers_on_turns_without_signal(self, mock_context):
        """Should trigger when turns_without_signal >= 3."""
        set_bb(mock_context, "turns_without_signal", 3)
        set_bb(mock_context, "last_signal", None)

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(mock_context) is True

    def test_triggers_on_low_confidence(self, mock_context):
        """Should trigger when signal confidence < 0.3."""
        set_bb(mock_context, "turns_without_signal", 0)
        set_bb(mock_context, "last_signal", {"type": "need_turn", "confidence": 0.2})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(mock_context) is True

    def test_triggers_on_stuck_signal(self, mock_context):
        """Should trigger when last signal is 'stuck'."""
        set_bb(mock_context, "turns_without_signal", 0)
        set_bb(mock_context, "last_signal", {"type": "stuck", "confidence": 0.9})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(mock_context) is True

    def test_no_trigger_on_normal_state(self, mock_context):
        """Should not trigger on normal state."""
        set_bb(mock_context, "turns_without_signal", 1)
        set_bb(mock_context, "last_signal", {"type": "need_turn", "confidence": 0.8})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(mock_context) is False

    def test_no_trigger_without_blackboard(self):
        """Should return False if no blackboard."""
        ctx = MagicMock()
        ctx.blackboard = None
        assert needs_fallback(ctx) is False


class TestNoSignalForNTurns:
    """Tests for the no_signal_for_n_turns condition."""

    def test_true_when_at_threshold(self, mock_context):
        """Returns True when exactly at threshold."""
        set_bb(mock_context, "turns_without_signal", 3)

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert no_signal_for_n_turns(mock_context, n=3) is True

    def test_true_when_above_threshold(self, mock_context):
        """Returns True when above threshold."""
        set_bb(mock_context, "turns_without_signal", 5)

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert no_signal_for_n_turns(mock_context, n=3) is True

    def test_false_when_below_threshold(self, mock_context):
        """Returns False when below threshold."""
        set_bb(mock_context, "turns_without_signal", 2)

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert no_signal_for_n_turns(mock_context, n=3) is False


class TestSignalConfidenceBelow:
    """Tests for the signal_confidence_below condition."""

    def test_true_when_below(self, mock_context):
        """Returns True when confidence below threshold."""
        set_bb(mock_context, "last_signal", {"confidence": 0.2})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert signal_confidence_below(mock_context, threshold=0.3) is True

    def test_false_when_above(self, mock_context):
        """Returns False when confidence above threshold."""
        set_bb(mock_context, "last_signal", {"confidence": 0.5})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert signal_confidence_below(mock_context, threshold=0.3) is False

    def test_false_when_no_signal(self, mock_context):
        """Returns False when no last signal."""
        set_bb(mock_context, "last_signal", None)

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert signal_confidence_below(mock_context) is False


class TestIsStuckSignal:
    """Tests for the is_stuck_signal condition."""

    def test_true_when_stuck(self, mock_context):
        """Returns True when last signal is stuck."""
        set_bb(mock_context, "last_signal", {"type": "stuck"})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert is_stuck_signal(mock_context) is True

    def test_false_when_other_type(self, mock_context):
        """Returns False for other signal types."""
        set_bb(mock_context, "last_signal", {"type": "need_turn"})

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert is_stuck_signal(mock_context) is False
```

### Acceptance Criteria Mapping

- **FR-019**: Fallback activates when no signal for 3+ turns
- **FR-020**: Fallback activates when signal confidence < 0.3
- **FR-021**: Fallback activates on explicit `stuck` signal

---

## T049: Integration test for 3 turns without signals triggering fallback

**File**: `backend/tests/integration/test_oracle_bt_integration.py` (extend)

### Test Case

```python
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from src.bt.wrappers.oracle_wrapper import OracleBTWrapper, OracleStreamChunk
from src.bt.state.blackboard import TypedBlackboard
from src.bt.core.context import TickContext


class TestFallbackIntegration:
    """Integration tests for fallback triggering."""

    @pytest.mark.asyncio
    async def test_three_turns_without_signals_triggers_fallback(self):
        """Verify 3 turns without signals triggers fallback classification."""

        # Create wrapper
        wrapper = OracleBTWrapper(
            user_id="test-user",
            project_id="test-project",
        )

        # Mock the tree to simulate 3 turns without signals
        mock_bb = TypedBlackboard(scope_name="test")

        # Simulate state after 3 turns without signal
        mock_bb._data["turns_without_signal"] = 3
        mock_bb._data["last_signal"] = None
        mock_bb._data["accumulated_content"] = ""
        mock_bb._data["tool_results"] = []
        mock_bb._data["query"] = "Test question"
        mock_bb._data["messages"] = []
        mock_bb._data["_pending_chunks"] = []

        # Create context
        ctx = TickContext(blackboard=mock_bb, tick_budget=100)

        # Import and test the condition
        from src.bt.conditions.fallback import needs_fallback

        # Patch bb_get to use our mock blackboard
        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(ctx) is True

        # Test the action
        from src.bt.actions.fallback_actions import trigger_fallback

        with patch("src.bt.actions.fallback_actions.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            with patch("src.bt.actions.fallback_actions.bb_set",
                       side_effect=lambda bb, k, v: bb._data.__setitem__(k, v)):
                result = trigger_fallback(ctx)

        # Verify fallback was triggered
        assert "fallback_classification" in mock_bb._data
        classification = mock_bb._data["fallback_classification"]
        assert classification is not None
        assert classification["action"] in ["continue", "force_response", "retry_with_hint", "escalate"]

    @pytest.mark.asyncio
    async def test_stuck_signal_triggers_immediate_fallback(self):
        """Verify stuck signal triggers fallback even on first occurrence."""

        mock_bb = TypedBlackboard(scope_name="test")
        mock_bb._data["turns_without_signal"] = 0  # Recent signal
        mock_bb._data["last_signal"] = {
            "type": "stuck",
            "confidence": 0.9,
            "attempted": ["search_code", "read_file"],
            "blocker": "Cannot find file"
        }

        ctx = TickContext(blackboard=mock_bb, tick_budget=100)

        from src.bt.conditions.fallback import needs_fallback

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(ctx) is True

    @pytest.mark.asyncio
    async def test_low_confidence_signal_triggers_fallback(self):
        """Verify low confidence signal triggers fallback."""

        mock_bb = TypedBlackboard(scope_name="test")
        mock_bb._data["turns_without_signal"] = 0
        mock_bb._data["last_signal"] = {
            "type": "need_turn",
            "confidence": 0.25,  # Below 0.3 threshold
            "reason": "Not sure what to do"
        }

        ctx = TickContext(blackboard=mock_bb, tick_budget=100)

        from src.bt.conditions.fallback import needs_fallback

        with patch("src.bt.conditions.fallback.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            assert needs_fallback(ctx) is True

    @pytest.mark.asyncio
    async def test_fallback_injects_system_message(self):
        """Verify fallback actions inject appropriate system messages."""

        mock_bb = TypedBlackboard(scope_name="test")
        mock_bb._data["fallback_classification"] = {
            "action": "force_response",
            "confidence": 0.8,
            "hint": None,
            "reason": "Test"
        }
        mock_bb._data["messages"] = []
        mock_bb._data["_pending_chunks"] = []

        ctx = TickContext(blackboard=mock_bb, tick_budget=100)

        from src.bt.actions.fallback_actions import apply_heuristic_classification

        with patch("src.bt.actions.fallback_actions.bb_get",
                   side_effect=lambda bb, k: bb._data.get(k)):
            with patch("src.bt.actions.fallback_actions.bb_set",
                       side_effect=lambda bb, k, v: bb._data.__setitem__(k, v)):
                result = apply_heuristic_classification(ctx)

        # Verify system message was injected
        messages = mock_bb._data["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert "final response" in messages[0]["content"].lower()
```

### Acceptance Criteria Mapping

- **US5.1**: Given agent response with no signal, when 3 turns pass, BERT fallback activates
- **US5.2**: Given signal confidence < 0.3, BERT fallback is consulted
- **US5.3**: Given explicit `stuck` signal, BERT attempts alternative strategy
- **Independent Test**: 3 turns without signals triggers fallback classification

---

## Summary

| Task | File | Key Function | Tests |
|------|------|--------------|-------|
| T042 | `fallback_classifier.py` | `heuristic_classify()` | T047 |
| T043 | `conditions/fallback.py` | `needs_fallback()` | T048 |
| T044 | `actions/fallback_actions.py` | `trigger_fallback()`, `apply_heuristic_classification()` | T048, T049 |
| T045 | `trees/oracle-agent.lua` | Fallback selector integration | T049 |
| T046 | Multiple | `log_fallback_trigger()` | Implicit in T047, T049 |
| T047 | `test_fallback_classifier.py` | Unit tests for classifier | - |
| T048 | `test_fallback_conditions.py` | Unit tests for conditions | - |
| T049 | `test_oracle_bt_integration.py` | Integration test | - |

### Dependencies

```
T042 (classifier) ──┬──> T044 (actions) ──> T045 (lua update)
                    │                           │
T043 (conditions) ──┘                           │
                                                v
T046 (logging) ─────────────────────────────────┘
                                                │
                                                v
                              T047, T048, T049 (tests)
```
