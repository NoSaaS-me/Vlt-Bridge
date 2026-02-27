# Expanded Tasks: User Story 3 - Budget and Loop Enforcement

**Feature**: 020-bt-oracle-agent
**User Story**: US3 - Budget and Loop Enforcement (Priority: P2)
**Goal**: BT enforces turn budgets and detects infinite loops from signal patterns

**Dependencies**: US2 (needs signals for loop detection), Phase 2 (Foundational)

---

## T027: Create budget.py conditions

**File**: `backend/src/bt/conditions/budget.py`

### Function Signatures

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.base import RunStatus

def turns_remaining(ctx: "TickContext") -> "RunStatus":
    """Check how many turns remain in budget. Returns SUCCESS if turns > 0."""
    ...

def is_at_budget_limit(ctx: "TickContext") -> "RunStatus":
    """Returns SUCCESS if turn == max_turns - 1 (last turn)."""
    ...

def is_over_budget(ctx: "TickContext") -> "RunStatus":
    """Returns SUCCESS if turn >= max_turns."""
    ...
```

### Core Algorithm

1. Import `get_oracle_config()` from config to get `ORACLE_MAX_TURNS`
2. Get `turn` from blackboard using `bb_get(bb, "turn", 0)`
3. Get `max_turns` from config (default 30)
4. For `turns_remaining`: return SUCCESS if `turn < max_turns`, else FAILURE
5. For `is_at_budget_limit`: return SUCCESS if `turn == max_turns - 1`, else FAILURE
6. For `is_over_budget`: return SUCCESS if `turn >= max_turns`, else FAILURE
7. Log condition evaluation for debugging

### Key Code Snippet

```python
"""Budget conditions for BT-controlled Oracle agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..state.base import RunStatus
from ...services.config import get_oracle_config

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


def _bb_get(bb: "TypedBlackboard", key: str, default=None):
    """Get value from blackboard without schema validation."""
    value = bb._lookup(key)
    return value if value is not None else default


def turns_remaining(ctx: "TickContext") -> RunStatus:
    """Check if turns remain in budget. SUCCESS if turn < max_turns."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    config = get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    remaining = max_turns - turn
    logger.debug(f"Budget check: {remaining} turns remaining ({turn}/{max_turns})")

    return RunStatus.SUCCESS if remaining > 0 else RunStatus.FAILURE


def is_at_budget_limit(ctx: "TickContext") -> RunStatus:
    """Check if at last turn. SUCCESS if turn == max_turns - 1."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    config = get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    at_limit = turn == max_turns - 1
    if at_limit:
        logger.warning(f"At budget limit: turn {turn} of {max_turns}")

    return RunStatus.SUCCESS if at_limit else RunStatus.FAILURE


def is_over_budget(ctx: "TickContext") -> RunStatus:
    """Check if over budget. SUCCESS if turn >= max_turns."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    config = get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    over = turn >= max_turns
    if over:
        logger.error(f"Over budget: turn {turn} >= max {max_turns}")

    return RunStatus.SUCCESS if over else RunStatus.FAILURE


__all__ = ["turns_remaining", "is_at_budget_limit", "is_over_budget"]
```

### Acceptance Criteria Mapping

| Criterion | Validation |
|-----------|------------|
| US3-AC1: Agent at 29/30 turns gets one more | `is_at_budget_limit` returns SUCCESS at turn 29 |
| US3-AC2: Agent at 30/30 turns forced completion | `is_over_budget` returns SUCCESS at turn 30 |
| FR-007: Configurable max turn limits | Uses `get_oracle_config().max_turns` |

---

## T028: Create loop_detection.py conditions

**File**: `backend/src/bt/conditions/loop_detection.py`

### Function Signatures

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.base import RunStatus

def is_stuck_loop(ctx: "TickContext") -> "RunStatus":
    """Check if agent is stuck in a loop. Returns SUCCESS if consecutive_same_reason >= 3."""
    ...

def has_repeated_signal(ctx: "TickContext") -> "RunStatus":
    """Check if last signal reason matches previous. Returns SUCCESS if match."""
    ...
```

### Core Algorithm

1. Get `last_signal` and `consecutive_same_reason` from blackboard
2. Get `signals_emitted` list for pattern analysis
3. For `is_stuck_loop`:
   - Return SUCCESS if `consecutive_same_reason >= 3`
   - Check for `need_turn` signals with same reason
4. For `has_repeated_signal`:
   - Compare `last_signal.fields.reason` with previous signal's reason
   - Return SUCCESS if they match
5. Log loop detection events for debugging
6. Consider both signal-based and tool-pattern-based loop detection

### Key Code Snippet

```python
"""Loop detection conditions for BT-controlled Oracle agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)

# Threshold for considering agent stuck
CONSECUTIVE_SAME_REASON_THRESHOLD = 3


def _bb_get(bb: "TypedBlackboard", key: str, default=None):
    """Get value from blackboard without schema validation."""
    value = bb._lookup(key)
    return value if value is not None else default


def _get_signal_reason(signal: Optional[Dict[str, Any]]) -> Optional[str]:
    """Extract reason from signal fields."""
    if signal is None:
        return None
    fields = signal.get("fields", {})
    return fields.get("reason")


def is_stuck_loop(ctx: "TickContext") -> RunStatus:
    """
    Check if agent is stuck in a loop.

    Returns SUCCESS if:
    - consecutive_same_reason >= 3 (same need_turn reason repeated)
    - OR tool_pattern loop detected (from existing loop detection)
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Check signal-based loop (from US2 signal tracking)
    consecutive = _bb_get(bb, "consecutive_same_reason", 0)
    if consecutive >= CONSECUTIVE_SAME_REASON_THRESHOLD:
        logger.warning(
            f"Stuck loop detected: same signal reason {consecutive} times"
        )
        return RunStatus.SUCCESS

    # Check existing tool-pattern loop detection
    loop_detected = _bb_get(bb, "loop_detected", False)
    if loop_detected:
        logger.warning("Stuck loop detected: tool pattern repetition")
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


def has_repeated_signal(ctx: "TickContext") -> RunStatus:
    """
    Check if last signal reason matches previous signal's reason.

    Used to increment consecutive_same_reason counter.
    Returns SUCCESS if reasons match.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    signals_emitted = _bb_get(bb, "signals_emitted", [])
    if len(signals_emitted) < 2:
        return RunStatus.FAILURE

    last_signal = signals_emitted[-1]
    prev_signal = signals_emitted[-2]

    # Only compare need_turn signals
    if last_signal.get("type") != "need_turn":
        return RunStatus.FAILURE
    if prev_signal.get("type") != "need_turn":
        return RunStatus.FAILURE

    last_reason = _get_signal_reason(last_signal)
    prev_reason = _get_signal_reason(prev_signal)

    if last_reason and last_reason == prev_reason:
        logger.debug(f"Repeated signal reason: {last_reason}")
        return RunStatus.SUCCESS

    return RunStatus.FAILURE


__all__ = ["is_stuck_loop", "has_repeated_signal", "CONSECUTIVE_SAME_REASON_THRESHOLD"]
```

### Acceptance Criteria Mapping

| Criterion | Validation |
|-----------|------------|
| US3-AC3: 3 consecutive same reason triggers stuck | `is_stuck_loop` returns SUCCESS when `consecutive_same_reason >= 3` |
| FR-008: Detect loop patterns | Checks both signal reasons and tool patterns |
| FR-009: Log all signals for debugging | Logger calls on detection |

---

## T029: Create budget_actions.py

**File**: `backend/src/bt/actions/budget_actions.py`

### Function Signatures

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.base import RunStatus

def force_completion(ctx: "TickContext") -> "RunStatus":
    """Force agent to complete with current accumulated content."""
    ...

def emit_budget_warning(ctx: "TickContext") -> "RunStatus":
    """Emit ANS budget warning event at threshold."""
    ...

def emit_budget_exceeded(ctx: "TickContext") -> "RunStatus":
    """Emit ANS budget exceeded event."""
    ...
```

### Core Algorithm

1. **force_completion**:
   - Get accumulated_content from blackboard
   - Append "[Response truncated due to budget limit]" if content exists
   - Set `force_complete` flag on blackboard
   - Emit `budget.iteration.exceeded` ANS event
   - Return SUCCESS to signal completion

2. **emit_budget_warning**:
   - Check if warning already emitted (`iteration_warning_emitted`)
   - Get current turn and max_turns
   - Emit `budget.iteration.warning` ANS event with payload
   - Set `iteration_warning_emitted = True`
   - Return SUCCESS

3. **emit_budget_exceeded**:
   - Emit `budget.iteration.exceeded` ANS event
   - Include turn count and percentage in payload
   - Return SUCCESS

### Key Code Snippet

```python
"""Budget enforcement actions for BT-controlled Oracle agent."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..state.base import RunStatus
from ...services.config import get_oracle_config

if TYPE_CHECKING:
    from ..core.context import TickContext
    from ..state.blackboard import TypedBlackboard

logger = logging.getLogger(__name__)


def _bb_get(bb: "TypedBlackboard", key: str, default=None):
    """Get value from blackboard without schema validation."""
    value = bb._lookup(key)
    return value if value is not None else default


def _bb_set(bb: "TypedBlackboard", key: str, value) -> None:
    """Set value in blackboard without schema validation."""
    bb._data[key] = value
    bb._writes.add(key)


def force_completion(ctx: "TickContext") -> RunStatus:
    """
    Force agent to complete with current accumulated content.

    Called when budget is exceeded or stuck loop detected.
    Appends truncation notice and emits exceeded event.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    config = get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    # Append truncation notice
    accumulated = _bb_get(bb, "accumulated_content", "")
    if accumulated:
        accumulated += "\n\n[Response truncated due to budget limit]"
    else:
        accumulated = "[Unable to complete: budget limit reached]"
    _bb_set(bb, "accumulated_content", accumulated)

    # Set force complete flag
    _bb_set(bb, "force_complete", True)

    # Emit exceeded event
    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.exceeded",
            source="oracle_bt",
            severity=Severity.ERROR,
            payload={
                "turn": turn,
                "max_turns": max_turns,
                "percentage": (turn / max_turns * 100) if max_turns > 0 else 100,
                "forced": True
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit budget exceeded event: {e}")

    logger.warning(f"Forced completion at turn {turn}/{max_turns}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_budget_warning(ctx: "TickContext") -> RunStatus:
    """
    Emit ANS budget warning event at threshold (70% default).

    Only emits once per session.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Check if already emitted
    if _bb_get(bb, "iteration_warning_emitted", False):
        ctx.mark_progress()
        return RunStatus.SUCCESS

    config = get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns
    warning_threshold = config.iteration_warning_threshold

    # Check if at warning threshold
    threshold_turn = int(max_turns * warning_threshold)
    if turn < threshold_turn:
        ctx.mark_progress()
        return RunStatus.SUCCESS

    # Mark as emitted
    _bb_set(bb, "iteration_warning_emitted", True)

    # Emit warning event
    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.warning",
            source="oracle_bt",
            severity=Severity.WARNING,
            payload={
                "turn": turn,
                "max_turns": max_turns,
                "percentage": (turn / max_turns * 100) if max_turns > 0 else 0,
                "remaining": max_turns - turn
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit budget warning: {e}")

    logger.info(f"Budget warning: {turn}/{max_turns} turns used")
    ctx.mark_progress()
    return RunStatus.SUCCESS


def emit_budget_exceeded(ctx: "TickContext") -> RunStatus:
    """Emit ANS budget exceeded event."""
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    # Check if already emitted
    if _bb_get(bb, "iteration_exceeded_emitted", False):
        ctx.mark_progress()
        return RunStatus.SUCCESS

    config = get_oracle_config()
    turn = _bb_get(bb, "turn", 0)
    max_turns = config.max_turns

    # Mark as emitted
    _bb_set(bb, "iteration_exceeded_emitted", True)

    # Emit exceeded event
    try:
        from ...services.ans.bus import get_event_bus
        from ...services.ans.event import Event, Severity

        bus = get_event_bus()
        bus.emit(Event(
            type="budget.iteration.exceeded",
            source="oracle_bt",
            severity=Severity.ERROR,
            payload={
                "turn": turn,
                "max_turns": max_turns,
                "percentage": 100.0
            }
        ))
    except Exception as e:
        logger.warning(f"Failed to emit budget exceeded: {e}")

    logger.error(f"Budget exceeded: {turn}/{max_turns}")
    ctx.mark_progress()
    return RunStatus.SUCCESS


__all__ = ["force_completion", "emit_budget_warning", "emit_budget_exceeded"]
```

### Acceptance Criteria Mapping

| Criterion | Validation |
|-----------|------------|
| US3-AC2: Force completion with partial | `force_completion` appends truncation notice |
| US3-AC4: BERT fallback on stuck | `force_completion` can be called from fallback |
| FR-007: Enforce maximum turn limits | `force_completion` enforces limit |
| FR-009: Log all signals | Logger calls for budget events |

---

## T030: Update oracle-agent.lua with budget guards

**File**: `backend/src/bt/trees/oracle-agent.lua`

### Changes Required

1. Add budget condition checks at loop entry
2. Add loop detection checks after signal parsing
3. Add force_completion action for budget exceeded
4. Integrate with existing budget warning actions

### Core Algorithm

1. At start of agent loop, check `is_over_budget`
2. If over budget, call `force_completion` and exit loop
3. After each LLM response, check `is_stuck_loop`
4. If stuck, call `force_completion` and exit loop
5. At 70% threshold, call `emit_budget_warning`
6. At last turn (`is_at_budget_limit`), inject warning into prompt

### Key Code Snippet (additions to oracle-agent.lua)

```lua
-- Add to blackboard schema
blackboard = {
    -- ... existing fields ...

    -- Signal state (from US2)
    signals_emitted = "list",
    last_signal = "Signal",
    consecutive_same_reason = "int",
    turns_without_signal = "int",

    -- Budget enforcement
    force_complete = "bool",
    iteration_warning_emitted = "bool",
    iteration_exceeded_emitted = "bool",
},

-- Update agent loop with budget guards
BT.selector({
    -- Check if forced to complete
    BT.sequence({
        BT.condition("is-force-complete", {
            expression = "bb.force_complete == true"
        }),
        BT.action("emit-done", {
            fn = "backend.src.bt.actions.oracle.emit_done"
        })
    }),

    -- Check budget exceeded FIRST
    BT.sequence({
        BT.condition("is-over-budget", {
            fn = "backend.src.bt.conditions.budget.is_over_budget"
        }),
        BT.action("force-completion", {
            fn = "backend.src.bt.actions.budget_actions.force_completion",
            description = "Force completion when budget exceeded"
        }),
        BT.action("emit-done", {
            fn = "backend.src.bt.actions.oracle.emit_done"
        })
    }),

    -- Check stuck loop
    BT.sequence({
        BT.condition("is-stuck-loop", {
            fn = "backend.src.bt.conditions.loop_detection.is_stuck_loop"
        }),
        BT.action("force-completion", {
            fn = "backend.src.bt.actions.budget_actions.force_completion",
            description = "Force completion when stuck in loop"
        }),
        BT.action("emit-done", {
            fn = "backend.src.bt.actions.oracle.emit_done"
        })
    }),

    -- Normal agent loop continues...
    BT.retry(30, BT.selector({
        -- ... existing loop content ...

        -- Add budget warning check at start of each turn
        BT.sequence({
            BT.always_succeed(
                BT.action("emit-budget-warning", {
                    fn = "backend.src.bt.actions.budget_actions.emit_budget_warning",
                    description = "Emit warning at 70% of max turns"
                })
            ),

            -- ... rest of turn ...
        })
    }))
})
```

### Acceptance Criteria Mapping

| Criterion | Validation |
|-----------|------------|
| US3-AC1: One more turn at 29/30 | Loop continues until `is_over_budget` |
| US3-AC2: Forced completion at 30/30 | `is-over-budget` triggers `force-completion` |
| US3-AC3: Loop detection triggers fallback | `is-stuck-loop` triggers `force-completion` |
| FR-006: BT controls loop continue/terminate | Selector checks budget/loop conditions |

---

## T031: Add turn budget configuration

**File**: `backend/src/services/config.py`

### Function Signatures

```python
from pydantic import BaseModel, Field

class OracleConfig(BaseModel):
    """Oracle agent configuration."""

    max_turns: int = Field(default=30, ge=1, le=100)
    iteration_warning_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    token_warning_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    context_warning_threshold: float = Field(default=0.70, ge=0.0, le=1.0)

def get_oracle_config() -> OracleConfig:
    """Get Oracle agent configuration from environment."""
    ...
```

### Core Algorithm

1. Add `ORACLE_MAX_TURNS` environment variable (default 30)
2. Add `ORACLE_ITERATION_WARNING_THRESHOLD` environment variable (default 0.70)
3. Create `OracleConfig` Pydantic model for validation
4. Add `get_oracle_config()` function with caching
5. Validate constraints: 1 <= max_turns <= 100

### Key Code Snippet

```python
# Add to backend/src/services/config.py

class OracleConfig(BaseModel):
    """Oracle agent configuration loaded from environment."""

    model_config = ConfigDict(frozen=True)

    max_turns: int = Field(
        default=30,
        ge=1,
        le=100,
        description="Maximum agent turns per query (ORACLE_MAX_TURNS)"
    )
    iteration_warning_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Warn at this percentage of max turns"
    )
    token_warning_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Warn at this percentage of token budget"
    )
    context_warning_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Warn at this percentage of context window"
    )


@lru_cache(maxsize=1)
def get_oracle_config() -> OracleConfig:
    """Load and cache Oracle agent configuration."""
    max_turns_str = _read_env("ORACLE_MAX_TURNS", "30")
    try:
        max_turns = int(max_turns_str)
    except ValueError:
        max_turns = 30

    iteration_warning_str = _read_env("ORACLE_ITERATION_WARNING_THRESHOLD", "0.70")
    try:
        iteration_warning = float(iteration_warning_str)
    except ValueError:
        iteration_warning = 0.70

    token_warning_str = _read_env("ORACLE_TOKEN_WARNING_THRESHOLD", "0.80")
    try:
        token_warning = float(token_warning_str)
    except ValueError:
        token_warning = 0.80

    context_warning_str = _read_env("ORACLE_CONTEXT_WARNING_THRESHOLD", "0.70")
    try:
        context_warning = float(context_warning_str)
    except ValueError:
        context_warning = 0.70

    return OracleConfig(
        max_turns=max_turns,
        iteration_warning_threshold=iteration_warning,
        token_warning_threshold=token_warning,
        context_warning_threshold=context_warning,
    )


def reload_oracle_config() -> OracleConfig:
    """Clear cached Oracle config and reload."""
    get_oracle_config.cache_clear()
    return get_oracle_config()


# Update __all__
__all__ = [
    "AppConfig", "get_config", "reload_config",
    "OracleConfig", "get_oracle_config", "reload_oracle_config",
    "PROJECT_ROOT", "DEFAULT_VAULT_BASE"
]
```

### Acceptance Criteria Mapping

| Criterion | Validation |
|-----------|------------|
| FR-007: Configurable max turn limits (default 30) | `OracleConfig.max_turns` with env var |
| SC-004: Completes within turn budget 99% of time | Config provides enforcement threshold |

---

## T032: Tests for budget conditions

**File**: `backend/tests/unit/bt/test_budget_conditions.py`

### Test Cases

```python
"""Tests for budget conditions."""

import pytest
from unittest.mock import MagicMock, patch

from backend.src.bt.conditions.budget import (
    turns_remaining,
    is_at_budget_limit,
    is_over_budget,
)
from backend.src.bt.state.base import RunStatus


class MockBlackboard:
    """Mock blackboard for testing."""

    def __init__(self, data: dict = None):
        self._data = data or {}

    def _lookup(self, key: str):
        return self._data.get(key)


class MockContext:
    """Mock TickContext for testing."""

    def __init__(self, blackboard_data: dict = None):
        self.blackboard = MockBlackboard(blackboard_data)

    def mark_progress(self):
        pass


@pytest.fixture
def mock_config():
    """Mock oracle config with max_turns=30."""
    config = MagicMock()
    config.max_turns = 30
    config.iteration_warning_threshold = 0.70
    return config


class TestTurnsRemaining:
    """Tests for turns_remaining condition."""

    def test_returns_success_when_turns_available(self, mock_config):
        """Should return SUCCESS when turn < max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 10})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_turns_available(self, mock_config):
        """Should return FAILURE when turn >= max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 30})
            result = turns_remaining(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_when_over_max_turns(self, mock_config):
        """Should return FAILURE when turn > max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 35})
            result = turns_remaining(ctx)
            assert result == RunStatus.FAILURE

    def test_defaults_turn_to_zero(self, mock_config):
        """Should default turn to 0 if not set."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({})
            result = turns_remaining(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_blackboard(self, mock_config):
        """Should return FAILURE when blackboard is None."""
        ctx = MockContext()
        ctx.blackboard = None
        result = turns_remaining(ctx)
        assert result == RunStatus.FAILURE


class TestIsAtBudgetLimit:
    """Tests for is_at_budget_limit condition."""

    def test_returns_success_at_last_turn(self, mock_config):
        """Should return SUCCESS when turn == max_turns - 1."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 29})
            result = is_at_budget_limit(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_before_last_turn(self, mock_config):
        """Should return FAILURE when turn < max_turns - 1."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 28})
            result = is_at_budget_limit(ctx)
            assert result == RunStatus.FAILURE

    def test_returns_failure_after_last_turn(self, mock_config):
        """Should return FAILURE when turn >= max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 30})
            result = is_at_budget_limit(ctx)
            assert result == RunStatus.FAILURE


class TestIsOverBudget:
    """Tests for is_over_budget condition."""

    def test_returns_success_when_at_max(self, mock_config):
        """Should return SUCCESS when turn == max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 30})
            result = is_over_budget(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_success_when_over_max(self, mock_config):
        """Should return SUCCESS when turn > max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 35})
            result = is_over_budget(ctx)
            assert result == RunStatus.SUCCESS

    def test_returns_failure_when_under_max(self, mock_config):
        """Should return FAILURE when turn < max_turns."""
        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=mock_config):
            ctx = MockContext({"turn": 29})
            result = is_over_budget(ctx)
            assert result == RunStatus.FAILURE


class TestBudgetWithCustomMaxTurns:
    """Tests with custom max_turns configuration."""

    def test_respects_custom_max_turns(self):
        """Should use max_turns from config."""
        config = MagicMock()
        config.max_turns = 5

        with patch("backend.src.bt.conditions.budget.get_oracle_config", return_value=config):
            # Turn 4 is last turn when max=5
            ctx = MockContext({"turn": 4})
            assert is_at_budget_limit(ctx) == RunStatus.SUCCESS

            # Turn 5 is over budget when max=5
            ctx = MockContext({"turn": 5})
            assert is_over_budget(ctx) == RunStatus.SUCCESS
```

### Acceptance Criteria Mapping

| Test | Criterion |
|------|-----------|
| `test_returns_success_at_last_turn` | US3-AC1 |
| `test_returns_success_when_at_max` | US3-AC2 |
| `test_respects_custom_max_turns` | FR-007 |

---

## T033: Tests for loop detection

**File**: `backend/tests/unit/bt/test_loop_detection.py`

### Test Cases

```python
"""Tests for loop detection conditions."""

import pytest
from unittest.mock import MagicMock

from backend.src.bt.conditions.loop_detection import (
    is_stuck_loop,
    has_repeated_signal,
    CONSECUTIVE_SAME_REASON_THRESHOLD,
)
from backend.src.bt.state.base import RunStatus


class MockBlackboard:
    """Mock blackboard for testing."""

    def __init__(self, data: dict = None):
        self._data = data or {}

    def _lookup(self, key: str):
        return self._data.get(key)


class MockContext:
    """Mock TickContext for testing."""

    def __init__(self, blackboard_data: dict = None):
        self.blackboard = MockBlackboard(blackboard_data)

    def mark_progress(self):
        pass


class TestIsStuckLoop:
    """Tests for is_stuck_loop condition."""

    def test_returns_success_when_consecutive_same_reason_exceeds_threshold(self):
        """Should return SUCCESS when consecutive_same_reason >= 3."""
        ctx = MockContext({
            "consecutive_same_reason": 3,
            "loop_detected": False
        })
        result = is_stuck_loop(ctx)
        assert result == RunStatus.SUCCESS

    def test_returns_success_when_tool_loop_detected(self):
        """Should return SUCCESS when loop_detected is True."""
        ctx = MockContext({
            "consecutive_same_reason": 0,
            "loop_detected": True
        })
        result = is_stuck_loop(ctx)
        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_no_loop(self):
        """Should return FAILURE when no loop indicators."""
        ctx = MockContext({
            "consecutive_same_reason": 2,
            "loop_detected": False
        })
        result = is_stuck_loop(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_with_empty_blackboard(self):
        """Should return FAILURE with defaults."""
        ctx = MockContext({})
        result = is_stuck_loop(ctx)
        assert result == RunStatus.FAILURE

    def test_threshold_constant_is_3(self):
        """Threshold should be 3."""
        assert CONSECUTIVE_SAME_REASON_THRESHOLD == 3

    def test_returns_failure_when_no_blackboard(self):
        """Should return FAILURE when blackboard is None."""
        ctx = MockContext()
        ctx.blackboard = None
        result = is_stuck_loop(ctx)
        assert result == RunStatus.FAILURE


class TestHasRepeatedSignal:
    """Tests for has_repeated_signal condition."""

    def test_returns_success_when_reasons_match(self):
        """Should return SUCCESS when last two need_turn signals have same reason."""
        signals = [
            {"type": "need_turn", "fields": {"reason": "waiting for tool"}},
            {"type": "need_turn", "fields": {"reason": "waiting for tool"}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.SUCCESS

    def test_returns_failure_when_reasons_differ(self):
        """Should return FAILURE when reasons are different."""
        signals = [
            {"type": "need_turn", "fields": {"reason": "waiting for tool"}},
            {"type": "need_turn", "fields": {"reason": "analyzing results"}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_not_need_turn(self):
        """Should return FAILURE when signals are not need_turn type."""
        signals = [
            {"type": "context_sufficient", "fields": {"sources_found": 3}},
            {"type": "context_sufficient", "fields": {"sources_found": 3}},
        ]
        ctx = MockContext({"signals_emitted": signals})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_less_than_two_signals(self):
        """Should return FAILURE when fewer than 2 signals."""
        ctx = MockContext({"signals_emitted": [
            {"type": "need_turn", "fields": {"reason": "test"}}
        ]})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE

    def test_returns_failure_when_no_signals(self):
        """Should return FAILURE when no signals emitted."""
        ctx = MockContext({"signals_emitted": []})
        result = has_repeated_signal(ctx)
        assert result == RunStatus.FAILURE


class TestLoopDetectionIntegration:
    """Integration tests for loop detection."""

    def test_three_consecutive_same_reason_triggers_stuck(self):
        """3 consecutive same reason signals should trigger stuck."""
        signals = [
            {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
            {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
            {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
        ]
        ctx = MockContext({
            "signals_emitted": signals,
            "consecutive_same_reason": 3,
            "loop_detected": False
        })
        assert is_stuck_loop(ctx) == RunStatus.SUCCESS

    def test_two_consecutive_does_not_trigger(self):
        """2 consecutive same reason signals should NOT trigger stuck."""
        signals = [
            {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
            {"type": "need_turn", "fields": {"reason": "retrying failed tool"}},
        ]
        ctx = MockContext({
            "signals_emitted": signals,
            "consecutive_same_reason": 2,
            "loop_detected": False
        })
        assert is_stuck_loop(ctx) == RunStatus.FAILURE
```

### Acceptance Criteria Mapping

| Test | Criterion |
|------|-----------|
| `test_returns_success_when_consecutive_same_reason_exceeds_threshold` | US3-AC3 |
| `test_three_consecutive_same_reason_triggers_stuck` | FR-008 |
| `test_returns_success_when_tool_loop_detected` | Existing loop detection |

---

## T034: Integration test for max_turns=5 forcing completion

**File**: `backend/tests/integration/test_oracle_bt_integration.py` (add to existing)

### Test Cases

```python
"""Integration tests for budget enforcement."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

from backend.src.bt.state.blackboard import TypedBlackboard
from backend.src.bt.core.context import TickContext


class TestBudgetEnforcementIntegration:
    """Integration tests for budget enforcement in oracle agent."""

    @pytest.fixture
    def mock_oracle_config(self):
        """Oracle config with max_turns=5 for testing."""
        config = MagicMock()
        config.max_turns = 5
        config.iteration_warning_threshold = 0.70
        config.token_warning_threshold = 0.80
        config.context_warning_threshold = 0.70
        return config

    @pytest.fixture
    def blackboard_at_turn_5(self):
        """Blackboard with turn=5 (at max_turns limit)."""
        bb = TypedBlackboard(scope_name="test")
        bb._data["turn"] = 5
        bb._data["accumulated_content"] = "Partial response so far..."
        bb._data["messages"] = [{"role": "user", "content": "test"}]
        bb._data["tool_calls"] = []
        return bb

    @pytest.fixture
    def blackboard_at_turn_4(self):
        """Blackboard with turn=4 (at last turn when max=5)."""
        bb = TypedBlackboard(scope_name="test")
        bb._data["turn"] = 4
        bb._data["accumulated_content"] = ""
        bb._data["messages"] = [{"role": "user", "content": "test"}]
        bb._data["iteration_warning_emitted"] = False
        return bb

    def test_max_turns_5_forces_completion_at_turn_5(
        self, mock_oracle_config, blackboard_at_turn_5
    ):
        """Agent should be forced to complete at turn 5 when max_turns=5."""
        from backend.src.bt.conditions.budget import is_over_budget
        from backend.src.bt.actions.budget_actions import force_completion

        with patch(
            "backend.src.bt.conditions.budget.get_oracle_config",
            return_value=mock_oracle_config
        ), patch(
            "backend.src.bt.actions.budget_actions.get_oracle_config",
            return_value=mock_oracle_config
        ):
            ctx = MagicMock()
            ctx.blackboard = blackboard_at_turn_5
            ctx.mark_progress = MagicMock()

            # Should detect over budget
            from backend.src.bt.state.base import RunStatus
            result = is_over_budget(ctx)
            assert result == RunStatus.SUCCESS

            # Should force completion
            result = force_completion(ctx)
            assert result == RunStatus.SUCCESS

            # Should have truncation notice
            accumulated = ctx.blackboard._data["accumulated_content"]
            assert "[Response truncated" in accumulated

            # Should have force_complete flag
            assert ctx.blackboard._data.get("force_complete") is True

    def test_max_turns_5_allows_turn_4(
        self, mock_oracle_config, blackboard_at_turn_4
    ):
        """Agent should be allowed one more turn at turn 4 when max_turns=5."""
        from backend.src.bt.conditions.budget import (
            is_over_budget,
            is_at_budget_limit,
            turns_remaining
        )

        with patch(
            "backend.src.bt.conditions.budget.get_oracle_config",
            return_value=mock_oracle_config
        ):
            ctx = MagicMock()
            ctx.blackboard = blackboard_at_turn_4

            from backend.src.bt.state.base import RunStatus

            # Should NOT be over budget
            assert is_over_budget(ctx) == RunStatus.FAILURE

            # SHOULD be at budget limit (last turn)
            assert is_at_budget_limit(ctx) == RunStatus.SUCCESS

            # Should have turns remaining
            assert turns_remaining(ctx) == RunStatus.SUCCESS

    def test_budget_warning_emitted_at_70_percent(
        self, mock_oracle_config, blackboard_at_turn_4
    ):
        """Budget warning should be emitted at 70% (turn 3 when max=5)."""
        from backend.src.bt.actions.budget_actions import emit_budget_warning

        # Set turn to 3 (70% of 5 = 3.5, rounds down to 3)
        blackboard_at_turn_4._data["turn"] = 3

        with patch(
            "backend.src.bt.actions.budget_actions.get_oracle_config",
            return_value=mock_oracle_config
        ), patch(
            "backend.src.services.ans.bus.get_event_bus"
        ) as mock_bus:
            mock_bus.return_value = MagicMock()

            ctx = MagicMock()
            ctx.blackboard = blackboard_at_turn_4
            ctx.mark_progress = MagicMock()

            from backend.src.bt.state.base import RunStatus
            result = emit_budget_warning(ctx)
            assert result == RunStatus.SUCCESS

            # Should have marked warning as emitted
            assert ctx.blackboard._data.get("iteration_warning_emitted") is True

    def test_stuck_loop_forces_completion(self, mock_oracle_config):
        """Stuck loop should force completion even before max turns."""
        from backend.src.bt.conditions.loop_detection import is_stuck_loop
        from backend.src.bt.actions.budget_actions import force_completion

        bb = TypedBlackboard(scope_name="test")
        bb._data["turn"] = 2  # Only turn 2, but stuck
        bb._data["consecutive_same_reason"] = 3  # Stuck!
        bb._data["loop_detected"] = False
        bb._data["accumulated_content"] = "Working on it..."

        with patch(
            "backend.src.bt.actions.budget_actions.get_oracle_config",
            return_value=mock_oracle_config
        ):
            ctx = MagicMock()
            ctx.blackboard = bb
            ctx.mark_progress = MagicMock()

            from backend.src.bt.state.base import RunStatus

            # Should detect stuck loop
            assert is_stuck_loop(ctx) == RunStatus.SUCCESS

            # Should force completion
            result = force_completion(ctx)
            assert result == RunStatus.SUCCESS
            assert "[Response truncated" in bb._data["accumulated_content"]


class TestEndToEndBudgetEnforcement:
    """End-to-end tests simulating full agent loop with budget."""

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response that keeps requesting more turns."""
        response = MagicMock()
        response.content = "I need more information..."
        response.tool_calls = [
            {"id": "call_1", "function": {"name": "search", "arguments": "{}"}}
        ]
        return response

    def test_agent_stops_at_max_turns_even_with_tool_calls(
        self, mock_llm_response
    ):
        """Agent should stop at max turns even if still making tool calls."""
        from backend.src.bt.conditions.budget import is_over_budget

        config = MagicMock()
        config.max_turns = 5

        bb = TypedBlackboard(scope_name="test")
        bb._data["turn"] = 5
        bb._data["tool_calls"] = mock_llm_response.tool_calls
        bb._data["accumulated_content"] = "Still working..."

        with patch(
            "backend.src.bt.conditions.budget.get_oracle_config",
            return_value=config
        ):
            ctx = MagicMock()
            ctx.blackboard = bb

            from backend.src.bt.state.base import RunStatus

            # Even with pending tool calls, should detect over budget
            assert is_over_budget(ctx) == RunStatus.SUCCESS
            assert len(bb._data["tool_calls"]) > 0  # Tool calls still present
```

### Acceptance Criteria Mapping

| Test | Criterion |
|------|-----------|
| `test_max_turns_5_forces_completion_at_turn_5` | US3 Independent Test |
| `test_max_turns_5_allows_turn_4` | US3-AC1 |
| `test_budget_warning_emitted_at_70_percent` | FR-007 warning threshold |
| `test_stuck_loop_forces_completion` | US3-AC3, US3-AC4 |
| `test_agent_stops_at_max_turns_even_with_tool_calls` | US3-AC2 |

---

## Summary

| Task | Status | Files | Dependencies |
|------|--------|-------|--------------|
| T027 | Ready | `backend/src/bt/conditions/budget.py` | T031 |
| T028 | Ready | `backend/src/bt/conditions/loop_detection.py` | US2 (signals_emitted) |
| T029 | Ready | `backend/src/bt/actions/budget_actions.py` | T031 |
| T030 | Ready | `backend/src/bt/trees/oracle-agent.lua` | T027, T028, T029 |
| T031 | Ready | `backend/src/services/config.py` | None |
| T032 | Parallel | `backend/tests/unit/bt/test_budget_conditions.py` | T027 |
| T033 | Parallel | `backend/tests/unit/bt/test_loop_detection.py` | T028 |
| T034 | Last | `backend/tests/integration/test_oracle_bt_integration.py` | All US3 tasks |

**Recommended Order**: T031 -> T027 -> T028 -> T029 -> (T032, T033 parallel) -> T030 -> T034
