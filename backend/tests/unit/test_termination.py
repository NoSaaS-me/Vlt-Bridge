"""Tests for DefaultDecisionTree termination conditions and warnings.

T042: Test termination conditions
T043: Test soft warning thresholds (70%, 80%)
"""

import time
import pytest
from dataclasses import replace

from backend.src.models.agent_state import AgentState
from backend.src.models.settings import AgentConfig
from backend.src.services.decision_tree.default import DefaultDecisionTree


class TestTerminationConditions:
    """Test termination conditions (T042)."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        """Create test config with low limits for easy testing."""
        return AgentConfig(
            max_iterations=10,
            soft_warning_percent=70,
            token_budget=1000,
            token_warning_percent=80,
            timeout_seconds=60,
            loop_detection_window_seconds=60,
        )

    @pytest.fixture
    def decision_tree(self, config: AgentConfig) -> DefaultDecisionTree:
        """Create decision tree with test config."""
        return DefaultDecisionTree(config)

    @pytest.fixture
    def base_state(self, config: AgentConfig) -> AgentState:
        """Create base agent state for testing."""
        return AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
            turn=0,
            tokens_used=0,
            start_time=time.time(),
        )

    def test_should_continue_at_start(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should continue at start with no limits reached."""
        should_continue, reason = decision_tree.should_continue(base_state)
        assert should_continue is True
        assert reason == ""

    def test_terminates_at_max_iterations(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should terminate when max iterations reached."""
        state = replace(base_state, turn=10)  # At limit
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "max_iterations"

    def test_terminates_over_max_iterations(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should terminate when max iterations exceeded."""
        state = replace(base_state, turn=15)  # Over limit
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "max_iterations"

    def test_continues_below_max_iterations(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should continue when below max iterations."""
        state = replace(base_state, turn=9)  # Just below limit
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is True
        assert reason == ""

    def test_terminates_at_token_budget(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should terminate when token budget reached."""
        state = replace(base_state, tokens_used=1000)  # At limit
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "token_budget"

    def test_terminates_over_token_budget(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should terminate when token budget exceeded."""
        state = replace(base_state, tokens_used=1500)  # Over limit
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "token_budget"

    def test_continues_below_token_budget(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should continue when below token budget."""
        state = replace(base_state, tokens_used=999)  # Just below limit
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is True
        assert reason == ""

    def test_terminates_on_timeout(self, decision_tree: DefaultDecisionTree, config: AgentConfig):
        """Should terminate when timeout exceeded."""
        # Create state with old start_time (past timeout)
        state = AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
            turn=1,
            tokens_used=0,
            start_time=time.time() - 61,  # 61 seconds ago (timeout is 60)
        )
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "timeout"

    def test_continues_before_timeout(self, decision_tree: DefaultDecisionTree, config: AgentConfig):
        """Should continue when timeout not reached."""
        state = AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
            turn=1,
            tokens_used=0,
            start_time=time.time() - 30,  # 30 seconds ago (timeout is 60)
        )
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is True
        assert reason == ""

    def test_terminates_on_no_progress_time_based(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should terminate after loop_detection_window_seconds of identical actions."""
        # Simulate identical actions with loop timing that exceeds window
        action_sig = 'vault_read:{"path":"test.md"}'
        # Set loop_start_time to 61 seconds ago (window is 60 in test config)
        loop_start = time.time() - 61
        state = replace(
            base_state,
            recent_actions=(action_sig, action_sig),
            loop_start_time=loop_start
        )
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "no_progress"

    def test_continues_with_identical_actions_before_window(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should continue if loop hasn't exceeded window yet."""
        action_sig = 'vault_read:{"path":"test.md"}'
        # Set loop_start_time to 30 seconds ago (under 60 second window)
        loop_start = time.time() - 30
        state = replace(
            base_state,
            recent_actions=(action_sig, action_sig),
            loop_start_time=loop_start
        )
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is True
        assert reason == ""

    def test_continues_with_varied_actions(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should continue when recent actions differ (no loop_start_time)."""
        sig1 = 'vault_read:{"path":"test1.md"}'
        sig2 = 'vault_read:{"path":"test2.md"}'
        sig3 = 'vault_read:{"path":"test3.md"}'
        state = replace(base_state, recent_actions=(sig1, sig2, sig3))
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is True
        assert reason == ""

    def test_terminates_on_error_limit(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should terminate after 3 consecutive errors."""
        # Simulate 3 consecutive errors
        decision_tree.on_tool_result(base_state, {"error": "error1", "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"error": "error2", "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"error": "error3", "tool_name": "test"})

        should_continue, reason = decision_tree.should_continue(base_state)
        assert should_continue is False
        assert reason == "error_limit"

    def test_terminates_on_is_error_flag(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should count errors flagged with is_error."""
        # Simulate 3 consecutive errors using is_error flag
        decision_tree.on_tool_result(base_state, {"is_error": True, "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"is_error": True, "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"is_error": True, "tool_name": "test"})

        should_continue, reason = decision_tree.should_continue(base_state)
        assert should_continue is False
        assert reason == "error_limit"

    def test_does_not_terminate_on_terminal_state(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not continue if already terminal."""
        state = replace(base_state, termination_reason="user_cancelled")
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        assert reason == "user_cancelled"

    def test_error_count_resets_on_success(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Error count should reset after successful tool call."""
        # Simulate 2 errors
        decision_tree.on_tool_result(base_state, {"error": "error1", "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"error": "error2", "tool_name": "test"})

        # Then a success
        decision_tree.on_tool_result(base_state, {"result": "ok", "tool_name": "test"})

        # Should still continue (error count reset)
        should_continue, _ = decision_tree.should_continue(base_state)
        assert should_continue is True

    def test_error_count_resets_after_success_then_errors(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Error count should reset and re-accumulate correctly."""
        # 2 errors
        decision_tree.on_tool_result(base_state, {"error": "error1", "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"error": "error2", "tool_name": "test"})

        # Success resets count
        decision_tree.on_tool_result(base_state, {"result": "ok", "tool_name": "test"})

        # 2 more errors (should still continue - only 2 consecutive)
        decision_tree.on_tool_result(base_state, {"error": "error3", "tool_name": "test"})
        decision_tree.on_tool_result(base_state, {"error": "error4", "tool_name": "test"})

        should_continue, _ = decision_tree.should_continue(base_state)
        assert should_continue is True

        # 3rd error triggers termination
        decision_tree.on_tool_result(base_state, {"error": "error5", "tool_name": "test"})
        should_continue, reason = decision_tree.should_continue(base_state)
        assert should_continue is False
        assert reason == "error_limit"

    def test_priority_order_terminal_first(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Terminal state should be checked before other conditions."""
        # Create state that would fail multiple conditions
        state = replace(
            base_state,
            turn=15,  # Over max iterations
            tokens_used=2000,  # Over token budget
            termination_reason="user_cancelled"  # But already terminated
        )
        should_continue, reason = decision_tree.should_continue(state)
        assert should_continue is False
        # Should return the existing termination reason, not a new one
        assert reason == "user_cancelled"


class TestSoftWarningThresholds:
    """Test soft warning thresholds (T043)."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        """Create test config."""
        return AgentConfig(
            max_iterations=10,
            soft_warning_percent=70,  # Warn at 70%
            token_budget=1000,
            token_warning_percent=80,  # Warn at 80%
            timeout_seconds=120,
        )

    @pytest.fixture
    def decision_tree(self, config: AgentConfig) -> DefaultDecisionTree:
        """Create decision tree with test config."""
        return DefaultDecisionTree(config)

    @pytest.fixture
    def base_state(self, config: AgentConfig) -> AgentState:
        """Create base agent state for testing."""
        return AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
            turn=0,
            tokens_used=0,
            start_time=time.time(),
        )

    def test_iteration_warning_at_70_percent(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should emit iteration warning at 70% (7/10 turns)."""
        state = replace(base_state, turn=7)
        warnings = decision_tree.get_warning_state(state)

        assert "iteration" in warnings
        assert warnings["iteration"]["type"] == "limit_warning"
        assert warnings["iteration"]["current_value"] == 7
        assert warnings["iteration"]["limit_value"] == 10
        assert warnings["iteration"]["percent"] == 70.0

    def test_iteration_warning_over_70_percent(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should emit iteration warning above 70%."""
        state = replace(base_state, turn=8)
        warnings = decision_tree.get_warning_state(state)

        assert "iteration" in warnings
        assert warnings["iteration"]["percent"] == 80.0

    def test_token_warning_at_80_percent(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should emit token warning at 80% (800/1000 tokens)."""
        state = replace(base_state, tokens_used=800)
        warnings = decision_tree.get_warning_state(state)

        assert "token" in warnings
        assert warnings["token"]["type"] == "limit_warning"
        assert warnings["token"]["current_value"] == 800
        assert warnings["token"]["limit_value"] == 1000
        assert warnings["token"]["percent"] == 80.0

    def test_token_warning_over_80_percent(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should emit token warning above 80%."""
        state = replace(base_state, tokens_used=900)
        warnings = decision_tree.get_warning_state(state)

        assert "token" in warnings
        assert warnings["token"]["percent"] == 90.0

    def test_no_iteration_warning_below_threshold(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not emit iteration warning below 70%."""
        state = replace(base_state, turn=6)  # 60%
        warnings = decision_tree.get_warning_state(state)

        assert "iteration" not in warnings

    def test_no_token_warning_below_threshold(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not emit token warning below 80%."""
        state = replace(base_state, tokens_used=790)  # 79%
        warnings = decision_tree.get_warning_state(state)

        assert "token" not in warnings

    def test_no_warning_below_both_thresholds(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not emit warnings below thresholds."""
        state = replace(base_state, turn=5, tokens_used=500)  # 50% each
        warnings = decision_tree.get_warning_state(state)

        assert warnings == {}

    def test_warning_emitted_only_once_iteration(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Iteration warning should only be emitted once."""
        state = replace(base_state, turn=7)

        # First call - should emit warning
        warnings1 = decision_tree.get_warning_state(state)
        assert "iteration" in warnings1

        # Second call - should NOT emit warning again
        warnings2 = decision_tree.get_warning_state(state)
        assert "iteration" not in warnings2

    def test_warning_emitted_only_once_token(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Token warning should only be emitted once."""
        state = replace(base_state, tokens_used=800)

        # First call - should emit warning
        warnings1 = decision_tree.get_warning_state(state)
        assert "token" in warnings1

        # Second call - should NOT emit warning again
        warnings2 = decision_tree.get_warning_state(state)
        assert "token" not in warnings2

    def test_both_warnings_can_trigger(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Both iteration and token warnings can trigger together."""
        state = replace(base_state, turn=7, tokens_used=800)
        warnings = decision_tree.get_warning_state(state)

        assert "iteration" in warnings
        assert "token" in warnings

    def test_one_warning_emitted_other_can_still_trigger(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """After one warning emitted, other can still trigger later."""
        # First: trigger only iteration warning
        state1 = replace(base_state, turn=7, tokens_used=500)
        warnings1 = decision_tree.get_warning_state(state1)
        assert "iteration" in warnings1
        assert "token" not in warnings1

        # Later: token reaches threshold, should trigger
        state2 = replace(base_state, turn=8, tokens_used=800)
        warnings2 = decision_tree.get_warning_state(state2)
        assert "iteration" not in warnings2  # Already emitted
        assert "token" in warnings2  # New warning


class TestActionSignature:
    """Test action signature generation for no-progress detection."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig()

    @pytest.fixture
    def decision_tree(self, config: AgentConfig) -> DefaultDecisionTree:
        return DefaultDecisionTree(config)

    def test_same_call_produces_same_signature(self, decision_tree: DefaultDecisionTree):
        """Same tool call should produce identical signature."""
        sig1 = decision_tree._action_signature("vault_read", {"path": "test.md"})
        sig2 = decision_tree._action_signature("vault_read", {"path": "test.md"})
        assert sig1 == sig2

    def test_different_args_produce_different_signature(self, decision_tree: DefaultDecisionTree):
        """Different arguments should produce different signatures."""
        sig1 = decision_tree._action_signature("vault_read", {"path": "test1.md"})
        sig2 = decision_tree._action_signature("vault_read", {"path": "test2.md"})
        assert sig1 != sig2

    def test_different_tools_produce_different_signature(self, decision_tree: DefaultDecisionTree):
        """Different tools should produce different signatures."""
        sig1 = decision_tree._action_signature("vault_read", {"path": "test.md"})
        sig2 = decision_tree._action_signature("vault_write", {"path": "test.md"})
        assert sig1 != sig2

    def test_argument_order_normalized(self, decision_tree: DefaultDecisionTree):
        """Argument order should not affect signature (JSON sorted)."""
        sig1 = decision_tree._action_signature("search", {"query": "test", "limit": 10})
        sig2 = decision_tree._action_signature("search", {"limit": 10, "query": "test"})
        assert sig1 == sig2

    def test_empty_arguments(self, decision_tree: DefaultDecisionTree):
        """Empty arguments should produce valid signature."""
        sig = decision_tree._action_signature("list_notes", {})
        assert sig == "list_notes:{}"

    def test_nested_arguments(self, decision_tree: DefaultDecisionTree):
        """Nested arguments should be properly serialized."""
        sig1 = decision_tree._action_signature("complex", {"nested": {"a": 1, "b": 2}})
        sig2 = decision_tree._action_signature("complex", {"nested": {"b": 2, "a": 1}})
        assert sig1 == sig2  # Sort keys should normalize nested objects


class TestOnToolResult:
    """Test on_tool_result state updates."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig()

    @pytest.fixture
    def decision_tree(self, config: AgentConfig) -> DefaultDecisionTree:
        return DefaultDecisionTree(config)

    @pytest.fixture
    def base_state(self, config: AgentConfig) -> AgentState:
        return AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
        )

    def test_tracks_recent_actions(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should track actions in recent_actions."""
        result = {"tool_name": "vault_read", "arguments": {"path": "test.md"}}

        state1 = decision_tree.on_tool_result(base_state, result)
        assert len(state1.recent_actions) == 1

        state2 = decision_tree.on_tool_result(state1, result)
        assert len(state2.recent_actions) == 2

        state3 = decision_tree.on_tool_result(state2, result)
        assert len(state3.recent_actions) == 3

    def test_keeps_only_last_3_actions(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should keep only last 3 actions."""
        result1 = {"tool_name": "action1", "arguments": {}}
        result2 = {"tool_name": "action2", "arguments": {}}
        result3 = {"tool_name": "action3", "arguments": {}}
        result4 = {"tool_name": "action4", "arguments": {}}

        state1 = decision_tree.on_tool_result(base_state, result1)
        state2 = decision_tree.on_tool_result(state1, result2)
        state3 = decision_tree.on_tool_result(state2, result3)
        state4 = decision_tree.on_tool_result(state3, result4)

        assert len(state4.recent_actions) == 3
        # First action should be dropped, actions 2, 3, 4 remain
        assert "action1" not in state4.recent_actions[-3]
        assert "action4" in state4.recent_actions[-1]

    def test_returns_new_state_instance(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should return new state instance, not modify original."""
        result = {"tool_name": "vault_read", "arguments": {"path": "test.md"}}

        new_state = decision_tree.on_tool_result(base_state, result)

        assert new_state is not base_state
        assert base_state.recent_actions == ()  # Original unchanged
        assert len(new_state.recent_actions) == 1

    def test_handles_missing_tool_name(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should handle result without tool_name."""
        result = {"arguments": {"path": "test.md"}}

        state = decision_tree.on_tool_result(base_state, result)

        # Should use "unknown" as default
        assert "unknown:" in state.recent_actions[-1]

    def test_handles_missing_arguments(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should handle result without arguments."""
        result = {"tool_name": "list_notes"}

        state = decision_tree.on_tool_result(base_state, result)

        assert "list_notes:{}" in state.recent_actions[-1]


class TestDetectNoProgress:
    """Test _detect_no_progress method directly with time-based loop detection."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig(loop_detection_window_seconds=60)

    @pytest.fixture
    def decision_tree(self, config: AgentConfig) -> DefaultDecisionTree:
        return DefaultDecisionTree(config)

    @pytest.fixture
    def base_state(self, config: AgentConfig) -> AgentState:
        return AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
        )

    def test_no_progress_with_empty_actions(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not detect no-progress with empty actions."""
        state = replace(base_state, recent_actions=())
        assert decision_tree._detect_no_progress(state) is False

    def test_no_progress_with_one_action(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not detect no-progress with one action."""
        state = replace(base_state, recent_actions=("action1",))
        assert decision_tree._detect_no_progress(state) is False

    def test_no_progress_without_loop_timer(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not detect no-progress when loop_start_time is None."""
        state = replace(
            base_state,
            recent_actions=("action", "action"),
            loop_start_time=None
        )
        assert decision_tree._detect_no_progress(state) is False

    def test_no_progress_within_window(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not detect no-progress when loop is under window threshold."""
        state = replace(
            base_state,
            recent_actions=("action", "action"),
            loop_start_time=time.time() - 30  # 30 seconds, under 60 second window
        )
        assert decision_tree._detect_no_progress(state) is False

    def test_no_progress_exceeds_window(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should detect no-progress when loop exceeds window threshold."""
        state = replace(
            base_state,
            recent_actions=("action", "action"),
            loop_start_time=time.time() - 61  # 61 seconds, over 60 second window
        )
        assert decision_tree._detect_no_progress(state) is True

    def test_no_progress_at_window_boundary(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should detect no-progress at exactly the window threshold."""
        state = replace(
            base_state,
            recent_actions=("action", "action"),
            loop_start_time=time.time() - 60  # Exactly at 60 second window
        )
        assert decision_tree._detect_no_progress(state) is True

    def test_no_progress_with_varied_actions_no_loop(self, decision_tree: DefaultDecisionTree, base_state: AgentState):
        """Should not detect no-progress with different actions (no loop timer)."""
        state = replace(
            base_state,
            recent_actions=("a", "b", "c"),
            loop_start_time=None
        )
        assert decision_tree._detect_no_progress(state) is False


class TestToolTimeoutConfig:
    """Test tool_timeout_seconds configuration field."""

    def test_tool_timeout_default_value(self):
        """Tool timeout should default to 30 seconds."""
        config = AgentConfig()
        assert config.tool_timeout_seconds == 30

    def test_tool_timeout_custom_value(self):
        """Tool timeout should accept custom value within bounds."""
        config = AgentConfig(tool_timeout_seconds=60)
        assert config.tool_timeout_seconds == 60

    def test_tool_timeout_minimum_bound(self):
        """Tool timeout minimum should be 5 seconds."""
        config = AgentConfig(tool_timeout_seconds=5)
        assert config.tool_timeout_seconds == 5

        # Below minimum should raise validation error
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentConfig(tool_timeout_seconds=4)

    def test_tool_timeout_maximum_bound(self):
        """Tool timeout maximum should be 120 seconds."""
        config = AgentConfig(tool_timeout_seconds=120)
        assert config.tool_timeout_seconds == 120

        # Above maximum should raise validation error
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AgentConfig(tool_timeout_seconds=121)

    def test_tool_timeout_in_agent_state(self):
        """AgentState should include tool_timeout from config."""
        config = AgentConfig(tool_timeout_seconds=45)
        state = AgentState(
            user_id="test-user",
            project_id="test-project",
            config=config,
        )
        assert state.config.tool_timeout_seconds == 45


class TestWithinTurnDuplicateDetection:
    """Test check_within_turn_loops() for within-turn deduplication."""

    @pytest.fixture
    def config(self) -> AgentConfig:
        return AgentConfig()

    @pytest.fixture
    def decision_tree(self, config: AgentConfig) -> DefaultDecisionTree:
        return DefaultDecisionTree(config)

    def test_no_duplicates_returns_none(self, decision_tree: DefaultDecisionTree):
        """Should return None when no duplicates present."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_read", "arguments": {"path": "file2.md"}},
            {"name": "search_code", "arguments": {"query": "test"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)
        assert result is None

    def test_single_call_returns_none(self, decision_tree: DefaultDecisionTree):
        """Should return None for single tool call."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)
        assert result is None

    def test_empty_list_returns_none(self, decision_tree: DefaultDecisionTree):
        """Should return None for empty list."""
        result = decision_tree.check_within_turn_loops([])
        assert result is None

    def test_detects_duplicate_tool_calls(self, decision_tree: DefaultDecisionTree):
        """Should detect identical tool calls."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_read", "arguments": {"path": "file1.md"}},  # Duplicate
            {"name": "search_code", "arguments": {"query": "test"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 1
        assert result["total_calls"] == 3
        assert "vault_read" in result["duplicated_tools"]

    def test_detects_multiple_duplicates(self, decision_tree: DefaultDecisionTree):
        """Should count multiple duplicates correctly."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_read", "arguments": {"path": "file1.md"}},  # Dup 1
            {"name": "vault_read", "arguments": {"path": "file1.md"}},  # Dup 2
            {"name": "search_code", "arguments": {"query": "test"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 2
        assert result["total_calls"] == 4

    def test_duplicate_rate_calculation(self, decision_tree: DefaultDecisionTree):
        """Should calculate duplicate rate correctly."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_read", "arguments": {"path": "file1.md"}},  # 1
            {"name": "vault_read", "arguments": {"path": "file1.md"}},  # 2
            {"name": "vault_read", "arguments": {"path": "file1.md"}},  # 3
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 3
        assert result["total_calls"] == 4
        assert result["duplicate_rate"] == 75.0  # 3/4 = 75%

    def test_different_args_not_duplicate(self, decision_tree: DefaultDecisionTree):
        """Same tool with different args should not be duplicate."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_read", "arguments": {"path": "file2.md"}},
            {"name": "vault_read", "arguments": {"path": "file3.md"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)
        assert result is None

    def test_different_tools_same_args_not_duplicate(self, decision_tree: DefaultDecisionTree):
        """Different tools with same args should not be duplicate."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_write", "arguments": {"path": "file1.md"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)
        assert result is None

    def test_argument_order_normalized(self, decision_tree: DefaultDecisionTree):
        """Should detect duplicates regardless of argument order."""
        tool_calls = [
            {"name": "search", "arguments": {"query": "test", "limit": 10}},
            {"name": "search", "arguments": {"limit": 10, "query": "test"}},  # Same args, different order
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 1

    def test_warning_message_format(self, decision_tree: DefaultDecisionTree):
        """Warning message should contain relevant info."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
            {"name": "vault_read", "arguments": {"path": "file1.md"}},
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert "warning" in result
        assert "vault_read" in result["warning"]
        assert "1" in result["warning"]  # duplicate count
        assert "2" in result["warning"]  # total calls

    def test_handles_missing_name(self, decision_tree: DefaultDecisionTree):
        """Should handle tool calls with missing name."""
        tool_calls = [
            {"arguments": {"path": "file1.md"}},  # Missing name
            {"arguments": {"path": "file1.md"}},  # Same, should be duplicate
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 1
        assert "unknown" in result["duplicated_tools"]

    def test_handles_missing_arguments(self, decision_tree: DefaultDecisionTree):
        """Should handle tool calls with missing arguments."""
        tool_calls = [
            {"name": "list_notes"},  # Missing arguments
            {"name": "list_notes"},  # Same, should be duplicate
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 1
        assert "list_notes" in result["duplicated_tools"]

    def test_multiple_different_duplicates(self, decision_tree: DefaultDecisionTree):
        """Should track multiple different tools that are duplicated."""
        tool_calls = [
            {"name": "vault_read", "arguments": {"path": "a.md"}},
            {"name": "vault_read", "arguments": {"path": "a.md"}},  # Dup of vault_read
            {"name": "search_code", "arguments": {"query": "test"}},
            {"name": "search_code", "arguments": {"query": "test"}},  # Dup of search_code
        ]
        result = decision_tree.check_within_turn_loops(tool_calls)

        assert result is not None
        assert result["duplicate_count"] == 2
        assert "vault_read" in result["duplicated_tools"]
        assert "search_code" in result["duplicated_tools"]
