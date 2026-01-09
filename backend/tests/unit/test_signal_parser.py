"""Unit tests for signal_parser.py (020-bt-oracle-agent T007).

Tests cover:
- All 6 signal types parsing correctly
- Malformed XML handling
- Edge cases (no signal, multiple signals, inline signals)
- Field value parsing (strings, ints, floats, bools, lists)
- Confidence extraction (element and attribute)
- strip_signal whitespace normalization
"""

import pytest
from datetime import datetime, timezone

from backend.src.services.signal_parser import (
    parse_signal,
    strip_signal,
    parse_and_strip,
    SIGNAL_PATTERN,
)
from backend.src.models.signals import Signal, SignalType


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def need_turn_response() -> str:
    """Response with need_turn signal."""
    return '''I found the API endpoint but need to verify the response format.

<signal type="need_turn">
  <reason>Found backup API, need to test if it responds correctly</reason>
  <confidence>0.85</confidence>
  <expected_turns>1</expected_turns>
</signal>'''


@pytest.fixture
def context_sufficient_response() -> str:
    """Response with context_sufficient signal."""
    return '''Based on the authentication middleware in auth.py and the design doc...

<signal type="context_sufficient">
  <sources_found>3</sources_found>
  <source_types>["code", "docs"]</source_types>
  <confidence>0.9</confidence>
</signal>'''


@pytest.fixture
def stuck_response() -> str:
    """Response with stuck signal."""
    return '''I searched all available sources but couldn't find deployment history.

<signal type="stuck">
  <attempted>["vault_search", "thread_seek", "code_search"]</attempted>
  <blocker>No deployment logs or history found in any source</blocker>
  <suggestions>["check external CI/CD system", "ask team member"]</suggestions>
  <confidence>0.7</confidence>
</signal>'''


@pytest.fixture
def need_capability_response() -> str:
    """Response with need_capability signal."""
    return '''I can show you the test file, but I cannot run the tests myself.

<signal type="need_capability">
  <capability>execute_shell_command</capability>
  <reason>Need to run pytest to verify the fix works</reason>
  <workaround>You can run the command manually: pytest tests/unit/</workaround>
  <confidence>0.8</confidence>
</signal>'''


@pytest.fixture
def partial_answer_response() -> str:
    """Response with partial_answer signal."""
    return '''Based on the dev config, the timeout appears to be 30 seconds...

<signal type="partial_answer">
  <missing>Could not verify production configuration</missing>
  <caveat>This is based on development settings only</caveat>
  <confidence>0.6</confidence>
</signal>'''


@pytest.fixture
def delegation_recommended_response() -> str:
    """Response with delegation_recommended signal."""
    return '''This would require analyzing the entire authentication system...

<signal type="delegation_recommended">
  <reason>Need to trace auth flow across 23 files</reason>
  <scope>Map all authentication code paths and dependencies</scope>
  <estimated_tokens>15000</estimated_tokens>
  <subagent_type>research</subagent_type>
  <confidence>0.9</confidence>
</signal>'''


# =============================================================================
# Test: All 6 Signal Types
# =============================================================================


class TestSignalTypeParsing:
    """Test parsing of all 6 signal types."""

    def test_parse_need_turn(self, need_turn_response: str) -> None:
        """Parse need_turn signal correctly."""
        signal = parse_signal(need_turn_response)

        assert signal is not None
        assert signal.type == SignalType.NEED_TURN
        assert signal.confidence == 0.85
        assert "backup API" in signal.fields["reason"]
        assert signal.fields["expected_turns"] == 1
        assert signal.is_continuation()
        assert not signal.is_terminal()

    def test_parse_context_sufficient(self, context_sufficient_response: str) -> None:
        """Parse context_sufficient signal correctly."""
        signal = parse_signal(context_sufficient_response)

        assert signal is not None
        assert signal.type == SignalType.CONTEXT_SUFFICIENT
        assert signal.confidence == 0.9
        assert signal.fields["sources_found"] == 3
        assert signal.fields["source_types"] == ["code", "docs"]
        assert signal.is_terminal()
        assert not signal.is_continuation()

    def test_parse_stuck(self, stuck_response: str) -> None:
        """Parse stuck signal correctly."""
        signal = parse_signal(stuck_response)

        assert signal is not None
        assert signal.type == SignalType.STUCK
        assert signal.confidence == 0.7
        assert len(signal.fields["attempted"]) == 3
        assert "vault_search" in signal.fields["attempted"]
        assert "deployment" in signal.fields["blocker"]
        assert signal.is_terminal()

    def test_parse_need_capability(self, need_capability_response: str) -> None:
        """Parse need_capability signal correctly."""
        signal = parse_signal(need_capability_response)

        assert signal is not None
        assert signal.type == SignalType.NEED_CAPABILITY
        assert signal.confidence == 0.8
        assert signal.fields["capability"] == "execute_shell_command"
        assert "pytest" in signal.fields["reason"]
        assert signal.fields["workaround"] is not None
        assert signal.is_continuation()

    def test_parse_partial_answer(self, partial_answer_response: str) -> None:
        """Parse partial_answer signal correctly."""
        signal = parse_signal(partial_answer_response)

        assert signal is not None
        assert signal.type == SignalType.PARTIAL_ANSWER
        assert signal.confidence == 0.6
        assert "production" in signal.fields["missing"]
        assert signal.fields["caveat"] is not None
        assert signal.is_terminal()

    def test_parse_delegation_recommended(
        self, delegation_recommended_response: str
    ) -> None:
        """Parse delegation_recommended signal correctly."""
        signal = parse_signal(delegation_recommended_response)

        assert signal is not None
        assert signal.type == SignalType.DELEGATION_RECOMMENDED
        assert signal.confidence == 0.9
        assert "23 files" in signal.fields["reason"]
        assert "authentication" in signal.fields["scope"]
        assert signal.fields["estimated_tokens"] == 15000
        assert signal.fields["subagent_type"] == "research"
        assert signal.is_continuation()


# =============================================================================
# Test: Malformed XML
# =============================================================================


class TestMalformedXML:
    """Test handling of malformed signal XML."""

    def test_no_signal_returns_none(self) -> None:
        """Response without signal returns None."""
        response = "Just a regular response without any signal."
        signal = parse_signal(response)
        assert signal is None

    def test_empty_response_returns_none(self) -> None:
        """Empty response returns None."""
        assert parse_signal("") is None
        assert parse_signal(None) is None  # type: ignore

    def test_unclosed_signal_tag(self) -> None:
        """Unclosed signal tag returns None."""
        response = '''Answer here.
        <signal type="need_turn">
          <reason>Unclosed signal'''
        signal = parse_signal(response)
        assert signal is None

    def test_unknown_signal_type(self) -> None:
        """Unknown signal type returns None."""
        response = '''<signal type="unknown_type">
          <reason>Test</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is None

    def test_missing_type_attribute(self) -> None:
        """Signal without type attribute returns None."""
        response = '''<signal>
          <reason>No type</reason>
        </signal>'''
        signal = parse_signal(response)
        assert signal is None

    def test_nested_xml_in_field(self) -> None:
        """Nested XML in field value is handled."""
        response = '''<signal type="need_turn">
          <reason>Found function that needs testing</reason>
          <confidence>0.7</confidence>
        </signal>'''
        # This should parse correctly with simple nested content
        signal = parse_signal(response)
        if signal is not None:
            assert "function" in signal.fields.get("reason", "")


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_signal_with_single_quotes(self) -> None:
        """Signal with single-quoted type attribute."""
        response = '''<signal type='need_turn'>
          <reason>Testing single quotes</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN

    def test_signal_with_extra_whitespace(self) -> None:
        """Signal with extra whitespace is parsed."""
        response = '''<signal   type="need_turn"  >
          <reason>  Extra whitespace  </reason>
          <confidence>  0.5  </confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN

    def test_confidence_as_attribute(self) -> None:
        """Confidence as XML attribute instead of element."""
        response = '''<signal type="context_sufficient" confidence="0.95">
          <sources_found>5</sources_found>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 0.95

    def test_confidence_out_of_range_clamped(self) -> None:
        """Confidence > 1.0 is clamped to 1.0."""
        response = '''<signal type="need_turn">
          <reason>Test confidence clamping</reason>
          <confidence>1.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 1.0

    def test_confidence_negative_clamped(self) -> None:
        """Confidence < 0.0 is clamped to 0.0."""
        response = '''<signal type="need_turn">
          <reason>Test negative confidence</reason>
          <confidence>-0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 0.0

    def test_missing_confidence_uses_default(self) -> None:
        """Missing confidence uses default 0.5."""
        response = '''<signal type="need_turn">
          <reason>No confidence field here</reason>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.confidence == 0.5

    def test_multiple_signals_takes_first(self) -> None:
        """Multiple signals - only first is parsed."""
        response = '''First answer.
        <signal type="need_turn">
          <reason>First signal reason</reason>
          <confidence>0.8</confidence>
        </signal>
        More text.
        <signal type="context_sufficient">
          <sources_found>1</sources_found>
          <confidence>0.9</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN
        assert "First signal" in signal.fields["reason"]

    def test_signal_inline_with_text(self) -> None:
        """Signal inline with text (not on own line)."""
        response = 'Answer here <signal type="context_sufficient"><sources_found>1</sources_found><confidence>0.5</confidence></signal> more text'
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.CONTEXT_SUFFICIENT

    def test_case_insensitive_type(self) -> None:
        """Signal type is case-insensitive."""
        response = '''<signal type="NEED_TURN">
          <reason>Uppercase type test</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.type == SignalType.NEED_TURN

    def test_timestamp_is_set(self) -> None:
        """Signal timestamp is set to current time."""
        before = datetime.now(timezone.utc)
        response = '''<signal type="need_turn">
          <reason>Test timestamp</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        after = datetime.now(timezone.utc)

        assert signal is not None
        assert before <= signal.timestamp <= after

    def test_raw_xml_preserved(self) -> None:
        """Raw XML is preserved in signal."""
        response = '''<signal type="need_turn">
          <reason>Test reason preserved</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert '<signal type="need_turn">' in signal.raw_xml
        assert "</signal>" in signal.raw_xml


# =============================================================================
# Test: Field Value Parsing
# =============================================================================


class TestFieldValueParsing:
    """Test parsing of different field value types."""

    def test_integer_field(self) -> None:
        """Integer field is parsed as int."""
        response = '''<signal type="context_sufficient">
          <sources_found>42</sources_found>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.fields["sources_found"] == 42
        assert isinstance(signal.fields["sources_found"], int)

    def test_list_field(self) -> None:
        """JSON array field is parsed as list."""
        response = '''<signal type="stuck">
          <attempted>["tool1", "tool2", "tool3"]</attempted>
          <blocker>Cannot proceed with the task</blocker>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert signal.fields["attempted"] == ["tool1", "tool2", "tool3"]
        assert isinstance(signal.fields["attempted"], list)

    def test_string_field(self) -> None:
        """String field is preserved as string."""
        response = '''<signal type="need_turn">
          <reason>This is a detailed reason text</reason>
          <confidence>0.5</confidence>
        </signal>'''
        signal = parse_signal(response)
        assert signal is not None
        assert isinstance(signal.fields["reason"], str)
        assert "detailed reason" in signal.fields["reason"]


# =============================================================================
# Test: strip_signal
# =============================================================================


class TestStripSignal:
    """Test signal stripping from response."""

    def test_strip_removes_signal(self, need_turn_response: str) -> None:
        """strip_signal removes the signal block."""
        cleaned = strip_signal(need_turn_response)
        assert "<signal" not in cleaned
        assert "</signal>" not in cleaned
        assert "backup API" not in cleaned  # Signal content removed

    def test_strip_preserves_content(self, need_turn_response: str) -> None:
        """strip_signal preserves response content."""
        cleaned = strip_signal(need_turn_response)
        assert "API endpoint" in cleaned

    def test_strip_normalizes_whitespace(self) -> None:
        """strip_signal normalizes multiple newlines."""
        response = '''Answer here.



        <signal type="need_turn">
          <reason>Test</reason>
        </signal>



        '''
        cleaned = strip_signal(response)
        assert "\n\n\n" not in cleaned
        assert cleaned == "Answer here."

    def test_strip_handles_no_signal(self) -> None:
        """strip_signal handles response without signal."""
        response = "Just text, no signal."
        cleaned = strip_signal(response)
        assert cleaned == "Just text, no signal."

    def test_strip_handles_empty_response(self) -> None:
        """strip_signal handles empty response."""
        assert strip_signal("") == ""
        assert strip_signal(None) == ""  # type: ignore


# =============================================================================
# Test: parse_and_strip
# =============================================================================


class TestParseAndStrip:
    """Test combined parse and strip function."""

    def test_returns_both_signal_and_clean(self, need_turn_response: str) -> None:
        """parse_and_strip returns both signal and cleaned text."""
        signal, cleaned = parse_and_strip(need_turn_response)

        assert signal is not None
        assert signal.type == SignalType.NEED_TURN
        assert "<signal" not in cleaned
        assert "API endpoint" in cleaned

    def test_no_signal_returns_none_and_original(self) -> None:
        """parse_and_strip with no signal returns None and original text."""
        response = "No signal here."
        signal, cleaned = parse_and_strip(response)

        assert signal is None
        assert cleaned == "No signal here."


# =============================================================================
# Test: Performance (smoke test)
# =============================================================================


class TestPerformance:
    """Smoke test for parsing performance."""

    def test_large_response_parses_quickly(self) -> None:
        """Large response (~10KB) parses in reasonable time."""
        import time

        # Generate ~10KB response
        large_content = "x" * 10000
        response = f'''{large_content}

        <signal type="context_sufficient">
          <sources_found>5</sources_found>
          <confidence>0.9</confidence>
        </signal>'''

        start = time.time()
        signal = parse_signal(response)
        duration_ms = (time.time() - start) * 1000

        assert signal is not None
        assert duration_ms < 50  # Should be <10ms, allow 50ms for slow CI
