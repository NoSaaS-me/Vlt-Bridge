# Tasks Expanded: User Story 1 - Intelligent Context Selection

**Feature**: 020-bt-oracle-agent
**Tasks**: T013-T019
**Goal**: Agent intelligently decides what context sources to use based on query classification

---

## T013: Add query_classification to Blackboard

**File**: `backend/src/bt/state/blackboard.py`

### Function Signature

No new function needed. Extend the oracle-agent.lua blackboard schema:

```lua
blackboard = {
    -- ... existing fields ...

    -- Query classification (NEW)
    query_classification = "QueryClassification",  -- or "dict" for untyped
}
```

### Core Algorithm

1. Open `backend/src/bt/trees/oracle-agent.lua`
2. Add `query_classification` to blackboard schema declaration (line ~26-67)
3. Define type as string for simplicity (avoid Pydantic schema requirement in Lua)
4. Initialize in `reset_state()` action to `nil`

### Key Code Snippet

```lua
-- In oracle-agent.lua blackboard schema
blackboard = {
    -- ... existing fields ...

    -- Query Classification (US1)
    query_classification = "dict",  -- {query_type, needs_code, needs_vault, needs_web, confidence}
}
```

```python
# In oracle.py reset_state()
def reset_state(ctx: "TickContext") -> RunStatus:
    bb = ctx.blackboard
    # ... existing resets ...

    # Reset query classification (US1)
    bb_set(bb, "query_classification", None)
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| query_classification field exists | `bb_get(bb, "query_classification")` returns None on fresh query |
| Field accessible in conditions | BT condition can read `bb.query_classification.needs_code` |

---

## T014: Create query_analysis.py Action

**File**: `backend/src/bt/actions/query_analysis.py` (NEW)

### Function Signature

```python
def analyze_query(ctx: TickContext) -> RunStatus:
    """
    Analyze user query and classify context needs.

    Reads: bb.query (str)
    Writes: bb.query_classification (dict)

    Returns:
        RunStatus.SUCCESS if classification succeeded
        RunStatus.FAILURE if query is missing
    """
```

### Core Algorithm

1. Get query from blackboard (`bb.query`)
2. Handle different query formats (str, OracleQuery, dict)
3. Call `classify_query()` from query_classifier service
4. Store result in `bb.query_classification`
5. Log classification for debugging
6. Return SUCCESS

### Key Code Snippet

```python
"""Query Analysis Action - Classifies user query for context selection."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)


def bb_get(bb, key, default=None):
    value = bb._lookup(key)
    return value if value is not None else default


def bb_set(bb, key, value):
    bb._data[key] = value
    bb._writes.add(key)


def analyze_query(ctx: "TickContext") -> RunStatus:
    """Analyze user query and classify context needs.

    Reads: bb.query
    Writes: bb.query_classification
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    query = bb_get(bb, "query")
    if not query:
        logger.warning("analyze_query: No query in blackboard")
        return RunStatus.FAILURE

    # Extract question text from various formats
    if isinstance(query, str):
        question = query
    elif hasattr(query, "question"):
        question = query.question
    elif isinstance(query, dict):
        question = query.get("question", str(query))
    else:
        question = str(query)

    # Classify using query_classifier service
    try:
        from src.services.query_classifier import classify_query
        classification = classify_query(question)
    except ImportError:
        # Fallback: inline heuristics if service not available
        classification = _fallback_classify(question)

    # Store as dict for Lua compatibility
    bb_set(bb, "query_classification", {
        "query_type": classification.query_type,
        "needs_code": classification.needs_code,
        "needs_vault": classification.needs_vault,
        "needs_web": classification.needs_web,
        "confidence": classification.confidence,
        "keywords_matched": classification.keywords_matched or [],
    })

    logger.debug(
        f"Query classified: type={classification.query_type}, "
        f"needs_code={classification.needs_code}, "
        f"needs_web={classification.needs_web}"
    )

    ctx.mark_progress()
    return RunStatus.SUCCESS


def _fallback_classify(question: str):
    """Fallback heuristic classification."""
    from dataclasses import dataclass

    @dataclass
    class FallbackClassification:
        query_type: str = "conversational"
        needs_code: bool = False
        needs_vault: bool = False
        needs_web: bool = False
        confidence: float = 0.5
        keywords_matched: list = None

    q = question.lower()

    # Web/research keywords
    if any(kw in q for kw in ["weather", "latest", "news", "search", "find online"]):
        return FallbackClassification(
            query_type="research",
            needs_web=True,
            confidence=0.8,
            keywords_matched=["weather/latest/news"]
        )

    # Code keywords
    if any(kw in q for kw in ["function", "class", "implement", "code", "where is", "how does"]):
        return FallbackClassification(
            query_type="code",
            needs_code=True,
            confidence=0.8,
            keywords_matched=["function/class/code"]
        )

    # Default: conversational
    return FallbackClassification()


__all__ = ["analyze_query"]
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| Classifies weather query as research | `classify("weather in Paris").query_type == "research"` |
| Classifies code query correctly | `classify("how does auth work").query_type == "code"` |
| Sets needs_web for research | `classify("weather").needs_web == True` |
| Sets needs_code for code | `classify("function").needs_code == True` |

---

## T015: Create context_needs.py Conditions

**File**: `backend/src/bt/conditions/context_needs.py` (NEW)

### Function Signatures

```python
def needs_code_context(ctx: TickContext) -> RunStatus:
    """Check if query needs code search."""

def needs_vault_context(ctx: TickContext) -> RunStatus:
    """Check if query needs vault/documentation search."""

def needs_web_context(ctx: TickContext) -> RunStatus:
    """Check if query needs web search."""

def has_query_classification(ctx: TickContext) -> RunStatus:
    """Check if query has been classified."""
```

### Core Algorithm

1. Get `query_classification` from blackboard
2. Return FAILURE if not classified
3. Check appropriate boolean field (needs_code, needs_vault, needs_web)
4. Return SUCCESS if True, FAILURE if False

### Key Code Snippet

```python
"""Context Needs Conditions - Check what context sources are needed."""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from ..state.base import RunStatus

if TYPE_CHECKING:
    from ..core.context import TickContext

logger = logging.getLogger(__name__)


def _get_classification(ctx: "TickContext") -> dict | None:
    """Get query classification from blackboard."""
    bb = ctx.blackboard
    if bb is None:
        return None
    classification = bb._lookup("query_classification")
    return classification if isinstance(classification, dict) else None


def has_query_classification(ctx: "TickContext") -> RunStatus:
    """Check if query has been classified.

    Returns:
        SUCCESS if query_classification exists
        FAILURE if not classified
    """
    classification = _get_classification(ctx)
    if classification is None:
        return RunStatus.FAILURE
    return RunStatus.SUCCESS


def needs_code_context(ctx: "TickContext") -> RunStatus:
    """Check if query needs code search.

    Reads: bb.query_classification.needs_code

    Returns:
        SUCCESS if needs_code is True
        FAILURE if False or not classified
    """
    classification = _get_classification(ctx)
    if classification is None:
        logger.debug("needs_code_context: No classification available")
        return RunStatus.FAILURE

    needs_code = classification.get("needs_code", False)
    logger.debug(f"needs_code_context: {needs_code}")
    return RunStatus.SUCCESS if needs_code else RunStatus.FAILURE


def needs_vault_context(ctx: "TickContext") -> RunStatus:
    """Check if query needs vault/documentation search.

    Reads: bb.query_classification.needs_vault

    Returns:
        SUCCESS if needs_vault is True
        FAILURE if False or not classified
    """
    classification = _get_classification(ctx)
    if classification is None:
        return RunStatus.FAILURE

    needs_vault = classification.get("needs_vault", False)
    logger.debug(f"needs_vault_context: {needs_vault}")
    return RunStatus.SUCCESS if needs_vault else RunStatus.FAILURE


def needs_web_context(ctx: "TickContext") -> RunStatus:
    """Check if query needs web search.

    Reads: bb.query_classification.needs_web

    Returns:
        SUCCESS if needs_web is True
        FAILURE if False or not classified
    """
    classification = _get_classification(ctx)
    if classification is None:
        return RunStatus.FAILURE

    needs_web = classification.get("needs_web", False)
    logger.debug(f"needs_web_context: {needs_web}")
    return RunStatus.SUCCESS if needs_web else RunStatus.FAILURE


def is_conversational(ctx: "TickContext") -> RunStatus:
    """Check if query is purely conversational (no tools needed).

    Returns:
        SUCCESS if query_type is "conversational"
        FAILURE otherwise
    """
    classification = _get_classification(ctx)
    if classification is None:
        return RunStatus.FAILURE

    query_type = classification.get("query_type", "")
    return RunStatus.SUCCESS if query_type == "conversational" else RunStatus.FAILURE


__all__ = [
    "has_query_classification",
    "needs_code_context",
    "needs_vault_context",
    "needs_web_context",
    "is_conversational",
]
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| needs_code_context returns SUCCESS | For query classified with needs_code=True |
| needs_web_context returns SUCCESS | For query classified with needs_web=True |
| Returns FAILURE without classification | When query_classification is None |

---

## T016: Modify oracle-agent.lua for Context Assessment Phase

**File**: `backend/src/bt/trees/oracle-agent.lua`

### Function Signature

N/A - Lua tree modification

### Core Algorithm

1. Add Context Assessment Phase after Initialization (Phase 1)
2. Insert `analyze_query` action before Context Loading
3. Add conditional tool selection based on classification
4. Modify tool filtering to skip unnecessary searches

### Key Code Snippet

Insert after Phase 1 (Initialization), before Phase 2 (Context Loading):

```lua
--[[
    Phase 1.5: Context Assessment (NEW - US1)
    Classify query to determine context needs
--]]
BT.action("analyze-query", {
    fn = "backend.src.bt.actions.query_analysis.analyze_query",
    description = "Classify query type and context needs"
}),

-- Optional: Skip code search for non-code queries
BT.selector({
    -- If needs code context, proceed with code loading
    BT.sequence({
        BT.condition("needs-code-context", {
            fn = "backend.src.bt.conditions.context_needs.needs_code_context"
        }),
        BT.action("prepare-code-context", {
            fn = "backend.src.bt.actions.oracle.noop",  -- placeholder for future
            description = "Prepare code search context"
        })
    }),
    -- Otherwise skip (always succeeds)
    BT.always_succeed(
        BT.action("skip-code-context", { fn = "backend.src.bt.actions.oracle.noop" })
    )
}),
```

**Full insertion point** (after line ~88, before Phase 2):

```lua
-- After emit-query-start, before Context Loading selector

--[[
    Phase 1.5: Context Assessment (US1)
    Classify query to determine what context sources are needed.
    This informs tool selection and prompt composition.
--]]
BT.action("analyze-query", {
    fn = "backend.src.bt.actions.query_analysis.analyze_query",
    description = "Classify query type and determine context needs"
}),
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| analyze-query action added | Tree contains analyze-query node |
| Runs before context loading | analyze-query precedes load-tree-node |
| Classification available for later phases | bb.query_classification set before message building |

---

## T017: Update build_system_prompt() to Use prompt_composer

**File**: `backend/src/bt/actions/oracle.py`

### Function Signature

```python
def build_system_prompt(ctx: TickContext) -> RunStatus:
    """Build system prompt using prompt_composer based on query classification.

    Reads:
        bb.query_classification (dict)
        bb.user_id (str)
        bb.project_id (str)
    Writes:
        bb.messages (list) - Adds system message at index 0
    """
```

### Core Algorithm

1. Get query_classification from blackboard
2. Call prompt_composer.compose_prompt() with classification
3. Compose_prompt selects segments based on query_type
4. Always includes: base.md, signals.md, tools-reference.md
5. Conditionally includes: code-analysis.md, documentation.md, research.md
6. Set composed prompt as system message

### Key Code Snippet

```python
def build_system_prompt(ctx: "TickContext") -> RunStatus:
    """Load vault files/threads, render system prompt via prompt_composer.

    US1 Enhancement: Uses query classification for dynamic composition.
    """
    bb = ctx.blackboard
    if bb is None:
        return RunStatus.FAILURE

    messages = bb_get(bb, "messages") or []
    user_id = bb_get(bb, "user_id")
    project_id = bb_get(bb, "project_id")
    classification = bb_get(bb, "query_classification")

    # Build system prompt using composer
    try:
        from src.services.prompt_composer import compose_prompt

        # Extract query_type, defaulting to "conversational"
        query_type = "conversational"
        if classification and isinstance(classification, dict):
            query_type = classification.get("query_type", "conversational")

        system_content = compose_prompt(
            query_type=query_type,
            user_id=user_id,
            project_id=project_id,
        )
    except ImportError:
        # Fallback to legacy prompt building
        logger.warning("prompt_composer not available, using legacy")
        system_content = _build_system_content(user_id, project_id)

    # Add system message
    messages.insert(0, {
        "role": "system",
        "content": system_content
    })

    bb_set(bb, "messages", messages)
    ctx.mark_progress()
    return RunStatus.SUCCESS
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| Code query includes code-analysis.md | Prompt contains code analysis instructions |
| Research query includes research.md | Prompt contains research methodology |
| Always includes signals.md | All prompts contain signal emission instructions |
| Fallback works without composer | Legacy prompt built if composer unavailable |

---

## T018: Tests for Query Analysis

**File**: `backend/tests/unit/bt/test_query_analysis.py` (NEW)

### Function Signature

```python
class TestQueryAnalysis:
    def test_analyze_query_weather() -> None: ...
    def test_analyze_query_code() -> None: ...
    def test_analyze_query_documentation() -> None: ...
    def test_analyze_query_conversational() -> None: ...
    def test_analyze_query_missing_query() -> None: ...
    def test_context_needs_conditions() -> None: ...
```

### Core Algorithm

1. Create mock TickContext with TypedBlackboard
2. Set query in blackboard
3. Call analyze_query action
4. Verify classification in blackboard
5. Test context_needs conditions against classification

### Key Code Snippet

```python
"""Tests for query analysis action and context needs conditions."""

import pytest
from unittest.mock import MagicMock

from src.bt.state.base import RunStatus
from src.bt.state.blackboard import TypedBlackboard
from src.bt.core.context import TickContext
from src.bt.actions.query_analysis import analyze_query
from src.bt.conditions.context_needs import (
    needs_code_context,
    needs_vault_context,
    needs_web_context,
    is_conversational,
)


@pytest.fixture
def ctx_with_query():
    """Create a TickContext with query in blackboard."""
    def _create(query: str):
        bb = TypedBlackboard(scope_name="test")
        bb._data["query"] = query
        return TickContext(blackboard=bb)
    return _create


class TestAnalyzeQuery:
    """Test suite for analyze_query action."""

    def test_weather_query_classified_as_research(self, ctx_with_query):
        """Weather query should route to web search."""
        ctx = ctx_with_query("What's the weather in Paris?")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "research"
        assert classification["needs_web"] is True
        assert classification["needs_code"] is False

    def test_code_query_classified_as_code(self, ctx_with_query):
        """Code question should route to code search."""
        ctx = ctx_with_query("How does the auth middleware work?")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "code"
        assert classification["needs_code"] is True
        assert classification["needs_web"] is False

    def test_thanks_classified_as_conversational(self, ctx_with_query):
        """Simple acknowledgment needs no tools."""
        ctx = ctx_with_query("Thanks, that helps!")

        result = analyze_query(ctx)

        assert result == RunStatus.SUCCESS
        classification = ctx.blackboard._data["query_classification"]
        assert classification["query_type"] == "conversational"
        assert classification["needs_code"] is False
        assert classification["needs_vault"] is False
        assert classification["needs_web"] is False

    def test_missing_query_returns_failure(self):
        """No query in blackboard should fail."""
        bb = TypedBlackboard(scope_name="test")
        ctx = TickContext(blackboard=bb)

        result = analyze_query(ctx)

        assert result == RunStatus.FAILURE


class TestContextNeedsConditions:
    """Test suite for context_needs conditions."""

    def test_needs_code_context_true(self, ctx_with_query):
        """needs_code_context returns SUCCESS when classification says so."""
        ctx = ctx_with_query("where is the login function")
        analyze_query(ctx)

        result = needs_code_context(ctx)

        assert result == RunStatus.SUCCESS

    def test_needs_web_context_true(self, ctx_with_query):
        """needs_web_context returns SUCCESS for research queries."""
        ctx = ctx_with_query("what's the latest news about Python 4")
        analyze_query(ctx)

        result = needs_web_context(ctx)

        assert result == RunStatus.SUCCESS

    def test_is_conversational_true(self, ctx_with_query):
        """is_conversational returns SUCCESS for acknowledgments."""
        ctx = ctx_with_query("ok thanks")
        analyze_query(ctx)

        result = is_conversational(ctx)

        assert result == RunStatus.SUCCESS

    def test_no_classification_returns_failure(self):
        """Conditions fail without prior classification."""
        bb = TypedBlackboard(scope_name="test")
        ctx = TickContext(blackboard=bb)

        assert needs_code_context(ctx) == RunStatus.FAILURE
        assert needs_web_context(ctx) == RunStatus.FAILURE
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| Weather query test | test_weather_query_classified_as_research |
| Code query test | test_code_query_classified_as_code |
| Conversational test | test_thanks_classified_as_conversational |
| Condition tests | TestContextNeedsConditions class |

---

## T019: Integration Test for "Weather in Paris"

**File**: `backend/tests/integration/test_oracle_bt_integration.py` (NEW or append)

### Function Signature

```python
class TestOracleBTContextSelection:
    async def test_weather_query_uses_web_search_only() -> None: ...
    async def test_code_query_does_not_use_web_search() -> None: ...
```

### Core Algorithm

1. Create OracleBTWrapper with test configuration
2. Submit "What's the weather in Paris?" query
3. Collect all streamed chunks
4. Verify tool_call chunks only contain web_search
5. Verify NO code search or vault search tools called

### Key Code Snippet

```python
"""Integration tests for Oracle BT context selection (US1)."""

import pytest
from unittest.mock import AsyncMock, patch

from src.bt.wrappers.oracle_wrapper import OracleBTWrapper
from src.models.oracle import OracleQuery


@pytest.fixture
def mock_llm():
    """Mock LLM that returns appropriate tool calls."""
    async def _mock_response(messages, tools, **kwargs):
        # Check if this is initial query or after tools
        last_msg = messages[-1]
        if last_msg.get("role") == "tool":
            # After tool results, return final answer
            return {
                "content": "The weather in Paris is 15C and sunny.",
                "tool_calls": None
            }

        # Initial query - return web_search tool call
        return {
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "weather in Paris"}'
                }
            }]
        }
    return _mock_response


@pytest.fixture
def mock_tool_executor():
    """Mock tool executor."""
    def _execute(name, args):
        if name == "web_search":
            return {"results": [{"title": "Paris Weather", "snippet": "15C sunny"}]}
        return {"error": "Unexpected tool call"}
    return _execute


class TestOracleBTContextSelection:
    """Integration tests for intelligent context selection (US1)."""

    @pytest.mark.asyncio
    async def test_weather_query_routes_to_web_search(self, mock_llm, mock_tool_executor):
        """
        GIVEN a weather query
        WHEN processed by Oracle BT
        THEN only web_search tool is called (not code/vault search)
        """
        query = OracleQuery(
            question="What's the weather in Paris?",
            user_id="test-user",
            project_id="test-project"
        )

        tool_calls_made = []

        with patch("src.services.llm_client.LLMClient.chat", mock_llm):
            with patch("src.services.tool_executor.ToolExecutor.execute") as mock_exec:
                mock_exec.side_effect = lambda name, args: (
                    tool_calls_made.append(name),
                    mock_tool_executor(name, args)
                )[1]

                wrapper = OracleBTWrapper()
                chunks = []
                async for chunk in wrapper.query(query):
                    chunks.append(chunk)

        # Verify ONLY web_search was called
        assert "web_search" in tool_calls_made
        assert "search_code" not in tool_calls_made
        assert "search_vault" not in tool_calls_made
        assert "read_thread" not in tool_calls_made

    @pytest.mark.asyncio
    async def test_code_query_does_not_use_web_search(self, mock_tool_executor):
        """
        GIVEN a code query
        WHEN processed by Oracle BT
        THEN code search tools are used (not web_search)
        """
        query = OracleQuery(
            question="How does the auth middleware work?",
            user_id="test-user",
            project_id="test-project"
        )

        tool_calls_made = []

        async def mock_code_llm(messages, tools, **kwargs):
            last_msg = messages[-1]
            if last_msg.get("role") == "tool":
                return {"content": "The auth middleware validates JWT tokens.", "tool_calls": None}
            return {
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "search_code",
                        "arguments": '{"query": "auth middleware"}'
                    }
                }]
            }

        with patch("src.services.llm_client.LLMClient.chat", mock_code_llm):
            with patch("src.services.tool_executor.ToolExecutor.execute") as mock_exec:
                mock_exec.side_effect = lambda name, args: (
                    tool_calls_made.append(name),
                    {"results": [{"path": "auth.py", "content": "..."}]}
                )[1]

                wrapper = OracleBTWrapper()
                chunks = []
                async for chunk in wrapper.query(query):
                    chunks.append(chunk)

        # Verify code search used, NOT web search
        assert "search_code" in tool_calls_made
        assert "web_search" not in tool_calls_made

    @pytest.mark.asyncio
    async def test_conversational_query_no_tools(self):
        """
        GIVEN a simple acknowledgment
        WHEN processed by Oracle BT
        THEN no tools are called
        """
        query = OracleQuery(
            question="Thanks, that helps!",
            user_id="test-user",
            project_id="test-project"
        )

        tool_calls_made = []

        async def mock_conversational_llm(messages, tools, **kwargs):
            return {"content": "You're welcome! Let me know if you have more questions.", "tool_calls": None}

        with patch("src.services.llm_client.LLMClient.chat", mock_conversational_llm):
            with patch("src.services.tool_executor.ToolExecutor.execute") as mock_exec:
                mock_exec.side_effect = lambda name, args: tool_calls_made.append(name)

                wrapper = OracleBTWrapper()
                chunks = []
                async for chunk in wrapper.query(query):
                    chunks.append(chunk)

        # Verify NO tools called
        assert len(tool_calls_made) == 0
```

### Acceptance Criteria Mapping

| Criterion | Verification |
|-----------|--------------|
| US1 Scenario 1 | test_weather_query_routes_to_web_search |
| US1 Scenario 2 | test_code_query_does_not_use_web_search |
| US1 Scenario 4 | test_conversational_query_no_tools |
| Agent uses correct tools | Tool call assertions in each test |

---

## Summary

| Task | File | Lines | Key Deliverable |
|------|------|-------|-----------------|
| T013 | oracle-agent.lua, oracle.py | ~10 | query_classification field |
| T014 | query_analysis.py | ~120 | analyze_query() action |
| T015 | context_needs.py | ~90 | 5 condition functions |
| T016 | oracle-agent.lua | ~20 | Context Assessment Phase |
| T017 | oracle.py | ~40 | prompt_composer integration |
| T018 | test_query_analysis.py | ~100 | Unit tests |
| T019 | test_oracle_bt_integration.py | ~120 | Integration tests |

**Total estimated lines**: ~500

**Dependencies**:
- T014, T015, T016 depend on T013 (blackboard field)
- T017 depends on T014 (classification available)
- T018, T019 depend on T014, T015 (implementation complete)

**Run order**: T013 -> T014 -> T015 -> T016 -> T017 -> T018 (parallel with T019) -> T019
