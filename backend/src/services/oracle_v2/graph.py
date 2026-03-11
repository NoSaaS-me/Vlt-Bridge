"""Oracle V2 LangGraph CodeAct graph builder.

Wraps create_codeact() from langgraph-codeact with:
  - OracleState schema (user_id, project_id, plan, recalled_facts)
  - make_sandbox_eval() for restricted REPL execution
  - Optional checkpointer (AsyncSqliteSaver or MemorySaver fallback)

Guards all imports with try/except ImportError so the module loads even
when langgraph-codeact or langchain-openai are not installed. If they are
absent, build_oracle_graph() will raise a clear ImportError at call time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query classifier (T038)
# ---------------------------------------------------------------------------

_DIRECT_PREFIXES = frozenset({
    "what is", "define", "when was", "who is", "list the", "show me", "where is",
})
_COMPLEX_KEYWORDS = frozenset({
    "refactor", "implement", "build", "analyze", "compare", "find all",
    "trace", "audit", "summarize all", "across the codebase",
})


def classify_query(query: str) -> Literal["direct", "plan"]:
    """Rule-based query classifier for the Oracle planner.

    Returns 'plan' for complex multi-step tasks, 'direct' for simple Q&A.

    Rules (in priority order):
      1. Complex keyword present → 'plan'
      2. Token count > 50 → 'plan'
      3. Token count < 15 AND starts with a direct prefix → 'direct'
      4. Default → 'direct'
    """
    q_lower = query.lower().strip()
    token_count = len(query.split())

    for kw in _COMPLEX_KEYWORDS:
        if kw in q_lower:
            return "plan"

    if token_count > 50:
        return "plan"

    if token_count < 15:
        for prefix in _DIRECT_PREFIXES:
            if q_lower.startswith(prefix):
                return "direct"

    return "direct"

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    from langgraph_codeact import create_codeact  # type: ignore[import]
    _CODEACT_AVAILABLE = True
except ImportError:
    _CODEACT_AVAILABLE = False
    create_codeact = None  # type: ignore[assignment]

try:
    from langchain_openai import ChatOpenAI  # type: ignore[import]
    _LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_OPENAI_AVAILABLE = False
    ChatOpenAI = None  # type: ignore[assignment]

try:
    from langgraph.checkpoint.memory import MemorySaver  # type: ignore[import]
    _MEMORY_SAVER_AVAILABLE = True
except ImportError:
    _MEMORY_SAVER_AVAILABLE = False
    MemorySaver = None  # type: ignore[assignment]


class _PickleSerde:
    """Pickle-based serde for MemorySaver.

    CodeAct stores Python function objects in the REPL ``context`` namespace.
    The default JsonPlusSerializer (msgpack) cannot serialize functions;
    pickle can, and is acceptable for in-process in-memory state.
    """

    def dumps_typed(self, obj: object) -> tuple[str, bytes]:
        import pickle
        return ("pickle", pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))

    def loads_typed(self, data: tuple[str, bytes]) -> object:
        import pickle
        encoding, raw = data
        if encoding == "pickle":
            return pickle.loads(raw)
        # Fallback: try JsonPlusSerializer for checkpoints written by another serde
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
        return JsonPlusSerializer().loads_typed(data)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_oracle_graph(
    tools: list[Callable],
    checkpointer: Any,
    model: Any,
):
    """Build and compile the Oracle CodeAct LangGraph.

    Args:
        tools: List of Python callables injected into the REPL namespace.
               Each callable must have a ``__name__`` and ``__doc__``.
        checkpointer: A LangGraph checkpointer (AsyncSqliteSaver, MemorySaver, …).
                      If None, falls back to MemorySaver.
        model: A pre-built LangChain chat model object (ChatOpenAI / ChatAnthropic).
               Created by OracleV2Wrapper from its api_key + model name.

    Returns:
        A compiled LangGraph CompiledStateGraph ready for ``astream_events()``.

    Raises:
        ImportError: If langgraph-codeact is not installed.
    """
    if not _CODEACT_AVAILABLE:
        raise ImportError(
            "langgraph-codeact is not installed. "
            "Install it with: uv pip install langgraph-codeact"
        )

    from .sandbox import make_sandbox_eval
    from .state import OracleState

    # Use MemorySaver fallback if no checkpointer provided
    effective_checkpointer = checkpointer
    if effective_checkpointer is None and _MEMORY_SAVER_AVAILABLE:
        # Use pickle serde so CodeAct's function-containing REPL context can be checkpointed
        effective_checkpointer = MemorySaver(serde=_PickleSerde())
        logger.warning("No checkpointer provided — using MemorySaver (non-persistent)")
    elif effective_checkpointer is None:
        raise ImportError(
            "No checkpointer provided and langgraph MemorySaver is unavailable. "
            "Install langgraph: uv pip install langgraph"
        )

    eval_fn = make_sandbox_eval(tools)

    graph = create_codeact(
        model=model,
        tools=tools,
        eval_fn=eval_fn,
        state_schema=OracleState,
    )

    compiled = graph.compile(checkpointer=effective_checkpointer)
    logger.debug("Oracle CodeAct graph compiled with %d tools", len(tools))
    return compiled


# ---------------------------------------------------------------------------
# Tool assembly
# ---------------------------------------------------------------------------

def build_oracle_tools(
    user_id: str,
    project_id: str,
    openrouter_api_key: Optional[str] = None,
    graphiti: Any = None,
    include_web_tools: bool = False,
    tavily_api_key: Optional[str] = None,
    search_provider: Optional[str] = None,
) -> list[Callable]:
    """Assemble the default tool list for a given user + project.

    Args:
        user_id: Authenticated user (for vault scoping).
        project_id: CodeRAG project identifier.
        openrouter_api_key: API key for vector search in coderag.
        graphiti: Optional Graphiti instance for memory tools.
        include_web_tools: Gate for web_search/web_fetch/deep_research tools.
        tavily_api_key: Tavily API key for web search (if include_web_tools=True).
        search_provider: Search provider override (tavily, openrouter, etc.).

    Returns:
        List of plain callable tools ready for REPL injection.
    """
    from .tools.code_tools import make_code_tools

    tools: list[Callable] = []
    tools.extend(make_code_tools(project_id, openrouter_api_key=openrouter_api_key))

    # Vault tools
    try:
        from .tools.vault_tools import make_vault_tools  # type: ignore[import]
        tools.extend(make_vault_tools(user_id=user_id, project_id=project_id))
        logger.debug("Vault tools loaded (%d total tools)", len(tools))
    except (ImportError, Exception) as exc:
        logger.debug("vault_tools unavailable: %s", exc)

    # Memory tools (T028) — require graphiti; degrade gracefully when None
    try:
        from .tools.memory_tools import make_memory_tools  # type: ignore[import]
        tools.extend(make_memory_tools(graphiti=graphiti, user_id=user_id, project_id=project_id))
        logger.debug("Memory tools loaded (%d total tools)", len(tools))
    except (ImportError, Exception) as exc:
        logger.debug("memory_tools unavailable: %s", exc)

    # Shell tools (T034)
    try:
        from .tools.shell_tools import make_shell_tools  # type: ignore[import]
        tools.extend(make_shell_tools())
        logger.debug("Shell tools loaded (%d total tools)", len(tools))
    except (ImportError, Exception) as exc:
        logger.debug("shell_tools unavailable: %s", exc)

    # Thread tools (T034)
    try:
        from .tools.thread_tools import make_thread_tools  # type: ignore[import]
        tools.extend(make_thread_tools(user_id=user_id))
        logger.debug("Thread tools loaded (%d total tools)", len(tools))
    except (ImportError, Exception) as exc:
        logger.debug("thread_tools unavailable: %s", exc)

    # Web tools (T034) — gated behind include_web_tools flag
    if include_web_tools:
        try:
            from .tools.web_tools import make_web_tools  # type: ignore[import]
            tools.extend(make_web_tools(
                search_provider=search_provider or "tavily",
                tavily_api_key=tavily_api_key or "",
                openrouter_api_key=openrouter_api_key or "",
            ))
            logger.debug("Web tools loaded (%d total tools)", len(tools))
        except (ImportError, Exception) as exc:
            logger.debug("web_tools unavailable: %s", exc)

    # Meta tools (T034) — must be last (takes full tool list as input)
    try:
        from .tools.meta_tools import make_meta_tools  # type: ignore[import]
        tools.extend(make_meta_tools(tools))
        logger.debug("Meta tools loaded (%d total tools)", len(tools))
    except (ImportError, Exception) as exc:
        logger.debug("meta_tools unavailable: %s", exc)

    return tools


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_chat_model(
    api_key: str,
    model_name: str,
    max_tokens: int = 16000,
    base_url: Optional[str] = None,
) -> Any:
    """Construct a LangChain ChatOpenAI pointed at OpenRouter.

    Args:
        api_key: OpenRouter (or GLM) API key.
        model_name: OpenRouter model identifier (e.g. ``anthropic/claude-sonnet-4-6``).
        max_tokens: Maximum tokens for model responses.
        base_url: Override base URL (None = OpenRouter default).

    Returns:
        ChatOpenAI instance configured for OpenRouter.

    Raises:
        ImportError: If langchain-openai is not installed.
    """
    if not _LANGCHAIN_OPENAI_AVAILABLE:
        raise ImportError(
            "langchain-openai is not installed. "
            "Install it with: uv pip install langchain-openai"
        )

    effective_base_url = base_url or OPENROUTER_BASE_URL

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url=effective_base_url,
        max_tokens=max_tokens,
        streaming=True,
    )
