"""Oracle V2 — CodeAct-based Oracle using LangGraph.

Drop-in replacement for RLMOracleWrapper (023-oracle-codeact-rework).

Public surface:
    OracleV2Wrapper — matches RLMOracleWrapper.__init__ / cancel() / process_query() API.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)


class OracleV2Wrapper:
    """LangGraph CodeAct Oracle — drop-in for RLMOracleWrapper.

    Thread identity maps to LangGraph thread_id (uuid). Threads are persisted
    in the oracle_threads SQLite table (schema added in T007).

    Security invariant: api_key is NEVER stored in LangGraph state.
    It is passed via ``config["configurable"]["api_key"]`` which
    AsyncSqliteSaver does not persist.
    """

    def __init__(
        self,
        user_id: str,
        api_key: str,
        project_id: str,
        model: str = "anthropic/claude-sonnet-4-6",
        max_tokens: int = 16000,
        base_url: Optional[str] = None,
        checkpointer: Any = None,
        graphiti: Any = None,  # reserved for Phase 4 (memory graph)
    ) -> None:
        self._user_id = user_id
        self._api_key = api_key
        self._project_id = project_id
        self._model = model
        self._max_tokens = max_tokens
        self._base_url = base_url
        self._checkpointer = checkpointer
        self._graphiti = graphiti  # unused in Phase 3

        self._cancelled = False
        self._active_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Cancel the active stream. Sets flag + cancels asyncio task if running."""
        self._cancelled = True
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            logger.info("OracleV2Wrapper: cancelled active task for user=%s", self._user_id)

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _get_db_path(self) -> str:
        """Return the path to the SQLite index DB."""
        from ..database import DEFAULT_DB_PATH
        return str(DEFAULT_DB_PATH)

    def _insert_thread(self, thread_id: str, now_iso: str, title: Optional[str] = None) -> None:
        """Insert a new row into oracle_threads."""
        try:
            db_path = self._get_db_path()
            with sqlite3.connect(db_path, timeout=5) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO oracle_threads
                        (thread_id, user_id, project_id, title, created_at, last_active_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (thread_id, self._user_id, self._project_id, title, now_iso, now_iso),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("oracle_threads insert failed (non-fatal): %s", exc)

    def _update_thread_activity(
        self,
        thread_id: str,
        now_iso: str,
        title: Optional[str] = None,
    ) -> None:
        """Update last_active_at and optionally set title if still null."""
        try:
            db_path = self._get_db_path()
            with sqlite3.connect(db_path, timeout=5) as conn:
                if title:
                    conn.execute(
                        """
                        UPDATE oracle_threads
                        SET last_active_at = ?,
                            title = COALESCE(title, ?)
                        WHERE thread_id = ?
                        """,
                        (now_iso, title, thread_id),
                    )
                else:
                    conn.execute(
                        "UPDATE oracle_threads SET last_active_at = ? WHERE thread_id = ?",
                        (now_iso, thread_id),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("oracle_threads update failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    async def process_query(
        self,
        query: str,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator:
        """Run the CodeAct graph and yield OracleStreamChunk events.

        Args:
            query: User question / instruction.
            context_id: Existing oracle_threads thread_id to resume, or None to create new.

        Yields:
            OracleStreamChunk — same types as RLMOracleWrapper:
              progress, content, tool_call, tool_result, done, error, context_update.
        """
        from ...models.oracle import OracleStreamChunk

        # Reset cancellation flag at the start of each query
        self._cancelled = False

        # Resolve thread ID
        is_new_thread = context_id is None
        thread_id = context_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Auto-title from first 60 chars of query
        auto_title = query[:60].strip() if query else None

        if is_new_thread:
            self._insert_thread(thread_id, now_iso, title=auto_title)
        else:
            # Ensure the thread row exists (may have been created in a previous session)
            self._insert_thread(thread_id, now_iso, title=None)

        # Build model object — raises ImportError clearly if langchain-openai absent
        try:
            from .graph import build_chat_model, build_oracle_graph, build_oracle_tools
        except ImportError as exc:
            yield OracleStreamChunk(
                type="error",
                error=f"OracleV2 dependency missing: {exc}. "
                      "Install with: uv pip install langgraph-codeact langchain-openai",
            )
            return

        try:
            model_obj = build_chat_model(
                api_key=self._api_key,
                model_name=self._model,
                max_tokens=self._max_tokens,
                base_url=self._base_url,
            )
        except ImportError as exc:
            yield OracleStreamChunk(type="error", error=str(exc))
            return

        # Assemble tools — pass graphiti for memory tools (T028/T034)
        tools = build_oracle_tools(
            user_id=self._user_id,
            project_id=self._project_id,
            openrouter_api_key=self._api_key,
            graphiti=self._graphiti,
        )

        # Build (or reuse) the compiled graph
        try:
            graph = build_oracle_graph(
                tools=tools,
                checkpointer=self._checkpointer,
                model=model_obj,
            )
        except ImportError as exc:
            yield OracleStreamChunk(type="error", error=str(exc))
            return
        except Exception as exc:
            logger.exception("Failed to build oracle graph")
            yield OracleStreamChunk(type="error", error=f"Graph build error: {exc}")
            return

        # LangGraph config — api_key in configurable so it is NOT checkpointed
        config = {
            "configurable": {
                "thread_id": thread_id,
                "api_key": self._api_key,
                "oracle_model": self._model,
            }
        }

        # T027: Pre-load recalled facts from Graphiti before the turn starts
        recalled_facts: list[str] = []
        if self._graphiti is not None:
            try:
                from ..memory.loader import load_recalled_facts
                recalled_facts = await load_recalled_facts(
                    graphiti=self._graphiti,
                    query=query,
                    user_id=self._user_id,
                    project_id=self._project_id,
                )
                logger.debug("process_query: loaded %d recalled facts", len(recalled_facts))
            except Exception as exc:
                logger.warning("process_query: failed to load recalled facts: %s", exc)

        # T039: Classify query and optionally run planner for complex tasks
        from .graph import classify_query
        query_type = classify_query(query)
        plan_steps: list[str] = []

        if query_type == "plan":
            from .nodes import planner_node
            try:
                plan_result = await planner_node(
                    state={"messages": [{"role": "user", "content": query}]},
                    config=config,
                )
                plan_steps = plan_result.get("plan", [])
            except Exception as exc:
                logger.warning("process_query: planner_node failed: %s", exc)

        # Build user message content, prepending recalled facts as context (T027)
        user_content = query
        if recalled_facts:
            facts_prefix = "Recalled facts from memory:\n" + "\n".join(
                f"- {f}" for f in recalled_facts
            ) + "\n\n"
            user_content = facts_prefix + query

        # Initial state for a new thread; for resume LangGraph reloads state from checkpoint
        initial_state: dict = {
            "messages": [{"role": "user", "content": user_content}],
            "context": {},
            "user_id": self._user_id,
            "project_id": self._project_id,
            "plan": plan_steps or None,
            "plan_step": 0,
            "recalled_facts": recalled_facts,
        }

        # For resumed threads, only send the new user message (with memory prefix if any)
        if not is_new_thread:
            initial_state = {"messages": [{"role": "user", "content": user_content}]}

        # Stream via oracle_to_sse adapter
        from .streaming import oracle_to_sse

        async def _run_stream() -> AsyncGenerator:
            try:
                events = graph.astream_events(
                    initial_state,
                    config=config,
                    version="v2",
                )
                async for chunk in oracle_to_sse(events, thread_id, is_new_thread):
                    if self._cancelled:
                        break
                    yield chunk
            except asyncio.CancelledError:
                # Return a done chunk so the frontend doesn't hang
                yield OracleStreamChunk(type="done", context_id=thread_id)
                return
            except Exception as exc:
                logger.exception("OracleV2Wrapper stream error")
                yield OracleStreamChunk(type="error", error=f"Stream error: {exc}")
                yield OracleStreamChunk(type="done", context_id=thread_id)

        stream_gen = _run_stream()

        # T041: Emit plan progress chunk before streaming if planner generated steps
        if plan_steps:
            plan_text = "Plan:\n" + "\n".join(
                f"{i}. {step}" for i, step in enumerate(plan_steps, 1)
            )
            yield OracleStreamChunk(type="progress", content=plan_text)

        # _cancelled flag is checked after each chunk — asyncio.Task cancellation
        # is handled via cancel() which sets _cancelled=True and cancels _active_task.
        # For generator-based streams the flag check is sufficient.

        answer_parts: list[str] = []
        try:
            async for chunk in stream_gen:
                if chunk.type == "content" and chunk.content:
                    answer_parts.append(chunk.content)
                yield chunk
                if self._cancelled:
                    break
        finally:
            finished_at = datetime.now(timezone.utc).isoformat()
            self._update_thread_activity(thread_id, finished_at, title=auto_title)
            # T027: Fire-and-forget memory writer for cross-session recall
            if self._graphiti is not None and answer_parts:
                try:
                    from ..memory.writer import write_episode
                    asyncio.create_task(
                        write_episode(
                            graphiti=self._graphiti,
                            episode_body="".join(answer_parts),
                            user_id=self._user_id,
                            project_id=self._project_id,
                            thread_id=thread_id,
                            turn_index=0,
                        )
                    )
                except Exception as exc:
                    logger.warning("process_query: failed to schedule memory write: %s", exc)
