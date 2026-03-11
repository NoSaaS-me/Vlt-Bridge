"""Streaming adapter: LangGraph astream_events v2 → OracleStreamChunk SSE stream.

Event mapping (version="v2"):
  on_chat_model_stream  → content chunk
  on_tool_start         → tool_call chunk
  on_tool_end           → tool_result chunk
  on_custom_event       → progress chunk (name="repl_stdout")
  graph completion      → done chunk (context_id=thread_id)

If is_new_thread=True, a context_update chunk is emitted first so the
frontend can update its thread ID before streaming begins.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator, AsyncIterator, Optional

logger = logging.getLogger(__name__)


def _extract_text_content(chunk_content) -> str:
    """Extract plain text from a LangChain message content (str or list of parts).

    Anthropic responses may send content as a list of typed parts:
        [{"type": "text", "text": "..."}, ...]
    OpenAI/OpenRouter responses send a plain string.
    """
    if isinstance(chunk_content, str):
        return chunk_content
    if isinstance(chunk_content, list):
        parts: list[str] = []
        for part in chunk_content:
            if isinstance(part, dict):
                text = part.get("text", "")
                if text:
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return ""


async def oracle_to_sse(
    events: AsyncIterator,
    thread_id: str,
    is_new_thread: bool,
) -> AsyncGenerator:
    """Convert LangGraph astream_events to OracleStreamChunk stream.

    Args:
        events: AsyncIterator from ``graph.astream_events(version="v2", ...)``.
        thread_id: The LangGraph / oracle_threads thread identifier.
        is_new_thread: If True, emit a context_update chunk first.

    Yields:
        OracleStreamChunk instances (not yet JSON-serialised).
    """
    # Late import to avoid hard dependency at module level
    from ...models.oracle import OracleStreamChunk

    # Emit context_update so the frontend knows the thread ID immediately
    if is_new_thread:
        yield OracleStreamChunk(type="context_update", context_id=thread_id)

    try:
        async for event in events:
            event_type: str = event.get("event", "")
            data: dict = event.get("data", {}) or {}

            if event_type == "on_chat_model_stream":
                # data["chunk"] is an AIMessageChunk
                chunk_obj = data.get("chunk")
                if chunk_obj is not None:
                    raw_content = getattr(chunk_obj, "content", None)
                    if raw_content:
                        text = _extract_text_content(raw_content)
                        if text:
                            yield OracleStreamChunk(type="content", content=text)

            elif event_type == "on_tool_start":
                tool_name: str = event.get("name", "")
                tool_input = data.get("input")
                tool_run_id: str = event.get("run_id", "")
                yield OracleStreamChunk(
                    type="tool_call",
                    tool_call={
                        "name": tool_name,
                        "arguments": tool_input,
                    },
                    tool_call_id=tool_run_id,
                )

            elif event_type == "on_tool_end":
                tool_run_id = event.get("run_id", "")
                output = data.get("output")
                # output may be a ToolMessage or plain string
                if hasattr(output, "content"):
                    output_str = str(output.content)
                elif output is not None:
                    output_str = str(output)
                else:
                    output_str = ""
                yield OracleStreamChunk(
                    type="tool_result",
                    tool_result=output_str,
                    tool_call_id=tool_run_id,
                )

            elif event_type == "on_custom_event":
                event_name: str = event.get("name", "")
                if event_name == "repl_stdout":
                    stdout_text = data if isinstance(data, str) else data.get("output", str(data))
                    if stdout_text:
                        yield OracleStreamChunk(type="progress", content=str(stdout_text))

            # Ignore other event types (on_chain_start, on_chain_end, etc.)

    except GeneratorExit:
        # Consumer stopped iterating — clean exit, no error chunk needed
        return

    except asyncio.CancelledError:
        # Cancellation is a clean termination — state remains resumable
        yield OracleStreamChunk(type="done", context_id=thread_id)
        return

    except Exception as exc:  # noqa: BLE001
        logger.exception("oracle_to_sse: unexpected error during streaming")
        yield OracleStreamChunk(
            type="error",
            error=f"Streaming error: {exc}",
        )
        return

    # Natural end of stream — emit done chunk
    yield OracleStreamChunk(type="done", context_id=thread_id)
