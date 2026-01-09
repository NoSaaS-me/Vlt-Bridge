"""OpenRouter LLM Client - HTTP client for OpenRouter API.

Implements LLMClientProtocol for use with BT LLMCallNode.
Extracted from oracle_agent.py for reuse in BT runtime.

Part of the BT Universal Runtime (spec 019).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)

# OpenRouter API base URL
OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class BTServices:
    """Container for services injected into TickContext.

    Provides dependency injection for BT nodes that need external services.
    """
    llm_client: Optional["OpenRouterClient"] = None
    tree_registry: Optional[Any] = None  # TreeRegistry for lazy subtree resolution

    # Future services can be added here:
    # tool_executor: Optional[ToolExecutor] = None
    # context_service: Optional[ContextService] = None


class OpenRouterClient:
    """HTTP client for OpenRouter API implementing LLMClientProtocol.

    Handles:
    - Non-streaming and streaming completions
    - Tool/function calling
    - Request cancellation
    - Reasoning/thinking trace extraction

    Example:
        >>> client = OpenRouterClient(api_key="sk-...")
        >>> result = await client.complete(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     model="deepseek/deepseek-chat"
        ... )
        >>> print(result["content"])
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE,
        timeout: float = 120.0,
        referer: str = "https://vlt.ai",
        title: str = "Vlt Oracle",
    ) -> None:
        """Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key.
            base_url: API base URL (default: OpenRouter).
            timeout: Request timeout in seconds.
            referer: HTTP-Referer header value.
            title: X-Title header value.
        """
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._referer = referer
        self._title = title

        # Track active requests for cancellation
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._cancelled: Set[str] = set()

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a non-streaming completion request.

        Args:
            messages: List of messages in OpenAI format.
            model: Model identifier.
            stream: Ignored (always False for this method).
            **kwargs: Additional parameters (tools, max_tokens, etc.)

        Returns:
            Response dict with:
            - content: Response text
            - tool_calls: List of tool calls (if any)
            - finish_reason: Why generation stopped
            - usage: Token usage stats
            - reasoning: Thinking/reasoning content (if model supports)
        """
        request_id = str(uuid.uuid4())

        request_body = self._build_request(messages, model, stream=False, **kwargs)

        logger.info(f"[OpenRouterClient] Non-streaming request to {model}")
        logger.debug(f"[OpenRouterClient] Messages: {len(messages)}, tools: {len(kwargs.get('tools', []))}")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=request_body,
                )
                response.raise_for_status()
                data = response.json()

                return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.TimeoutException:
            logger.error("OpenRouter API timeout")
            raise

    async def stream_complete(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        on_chunk: Optional[Callable[[str, int], None]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a streaming completion request.

        Args:
            messages: List of messages in OpenAI format.
            model: Model identifier.
            on_chunk: Callback for content chunks (content, index).
            **kwargs: Additional parameters (tools, max_tokens, etc.)

        Returns:
            Final response dict with accumulated content and tool_calls.
        """
        request_id = str(uuid.uuid4())

        request_body = self._build_request(messages, model, stream=True, **kwargs)

        logger.info(f"[OpenRouterClient] Streaming request to {model}")
        logger.debug(f"[OpenRouterClient] Messages: {len(messages)}, tools: {len(kwargs.get('tools', []))}")

        # Buffers for accumulation
        content_buffer = ""
        reasoning_buffer = ""
        tool_calls_buffer: Dict[int, Dict[str, Any]] = {}
        finish_reason = None
        chunk_index = 0
        usage = {}

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=request_body,
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        # Check for cancellation
                        if request_id in self._cancelled:
                            logger.info(f"[OpenRouterClient] Request {request_id} cancelled")
                            self._cancelled.discard(request_id)
                            raise asyncio.CancelledError("Request cancelled")

                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            logger.debug("[OpenRouterClient] Stream [DONE]")
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning(f"[OpenRouterClient] Failed to parse SSE: {data_str[:200]}")
                            continue

                        choices = data.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})

                        # Update finish_reason only if truthy
                        if choice.get("finish_reason"):
                            finish_reason = choice.get("finish_reason")

                        # Extract usage if present
                        if "usage" in data:
                            usage = data["usage"]

                        # Handle content
                        if "content" in delta and delta["content"]:
                            content_chunk = delta["content"]
                            content_buffer += content_chunk

                            # Call chunk callback
                            if on_chunk:
                                on_chunk(content_chunk, chunk_index)
                                chunk_index += 1

                        # Handle reasoning/thinking (DeepSeek, o1, etc.)
                        reasoning_chunk = self._extract_reasoning(delta)
                        if reasoning_chunk:
                            reasoning_buffer += reasoning_chunk

                        # Handle tool calls
                        if "tool_calls" in delta:
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_buffer:
                                    tool_calls_buffer[idx] = {
                                        "id": tc.get("id", ""),
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }

                                if "id" in tc and tc["id"]:
                                    tool_calls_buffer[idx]["id"] = tc["id"]

                                if "function" in tc:
                                    if "name" in tc["function"] and tc["function"]["name"]:
                                        tool_calls_buffer[idx]["function"]["name"] = tc["function"]["name"]
                                    if "arguments" in tc["function"] and tc["function"]["arguments"]:
                                        tool_calls_buffer[idx]["function"]["arguments"] += tc["function"]["arguments"]

            # Convert tool calls buffer to list
            tool_calls = None
            if tool_calls_buffer:
                tool_calls = [tool_calls_buffer[i] for i in sorted(tool_calls_buffer.keys())]

            logger.info(
                f"[OpenRouterClient] Stream complete: "
                f"content_len={len(content_buffer)}, "
                f"reasoning_len={len(reasoning_buffer)}, "
                f"tool_calls={len(tool_calls) if tool_calls else 0}, "
                f"finish_reason={finish_reason}"
            )

            return {
                "content": content_buffer,
                "reasoning": reasoning_buffer if reasoning_buffer else None,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason or "stop",
                "usage": usage,
                "request_id": request_id,
            }

        except httpx.HTTPStatusError as e:
            logger.error(f"OpenRouter API error: {e.response.status_code}")
            raise
        except httpx.TimeoutException:
            logger.error("OpenRouter API timeout")
            raise

    def cancel(self, request_id: str) -> bool:
        """Cancel an in-flight request.

        Args:
            request_id: The request to cancel.

        Returns:
            True if cancellation was requested.
        """
        self._cancelled.add(request_id)

        if request_id in self._active_requests:
            task = self._active_requests[request_id]
            task.cancel()
            return True

        return False

    def _build_request(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the API request body."""
        request = {
            "model": model,
            "messages": messages,
            "stream": stream,
        }

        # Add tools if provided
        tools = kwargs.get("tools")
        if tools:
            request["tools"] = tools
            request["tool_choice"] = kwargs.get("tool_choice", "auto")
            request["parallel_tool_calls"] = kwargs.get("parallel_tool_calls", True)

        # Add max_tokens if provided
        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            request["max_tokens"] = max_tokens

        # Add temperature if provided
        temperature = kwargs.get("temperature")
        if temperature is not None:
            request["temperature"] = temperature

        return request

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for the request."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._referer,
            "X-Title": self._title,
            "Content-Type": "application/json",
        }

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse non-streaming API response."""
        choices = data.get("choices", [])
        if not choices:
            return {
                "content": "",
                "tool_calls": None,
                "finish_reason": "error",
                "usage": data.get("usage", {}),
            }

        choice = choices[0]
        message = choice.get("message", {})

        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason", "stop"),
            "usage": data.get("usage", {}),
            "reasoning": self._extract_reasoning(message),
        }

    def _extract_reasoning(self, delta_or_message: Dict[str, Any]) -> Optional[str]:
        """Extract reasoning/thinking content from response delta or message.

        Different models use different formats:
        - DeepSeek: "reasoning" field
        - Some models: "reasoning_details" with list of dicts
        """
        # Direct reasoning field
        if "reasoning" in delta_or_message and delta_or_message["reasoning"]:
            return delta_or_message["reasoning"]

        # reasoning_details format
        if "reasoning_details" in delta_or_message:
            details = delta_or_message["reasoning_details"]
            if isinstance(details, list):
                parts = []
                for detail in details:
                    if isinstance(detail, dict) and detail.get("text"):
                        parts.append(detail["text"])
                if parts:
                    return "".join(parts)

        return None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "OpenRouterClient",
    "BTServices",
    "OPENROUTER_BASE",
]
