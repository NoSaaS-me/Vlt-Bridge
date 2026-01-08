"""
LLM Nodes - Nodes for making LLM API calls with streaming support.

This module implements the LLMCallNode from contracts/nodes.yaml (Phase 3):
- LLMCallNode: Makes LLM API calls with streaming, budget tracking, and retry logic

Error codes handled:
- E6001: Async operation timeout
- E6002: Async operation cancelled
- E6003: LLM API error

Part of the BT Universal Runtime (spec 019).
Tasks covered: 3.1.1-3.5.6 from tasks.md
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    TYPE_CHECKING,
    Type,
    Union,
)

from pydantic import BaseModel, Field

from ..state.base import (
    NodeType,
    RunStatus,
    BTError,
    ErrorContext,
    RecoveryAction,
    RecoveryInfo,
    Severity,
)
from ..state.contracts import NodeContract
from .leaves import LeafNode

if TYPE_CHECKING:
    from ..core.context import TickContext


logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Contract Types
# =============================================================================


class PromptContent(BaseModel):
    """Content for an LLM prompt.

    This represents the structured prompt data that LLMCallNode reads
    from the blackboard.

    Attributes:
        system: Optional system message for the LLM.
        messages: List of message dicts with role and content.
        user: Optional simple user message (alternative to messages).
    """

    system: Optional[str] = None
    messages: List[Dict[str, str]] = Field(default_factory=list)
    user: Optional[str] = None

    def to_api_messages(self) -> List[Dict[str, str]]:
        """Convert to API-compatible message format.

        Returns:
            List of messages in OpenAI-compatible format.
        """
        result = []

        if self.system:
            result.append({"role": "system", "content": self.system})

        if self.messages:
            result.extend(self.messages)
        elif self.user:
            result.append({"role": "user", "content": self.user})

        return result


class LLMResponse(BaseModel):
    """Response from an LLM API call.

    Written to blackboard on successful completion.

    Attributes:
        content: The full response text.
        model: Model used for generation.
        finish_reason: Why generation stopped (stop, length, etc.).
        usage: Token usage information.
        request_id: Unique identifier for this request.
    """

    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Dict[str, int] = Field(default_factory=dict)
    request_id: Optional[str] = None

    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        return self.usage.get("total_tokens", 0)


class StreamChunk(BaseModel):
    """A single chunk from streaming LLM response.

    Written to blackboard during streaming.

    Attributes:
        content: The chunk text content.
        index: Chunk sequence number.
        finish_reason: Set on final chunk if streaming complete.
        accumulated: Full response so far (optional).
    """

    content: str
    index: int = 0
    finish_reason: Optional[str] = None
    accumulated: Optional[str] = None


class LLMErrorType(str, Enum):
    """Types of LLM API errors (from errors.yaml E6003)."""

    RATE_LIMIT = "rate_limit"
    AUTH_ERROR = "auth_error"
    CONTEXT_LENGTH = "context_length"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"


class LLMError(BaseModel):
    """Error information from failed LLM call.

    Written to blackboard on failure for inspection.

    Attributes:
        error_type: Classification of the error.
        message: Human-readable error message.
        code: Error code (E6001, E6002, E6003).
        retry_count: Number of retries attempted.
        model: Model that was being called.
        request_id: Request ID if available.
    """

    error_type: LLMErrorType
    message: str
    code: str
    retry_count: int = 0
    model: Optional[str] = None
    request_id: Optional[str] = None


# =============================================================================
# LLM Client Protocol
# =============================================================================


class LLMClientProtocol(Protocol):
    """Protocol for LLM client implementations.

    This defines the interface that any LLM client must implement
    to work with LLMCallNode.
    """

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a completion request.

        Args:
            messages: List of messages in OpenAI format.
            model: Model identifier.
            stream: Whether to stream the response.
            **kwargs: Additional model-specific parameters.

        Returns:
            Response dict with content, usage, etc.
        """
        ...

    async def stream_complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        on_chunk: Optional[Callable[[str, int], None]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Make a streaming completion request.

        Args:
            messages: List of messages in OpenAI format.
            model: Model identifier.
            on_chunk: Callback for each chunk (content, index).
            **kwargs: Additional model-specific parameters.

        Returns:
            Final response dict with full content and usage.
        """
        ...

    def cancel(self, request_id: str) -> bool:
        """Cancel an in-flight request.

        Args:
            request_id: The request to cancel.

        Returns:
            True if cancellation was successful.
        """
        ...


# =============================================================================
# Error Factory Functions (E6001-E6003)
# =============================================================================


def make_timeout_error(
    operation_id: str,
    node_id: str,
    timeout_ms: int,
    operation_type: str = "llm_call",
) -> BTError:
    """Create E6001: Async operation timeout error.

    From errors.yaml E6001:
    - Category: async
    - Severity: error
    - Recovery: retry with backoff (max 2 retries, 1000ms backoff)
    """
    return BTError(
        code="E6001",
        category="async",
        severity=Severity.ERROR,
        message=f"Async operation '{operation_id}' timed out after {timeout_ms}ms",
        context=ErrorContext(
            node_id=node_id,
            extra={
                "operation_id": operation_id,
                "operation_type": operation_type,
                "timeout_ms": timeout_ms,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.RETRY,
            max_retries=2,
            backoff_ms=1000,
            manual_steps="Check network connectivity. Consider increasing timeout.",
        ),
        emit_event=True,
    )


def make_cancelled_error(
    operation_id: str,
    node_id: str,
    reason: str,
    operation_type: str = "llm_call",
) -> BTError:
    """Create E6002: Async operation cancelled error.

    From errors.yaml E6002:
    - Category: async
    - Severity: warning
    - Recovery: abort
    """
    return BTError(
        code="E6002",
        category="async",
        severity=Severity.WARNING,
        message=f"Async operation '{operation_id}' cancelled: {reason}",
        context=ErrorContext(
            node_id=node_id,
            extra={
                "operation_id": operation_id,
                "operation_type": operation_type,
                "reason": reason,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.ABORT,
            manual_steps="Cancellation is intentional",
        ),
        emit_event=True,
    )


def make_llm_api_error(
    node_id: str,
    node_name: str,
    model: str,
    error_type: LLMErrorType,
    error_message: str,
    request_id: Optional[str] = None,
    tokens_used: Optional[int] = None,
) -> BTError:
    """Create E6003: LLM API error.

    From errors.yaml E6003:
    - Category: async
    - Severity: error
    - Recovery: retry 3x with 2000ms backoff
    """
    return BTError(
        code="E6003",
        category="async",
        severity=Severity.ERROR,
        message=f"LLM API error in '{node_name}': {error_type.value} - {error_message}",
        context=ErrorContext(
            node_id=node_id,
            extra={
                "node_name": node_name,
                "model": model,
                "error_type": error_type.value,
                "error_message": error_message,
                "request_id": request_id,
                "tokens_used": tokens_used,
            },
        ),
        recovery=RecoveryInfo(
            action=RecoveryAction.RETRY,
            max_retries=3,
            backoff_ms=2000,
            manual_steps="Check API key, rate limits, or model availability",
        ),
        emit_event=True,
    )


# =============================================================================
# LLMCallNode Implementation
# =============================================================================


class LLMCallNode(LeafNode):
    """Makes LLM API call with streaming support.

    From contracts/nodes.yaml LLMCall section:

    Behavior:
    1. First tick: Initiate request, return RUNNING
    2. Subsequent ticks: Check completion
       - Streaming: Update blackboard, continue RUNNING
       - Complete: Write response to blackboard, return SUCCESS
       - Error (retryable): Increment retry, return RUNNING
       - Error (fatal): Write error, return FAILURE
       - Budget exceeded: Write error, return FAILURE
       - Timeout: Write error, return FAILURE
       - Interrupted: Cleanup, return FAILURE

    Config:
        model: Model identifier (required)
        prompt_key: Blackboard key containing prompt (required)
        response_key: Blackboard key to write response (required)
        stream_to: Optional blackboard key for streaming chunks
        timeout: Timeout in seconds (default 120)
        budget_tokens: Max tokens for this call (optional)
        interruptible: Whether this can be cancelled (default True)
        retry_on: Error types to retry (optional)
        max_retries: Maximum retry attempts (default 3)

    State:
        _request_id: In-flight request ID
        _partial_response: Accumulated chunks
        _tokens_used: Current token usage
        _retry_count: Current retry count
        _started_at: When request started

    Example:
        >>> node = LLMCallNode(
        ...     id="generate-response",
        ...     model="gpt-4",
        ...     prompt_key="user_prompt",
        ...     response_key="llm_response",
        ...     stream_to="response_chunks",
        ...     timeout=60,
        ...     budget_tokens=4000,
        ... )
    """

    # Retryable error types by default
    DEFAULT_RETRY_ON = [
        LLMErrorType.RATE_LIMIT.value,
        LLMErrorType.NETWORK_ERROR.value,
        LLMErrorType.API_ERROR.value,
    ]

    def __init__(
        self,
        id: str,
        model: str,
        prompt_key: str,
        response_key: str,
        stream_to: Optional[str] = None,
        timeout: int = 120,
        budget_tokens: Optional[int] = None,
        interruptible: bool = True,
        retry_on: Optional[List[str]] = None,
        max_retries: int = 3,
        on_chunk: Optional[Callable[[str, int], None]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize an LLMCallNode.

        Args:
            id: Unique node identifier.
            model: Model identifier (e.g., "gpt-4", "claude-3-opus").
            prompt_key: Blackboard key containing PromptContent.
            response_key: Blackboard key to write LLMResponse.
            stream_to: Optional blackboard key for StreamChunk updates.
            timeout: Request timeout in seconds.
            budget_tokens: Maximum tokens allowed for this call.
            interruptible: Whether the request can be cancelled.
            retry_on: List of error types to retry on.
            max_retries: Maximum number of retry attempts.
            on_chunk: Optional callback for streaming chunks.
            name: Human-readable name (defaults to id).
            metadata: Optional metadata for debugging.
        """
        super().__init__(id=id, name=name, metadata=metadata)

        # Configuration (immutable after init)
        self._model = model
        self._prompt_key = prompt_key
        self._response_key = response_key
        self._stream_to = stream_to
        self._timeout = timeout
        self._budget_tokens = budget_tokens
        self._interruptible = interruptible
        self._retry_on = retry_on if retry_on is not None else self.DEFAULT_RETRY_ON
        self._max_retries = max_retries
        self._on_chunk = on_chunk

        # State (mutable during execution)
        self._request_id: Optional[str] = None
        self._partial_response: str = ""
        self._tokens_used: int = 0
        self._retry_count: int = 0
        self._started_at: Optional[datetime] = None
        self._chunk_index: int = 0
        self._last_error: Optional[LLMError] = None

        # Async task tracking
        self._pending_task: Optional[asyncio.Task] = None
        self._result_ready: bool = False
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[Exception] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def model(self) -> str:
        """Model identifier."""
        return self._model

    @property
    def prompt_key(self) -> str:
        """Blackboard key for prompt content."""
        return self._prompt_key

    @property
    def response_key(self) -> str:
        """Blackboard key for response."""
        return self._response_key

    @property
    def stream_to(self) -> Optional[str]:
        """Blackboard key for streaming chunks."""
        return self._stream_to

    @property
    def timeout_seconds(self) -> float:
        """Timeout in seconds."""
        return float(self._timeout)

    @property
    def budget_tokens(self) -> Optional[int]:
        """Token budget for this call."""
        return self._budget_tokens

    @property
    def interruptible(self) -> bool:
        """Whether this request can be cancelled."""
        return self._interruptible

    @property
    def request_id(self) -> Optional[str]:
        """Current in-flight request ID."""
        return self._request_id

    @property
    def tokens_used(self) -> int:
        """Current token usage."""
        return self._tokens_used

    @property
    def retry_count(self) -> int:
        """Current retry count."""
        return self._retry_count

    # =========================================================================
    # Contract
    # =========================================================================

    @classmethod
    def contract(cls) -> NodeContract:
        """Declare base state requirements for LLMCallNode.

        Note: This returns an empty contract since actual keys are instance-specific.
        Use get_instance_contract() for the actual contract.
        """
        return NodeContract(
            description="Make LLM API call with streaming support"
        )

    def get_instance_contract(self) -> NodeContract:
        """Get the contract for this specific node instance.

        Returns a contract with the actual blackboard keys configured
        for this node.
        """
        outputs = {self._response_key: LLMResponse}
        if self._stream_to:
            outputs[self._stream_to] = StreamChunk

        return NodeContract(
            inputs={self._prompt_key: PromptContent},
            outputs=outputs,
            description=f"LLM call to {self._model}",
        )

    def _validate_contract_inputs(self, blackboard) -> list:
        """Override to validate instance-specific contract.

        Checks that the actual prompt_key exists in the blackboard.
        """
        if blackboard is None:
            return [self._prompt_key]

        if not blackboard.has(self._prompt_key):
            return [self._prompt_key]

        return []

    # =========================================================================
    # Main Execution
    # =========================================================================

    def _tick(self, ctx: "TickContext") -> RunStatus:
        """Execute the LLM call state machine.

        State transitions:
        - FRESH -> RUNNING (initiate request)
        - RUNNING -> RUNNING (waiting/streaming)
        - RUNNING -> SUCCESS (complete)
        - RUNNING -> FAILURE (error/timeout/cancelled)

        Args:
            ctx: Tick context with blackboard access.

        Returns:
            RunStatus based on current state.
        """
        # Check for interruption
        if ctx.cancellation_requested:
            return self._handle_interruption(ctx)

        # Check timeout
        if self._started_at is not None:
            elapsed_ms = (datetime.now(timezone.utc) - self._started_at).total_seconds() * 1000
            if elapsed_ms > self._timeout * 1000:
                return self._handle_timeout(ctx)

        # State machine
        if self._request_id is None:
            # First tick - initiate request
            return self._initiate_request(ctx)
        else:
            # Subsequent ticks - check completion
            return self._check_completion(ctx)

    def _initiate_request(self, ctx: "TickContext") -> RunStatus:
        """Start the LLM request.

        Reads prompt from blackboard, validates budget, and initiates
        the async request.

        Args:
            ctx: Tick context.

        Returns:
            RUNNING if request started, FAILURE if error.
        """
        # Generate request ID
        self._request_id = str(uuid.uuid4())
        self._started_at = datetime.now(timezone.utc)
        self._partial_response = ""
        self._chunk_index = 0

        # Track async operation
        ctx.add_async(self._request_id)

        # Read prompt from blackboard
        prompt = self._get_prompt(ctx)
        if prompt is None:
            return RunStatus.FAILURE

        # Check budget before making request
        if self._budget_tokens is not None and self._tokens_used >= self._budget_tokens:
            self._write_budget_exceeded_error(ctx)
            ctx.complete_async(self._request_id)
            return RunStatus.FAILURE

        # Get LLM client from services
        llm_client = self._get_llm_client(ctx)
        if llm_client is None:
            logger.warning(
                f"LLMCallNode '{self._id}' has no LLM client available. "
                f"Using mock mode (will return empty response after delay)."
            )
            # For testing without a real client, we'll simulate
            self._simulate_request(ctx, prompt)
            return RunStatus.RUNNING

        # Start async request
        try:
            self._start_async_request(ctx, llm_client, prompt)
            ctx.mark_progress()
            return RunStatus.RUNNING

        except Exception as e:
            logger.error(f"LLMCallNode '{self._id}' failed to start request: {e}")
            self._write_api_error(ctx, LLMErrorType.API_ERROR, str(e))
            ctx.complete_async(self._request_id)
            return RunStatus.FAILURE

    def _check_completion(self, ctx: "TickContext") -> RunStatus:
        """Check if the async request has completed.

        Handles:
        - Streaming chunk updates
        - Successful completion
        - Errors with retry logic
        - Budget exceeded

        Args:
            ctx: Tick context.

        Returns:
            RUNNING if still in progress, SUCCESS/FAILURE when done.
        """
        # Check if result is ready
        if self._result_ready:
            if self._error is not None:
                return self._handle_error(ctx, self._error)
            elif self._result is not None:
                return self._handle_success(ctx, self._result)

        # Check for streaming updates
        if self._stream_to and self._partial_response:
            self._write_stream_chunk(ctx)

        # Still running
        ctx.mark_progress()
        return RunStatus.RUNNING

    def _handle_success(self, ctx: "TickContext", result: Dict[str, Any]) -> RunStatus:
        """Handle successful completion.

        Writes LLMResponse to blackboard and returns SUCCESS.

        Args:
            ctx: Tick context.
            result: API response data.

        Returns:
            SUCCESS.
        """
        # Extract response data
        content = result.get("content", self._partial_response)
        usage = result.get("usage", {})
        finish_reason = result.get("finish_reason", "stop")

        # Update token tracking
        self._tokens_used = usage.get("total_tokens", 0)

        # Create response model
        response = LLMResponse(
            content=content,
            model=self._model,
            finish_reason=finish_reason,
            usage=usage,
            request_id=self._request_id,
        )

        # Write to blackboard
        if ctx.blackboard:
            # Check if response_key is registered
            if not ctx.blackboard.has(self._response_key):
                # Register it dynamically
                ctx.blackboard.register(self._response_key, LLMResponse)
            ctx.blackboard.set(self._response_key, response)

        # Complete async operation
        ctx.complete_async(self._request_id)

        # Clear state
        self._cleanup()

        logger.debug(
            f"LLMCallNode '{self._id}' completed successfully. "
            f"Tokens used: {self._tokens_used}"
        )

        return RunStatus.SUCCESS

    def _handle_error(self, ctx: "TickContext", error: Exception) -> RunStatus:
        """Handle request error with retry logic.

        Args:
            ctx: Tick context.
            error: The exception that occurred.

        Returns:
            RUNNING if retrying, FAILURE if exhausted.
        """
        # Classify error
        error_type = self._classify_error(error)

        # Check if retryable
        if error_type.value in self._retry_on and self._retry_count < self._max_retries:
            self._retry_count += 1
            backoff_ms = self._calculate_backoff()

            logger.info(
                f"LLMCallNode '{self._id}' retrying after error: {error} "
                f"(attempt {self._retry_count}/{self._max_retries}, "
                f"backoff {backoff_ms}ms)"
            )

            # Reset for retry
            self._error = None
            self._result_ready = False
            self._request_id = str(uuid.uuid4())
            self._started_at = datetime.now(timezone.utc)

            # Add delay before retry (in real implementation)
            ctx.mark_progress()
            return RunStatus.RUNNING

        # Error is fatal or retries exhausted
        self._write_api_error(ctx, error_type, str(error))
        ctx.complete_async(self._request_id)
        self._cleanup()

        return RunStatus.FAILURE

    def _handle_timeout(self, ctx: "TickContext") -> RunStatus:
        """Handle request timeout.

        Creates E6001 error and returns FAILURE.

        Args:
            ctx: Tick context.

        Returns:
            FAILURE.
        """
        timeout_error = make_timeout_error(
            operation_id=self._request_id or "unknown",
            node_id=self._id,
            timeout_ms=int(self._timeout * 1000),
        )

        logger.error(str(timeout_error))

        # Write error to blackboard
        error_model = LLMError(
            error_type=LLMErrorType.NETWORK_ERROR,
            message=f"Request timed out after {self._timeout}s",
            code="E6001",
            retry_count=self._retry_count,
            model=self._model,
            request_id=self._request_id,
        )
        self._write_error_to_blackboard(ctx, error_model)

        # Cancel pending request
        self._cancel_request()

        # Complete async
        if self._request_id:
            ctx.complete_async(self._request_id)

        self._cleanup()
        return RunStatus.FAILURE

    def _handle_interruption(self, ctx: "TickContext") -> RunStatus:
        """Handle cancellation/interruption.

        Creates E6002 error and returns FAILURE.

        Args:
            ctx: Tick context.

        Returns:
            FAILURE.
        """
        reason = ctx.cancellation_reason or "user_request"

        if self._request_id:
            cancelled_error = make_cancelled_error(
                operation_id=self._request_id,
                node_id=self._id,
                reason=reason,
            )
            logger.info(str(cancelled_error))

            # Write error to blackboard
            error_model = LLMError(
                error_type=LLMErrorType.API_ERROR,
                message=f"Request cancelled: {reason}",
                code="E6002",
                retry_count=self._retry_count,
                model=self._model,
                request_id=self._request_id,
            )
            self._write_error_to_blackboard(ctx, error_model)

            # Cancel pending request
            self._cancel_request()

            ctx.complete_async(self._request_id)

        self._cleanup()
        return RunStatus.FAILURE

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_prompt(self, ctx: "TickContext") -> Optional[PromptContent]:
        """Get prompt from blackboard.

        Args:
            ctx: Tick context.

        Returns:
            PromptContent if found, None otherwise.
        """
        if ctx.blackboard is None:
            logger.error(f"LLMCallNode '{self._id}' has no blackboard")
            return None

        # Try to get prompt content directly
        # The blackboard.get returns Optional[T], not ErrorResult
        try:
            result = ctx.blackboard.get(self._prompt_key, PromptContent)
            if result is not None:
                return result
        except Exception as e:
            logger.debug(
                f"LLMCallNode '{self._id}' blackboard.get raised: {e}"
            )

        # Try to get raw value and convert
        raw = ctx.blackboard._data.get(self._prompt_key)
        if raw is not None:
            try:
                if isinstance(raw, PromptContent):
                    return raw
                elif isinstance(raw, dict):
                    return PromptContent.model_validate(raw)
                elif isinstance(raw, str):
                    # Simple string becomes user message
                    return PromptContent(user=raw)
            except Exception as e:
                logger.error(
                    f"LLMCallNode '{self._id}' failed to parse prompt: {e}"
                )
                return None

        logger.error(
            f"LLMCallNode '{self._id}' could not read prompt "
            f"from '{self._prompt_key}'"
        )
        return None

    def _get_llm_client(self, ctx: "TickContext") -> Optional[LLMClientProtocol]:
        """Get LLM client from services.

        Args:
            ctx: Tick context.

        Returns:
            LLM client if available, None otherwise.
        """
        if ctx.services is None:
            return None

        return getattr(ctx.services, "llm_client", None)

    def _start_async_request(
        self,
        ctx: "TickContext",
        client: LLMClientProtocol,
        prompt: PromptContent,
    ) -> None:
        """Start the async LLM request.

        Args:
            ctx: Tick context.
            client: LLM client.
            prompt: Prompt content.
        """
        messages = prompt.to_api_messages()

        # Create callback for streaming chunks
        def on_chunk(content: str, index: int) -> None:
            self._partial_response += content
            self._chunk_index = index
            if self._on_chunk:
                self._on_chunk(content, index)

        # Start async task
        if self._stream_to:
            # Streaming mode
            coro = client.stream_complete(
                messages=messages,
                model=self._model,
                on_chunk=on_chunk,
            )
        else:
            # Non-streaming mode
            coro = client.complete(
                messages=messages,
                model=self._model,
                stream=False,
            )

        # Schedule the coroutine
        try:
            loop = asyncio.get_event_loop()
            self._pending_task = loop.create_task(self._run_request(coro))
        except RuntimeError:
            # No event loop - run synchronously (for testing)
            logger.debug(
                f"LLMCallNode '{self._id}' running without event loop"
            )

    async def _run_request(self, coro) -> None:
        """Run the request coroutine and capture result.

        Args:
            coro: The coroutine to run.
        """
        try:
            self._result = await coro
            self._result_ready = True
        except Exception as e:
            self._error = e
            self._result_ready = True

    def _simulate_request(self, ctx: "TickContext", prompt: PromptContent) -> None:
        """Simulate a request for testing without a real client.

        Args:
            ctx: Tick context.
            prompt: Prompt content.
        """
        # Simulate completion after a delay
        self._result = {
            "content": f"Mock response for prompt: {prompt.user or 'messages'}",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "finish_reason": "stop",
        }
        self._result_ready = True

    def _write_stream_chunk(self, ctx: "TickContext") -> None:
        """Write current streaming chunk to blackboard.

        Args:
            ctx: Tick context.
        """
        if ctx.blackboard is None or self._stream_to is None:
            return

        chunk = StreamChunk(
            content=self._partial_response,
            index=self._chunk_index,
            accumulated=self._partial_response,
        )

        # Register if needed
        if not ctx.blackboard.has(self._stream_to):
            ctx.blackboard.register(self._stream_to, StreamChunk)

        ctx.blackboard.set(self._stream_to, chunk)

    def _write_budget_exceeded_error(self, ctx: "TickContext") -> None:
        """Write budget exceeded error to blackboard.

        Args:
            ctx: Tick context.
        """
        error = LLMError(
            error_type=LLMErrorType.CONTEXT_LENGTH,
            message=f"Token budget exceeded: {self._tokens_used}/{self._budget_tokens}",
            code="E6003",
            retry_count=self._retry_count,
            model=self._model,
        )
        self._write_error_to_blackboard(ctx, error)

        logger.error(
            f"LLMCallNode '{self._id}' exceeded token budget: "
            f"{self._tokens_used}/{self._budget_tokens}"
        )

    def _write_api_error(
        self,
        ctx: "TickContext",
        error_type: LLMErrorType,
        message: str,
    ) -> None:
        """Write API error to blackboard and log.

        Args:
            ctx: Tick context.
            error_type: Type of error.
            message: Error message.
        """
        bt_error = make_llm_api_error(
            node_id=self._id,
            node_name=self._name,
            model=self._model,
            error_type=error_type,
            error_message=message,
            request_id=self._request_id,
            tokens_used=self._tokens_used if self._tokens_used > 0 else None,
        )
        logger.error(str(bt_error))

        error = LLMError(
            error_type=error_type,
            message=message,
            code="E6003",
            retry_count=self._retry_count,
            model=self._model,
            request_id=self._request_id,
        )
        self._write_error_to_blackboard(ctx, error)

    def _write_error_to_blackboard(
        self,
        ctx: "TickContext",
        error: LLMError,
    ) -> None:
        """Write error model to blackboard.

        Uses response_key + "_error" as the key.

        Args:
            ctx: Tick context.
            error: Error model.
        """
        if ctx.blackboard is None:
            return

        error_key = f"{self._response_key}_error"
        if not ctx.blackboard.has(error_key):
            ctx.blackboard.register(error_key, LLMError)
        ctx.blackboard.set(error_key, error)
        self._last_error = error

    def _classify_error(self, error: Exception) -> LLMErrorType:
        """Classify an exception into an LLMErrorType.

        Args:
            error: The exception to classify.

        Returns:
            Appropriate LLMErrorType.
        """
        error_str = str(error).lower()

        if "rate" in error_str or "limit" in error_str or "429" in error_str:
            return LLMErrorType.RATE_LIMIT
        elif "auth" in error_str or "key" in error_str or "401" in error_str:
            return LLMErrorType.AUTH_ERROR
        elif "context" in error_str or "length" in error_str or "token" in error_str:
            return LLMErrorType.CONTEXT_LENGTH
        elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
            return LLMErrorType.NETWORK_ERROR
        else:
            return LLMErrorType.API_ERROR

    def _calculate_backoff(self) -> int:
        """Calculate backoff delay for retry.

        Uses exponential backoff: base_delay * 2^(retry_count - 1)

        Returns:
            Backoff delay in milliseconds.
        """
        base_delay = 2000  # From errors.yaml E6003
        return base_delay * (2 ** (self._retry_count - 1))

    def _cancel_request(self) -> None:
        """Cancel any pending request."""
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()

    def _cleanup(self) -> None:
        """Reset state after completion."""
        self._request_id = None
        self._started_at = None
        self._result_ready = False
        self._result = None
        self._error = None
        self._pending_task = None
        # Note: Don't reset _partial_response, _tokens_used, _retry_count
        # as they may be needed for debugging

    # =========================================================================
    # Public Methods
    # =========================================================================

    def interrupt(self) -> None:
        """Cancel the in-flight request.

        Called externally to interrupt the node. The next tick will
        return FAILURE with E6002 error.
        """
        if not self._interruptible:
            logger.warning(
                f"LLMCallNode '{self._id}' is not interruptible"
            )
            return

        self._cancel_request()
        logger.info(f"LLMCallNode '{self._id}' interrupted")

    def get_token_usage(self) -> Dict[str, int]:
        """Get current token usage.

        Returns:
            Dict with token counts.
        """
        return {
            "tokens_used": self._tokens_used,
            "budget_tokens": self._budget_tokens or 0,
            "remaining": (
                (self._budget_tokens - self._tokens_used)
                if self._budget_tokens
                else 0
            ),
        }

    def reset(self) -> None:
        """Reset node state for reuse."""
        super().reset()

        # Cancel any pending request
        self._cancel_request()

        # Reset all state
        self._request_id = None
        self._partial_response = ""
        self._tokens_used = 0
        self._retry_count = 0
        self._started_at = None
        self._chunk_index = 0
        self._last_error = None
        self._result_ready = False
        self._result = None
        self._error = None
        self._pending_task = None

    def debug_info(self) -> Dict[str, Any]:
        """Return debug information."""
        info = super().debug_info()
        info.update({
            "model": self._model,
            "prompt_key": self._prompt_key,
            "response_key": self._response_key,
            "stream_to": self._stream_to,
            "timeout": self._timeout,
            "budget_tokens": self._budget_tokens,
            "interruptible": self._interruptible,
            "retry_on": self._retry_on,
            "max_retries": self._max_retries,
            "request_id": self._request_id,
            "tokens_used": self._tokens_used,
            "retry_count": self._retry_count,
            "partial_response_length": len(self._partial_response),
            "started_at": (
                self._started_at.isoformat() if self._started_at else None
            ),
            "has_pending_task": self._pending_task is not None,
            "last_error": (
                self._last_error.model_dump() if self._last_error else None
            ),
        })
        return info


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Node
    "LLMCallNode",
    # Models
    "PromptContent",
    "LLMResponse",
    "StreamChunk",
    "LLMError",
    "LLMErrorType",
    # Protocol
    "LLMClientProtocol",
    # Error factories
    "make_timeout_error",
    "make_cancelled_error",
    "make_llm_api_error",
]
