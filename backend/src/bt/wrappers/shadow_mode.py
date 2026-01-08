"""
Shadow Mode - Parallel execution of old and new Oracle for regression testing.

Shadow mode runs the original OracleAgent alongside the new BT-based Oracle,
compares their outputs, and logs any discrepancies for analysis.

Features:
1. Parallel execution of both implementations
2. Output comparison with detailed diff reporting
3. Discrepancy logging for debugging
4. Toggle via feature flag

Part of the BT Universal Runtime (spec 019).
Tasks covered: 5.3.1-5.3.4 from tasks.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
)

from .oracle_wrapper import OracleBTWrapper, OracleStreamChunk

if TYPE_CHECKING:
    from ...services.oracle_agent import OracleAgent

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Flag
# =============================================================================


def is_bt_oracle_enabled() -> bool:
    """Check if BT-based Oracle is enabled.

    Reads from environment variable ORACLE_USE_BT.
    Values: "true", "shadow", "false" (default)

    Returns:
        True if BT Oracle should be used exclusively.
    """
    value = os.environ.get("ORACLE_USE_BT", "false").lower()
    return value == "true"


def is_shadow_mode_enabled() -> bool:
    """Check if shadow mode is enabled.

    Shadow mode runs both implementations in parallel for comparison.

    Returns:
        True if shadow mode is enabled.
    """
    value = os.environ.get("ORACLE_USE_BT", "false").lower()
    return value == "shadow"


def get_oracle_mode() -> str:
    """Get current Oracle mode.

    Returns:
        "bt" - Use BT Oracle only
        "shadow" - Run both, compare, use legacy output
        "legacy" - Use legacy Oracle only (default)
    """
    value = os.environ.get("ORACLE_USE_BT", "false").lower()
    if value == "true":
        return "bt"
    elif value == "shadow":
        return "shadow"
    else:
        return "legacy"


# =============================================================================
# Discrepancy Types
# =============================================================================


@dataclass
class ChunkDiscrepancy:
    """Record of a discrepancy between BT and legacy outputs."""

    field: str
    bt_value: Any
    legacy_value: Any
    index: Optional[int] = None
    severity: str = "warning"  # warning, error, info


@dataclass
class ComparisonReport:
    """Full comparison report between BT and legacy execution."""

    timestamp: datetime
    user_id: str
    query_preview: str

    bt_chunks: List[Dict[str, Any]]
    legacy_chunks: List[Dict[str, Any]]

    discrepancies: List[ChunkDiscrepancy] = field(default_factory=list)
    match_rate: float = 0.0

    bt_duration_ms: float = 0.0
    legacy_duration_ms: float = 0.0

    bt_error: Optional[str] = None
    legacy_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "query_preview": self.query_preview,
            "bt_chunk_count": len(self.bt_chunks),
            "legacy_chunk_count": len(self.legacy_chunks),
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": [
                {
                    "field": d.field,
                    "bt_value": str(d.bt_value)[:100],
                    "legacy_value": str(d.legacy_value)[:100],
                    "index": d.index,
                    "severity": d.severity,
                }
                for d in self.discrepancies
            ],
            "match_rate": self.match_rate,
            "bt_duration_ms": self.bt_duration_ms,
            "legacy_duration_ms": self.legacy_duration_ms,
            "bt_error": self.bt_error,
            "legacy_error": self.legacy_error,
        }


# =============================================================================
# Shadow Mode Runner
# =============================================================================


class ShadowModeRunner:
    """Runs both Oracle implementations in parallel for comparison.

    Example:
        >>> runner = ShadowModeRunner(user_id="user1")
        >>> async for chunk in runner.run_parallel(query="Hello"):
        ...     print(chunk)  # Legacy chunks are yielded
        >>> report = runner.get_comparison_report()
        >>> print(report.discrepancies)
    """

    def __init__(
        self,
        user_id: str,
        project_id: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
    ) -> None:
        """Initialize shadow mode runner.

        Args:
            user_id: User ID for both implementations.
            project_id: Optional project ID.
            model: LLM model identifier.
            max_tokens: Maximum tokens for response.
        """
        self._user_id = user_id
        self._project_id = project_id
        self._model = model
        self._max_tokens = max_tokens

        # Collected chunks
        self._bt_chunks: List[Dict[str, Any]] = []
        self._legacy_chunks: List[Dict[str, Any]] = []

        # Timing
        self._bt_start: Optional[float] = None
        self._bt_end: Optional[float] = None
        self._legacy_start: Optional[float] = None
        self._legacy_end: Optional[float] = None

        # Errors
        self._bt_error: Optional[str] = None
        self._legacy_error: Optional[str] = None

        # Report
        self._report: Optional[ComparisonReport] = None
        self._query: str = ""

    async def run_parallel(
        self,
        query: str,
        context_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run both implementations in parallel, yield legacy output.

        The BT output is collected for comparison but not yielded.
        Legacy output is yielded to maintain backwards compatibility.

        Args:
            query: User query text.
            context_id: Optional context ID.

        Yields:
            Legacy Oracle chunks (for backwards compatibility).
        """
        self._query = query
        self._bt_chunks = []
        self._legacy_chunks = []

        # Create tasks for both implementations
        bt_task = asyncio.create_task(self._run_bt(query, context_id))
        legacy_gen = self._run_legacy(query, context_id)

        # Yield legacy chunks while collecting BT chunks in background
        try:
            async for chunk in legacy_gen:
                self._legacy_chunks.append(chunk)
                yield chunk

        except Exception as e:
            self._legacy_error = str(e)
            logger.error(f"Shadow mode legacy error: {e}")

        # Wait for BT task to complete
        try:
            await bt_task
        except Exception as e:
            self._bt_error = str(e)
            logger.error(f"Shadow mode BT error: {e}")

        # Generate comparison report
        self._generate_report()

        # Log discrepancies
        if self._report and self._report.discrepancies:
            logger.warning(
                f"Shadow mode found {len(self._report.discrepancies)} discrepancies "
                f"(match rate: {self._report.match_rate:.1%})"
            )
            self._log_discrepancies()

    async def _run_bt(
        self,
        query: str,
        context_id: Optional[str],
    ) -> None:
        """Run BT-based Oracle and collect chunks."""
        import time

        self._bt_start = time.time()

        try:
            wrapper = OracleBTWrapper(
                user_id=self._user_id,
                project_id=self._project_id,
                model=self._model,
                max_tokens=self._max_tokens,
                enable_shadow_mode=True,
            )

            async for chunk in wrapper.process_query(query, context_id):
                self._bt_chunks.append(chunk.model_dump())

        except Exception as e:
            self._bt_error = str(e)
            logger.error(f"BT Oracle error in shadow mode: {e}")

        finally:
            self._bt_end = time.time()

    async def _run_legacy(
        self,
        query: str,
        context_id: Optional[str],
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run legacy Oracle and yield chunks."""
        import time

        self._legacy_start = time.time()

        try:
            from ...services.oracle_agent import OracleAgent

            agent = OracleAgent(
                user_id=self._user_id,
                project_id=self._project_id,
            )

            async for chunk in agent.query(
                question=query,
                context_id=context_id,
                model=self._model,
                max_tokens=self._max_tokens,
            ):
                # Convert OracleStreamChunk to dict
                if hasattr(chunk, "model_dump"):
                    yield chunk.model_dump()
                elif isinstance(chunk, dict):
                    yield chunk
                else:
                    yield {"type": "unknown", "data": str(chunk)}

        finally:
            self._legacy_end = time.time()

    def _generate_report(self) -> None:
        """Generate comparison report from collected chunks."""
        discrepancies = []

        # Compare chunk counts
        bt_count = len(self._bt_chunks)
        legacy_count = len(self._legacy_chunks)

        if bt_count != legacy_count:
            discrepancies.append(ChunkDiscrepancy(
                field="chunk_count",
                bt_value=bt_count,
                legacy_value=legacy_count,
                severity="warning",
            ))

        # Compare chunk types
        bt_types = [c.get("type") for c in self._bt_chunks]
        legacy_types = [c.get("type") for c in self._legacy_chunks]

        if bt_types != legacy_types:
            discrepancies.append(ChunkDiscrepancy(
                field="chunk_type_sequence",
                bt_value=bt_types,
                legacy_value=legacy_types,
                severity="warning",
            ))

        # Compare individual chunks
        matches = 0
        for i in range(min(bt_count, legacy_count)):
            bt_chunk = self._bt_chunks[i]
            legacy_chunk = self._legacy_chunks[i]

            if bt_chunk.get("type") == legacy_chunk.get("type"):
                matches += 1

                # Deep compare content
                chunk_discrepancies = self._compare_chunks(
                    bt_chunk, legacy_chunk, i
                )
                discrepancies.extend(chunk_discrepancies)

        # Compare final accumulated content
        bt_content = self._get_final_content(self._bt_chunks)
        legacy_content = self._get_final_content(self._legacy_chunks)

        if bt_content != legacy_content:
            # Check if content is similar (allows for minor formatting differences)
            similarity = self._text_similarity(bt_content, legacy_content)
            if similarity < 0.95:
                discrepancies.append(ChunkDiscrepancy(
                    field="accumulated_content",
                    bt_value=bt_content[:500],
                    legacy_value=legacy_content[:500],
                    severity="error" if similarity < 0.8 else "warning",
                ))

        # Calculate match rate
        total = max(bt_count, legacy_count)
        match_rate = matches / total if total > 0 else 1.0

        # Calculate durations
        bt_duration = (
            (self._bt_end - self._bt_start) * 1000
            if self._bt_start and self._bt_end
            else 0.0
        )
        legacy_duration = (
            (self._legacy_end - self._legacy_start) * 1000
            if self._legacy_start and self._legacy_end
            else 0.0
        )

        self._report = ComparisonReport(
            timestamp=datetime.now(timezone.utc),
            user_id=self._user_id,
            query_preview=self._query[:100],
            bt_chunks=self._bt_chunks,
            legacy_chunks=self._legacy_chunks,
            discrepancies=discrepancies,
            match_rate=match_rate,
            bt_duration_ms=bt_duration,
            legacy_duration_ms=legacy_duration,
            bt_error=self._bt_error,
            legacy_error=self._legacy_error,
        )

    def _compare_chunks(
        self,
        bt_chunk: Dict[str, Any],
        legacy_chunk: Dict[str, Any],
        index: int,
    ) -> List[ChunkDiscrepancy]:
        """Compare two chunks field by field."""
        discrepancies = []
        chunk_type = bt_chunk.get("type", "unknown")

        # Fields to compare based on chunk type
        compare_fields = {
            "content": ["content"],
            "reasoning": ["reasoning"],
            "tool_call": ["tool_call"],
            "tool_result": ["tool_result"],
            "done": ["accumulated_content", "context_id"],
            "error": ["error"],
        }

        fields = compare_fields.get(chunk_type, [])

        for field in fields:
            bt_value = bt_chunk.get(field)
            legacy_value = legacy_chunk.get(field)

            if bt_value != legacy_value:
                # Allow for minor content differences
                if isinstance(bt_value, str) and isinstance(legacy_value, str):
                    similarity = self._text_similarity(bt_value, legacy_value)
                    if similarity >= 0.95:
                        continue  # Close enough

                discrepancies.append(ChunkDiscrepancy(
                    field=field,
                    bt_value=bt_value,
                    legacy_value=legacy_value,
                    index=index,
                    severity="info" if field == "context_id" else "warning",
                ))

        return discrepancies

    def _get_final_content(self, chunks: List[Dict[str, Any]]) -> str:
        """Extract final accumulated content from chunks."""
        for chunk in reversed(chunks):
            if chunk.get("type") == "done":
                return chunk.get("accumulated_content", "")
        return ""

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity ratio."""
        if not text1 and not text2:
            return 1.0
        if not text1 or not text2:
            return 0.0

        # Simple character-level similarity
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(text1, text2))
        return matches / max(len1, len2)

    def _log_discrepancies(self) -> None:
        """Log discrepancies for debugging."""
        if not self._report:
            return

        for d in self._report.discrepancies[:10]:  # Limit logged discrepancies
            logger.warning(
                f"Shadow mode discrepancy [{d.severity}] {d.field}: "
                f"BT={str(d.bt_value)[:50]!r} vs "
                f"Legacy={str(d.legacy_value)[:50]!r}"
            )

        # Log to file for detailed analysis
        try:
            from pathlib import Path

            log_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "shadow_logs"
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = log_dir / f"shadow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_file.write_text(json.dumps(self._report.to_dict(), indent=2))
            logger.info(f"Shadow mode report saved to: {log_file}")

        except Exception as e:
            logger.warning(f"Failed to save shadow mode report: {e}")

    def get_comparison_report(self) -> Optional[ComparisonReport]:
        """Get the comparison report after execution.

        Returns:
            ComparisonReport or None if not yet generated.
        """
        return self._report


# =============================================================================
# Convenience Function
# =============================================================================


async def run_with_shadow_mode(
    user_id: str,
    query: str,
    project_id: Optional[str] = None,
    context_id: Optional[str] = None,
    model: Optional[str] = None,
    max_tokens: int = 4096,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Run Oracle with appropriate mode based on feature flag.

    Automatically selects between:
    - Legacy Oracle (default)
    - BT Oracle (ORACLE_USE_BT=true)
    - Shadow mode (ORACLE_USE_BT=shadow)

    Args:
        user_id: User ID.
        query: Query text.
        project_id: Optional project ID.
        context_id: Optional context ID.
        model: LLM model.
        max_tokens: Max tokens.

    Yields:
        Oracle response chunks.
    """
    mode = get_oracle_mode()

    if mode == "bt":
        # Use BT Oracle exclusively
        wrapper = OracleBTWrapper(
            user_id=user_id,
            project_id=project_id,
            model=model,
            max_tokens=max_tokens,
        )
        async for chunk in wrapper.process_query(query, context_id):
            yield chunk.model_dump()

    elif mode == "shadow":
        # Run both in parallel
        runner = ShadowModeRunner(
            user_id=user_id,
            project_id=project_id,
            model=model,
            max_tokens=max_tokens,
        )
        async for chunk in runner.run_parallel(query, context_id):
            yield chunk

    else:
        # Use legacy Oracle
        from ...services.oracle_agent import OracleAgent

        agent = OracleAgent(user_id=user_id, project_id=project_id)
        async for chunk in agent.query(
            question=query,
            context_id=context_id,
            model=model,
            max_tokens=max_tokens,
        ):
            if hasattr(chunk, "model_dump"):
                yield chunk.model_dump()
            else:
                yield chunk


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Feature flags
    "is_bt_oracle_enabled",
    "is_shadow_mode_enabled",
    "get_oracle_mode",
    # Classes
    "ShadowModeRunner",
    "ComparisonReport",
    "ChunkDiscrepancy",
    # Functions
    "run_with_shadow_mode",
]
