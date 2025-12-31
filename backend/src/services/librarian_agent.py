"""Librarian Agent - Subagent for content summarization and vault organization.

The Librarian is a specialized subagent that:
1. Summarizes content to save Oracle's context window
2. Caches summaries in markdown files for reuse
3. Organizes vault with proper folder structures
4. Creates indexes with wikilinks for discoverability
5. Returns streaming blocks back to Oracle

Does NOT modify source data - only creates new summaries and indexes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, date
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import yaml

from ..models.librarian import (
    CachedSummaryMetadata,
    ContentItem,
    LibrarianStreamChunk,
    OrganizeResult,
    SummaryResult,
)

logger = logging.getLogger(__name__)


# Lazy imports to avoid circular dependencies
_tool_executor = None
_prompt_loader = None


def _get_tool_executor():
    """Get ToolExecutor instance lazily."""
    global _tool_executor
    if _tool_executor is None:
        from .tool_executor import ToolExecutor
        _tool_executor = ToolExecutor()
    return _tool_executor


def _get_prompt_loader():
    """Get PromptLoader instance lazily."""
    global _prompt_loader
    if _prompt_loader is None:
        from .prompt_loader import PromptLoader
        _prompt_loader = PromptLoader()
    return _prompt_loader


class LibrarianAgentError(Exception):
    """Raised when Librarian agent operations fail."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class LibrarianAgent:
    """Subagent specialized for content summarization and vault organization.

    Primary responsibilities:
    - Summarize large content to save Oracle's context
    - Cache summaries in organized folder structure
    - Create/maintain vault indexes with wikilinks
    - Return results as streaming blocks

    Does NOT modify source data - only creates new summaries and indexes.

    Example:
        >>> librarian = LibrarianAgent(
        ...     api_key="sk-...",
        ...     model="anthropic/claude-sonnet-4",
        ...     user_id="user-123",
        ... )
        >>> async for chunk in librarian.summarize(
        ...     task="Summarize the architecture folder",
        ...     content=[{"path": "arch/api.md", "content": "...", "type": "vault"}],
        ... ):
        ...     print(chunk)
    """

    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    MAX_TURNS = 5  # Librarian tasks should be simpler than Oracle
    DEFAULT_MODEL = "anthropic/claude-sonnet-4"
    CACHE_ROOT = "oracle-cache"

    def __init__(
        self,
        api_key: str,
        model: Optional[str] = None,
        project_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tool_executor: Optional[Any] = None,
    ):
        """Initialize the Librarian agent.

        Args:
            api_key: OpenRouter API key
            model: Model to use (from user settings, NOT hardcoded default)
            project_id: Project context for tool scoping
            user_id: User ID for vault access
            tool_executor: Optional ToolExecutor instance (for testing)
        """
        self.api_key = api_key
        self.model = model or self.DEFAULT_MODEL
        self.project_id = project_id
        self.user_id = user_id
        self._tool_executor = tool_executor

    def _get_executor(self):
        """Get the tool executor (injected or global)."""
        return self._tool_executor or _get_tool_executor()

    # =========================================================================
    # Cache Path Generation
    # =========================================================================

    def _get_cache_path(
        self,
        source_type: str,
        source_id: str,
        date_str: Optional[str] = None,
    ) -> str:
        """Generate cache path for a summary.

        Cache structure:
            oracle-cache/summaries/{type}/{date}/{id}.md

        Args:
            source_type: Type of source (vault, thread, code, web)
            source_id: Identifier for the content (sanitized for filesystem)
            date_str: Date string for organization (defaults to today)

        Returns:
            Vault-relative path for the cached summary
        """
        if date_str is None:
            date_str = date.today().isoformat()

        # Sanitize source_id for filesystem
        safe_id = re.sub(r'[^\w\-]', '-', source_id)
        safe_id = re.sub(r'-+', '-', safe_id).strip('-')[:64]

        return f"{self.CACHE_ROOT}/summaries/{source_type}/{date_str}/{safe_id}.md"

    def _generate_cache_key(
        self,
        task: str,
        content: List[Dict[str, Any]],
    ) -> str:
        """Generate a unique cache key for the summarization request.

        The cache key is based on:
        - Task description
        - Sorted list of source paths
        - Hash of content (first 1000 chars of each)

        Args:
            task: The summarization task
            content: List of content items

        Returns:
            A hash-based cache key
        """
        # Build a deterministic string from inputs
        paths = sorted(c.get("path", "") for c in content)
        content_preview = "".join(
            c.get("content", "")[:1000] for c in sorted(content, key=lambda x: x.get("path", ""))
        )

        cache_input = f"{task}|{','.join(paths)}|{content_preview}"
        return hashlib.sha256(cache_input.encode()).hexdigest()[:16]

    # =========================================================================
    # Cache Operations
    # =========================================================================

    async def _check_cache(
        self,
        cache_path: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if a cached summary exists and is valid.

        Args:
            cache_path: Vault-relative path to the cache file

        Returns:
            Dictionary with {summary, metadata} if cache hit, None otherwise
        """
        if not self.user_id:
            return None

        executor = self._get_executor()

        try:
            result = await executor.execute(
                name="vault_read",
                arguments={"path": cache_path},
                user_id=self.user_id,
            )
            data = json.loads(result)

            if "error" in data:
                return None

            content = data.get("content", "")
            if not content:
                return None

            # Parse YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    try:
                        metadata = yaml.safe_load(parts[1])
                        summary = parts[2].strip()
                        return {
                            "summary": summary,
                            "metadata": metadata,
                        }
                    except yaml.YAMLError:
                        pass

            # No valid frontmatter, treat entire content as summary
            return {"summary": content, "metadata": {}}

        except Exception as e:
            logger.debug(f"Cache check failed for {cache_path}: {e}")
            return None

    async def _save_to_cache(
        self,
        cache_path: str,
        summary: str,
        sources: List[str],
        task: str,
        source_type: str,
        cache_key: str,
    ) -> bool:
        """Save a summary to the cache with frontmatter metadata.

        Args:
            cache_path: Vault-relative path for the cache file
            summary: The summary content to cache
            sources: List of source paths
            task: The summarization task
            source_type: Primary source type
            cache_key: Unique cache key

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.user_id:
            return False

        executor = self._get_executor()

        # Estimate token count (rough: 1 token ~= 4 chars)
        token_count = len(summary) // 4

        # Build frontmatter
        metadata = CachedSummaryMetadata(
            created=datetime.utcnow(),
            sources=sources,
            token_count=token_count,
            cache_key=cache_key,
            task=task,
            source_type=source_type,
        )

        # Convert to YAML frontmatter
        frontmatter_dict = {
            "created": metadata.created.isoformat(),
            "sources": metadata.sources,
            "token_count": metadata.token_count,
            "cache_key": metadata.cache_key,
        }
        if metadata.task:
            frontmatter_dict["task"] = metadata.task
        if metadata.source_type:
            frontmatter_dict["source_type"] = metadata.source_type

        frontmatter_yaml = yaml.dump(frontmatter_dict, default_flow_style=False)

        # Build full content
        full_content = f"---\n{frontmatter_yaml}---\n\n{summary}"

        try:
            # Extract folder path for title
            path_parts = cache_path.rsplit("/", 1)
            title = path_parts[-1].replace(".md", "") if path_parts else "summary"

            result = await executor.execute(
                name="vault_write",
                arguments={
                    "path": cache_path,
                    "body": full_content,
                    "title": f"Summary: {title}",
                },
                user_id=self.user_id,
            )
            data = json.loads(result)

            if "error" in data:
                logger.warning(f"Failed to cache summary: {data['error']}")
                return False

            logger.info(f"Cached summary to {cache_path}")
            return True

        except Exception as e:
            logger.warning(f"Failed to cache summary to {cache_path}: {e}")
            return False

    # =========================================================================
    # Summarization
    # =========================================================================

    async def summarize(
        self,
        task: str,
        content: List[Dict[str, Any]],
        max_summary_tokens: int = 1000,
        force_refresh: bool = False,
    ) -> AsyncGenerator[LibrarianStreamChunk, None]:
        """Summarize content, cache result, return streaming blocks.

        This is the primary entry point for content summarization. It:
        1. Checks cache for existing summary
        2. If cache miss, calls LLM to generate summary
        3. Caches the result for future use
        4. Yields streaming chunks throughout

        Args:
            task: Description of what to summarize or focus on
            content: List of {path, content, source_type} dicts
            max_summary_tokens: Maximum tokens for the summary
            force_refresh: Bypass cache and regenerate

        Yields:
            LibrarianStreamChunk objects with progress and results
        """
        # Validate inputs
        if not content:
            yield LibrarianStreamChunk(
                type="error",
                content="No content provided to summarize",
            )
            return

        # Determine primary source type
        source_types = [c.get("source_type", "vault") for c in content]
        primary_type = max(set(source_types), key=source_types.count)
        if len(set(source_types)) > 1:
            primary_type = "mixed"

        # Generate cache key and path
        cache_key = self._generate_cache_key(task, content)
        source_id = f"{task[:30]}-{cache_key}"
        cache_path = self._get_cache_path(primary_type, source_id)

        # Check cache unless force refresh
        if not force_refresh:
            yield LibrarianStreamChunk(
                type="thinking",
                content="Checking cache for existing summary...",
            )

            cached = await self._check_cache(cache_path)
            if cached:
                yield LibrarianStreamChunk(
                    type="cache_hit",
                    content=cached["summary"],
                    sources=[c.get("path", "") for c in content],
                    cache_path=cache_path,
                    metadata=cached.get("metadata", {}),
                )
                yield LibrarianStreamChunk(
                    type="done",
                    metadata={"from_cache": True, "cache_path": cache_path},
                )
                return

        # No cache hit - generate summary
        yield LibrarianStreamChunk(
            type="thinking",
            content=f"Summarizing {len(content)} source(s)...",
        )

        # Build prompt
        prompt_loader = _get_prompt_loader()

        try:
            system_prompt = prompt_loader.load(
                "librarian/summarize.md",
                {
                    "task": task,
                    "max_tokens": max_summary_tokens,
                    "source_count": len(content),
                    "primary_type": primary_type,
                },
            )
        except Exception:
            # Fallback to inline prompt
            system_prompt = self._get_summarize_prompt(task, max_summary_tokens)

        # Build user message with content
        content_parts = []
        for item in content:
            path = item.get("path", "unknown")
            text = item.get("content", "")
            s_type = item.get("source_type", "vault")
            content_parts.append(f"## Source: {path} ({s_type})\n\n{text}\n")

        user_message = "\n---\n".join(content_parts)

        # Call LLM for summarization
        try:
            summary = await self._call_llm_for_summary(
                system_prompt=system_prompt,
                user_message=user_message,
                max_tokens=max_summary_tokens,
            )

            # Extract sources mentioned in the summary
            sources = [c.get("path", "") for c in content if c.get("path")]

            # Yield the summary
            yield LibrarianStreamChunk(
                type="summary",
                content=summary,
                sources=sources,
                cache_path=cache_path,
            )

            # Cache the summary
            yield LibrarianStreamChunk(
                type="thinking",
                content="Caching summary for future use...",
            )

            cached_successfully = await self._save_to_cache(
                cache_path=cache_path,
                summary=summary,
                sources=sources,
                task=task,
                source_type=primary_type,
                cache_key=cache_key,
            )

            # Done
            yield LibrarianStreamChunk(
                type="done",
                metadata={
                    "from_cache": False,
                    "cache_path": cache_path if cached_successfully else None,
                    "token_count": len(summary) // 4,
                },
            )

        except Exception as e:
            logger.exception(f"Summarization failed: {e}")
            yield LibrarianStreamChunk(
                type="error",
                content=f"Summarization failed: {str(e)}",
            )

    def _get_summarize_prompt(self, task: str, max_tokens: int) -> str:
        """Get inline summarization prompt as fallback.

        Args:
            task: The summarization task
            max_tokens: Maximum tokens for summary

        Returns:
            System prompt for summarization
        """
        return f"""# Librarian Summarization Task

You are the **Librarian**, a specialized summarization agent. Your task is to create a concise summary of the provided content.

## Task

{task}

## Guidelines

1. **Preserve Key Information**: Include all important facts, decisions, and references
2. **Cite Sources**: Always mention which source(s) information comes from using [[wikilinks]]
3. **Be Concise**: Target approximately {max_tokens} tokens
4. **Structure Well**: Use headers and bullet points for readability
5. **Link Related**: Use [[wikilinks]] to connect related concepts

## Summary Format

```markdown
# Summary: [Topic]

## Key Documents
- [[source1]] - Brief description
- [[source2]] - Brief description

## Main Points
1. First major point (from [[source]])
2. Second major point (from [[source]])
...

## Sources Cited
All information from: [[source1]], [[source2]], ...
```

## Important

- DO NOT add information not present in the sources
- DO NOT hallucinate citations
- DO cite every piece of information to its source
"""

    async def _call_llm_for_summary(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> str:
        """Call the LLM to generate a summary.

        Args:
            system_prompt: System instructions
            user_message: Content to summarize
            max_tokens: Maximum response tokens

        Returns:
            The generated summary text

        Raises:
            LibrarianAgentError: If LLM call fails
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://vlt.ai",
                        "X-Title": "Vlt Librarian",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.3,  # Lower temp for consistent summaries
                    },
                )
                response.raise_for_status()
                data = response.json()

            choices = data.get("choices", [])
            if not choices:
                raise LibrarianAgentError("No response from model")

            return choices[0].get("message", {}).get("content", "")

        except httpx.HTTPStatusError as e:
            logger.error(f"LLM API error: {e.response.status_code}")
            raise LibrarianAgentError(f"API error: {e.response.status_code}")
        except httpx.TimeoutException:
            raise LibrarianAgentError("Request timeout")
        except Exception as e:
            raise LibrarianAgentError(f"LLM call failed: {str(e)}")

    # =========================================================================
    # Vault Organization
    # =========================================================================

    async def organize(
        self,
        folder: str,
        create_index: bool = True,
        task: Optional[str] = None,
    ) -> AsyncGenerator[LibrarianStreamChunk, None]:
        """Organize a vault folder and optionally create an index.

        This method:
        1. Lists all notes in the folder
        2. Analyzes their content and relationships
        3. Creates an index.md with wikilinks to all notes
        4. Optionally reorganizes notes into subfolders

        Args:
            folder: Folder path to organize
            create_index: Whether to create/update index.md
            task: Optional specific organization instructions

        Yields:
            LibrarianStreamChunk objects with progress and results
        """
        if not self.user_id:
            yield LibrarianStreamChunk(
                type="error",
                content="User ID required for vault operations",
            )
            return

        executor = self._get_executor()

        # Phase 1: Discovery
        yield LibrarianStreamChunk(
            type="thinking",
            content=f"Listing notes in {folder}...",
        )

        try:
            result = await executor.execute(
                name="vault_list",
                arguments={"folder": folder},
                user_id=self.user_id,
            )
            list_data = json.loads(result)

            if "error" in list_data:
                yield LibrarianStreamChunk(
                    type="error",
                    content=f"Failed to list folder: {list_data['error']}",
                )
                return

            notes = list_data.get("notes", [])

            if not notes:
                yield LibrarianStreamChunk(
                    type="thinking",
                    content=f"Folder {folder} is empty, nothing to organize.",
                )
                yield LibrarianStreamChunk(
                    type="done",
                    metadata={"files_organized": 0},
                )
                return

            yield LibrarianStreamChunk(
                type="thinking",
                content=f"Found {len(notes)} note(s) in {folder}",
            )

        except Exception as e:
            yield LibrarianStreamChunk(
                type="error",
                content=f"Failed to list folder: {str(e)}",
            )
            return

        # Phase 2: Read notes and build index
        if create_index:
            yield LibrarianStreamChunk(
                type="thinking",
                content="Reading notes to build index...",
            )

            note_summaries = []
            for note in notes:
                note_path = note.get("path", "")
                if not note_path or note_path.endswith("index.md"):
                    continue

                try:
                    read_result = await executor.execute(
                        name="vault_read",
                        arguments={"path": note_path},
                        user_id=self.user_id,
                    )
                    read_data = json.loads(read_result)

                    if "error" not in read_data:
                        title = read_data.get("title", "") or Path(note_path).stem
                        content = read_data.get("content", "")
                        # Get first paragraph as summary
                        first_para = content.split("\n\n")[0][:200] if content else ""

                        note_summaries.append({
                            "path": note_path,
                            "title": title,
                            "summary": first_para,
                        })

                except Exception as e:
                    logger.warning(f"Failed to read {note_path}: {e}")

            # Generate index content
            index_content = self._generate_index_content(
                folder=folder,
                notes=note_summaries,
                task=task,
            )

            yield LibrarianStreamChunk(
                type="index",
                content=index_content,
                sources=[n["path"] for n in note_summaries],
            )

            # Write index file
            index_path = f"{folder}/index.md".lstrip("/")
            if index_path.startswith("/"):
                index_path = index_path[1:]

            try:
                write_result = await executor.execute(
                    name="vault_write",
                    arguments={
                        "path": index_path,
                        "body": index_content,
                        "title": f"Index: {folder}",
                    },
                    user_id=self.user_id,
                )
                write_data = json.loads(write_result)

                if "error" in write_data:
                    yield LibrarianStreamChunk(
                        type="error",
                        content=f"Failed to write index: {write_data['error']}",
                    )
                    return

                yield LibrarianStreamChunk(
                    type="done",
                    metadata={
                        "index_path": index_path,
                        "files_organized": len(note_summaries),
                        "wikilinks_created": len(note_summaries),
                    },
                )

            except Exception as e:
                yield LibrarianStreamChunk(
                    type="error",
                    content=f"Failed to write index: {str(e)}",
                )
                return

        else:
            yield LibrarianStreamChunk(
                type="done",
                metadata={"files_organized": len(notes)},
            )

    def _generate_index_content(
        self,
        folder: str,
        notes: List[Dict[str, Any]],
        task: Optional[str] = None,
    ) -> str:
        """Generate index.md content with wikilinks.

        Args:
            folder: Folder being indexed
            notes: List of note info dicts
            task: Optional organization task description

        Returns:
            Markdown content for index.md
        """
        folder_name = folder.strip("/").split("/")[-1] if folder else "Root"
        created_date = datetime.utcnow().strftime("%Y-%m-%d")

        lines = [
            f"# {folder_name.replace('-', ' ').title()}",
            "",
            f"> Index generated on {created_date}",
            "",
        ]

        if task:
            lines.extend([
                "## Overview",
                "",
                task,
                "",
            ])

        lines.extend([
            "## Contents",
            "",
        ])

        for note in sorted(notes, key=lambda n: n.get("title", "")):
            title = note.get("title", "Untitled")
            summary = note.get("summary", "")

            # Use wikilink format
            lines.append(f"- [[{title}]]")
            if summary:
                # Add summary as indented text
                summary_clean = summary.replace("\n", " ").strip()
                if len(summary_clean) > 100:
                    summary_clean = summary_clean[:100] + "..."
                lines.append(f"  - {summary_clean}")

        lines.extend([
            "",
            "---",
            f"*{len(notes)} note(s) in this folder*",
        ])

        return "\n".join(lines)


# Factory function
def get_librarian_agent(
    api_key: str,
    model: Optional[str] = None,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> LibrarianAgent:
    """Create a LibrarianAgent instance.

    Args:
        api_key: OpenRouter API key
        model: Model override (from user settings)
        project_id: Project context
        user_id: User ID for vault access

    Returns:
        Configured LibrarianAgent
    """
    return LibrarianAgent(
        api_key=api_key,
        model=model,
        project_id=project_id,
        user_id=user_id,
    )


__all__ = ["LibrarianAgent", "LibrarianAgentError", "get_librarian_agent"]
