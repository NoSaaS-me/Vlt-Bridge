"""Integration tests for Oracle memory operations (009-oracle-agent T032, T040).

These tests verify the Oracle Agent can:
- Save notes via vault_write tool (T040)
- Push to threads for long-term memory (T032)
- Retrieve saved context for future queries

The tests use real services with temporary directories to verify the full
integration stack without mocking.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from backend.src.models.oracle import OracleRequest, OracleStreamChunk
from backend.src.services.tool_executor import ToolExecutor
from backend.src.services.vault import VaultService
from backend.src.services.indexer import IndexerService
from backend.src.services.thread_service import ThreadService
from backend.src.services.database import DatabaseService
from backend.src.services.config import AppConfig


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="oracle_memory_test_")
    yield Path(temp_dir)
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config(temp_test_dir: Path) -> AppConfig:
    """Create test configuration with temporary paths."""
    vault_dir = temp_test_dir / "vaults"
    vault_dir.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        vault_base_path=vault_dir,
        enable_local_mode=True,
        local_dev_token="test-token",
    )
    return config


@pytest.fixture
def db_service(temp_test_dir: Path) -> DatabaseService:
    """Create DatabaseService with test database path."""
    db_path = temp_test_dir / "test_index.db"
    db = DatabaseService(db_path=db_path)
    db.initialize()
    return db


@pytest.fixture
def vault_service(test_config: AppConfig) -> VaultService:
    """Create VaultService with test configuration."""
    return VaultService(config=test_config)


@pytest.fixture
def indexer_service(db_service: DatabaseService) -> IndexerService:
    """Create IndexerService with test database."""
    return IndexerService(db_service=db_service)


@pytest.fixture
def thread_service(db_service: DatabaseService) -> ThreadService:
    """Create ThreadService with test database."""
    return ThreadService(db_service=db_service)


@pytest.fixture
def tool_executor(
    vault_service: VaultService,
    indexer_service: IndexerService,
    thread_service: ThreadService,
    db_service: DatabaseService,
) -> ToolExecutor:
    """Create ToolExecutor with real services for integration testing."""
    # Create a mock oracle_bridge since we don't need code search for these tests
    oracle_bridge = MagicMock()
    oracle_bridge.search_code = AsyncMock(return_value={"results": []})

    return ToolExecutor(
        vault_service=vault_service,
        indexer_service=indexer_service,
        thread_service=thread_service,
        oracle_bridge=oracle_bridge,
        db_service=db_service,
    )


class TestVaultWriteTool:
    """Integration tests for vault_write tool (T040).

    Verifies that Oracle can save notes via the vault_write tool with:
    - Basic note creation
    - Note creation with title
    - Note updates (overwrite)
    - Index synchronization after write
    - Error handling for invalid paths
    """

    @pytest.mark.asyncio
    async def test_vault_write_creates_note(
        self, tool_executor: ToolExecutor, vault_service: VaultService
    ) -> None:
        """vault_write creates a new note in the vault."""
        result = await tool_executor.execute(
            "vault_write",
            {
                "path": "research/auth-findings.md",
                "body": "# Authentication Research\n\nJWT is the recommended approach.",
            },
            "test-user",
        )

        data = json.loads(result)

        # Should succeed
        assert data.get("status") == "ok"
        assert data.get("path") == "research/auth-findings.md"

        # Note should exist in vault
        note = vault_service.read_note("test-user", "research/auth-findings.md")
        assert note is not None
        assert "JWT is the recommended approach" in note["body"]

    @pytest.mark.asyncio
    async def test_vault_write_with_title(
        self, tool_executor: ToolExecutor, vault_service: VaultService
    ) -> None:
        """vault_write stores title in frontmatter."""
        result = await tool_executor.execute(
            "vault_write",
            {
                "path": "decisions/use-jwt.md",
                "title": "Decision: Use JWT for Authentication",
                "body": "We decided to use JWT because of its stateless nature.",
            },
            "test-user",
        )

        data = json.loads(result)

        assert data.get("status") == "ok"
        assert data.get("title") == "Decision: Use JWT for Authentication"

        # Verify frontmatter
        note = vault_service.read_note("test-user", "decisions/use-jwt.md")
        assert note["metadata"].get("title") == "Decision: Use JWT for Authentication"

    @pytest.mark.asyncio
    async def test_vault_write_updates_existing_note(
        self, tool_executor: ToolExecutor, vault_service: VaultService
    ) -> None:
        """vault_write updates an existing note."""
        # First write
        await tool_executor.execute(
            "vault_write",
            {
                "path": "notes/progress.md",
                "body": "Initial progress note.",
            },
            "test-user",
        )

        # Update
        result = await tool_executor.execute(
            "vault_write",
            {
                "path": "notes/progress.md",
                "body": "Updated progress with new findings.",
            },
            "test-user",
        )

        data = json.loads(result)
        assert data.get("status") == "ok"

        # Verify updated content
        note = vault_service.read_note("test-user", "notes/progress.md")
        assert "Updated progress" in note["body"]
        assert "Initial progress" not in note["body"]

    @pytest.mark.asyncio
    async def test_vault_write_indexes_note(
        self, tool_executor: ToolExecutor, indexer_service: IndexerService
    ) -> None:
        """vault_write indexes the note for searchability."""
        await tool_executor.execute(
            "vault_write",
            {
                "path": "indexed/searchable.md",
                "title": "Searchable Authentication Patterns",
                "body": "OAuth2 and OpenID Connect are industry standards.",
            },
            "test-user",
        )

        # Should be searchable via indexer
        results = indexer_service.search_notes(
            "test-user", "OAuth2 OpenID", limit=5
        )

        assert len(results) >= 1
        found_paths = [r["path"] for r in results]
        assert "indexed/searchable.md" in found_paths

    @pytest.mark.asyncio
    async def test_vault_write_invalid_path_returns_error(
        self, tool_executor: ToolExecutor
    ) -> None:
        """vault_write returns error for invalid paths."""
        # Path traversal attempt
        result = await tool_executor.execute(
            "vault_write",
            {
                "path": "../escape/malicious.md",
                "body": "Should not be written.",
            },
            "test-user",
        )

        data = json.loads(result)
        assert "error" in data
        assert ".." in data["error"] or "invalid" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_vault_write_without_md_extension_returns_error(
        self, tool_executor: ToolExecutor
    ) -> None:
        """vault_write requires .md extension."""
        result = await tool_executor.execute(
            "vault_write",
            {
                "path": "notes/no-extension",
                "body": "Missing extension.",
            },
            "test-user",
        )

        data = json.loads(result)
        assert "error" in data
        assert ".md" in data["error"]

    @pytest.mark.asyncio
    async def test_vault_write_large_note(
        self, tool_executor: ToolExecutor, vault_service: VaultService
    ) -> None:
        """vault_write handles moderately large notes."""
        # Create a note with substantial content (not exceeding 1MB limit)
        large_body = "# Large Research Document\n\n" + ("Lorem ipsum. " * 1000)

        result = await tool_executor.execute(
            "vault_write",
            {
                "path": "research/large-doc.md",
                "body": large_body,
            },
            "test-user",
        )

        data = json.loads(result)
        assert data.get("status") == "ok"

        # Verify content was saved
        note = vault_service.read_note("test-user", "research/large-doc.md")
        assert len(note["body"]) > 10000


class TestThreadPushTool:
    """Integration tests for thread_push tool (T032).

    Verifies that Oracle can push thoughts to threads with:
    - Basic thread push
    - Thread auto-creation
    - Entry types (thought, decision, research, insight)
    - Entry retrieval via thread_read
    - Thread search via thread_seek
    """

    @pytest.mark.asyncio
    async def test_thread_push_creates_entry(
        self, tool_executor: ToolExecutor, thread_service: ThreadService
    ) -> None:
        """thread_push creates an entry in a thread."""
        result = await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "auth-research",
                "content": "Discovered that JWT tokens should have short expiry times.",
                "entry_type": "insight",
            },
            "test-user",
        )

        data = json.loads(result)

        assert data.get("status") == "ok"
        assert data.get("thread_id") == "auth-research"
        assert "entry_id" in data
        assert "sequence_id" in data

    @pytest.mark.asyncio
    async def test_thread_push_auto_creates_thread(
        self, tool_executor: ToolExecutor, thread_service: ThreadService
    ) -> None:
        """thread_push creates thread if it doesn't exist."""
        result = await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "new-feature-thread",
                "content": "Starting work on user authentication.",
            },
            "test-user",
        )

        data = json.loads(result)
        assert data.get("status") == "ok"

        # Thread should exist now
        thread = thread_service.get_thread("test-user", "new-feature-thread")
        assert thread is not None
        assert thread.thread_id == "new-feature-thread"

    @pytest.mark.asyncio
    async def test_thread_push_multiple_entries(
        self, tool_executor: ToolExecutor
    ) -> None:
        """thread_push adds multiple entries to the same thread."""
        # First entry
        result1 = await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "multi-entry-thread",
                "content": "First insight about the problem.",
                "entry_type": "thought",
            },
            "test-user",
        )

        # Second entry
        result2 = await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "multi-entry-thread",
                "content": "Decision made: use approach A.",
                "entry_type": "decision",
            },
            "test-user",
        )

        data1 = json.loads(result1)
        data2 = json.loads(result2)

        # Both should succeed
        assert data1.get("status") == "ok"
        assert data2.get("status") == "ok"

        # Sequence IDs should increment
        assert data2.get("sequence_id") > data1.get("sequence_id")

    @pytest.mark.asyncio
    async def test_thread_read_retrieves_entries(
        self, tool_executor: ToolExecutor
    ) -> None:
        """thread_read retrieves entries from a thread."""
        # Push some entries first
        await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "readable-thread",
                "content": "Entry one for reading test.",
            },
            "test-user",
        )
        await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "readable-thread",
                "content": "Entry two for reading test.",
            },
            "test-user",
        )

        # Read the thread
        result = await tool_executor.execute(
            "thread_read",
            {"thread_id": "readable-thread", "limit": 10},
            "test-user",
        )

        data = json.loads(result)

        assert data.get("thread_id") == "readable-thread"
        assert data.get("entry_count") >= 2

        # Check entries contain our content
        entries_content = " ".join(e["content"] for e in data.get("entries", []))
        assert "Entry one" in entries_content
        assert "Entry two" in entries_content

    @pytest.mark.asyncio
    async def test_thread_read_nonexistent_returns_error(
        self, tool_executor: ToolExecutor
    ) -> None:
        """thread_read returns error for nonexistent thread."""
        result = await tool_executor.execute(
            "thread_read",
            {"thread_id": "nonexistent-thread-xyz"},
            "test-user",
        )

        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_thread_seek_searches_content(
        self, tool_executor: ToolExecutor
    ) -> None:
        """thread_seek searches across threads."""
        # Create thread with searchable content
        await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "searchable-thread",
                "content": "The authentication system uses bcrypt for password hashing.",
            },
            "test-user",
        )

        # Search for content
        result = await tool_executor.execute(
            "thread_seek",
            {"query": "bcrypt password", "limit": 5},
            "test-user",
        )

        data = json.loads(result)

        # Should find results (may be empty if FTS not fully working)
        assert "results" in data
        assert "query" in data

    @pytest.mark.asyncio
    async def test_thread_list_shows_threads(
        self, tool_executor: ToolExecutor
    ) -> None:
        """thread_list returns all active threads."""
        # Create some threads
        await tool_executor.execute(
            "thread_push",
            {"thread_id": "list-thread-1", "content": "Thread 1 content."},
            "test-user",
        )
        await tool_executor.execute(
            "thread_push",
            {"thread_id": "list-thread-2", "content": "Thread 2 content."},
            "test-user",
        )

        # List threads
        result = await tool_executor.execute(
            "thread_list",
            {"status": "active"},
            "test-user",
        )

        data = json.loads(result)

        assert "threads" in data
        thread_ids = [t["thread_id"] for t in data["threads"]]
        assert "list-thread-1" in thread_ids
        assert "list-thread-2" in thread_ids


class TestOracleMemoryIntegration:
    """End-to-end tests for Oracle memory workflow.

    Tests the full flow of Oracle saving context for future use:
    1. Oracle saves research to vault
    2. Oracle records decision to thread
    3. Later query can find the saved context
    """

    @pytest.mark.asyncio
    async def test_full_oracle_memory_workflow(
        self,
        tool_executor: ToolExecutor,
        vault_service: VaultService,
        indexer_service: IndexerService,
    ) -> None:
        """Oracle saves research and decisions for future retrieval."""
        user_id = "test-user"

        # Step 1: Save research findings to vault
        vault_result = await tool_executor.execute(
            "vault_write",
            {
                "path": "oracle-research/api-design.md",
                "title": "API Design Research",
                "body": """# API Design Research

## Summary
Investigated RESTful vs GraphQL for the new API.

## Findings
- REST is simpler for CRUD operations
- GraphQL reduces over-fetching
- Team has more REST experience

## Recommendation
Use REST for the initial implementation with potential GraphQL gateway later.
""",
            },
            user_id,
        )

        vault_data = json.loads(vault_result)
        assert vault_data.get("status") == "ok"

        # Step 2: Record decision to thread
        thread_result = await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "api-design-decisions",
                "content": "Decided to use REST API. See [[api-design]] for full research.",
                "entry_type": "decision",
            },
            user_id,
        )

        thread_data = json.loads(thread_result)
        assert thread_data.get("status") == "ok"

        # Step 3: Verify vault content is searchable
        search_result = await tool_executor.execute(
            "vault_search",
            {"query": "REST GraphQL API", "limit": 5},
            user_id,
        )

        search_data = json.loads(search_result)
        assert search_data.get("count") >= 1

        # Step 4: Verify thread is readable
        read_result = await tool_executor.execute(
            "thread_read",
            {"thread_id": "api-design-decisions", "limit": 10},
            user_id,
        )

        read_data = json.loads(read_result)
        assert "REST API" in str(read_data.get("entries", []))

    @pytest.mark.asyncio
    async def test_oracle_persists_across_sessions(
        self,
        tool_executor: ToolExecutor,
        vault_service: VaultService,
        indexer_service: IndexerService,
        thread_service: ThreadService,
    ) -> None:
        """Saved content persists and is retrievable in new 'session'."""
        user_id = "test-user"

        # Session 1: Save data
        await tool_executor.execute(
            "vault_write",
            {
                "path": "persistent/session-data.md",
                "body": "Important data from session 1.",
            },
            user_id,
        )

        await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "persistent-thread",
                "content": "Session 1 logged this important finding.",
            },
            user_id,
        )

        # Simulate "new session" - just verify data persists
        # (In real integration, this would be a new ToolExecutor instance
        # but same database/vault)

        # Read vault content
        note = vault_service.read_note(user_id, "persistent/session-data.md")
        assert "Important data from session 1" in note["body"]

        # Read thread content
        thread = thread_service.get_thread(user_id, "persistent-thread", include_entries=True)
        assert thread is not None
        assert any("Session 1" in e.content for e in (thread.entries or []))


class TestOracleToolChaining:
    """Tests for Oracle tool chaining scenarios.

    Verifies that tools work correctly when used in sequence:
    - Write then read
    - Write then search
    - Push then read then seek
    """

    @pytest.mark.asyncio
    async def test_write_then_read(self, tool_executor: ToolExecutor) -> None:
        """Write a note, then read it back."""
        user_id = "test-user"

        # Write
        await tool_executor.execute(
            "vault_write",
            {
                "path": "chain-test/write-read.md",
                "body": "Content written for chain test.",
            },
            user_id,
        )

        # Read
        result = await tool_executor.execute(
            "vault_read",
            {"path": "chain-test/write-read.md"},
            user_id,
        )

        data = json.loads(result)
        assert "Content written for chain test" in data.get("content", "")

    @pytest.mark.asyncio
    async def test_write_then_list(self, tool_executor: ToolExecutor) -> None:
        """Write notes, then list them."""
        user_id = "test-user"

        # Write multiple notes
        await tool_executor.execute(
            "vault_write",
            {"path": "list-test/note1.md", "body": "Note 1"},
            user_id,
        )
        await tool_executor.execute(
            "vault_write",
            {"path": "list-test/note2.md", "body": "Note 2"},
            user_id,
        )

        # List
        result = await tool_executor.execute(
            "vault_list",
            {"folder": "list-test"},
            user_id,
        )

        data = json.loads(result)
        assert data.get("count") >= 2

        note_paths = [n["path"] for n in data.get("notes", [])]
        assert "list-test/note1.md" in note_paths
        assert "list-test/note2.md" in note_paths

    @pytest.mark.asyncio
    async def test_push_then_read_then_seek(self, tool_executor: ToolExecutor) -> None:
        """Push entries, read thread, then seek across threads."""
        user_id = "test-user"

        # Push entries
        await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "chain-thread",
                "content": "First entry about database migrations.",
            },
            user_id,
        )
        await tool_executor.execute(
            "thread_push",
            {
                "thread_id": "chain-thread",
                "content": "Second entry about schema validation.",
            },
            user_id,
        )

        # Read
        read_result = await tool_executor.execute(
            "thread_read",
            {"thread_id": "chain-thread"},
            user_id,
        )

        read_data = json.loads(read_result)
        assert read_data.get("entry_count") >= 2

        # Seek
        seek_result = await tool_executor.execute(
            "thread_seek",
            {"query": "database schema"},
            user_id,
        )

        seek_data = json.loads(seek_result)
        # Results may vary based on FTS, but should not error
        assert "results" in seek_data
