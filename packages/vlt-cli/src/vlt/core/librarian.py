"""
Librarian - Thread summarization and state management.

DEPRECATION NOTICE:
As of the server-side architecture refactor, the Librarian no longer makes
direct LLM API calls. Instead, summarization is handled by the backend server
via the POST /api/threads/{thread_id}/summarize endpoint.

This module now provides:
1. ServerLibrarian: Calls the backend for summarization (recommended)
2. Librarian: Legacy class that still accepts an LLM provider (deprecated)

To use server-side summarization, configure your sync token with:
    vlt config set-key <your-sync-token>
"""

import uuid
import logging
from typing import List, Optional
import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from vlt.core.interfaces import ILibrarian, ILLMProvider
from vlt.core.models import Node, State, Thread, Project
from vlt.core.vector import VectorService
from vlt.db import get_db
from vlt.config import Settings

logger = logging.getLogger(__name__)


class ServerLibrarian:
    """
    Server-side Librarian that delegates summarization to the backend.

    This is the recommended way to use the Librarian. It calls the backend's
    /api/threads/{thread_id}/summarize endpoint, which handles LLM operations
    with centralized API key management.
    """

    def __init__(self, db: Session = None, project_id: str = None):
        self._db = db
        self._settings = Settings()
        self._project_id = project_id

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = next(get_db())
        return self._db

    @property
    def vault_url(self) -> str:
        """Get the backend server URL from config."""
        # Try environment variable first, then fall back to default
        import os
        url = os.getenv("VLT_VAULT_URL") or self._settings.vault_url
        if url:
            return url.rstrip("/")
        return "http://localhost:8000"

    @property
    def sync_token(self) -> Optional[str]:
        """Get the sync token for authentication."""
        return self._settings.sync_token

    @property
    def project_id(self) -> Optional[str]:
        """Get the project ID for scoping operations."""
        if self._project_id:
            return self._project_id
        # Try to load from vlt.toml if not explicitly set
        from .identity import load_project_identity
        identity = load_project_identity()
        return identity.id if identity else None

    async def sync_all_threads_to_server(self) -> dict:
        """
        Sync project threads to the server before summarization.

        This ensures threads exist on the server before we request summarization.
        The server needs to have the thread entries to generate summaries.

        Only syncs threads belonging to the current project (from vlt.toml).

        Returns:
            Dict with sync statistics
        """
        from vlt.core.sync import sync_all_threads

        return await sync_all_threads(self.db, project_id=self.project_id)

    async def summarize_thread_on_server(self, thread_id: str, current_summary: Optional[str] = None) -> dict:
        """
        Call the backend to summarize a thread.

        Args:
            thread_id: Thread identifier
            current_summary: Existing summary for incremental updates

        Returns:
            Dict with summary, model, tokens_used, success, error
        """
        if not self.sync_token:
            return {
                "summary": current_summary or "No sync token configured. Run: vlt config set-key <token>",
                "model": None,
                "tokens_used": 0,
                "success": False,
                "error": "No sync token configured",
            }

        url = f"{self.vault_url}/api/threads/{thread_id}/summarize"
        headers = {
            "Authorization": f"Bearer {self.sync_token}",
            "Content-Type": "application/json",
        }

        body = {}
        if current_summary:
            body["current_summary"] = current_summary

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=body)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"Server summarization failed: {e.response.status_code} - {e.response.text}")
            return {
                "summary": current_summary or "Server summarization failed.",
                "model": None,
                "tokens_used": 0,
                "success": False,
                "error": f"HTTP {e.response.status_code}: {e.response.text[:100]}",
            }
        except Exception as e:
            logger.error(f"Server summarization failed: {e}")
            return {
                "summary": current_summary or "Server summarization failed.",
                "model": None,
                "tokens_used": 0,
                "success": False,
                "error": str(e),
            }

    def process_pending_nodes_via_server(self) -> int:
        """
        Process pending nodes by calling the backend for summarization.

        This first syncs project threads to the server, then triggers server-side
        summarization for each thread with pending nodes. The server handles
        all LLM operations.

        Only processes threads belonging to the current project (from vlt.toml).

        Returns:
            Number of nodes processed
        """
        import asyncio
        processed_count = 0

        # Get project scope
        current_project = self.project_id
        if current_project:
            print(f"[bold]Project scope:[/bold] {current_project}")
        else:
            print("[yellow]Warning: No project scope (no vlt.toml found). Processing all threads.[/yellow]")

        # STEP 1: Sync project threads to the server first
        # This ensures the server has all entries before we request summarization
        print("Syncing threads to server...")
        sync_stats = asyncio.run(self.sync_all_threads_to_server())

        # Show sync errors prominently
        if sync_stats.get("errors"):
            print(f"[yellow]Sync warnings ({len(sync_stats['errors'])} issues):[/yellow]")
            for error in sync_stats["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(sync_stats["errors"]) > 5:
                print(f"  ... and {len(sync_stats['errors']) - 5} more")

        threads_synced = sync_stats.get('threads_synced', 0)
        threads_failed = sync_stats.get('threads_failed', 0)
        total_entries = sync_stats.get('total_entries', 0)

        print(
            f"[green]Sync complete:[/green] {threads_synced} threads synced, "
            f"{threads_failed} failed, {total_entries} total entries"
        )

        # ABORT if no threads were synced and there were errors
        if threads_synced == 0 and sync_stats.get("errors"):
            print("[red]No threads synced. Cannot proceed with summarization.[/red]")
            print("[dim]Check your sync token: vlt config set-key <token>[/dim]")
            return 0

        # STEP 2: Get threads with pending nodes (scoped to project if set)
        query = select(Thread)
        if current_project:
            query = query.where(Thread.project_id == current_project)
        threads = self.db.scalars(query).all()

        for thread in threads:
            # Get current state
            state = self.db.scalars(
                select(State)
                .where(State.target_id == thread.id)
                .where(State.target_type == "thread")
            ).first()

            last_head_id = state.head_node_id if state else None

            # Find nodes AFTER the last head
            query = select(Node).where(Node.thread_id == thread.id).order_by(Node.sequence_id)
            if last_head_id:
                head_node = self.db.get(Node, last_head_id)
                if head_node:
                    query = query.where(Node.sequence_id > head_node.sequence_id)

            new_nodes = self.db.scalars(query).all()

            if not new_nodes:
                continue

            current_summary = state.summary if state else None

            # STEP 3: Call server for summarization
            result = asyncio.run(self.summarize_thread_on_server(
                thread_id=thread.id,
                current_summary=current_summary,
            ))

            if result["success"]:
                updated_summary = result["summary"]

                # Update State locally
                if not state:
                    state = State(
                        id=str(uuid.uuid4()),
                        target_id=thread.id,
                        target_type="thread",
                        summary=updated_summary,
                        head_node_id=new_nodes[-1].id
                    )
                    self.db.add(state)
                else:
                    state.summary = updated_summary
                    state.head_node_id = new_nodes[-1].id

                processed_count += len(new_nodes)
                print(f"[green]✓[/green] Summarized thread {thread.id}: {len(new_nodes)} new nodes")
            else:
                print(f"[red]✗[/red] Failed to summarize thread {thread.id}: {result.get('error')}")

        self.db.commit()
        return processed_count


class Librarian(ILibrarian):
    """
    DEPRECATED: Legacy Librarian that makes direct LLM API calls.

    This class is preserved for backward compatibility but is deprecated.
    Use ServerLibrarian instead, which delegates to the backend.

    The legacy workflow requires users to configure their own OpenRouter API key
    via `vlt config set-key`. The new server-side approach centralizes API key
    management and billing on the backend.
    """

    def __init__(self, llm_provider: ILLMProvider = None, db: Session = None):
        """
        Initialize the legacy Librarian.

        Args:
            llm_provider: LLM provider for summarization (optional, deprecated)
            db: Database session (optional)
        """
        self.llm = llm_provider
        self._db = db
        logger.warning(
            "Librarian is deprecated. Use ServerLibrarian for server-side summarization. "
            "Direct LLM calls from the CLI will be removed in a future version."
        )

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = next(get_db())
        return self._db

    def process_pending_nodes(self) -> int:
        """
        DEPRECATED: Process pending nodes using local LLM calls.

        This method makes direct API calls to OpenRouter, which requires
        the user to configure their own API key. Consider using
        ServerLibrarian.process_pending_nodes_via_server() instead.
        """
        if self.llm is None:
            logger.error("No LLM provider configured. Cannot process nodes.")
            return 0

        processed_count = 0

        # 1. Get all threads
        threads = self.db.scalars(select(Thread)).all()

        for thread in threads:
            # Get current state
            state = self.db.scalars(
                select(State)
                .where(State.target_id == thread.id)
                .where(State.target_type == "thread")
            ).first()

            last_head_id = state.head_node_id if state else None

            # Find nodes AFTER the last head
            query = select(Node).where(Node.thread_id == thread.id).order_by(Node.sequence_id)
            if last_head_id:
                head_node = self.db.get(Node, last_head_id)
                if head_node:
                    query = query.where(Node.sequence_id > head_node.sequence_id)

            new_nodes = self.db.scalars(query).all()

            if not new_nodes:
                continue

            # Collapse new nodes content
            new_content = "\n".join([f"- {n.content}" for n in new_nodes])
            current_summary = state.summary if state else "No history."

            # Generate new summary (DEPRECATED: direct LLM call)
            updated_summary = self.llm.generate_summary(current_summary, new_content)

            # Embed NEW nodes (DEPRECATED: direct embedding call)
            for node in new_nodes:
                if not node.embedding:
                    emb = self.llm.get_embedding(node.content)
                    node.embedding = VectorService.serialize(emb)

            # Update State
            if not state:
                state = State(
                    id=str(uuid.uuid4()),
                    target_id=thread.id,
                    target_type="thread",
                    summary=updated_summary,
                    head_node_id=new_nodes[-1].id
                )
                self.db.add(state)
            else:
                state.summary = updated_summary
                state.head_node_id = new_nodes[-1].id

            processed_count += len(new_nodes)

        self.db.commit()
        return processed_count

    def update_project_overviews(self) -> int:
        """
        DEPRECATED: Aggregate thread summaries into project summaries.

        This method makes direct API calls to OpenRouter.
        """
        if self.llm is None:
            logger.error("No LLM provider configured. Cannot update overviews.")
            return 0

        updated_count = 0
        projects = self.db.scalars(select(Project)).all()

        for project in projects:
            # Get all thread summaries
            thread_states = self.db.scalars(
                select(State)
                .join(Thread, Thread.id == State.target_id)
                .where(Thread.project_id == project.id)
                .where(State.target_type == "thread")
            ).all()

            if not thread_states:
                continue

            combined_context = "\n\n".join([f"Thread {s.target_id}:\n{s.summary}" for s in thread_states])

            # Get current project state
            proj_state = self.db.scalars(
                select(State)
                .where(State.target_id == project.id)
                .where(State.target_type == "project")
            ).first()

            current_summary = proj_state.summary if proj_state else "New Project."

            # Generate Overview (DEPRECATED: direct LLM call)
            new_summary = self.llm.generate_summary(current_summary, combined_context)

            if not proj_state:
                proj_state = State(
                    id=str(uuid.uuid4()),
                    target_id=project.id,
                    target_type="project",
                    summary=new_summary
                )
                self.db.add(proj_state)
            else:
                proj_state.summary = new_summary

            updated_count += 1

        self.db.commit()
        return updated_count