import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc
from sqlalchemy.exc import SQLAlchemyError

from vlt.core.interfaces import IVaultService, ThreadStateView, ProjectOverviewView, SearchResult, NodeView
from vlt.core.models import Project, Thread, Node, State, Tag, Reference
from vlt.db import get_db
from vlt.core.vector import VectorService
from vlt.lib.llm import OpenRouterLLMProvider

class VaultError(Exception):
    pass

class SqliteVaultService(IVaultService):
    def add_tag(self, node_id: str, tag_name: str) -> Tag:
        try:
            # Check if tag exists
            tag = self.db.scalars(select(Tag).where(Tag.name == tag_name)).first()
            if not tag:
                tag = Tag(name=tag_name)
                self.db.add(tag)
            
            node = self.db.get(Node, node_id)
            if not node:
                raise VaultError(f"Node {node_id} not found.")
            
            if tag not in node.tags:
                node.tags.append(tag)
            
            self.db.commit()
            self.db.refresh(tag)
            return tag
        except SQLAlchemyError as e:
            self.db.rollback()
            raise VaultError(f"Database error adding tag: {str(e)}")

    def add_reference(self, source_node_id: str, target_thread_id: str, note: str) -> Reference:
        try:
            # Check if source node exists
            node = self.db.get(Node, source_node_id)
            if not node:
                raise VaultError(f"Source Node {source_node_id} not found.")
            
            # Check if target thread exists (handle project/thread slug)
            if "/" in target_thread_id:
                _, target_slug = target_thread_id.split("/")
            else:
                target_slug = target_thread_id
                
            thread = self.db.get(Thread, target_slug)
            if not thread:
                raise VaultError(f"Target Thread {target_slug} not found.")

            ref = Reference(
                id=str(uuid.uuid4()),
                source_node_id=source_node_id,
                target_thread_id=target_slug,
                note=note
            )
            self.db.add(ref)
            self.db.commit()
            self.db.refresh(ref)
            return ref
        except SQLAlchemyError as e:
            self.db.rollback()
            raise VaultError(f"Database error adding reference: {str(e)}")

    def move_thread(self, thread_id: str, new_project_id: str) -> Thread:
        try:
            if "/" in thread_id:
                _, thread_slug = thread_id.split("/")
            else:
                thread_slug = thread_id
                
            thread = self.db.get(Thread, thread_slug)
            if not thread:
                raise VaultError(f"Thread {thread_slug} not found.")
                
            # Ensure target project exists
            project = self.db.get(Project, new_project_id)
            if not project:
                # Auto-create if moving to a new project?
                # For safety, let's auto-create.
                project = Project(id=new_project_id, name=new_project_id, description="Auto-created via move")
                self.db.add(project)
            
            thread.project_id = new_project_id
            self.db.commit()
            self.db.refresh(thread)
            return thread
        except SQLAlchemyError as e:
            self.db.rollback()
            raise VaultError(f"Database error moving thread: {str(e)}")

    def __init__(self, db: Session = None):
        self._db = db

    @property
    def db(self) -> Session:
        if self._db is None:
            self._db = next(get_db())
        return self._db
        
    def __del__(self):
        # Optional cleanup if we created the session
        # Realistically, for a CLI tool, process exit cleans up.
        pass

    def create_project(self, name: str, description: str) -> Project:
        try:
            project_id = name.lower().replace(" ", "-")
            project = Project(id=project_id, name=name, description=description)
            self.db.add(project)
            self.db.commit()
            self.db.refresh(project)
            return project
        except SQLAlchemyError as e:
            self.db.rollback()
            raise VaultError(f"Database error creating project: {str(e)}")

    def create_thread(self, project_id: str, name: str, initial_thought: str, author: str = "user") -> Thread:
        try:
            thread_id = name.lower().replace(" ", "-")
            thread = Thread(id=thread_id, project_id=project_id, status="active")
            self.db.add(thread)
            self.db.commit() 
            
            self.add_thought(thread_id, initial_thought, author=author)
            
            self.db.refresh(thread)
            return thread
        except SQLAlchemyError as e:
            self.db.rollback()
            raise VaultError(f"Database error creating thread: {str(e)}")

    def add_thought(self, thread_id: str, content: str, author: str = "user") -> Node:
        try:
            last_node = self.db.scalars(
                select(Node)
                .where(Node.thread_id == thread_id)
                .order_by(desc(Node.sequence_id))
                .limit(1)
            ).first()

            new_sequence_id = (last_node.sequence_id + 1) if last_node else 0
            prev_node_id = last_node.id if last_node else None

            node = Node(
                id=str(uuid.uuid4()),
                thread_id=thread_id,
                sequence_id=new_sequence_id,
                content=content,
                author=author,
                prev_node_id=prev_node_id,
                timestamp=datetime.now(timezone.utc)
            )
            self.db.add(node)
            self.db.commit()
            self.db.refresh(node)
            return node
        except SQLAlchemyError as e:
            self.db.rollback()
            raise VaultError(f"Database error adding thought: {str(e)}")

    def get_thread_state(self, thread_id: str, limit: int = 5, current_project_id: str = "orphaned") -> ThreadStateView:
        if "/" in thread_id:
            _, thread_slug = thread_id.split("/")
        else:
            thread_slug = thread_id

        # Fetch thread to get project_id
        thread = self.db.get(Thread, thread_slug)
        if not thread:
             # Check for orphans
             node_exists = self.db.scalars(select(Node).where(Node.thread_id == thread_slug).limit(1)).first()
             if node_exists:
                 # Auto-repair
                 thread = Thread(id=thread_slug, project_id=current_project_id, status="recovered")
                 self.db.add(thread)
                 self.db.commit()
                 self.db.refresh(thread)
             else:
                 raise VaultError(f"Thread {thread_slug} not found.")

        # Get summary from local State cache
        # NOTE: Summaries are generated by `vlt librarian run`, not on read.
        # This keeps `vlt thread read` fast (no server calls).
        from vlt.config import settings

        summary = None

        # Check if we have a cached State-based summary
        state = self.db.scalars(
            select(State)
            .where(State.target_id == thread_slug)
            .where(State.target_type == "thread")
        ).first()

        # Error messages that indicate stale/bad cached summaries
        error_indicators = [
            "LLM API Key missing",
            "Cannot summarize",
            "Error generating summary",
            "Server summarization failed",
            "Summarization unavailable",
        ]

        # Use cached summary only if it's valid (not an error message)
        if state and state.summary:
            is_error_summary = any(err in state.summary for err in error_indicators)
            if not is_error_summary:
                summary = state.summary

        # If no valid summary, provide helpful guidance
        if not summary:
            if settings.is_server_configured:
                summary = (
                    "Summary pending. Run: vlt librarian run\n"
                    "(This will sync threads and generate summaries)"
                )
            else:
                summary = (
                    "No summary available. To enable summaries:\n"
                    "1. Configure server sync: vlt config set-key <token> --server <url>\n"
                    "2. Run: vlt librarian run"
                )

        query = select(Node).where(Node.thread_id == thread_slug).order_by(desc(Node.sequence_id))

        if limit > 0:
            query = query.limit(limit)

        nodes = self.db.scalars(query).all()

        node_views = [
            NodeView(
                id=n.id, content=n.content, author=n.author,
                timestamp=n.timestamp, sequence_id=n.sequence_id
            ) for n in reversed(nodes)
        ]

        return ThreadStateView(
            thread_id=thread_slug,
            project_id=thread.project_id,
            summary=summary,
            recent_nodes=node_views,
            meta={}  # Legacy State.meta no longer used with lazy eval
        )

    def search_thread(self, thread_id: str, query: str) -> List[SearchResult]:
        if "/" in thread_id:
            _, thread_slug = thread_id.split("/")
        else:
            thread_slug = thread_id
            
        llm = OpenRouterLLMProvider()
        query_vec = llm.get_embedding(query)
        
        # Scoped to thread
        stmt = select(Node.id, Node.embedding).where(Node.thread_id == thread_slug).where(Node.embedding.is_not(None))
        candidates = self.db.execute(stmt).all()
        
        vec_service = VectorService()
        matches = vec_service.search_memory(query_vec, candidates)
        
        if not matches:
            return []

        node_ids = [m[0] for m in matches]
        nodes = self.db.scalars(select(Node).where(Node.id.in_(node_ids))).all()
        node_map = {n.id: n for n in nodes}
        
        results = []
        for node_id, score in matches:
            if node_id in node_map:
                node = node_map[node_id]
                results.append(SearchResult(
                    node_id=node.id,
                    content=node.content,
                    score=score,
                    thread_id=node.thread_id
                ))
        return results

    def get_project_overview(self, project_id: str) -> ProjectOverviewView:
        state = self.db.scalars(
            select(State)
            .where(State.target_id == project_id)
            .where(State.target_type == "project")
        ).first()

        threads = self.db.scalars(
            select(Thread).where(Thread.project_id == project_id)
        ).all()

        active_threads = [
            {"id": t.id, "status": t.status, "last_activity": "N/A"}
            for t in threads
        ]

        return ProjectOverviewView(
            project_id=project_id,
            summary=state.summary if state else "No project summary available.",
            active_threads=active_threads
        )

    def search(self, query: str, project_id: Optional[str] = None) -> List[SearchResult]:
        llm = OpenRouterLLMProvider()
        query_vec = llm.get_embedding(query)
        
        stmt = select(Node.id, Node.embedding).where(Node.embedding.is_not(None))
        if project_id:
            stmt = stmt.join(Thread).where(Thread.project_id == project_id)
            
        candidates = self.db.execute(stmt).all()
        
        vec_service = VectorService()
        matches = vec_service.search_memory(query_vec, candidates)
        
        if not matches:
            return []

        # Batch fetch matching nodes (Fix N+1)
        node_ids = [m[0] for m in matches]
        nodes = self.db.scalars(select(Node).where(Node.id.in_(node_ids))).all()
        node_map = {n.id: n for n in nodes}
        
        results = []
        for node_id, score in matches:
            if node_id in node_map:
                node = node_map[node_id]
                results.append(SearchResult(
                    node_id=node.id,
                    content=node.content,
                    score=score,
                    thread_id=node.thread_id
                ))

        return results

    def list_threads(self, project_id: str, db: Optional[Session] = None) -> List[Thread]:
        """List all threads for a project.

        Args:
            project_id: Project identifier
            db: Optional database session (uses self.db if not provided)

        Returns:
            List of Thread objects
        """
        session = db or self.db
        threads = session.scalars(
            select(Thread).where(Thread.project_id == project_id)
        ).all()
        return list(threads)

    def seek_threads(
        self,
        project_id: str,
        query: str,
        limit: int = 20,
        db: Optional[Session] = None
    ) -> List[dict]:
        """Search threads using semantic vector search (T061).

        This is called by ThreadRetriever for oracle integration.
        Uses lazy evaluation - only generates summaries for matching threads.

        Args:
            project_id: Project identifier to scope search
            query: Natural language search query
            limit: Maximum results to return
            db: Optional database session

        Returns:
            List of dicts with thread search results:
            {
                "thread_id": "auth-design",
                "node_id": "uuid-42",
                "content": "Decided to use JWT...",
                "author": "claude",
                "timestamp": "2025-01-15T10:30:00Z",
                "score": 0.85
            }
        """
        session = db or self.db

        # Get embedding for query
        llm = OpenRouterLLMProvider()
        query_vec = llm.get_embedding(query)

        # Search across nodes in this project's threads
        # T061: This triggers lazy summary generation for matching threads
        stmt = (
            select(Node.id, Node.embedding, Node.thread_id, Node.content,
                   Node.author, Node.timestamp, Node.sequence_id)
            .join(Thread, Node.thread_id == Thread.id)
            .where(Thread.project_id == project_id)
            .where(Node.embedding.is_not(None))
        )

        candidates = session.execute(stmt).all()

        if not candidates:
            return []

        # Build candidate list for vector search (id, embedding)
        vec_candidates = [(c.id, c.embedding) for c in candidates]

        # Perform vector similarity search
        vec_service = VectorService()
        matches = vec_service.search_memory(query_vec, vec_candidates)

        if not matches:
            return []

        # Build lookup map for matched nodes
        node_map = {c.id: c for c in candidates}

        # Format results
        results = []
        for node_id, score in matches[:limit]:
            if node_id in node_map:
                node_data = node_map[node_id]
                results.append({
                    "thread_id": node_data.thread_id,
                    "node_id": node_data.sequence_id,  # Use sequence_id as node identifier
                    "content": node_data.content,
                    "author": node_data.author,
                    "timestamp": node_data.timestamp.isoformat(),
                    "score": score
                })

        # T061: Trigger lazy summary generation for threads that matched
        # This ensures summaries are fresh when oracle uses them
        from vlt.core.lazy_eval import ThreadSummaryManager

        matched_thread_ids = list(set(r["thread_id"] for r in results))
        if matched_thread_ids:
            try:
                summary_manager = ThreadSummaryManager(llm, session)
                for thread_id in matched_thread_ids:
                    # Generate summary on-demand (uses cache if fresh)
                    summary_manager.generate_summary(thread_id)
            except Exception as e:
                # Don't fail the search if summary generation fails
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to generate summaries for matched threads: {e}")

        return results