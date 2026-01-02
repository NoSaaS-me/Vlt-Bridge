"""
BM25 keyword indexer using SQLite FTS5 for CodeRAG.

T022: BM25 indexer using the code_chunk_fts table created in migrations.
Provides keyword-based code search to complement vector search.
"""

from typing import List, Tuple, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session
from vlt.db import engine, SessionLocal


class BM25Indexer:
    """
    BM25 keyword search indexer using SQLite FTS5.

    Uses the code_chunk_fts virtual table created in migrations.
    All queries use parameterized statements to prevent SQL injection.
    """

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize BM25 indexer.

        Args:
            db: Optional SQLAlchemy session (creates new if None)
        """
        self._db = db
        self._owns_db = db is None

    def __enter__(self):
        if self._owns_db:
            self._db = SessionLocal()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._owns_db and self._db:
            self._db.close()

    @property
    def db(self) -> Session:
        """Get database session, creating if needed."""
        if self._db is None:
            self._db = SessionLocal()
            self._owns_db = True
        return self._db

    def index_chunk(
        self,
        chunk_id: str,
        name: str,
        qualified_name: str,
        signature: Optional[str],
        docstring: Optional[str],
        body: str
    ) -> None:
        """
        Index a code chunk for BM25 keyword search.

        This inserts or replaces the chunk in the FTS5 table.
        The FTS5 table uses porter stemming and unicode tokenization.

        Args:
            chunk_id: Unique identifier for the chunk
            name: Simple name (function/class name)
            qualified_name: Fully qualified name (module.Class.method)
            signature: Function/method signature (can be None)
            docstring: Documentation string (can be None)
            body: Code body text

        Note:
            All fields are indexed for full-text search.
            chunk_id is UNINDEXED (stored but not searchable).
        """
        # Sanitize None values to empty strings for FTS5
        signature = signature or ""
        docstring = docstring or ""

        # Use INSERT OR REPLACE to handle updates
        query = text("""
            INSERT OR REPLACE INTO code_chunk_fts(chunk_id, name, qualified_name, signature, docstring, body)
            VALUES (:chunk_id, :name, :qualified_name, :signature, :docstring, :body)
        """)

        self.db.execute(query, {
            "chunk_id": chunk_id,
            "name": name,
            "qualified_name": qualified_name,
            "signature": signature,
            "docstring": docstring,
            "body": body
        })
        self.db.commit()

    def delete_chunk(self, chunk_id: str) -> None:
        """
        Remove a chunk from the BM25 index.

        Args:
            chunk_id: Unique identifier of chunk to delete
        """
        query = text("""
            DELETE FROM code_chunk_fts
            WHERE chunk_id = :chunk_id
        """)

        self.db.execute(query, {"chunk_id": chunk_id})
        self.db.commit()

    def delete_chunks_by_file(self, project_id: str, file_path: str) -> int:
        """
        Delete all chunks from a specific file (for re-indexing).

        Args:
            project_id: Project identifier
            file_path: Path to the file

        Returns:
            Number of chunks deleted

        Note:
            This requires joining with code_chunks table since FTS5 doesn't
            store project_id or file_path.
        """
        # First get chunk_ids for this file
        find_query = text("""
            SELECT id FROM code_chunks
            WHERE project_id = :project_id AND file_path = :file_path
        """)

        result = self.db.execute(find_query, {
            "project_id": project_id,
            "file_path": file_path
        })

        chunk_ids = [row[0] for row in result]

        if not chunk_ids:
            return 0

        # Delete from FTS5 table
        # SQLite doesn't support IN with parameters easily, so we do individual deletes
        deleted = 0
        for chunk_id in chunk_ids:
            self.delete_chunk(chunk_id)
            deleted += 1

        return deleted

    def search_bm25(
        self,
        query: str,
        limit: int = 20,
        project_id: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Search code chunks using BM25 ranking.

        Args:
            query: Search query (supports FTS5 query syntax)
            limit: Maximum number of results (default: 20)
            project_id: Optional project filter

        Returns:
            List of (chunk_id, score) tuples, sorted by relevance (descending)

        Note:
            FTS5 MATCH syntax:
            - "exact phrase" for phrase search
            - word1 OR word2 for boolean OR
            - word1 AND word2 for boolean AND (default)
            - word* for prefix matching
            - Special characters are escaped automatically
        """
        # Sanitize query to prevent FTS5 syntax errors
        sanitized_query = self._sanitize_query(query)

        if not sanitized_query:
            return []

        # Base FTS5 query using BM25 ranking
        # FTS5 provides bm25() as a built-in ranking function
        if project_id:
            # Join with code_chunks to filter by project
            sql_query = text("""
                SELECT fts.chunk_id, bm25(code_chunk_fts) as score
                FROM code_chunk_fts AS fts
                INNER JOIN code_chunks AS cc ON fts.chunk_id = cc.id
                WHERE code_chunk_fts MATCH :query AND cc.project_id = :project_id
                ORDER BY score
                LIMIT :limit
            """)
            result = self.db.execute(sql_query, {
                "query": sanitized_query,
                "project_id": project_id,
                "limit": limit
            })
        else:
            # No project filter - search all
            sql_query = text("""
                SELECT chunk_id, bm25(code_chunk_fts) as score
                FROM code_chunk_fts
                WHERE code_chunk_fts MATCH :query
                ORDER BY score
                LIMIT :limit
            """)
            result = self.db.execute(sql_query, {
                "query": sanitized_query,
                "limit": limit
            })

        # FTS5 bm25() returns negative scores (lower = better)
        # We negate to get positive scores where higher = better
        results = [(row[0], -float(row[1])) for row in result]

        return results

    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize user query for FTS5 MATCH using token extraction.

        Extracts alphanumeric tokens (including underscores for identifiers)
        and wraps each in quotes to prevent FTS5 syntax errors from special
        characters like =, <, >, /, etc.

        This approach:
        - Automatically handles ALL special characters (=, <, >, /, !, @, etc.)
        - Preserves word boundaries correctly
        - Supports prefix matching with trailing *
        - Works well for code search (keeps underscores in identifiers)

        Args:
            query: Raw user query (may contain code syntax, operators, etc.)

        Returns:
            Sanitized query safe for FTS5 MATCH, or empty string if no valid tokens

        Examples:
            >>> _sanitize_query("from fastapi import FastAPI app = FastAPI")
            '"from" "fastapi" "import" "FastAPI" "app" "FastAPI"'

            >>> _sanitize_query("VaultService")
            '"VaultService"'

            >>> _sanitize_query("find_definition*")
            '"find_definition"*'
        """
        import re

        if not query or not query.strip():
            return ""

        # Token pattern: alphanumeric + underscores + optional trailing *
        # Underscores are included for code identifiers (e.g., my_function)
        TOKEN_PATTERN = re.compile(r"[0-9A-Za-z_]+(?:\*)?")

        sanitized_terms = []

        for match in TOKEN_PATTERN.finditer(query):
            token = match.group()
            has_prefix_star = token.endswith("*")
            core = token[:-1] if has_prefix_star else token

            if not core:
                continue

            # Wrap token in quotes to neutralize FTS5 operators
            # Preserve trailing * for prefix matching
            sanitized_terms.append(f'"{core}"{"*" if has_prefix_star else ""}')

        if not sanitized_terms:
            return ""

        return " ".join(sanitized_terms)

    def get_stats(self, project_id: Optional[str] = None) -> dict:
        """
        Get BM25 index statistics.

        Args:
            project_id: Optional project filter

        Returns:
            Dictionary with index stats (total_chunks, indexed_chunks)
        """
        # Count total chunks
        if project_id:
            total_query = text("""
                SELECT COUNT(*) FROM code_chunks
                WHERE project_id = :project_id
            """)
            result = self.db.execute(total_query, {"project_id": project_id})
        else:
            total_query = text("SELECT COUNT(*) FROM code_chunks")
            result = self.db.execute(total_query)

        total_chunks = result.scalar() or 0

        # Count indexed chunks (in FTS5 table)
        indexed_query = text("SELECT COUNT(*) FROM code_chunk_fts")
        result = self.db.execute(indexed_query)
        indexed_chunks = result.scalar() or 0

        return {
            "total_chunks": total_chunks,
            "indexed_chunks": indexed_chunks,
            "coverage_percent": (indexed_chunks / total_chunks * 100) if total_chunks > 0 else 0
        }

    def rebuild_index(self, project_id: Optional[str] = None) -> int:
        """
        Rebuild entire BM25 index from code_chunks table.

        Args:
            project_id: Optional project filter (rebuilds only that project)

        Returns:
            Number of chunks indexed
        """
        # Clear existing FTS5 entries
        if project_id:
            # Get chunk IDs for this project
            find_query = text("""
                SELECT id FROM code_chunks
                WHERE project_id = :project_id
            """)
            result = self.db.execute(find_query, {"project_id": project_id})
            chunk_ids = [row[0] for row in result]

            # Delete them from FTS5
            for chunk_id in chunk_ids:
                delete_query = text("""
                    DELETE FROM code_chunk_fts WHERE chunk_id = :chunk_id
                """)
                self.db.execute(delete_query, {"chunk_id": chunk_id})
        else:
            # Clear entire FTS5 table
            self.db.execute(text("DELETE FROM code_chunk_fts"))

        self.db.commit()

        # Rebuild from code_chunks
        if project_id:
            chunks_query = text("""
                SELECT id, name, qualified_name, signature, docstring, body
                FROM code_chunks
                WHERE project_id = :project_id
            """)
            result = self.db.execute(chunks_query, {"project_id": project_id})
        else:
            chunks_query = text("""
                SELECT id, name, qualified_name, signature, docstring, body
                FROM code_chunks
            """)
            result = self.db.execute(chunks_query)

        count = 0
        for row in result:
            chunk_id, name, qualified_name, signature, docstring, body = row
            self.index_chunk(chunk_id, name, qualified_name, signature, docstring, body)
            count += 1

        return count


# Convenience functions for standalone use
def index_chunk(
    chunk_id: str,
    name: str,
    qualified_name: str,
    signature: Optional[str],
    docstring: Optional[str],
    body: str,
    db: Optional[Session] = None
) -> None:
    """Standalone function to index a single chunk."""
    with BM25Indexer(db) as indexer:
        indexer.index_chunk(chunk_id, name, qualified_name, signature, docstring, body)


def search_bm25(
    query: str,
    limit: int = 20,
    project_id: Optional[str] = None,
    db: Optional[Session] = None
) -> List[dict]:
    """
    Standalone function to search BM25 index.

    Returns list of dicts with chunk metadata instead of just (chunk_id, score) tuples.
    """
    from vlt.core.models import CodeChunk

    with BM25Indexer(db) as indexer:
        # Get (chunk_id, score) pairs
        raw_results = indexer.search_bm25(query, limit, project_id)

        if not raw_results:
            return []

        # Fetch full chunk metadata
        chunk_ids = [chunk_id for chunk_id, _ in raw_results]
        score_map = {chunk_id: score for chunk_id, score in raw_results}

        # Query database for chunks
        chunks_query = text("""
            SELECT id, file_path, qualified_name, signature, docstring, body, lineno
            FROM code_chunks
            WHERE id IN :chunk_ids
        """)

        # SQLite doesn't support tuples in IN clause directly with parameters
        # We need to use a different approach
        results = []
        for chunk_id in chunk_ids:
            chunk_query = text("""
                SELECT id, file_path, qualified_name, signature, docstring, body, lineno
                FROM code_chunks
                WHERE id = :chunk_id
            """)
            result = indexer.db.execute(chunk_query, {"chunk_id": chunk_id})
            row = result.first()

            if row:
                results.append({
                    "chunk_id": row[0],
                    "file_path": row[1],
                    "qualified_name": row[2],
                    "signature": row[3],
                    "docstring": row[4],
                    "body": row[5],
                    "lineno": row[6],
                    "score": score_map[row[0]]
                })

        # Sort by score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results


def delete_chunk(chunk_id: str, db: Optional[Session] = None) -> None:
    """Standalone function to delete a chunk from index."""
    with BM25Indexer(db) as indexer:
        indexer.delete_chunk(chunk_id)
