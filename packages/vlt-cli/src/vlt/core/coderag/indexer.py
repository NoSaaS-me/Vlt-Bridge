"""CodeRAG Indexer service for indexing project codebases.

This module provides the main indexing service that orchestrates:
- File discovery and parsing
- Semantic chunking with context enrichment
- Embedding generation (vector search)
- BM25 keyword indexing
- Code graph construction
- Repository map generation
- ctags symbol indexing
- Incremental indexing with content hash comparison

Based on tasks T024-T026 and T030 (incremental indexing).
"""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from uuid import uuid4
import json

from sqlalchemy import select, delete, func
from sqlalchemy.orm import Session

from vlt.db import engine
from vlt.core.models import (
    CodeChunk, CodeNode, CodeEdge, SymbolDefinition, RepoMap,
    IndexDeltaQueue, ChunkType, NodeType, EdgeType,
    ChangeType, QueueStatus
)
from vlt.core.identity import load_vlt_config, load_coderag_config
from vlt.config import Settings

from .parser import detect_language, SUPPORTED_LANGUAGES
from .chunker import chunk_file, is_available as chunker_available
from .embedder import get_embeddings_batch, EmbeddingError
from .bm25 import index_chunk as bm25_index_chunk, delete_chunk as bm25_delete_chunk
from .graph import build_graph
from .ctags import generate_ctags, parse_ctags
from .delta import (
    DeltaQueueManager, DeltaConfig,
    detect_file_changes, scan_directory_for_changes
)

logger = logging.getLogger(__name__)


class IndexerStats:
    """Statistics from indexing operation."""
    def __init__(self):
        self.files_discovered = 0
        self.files_indexed = 0
        self.files_skipped = 0
        self.files_failed = 0
        self.chunks_created = 0
        self.symbols_indexed = 0
        self.graph_nodes = 0
        self.graph_edges = 0
        self.embeddings_generated = 0
        self.start_time = datetime.now(timezone.utc)
        self.end_time = None
        self.errors: List[str] = []

    def finish(self):
        self.end_time = datetime.now(timezone.utc)

    @property
    def duration_seconds(self) -> float:
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_discovered": self.files_discovered,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "files_failed": self.files_failed,
            "chunks_created": self.chunks_created,
            "symbols_indexed": self.symbols_indexed,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
            "embeddings_generated": self.embeddings_generated,
            "duration_seconds": self.duration_seconds,
            "errors": self.errors
        }


class CodeRAGIndexer:
    """Service for indexing codebases with hybrid retrieval support."""

    def __init__(self, project_path: Path, project_id: str, settings: Optional[Settings] = None):
        """Initialize indexer for a project.

        Args:
            project_path: Root path of the project to index
            project_id: Project identifier
            settings: Optional settings instance (uses default if None)
        """
        self.project_path = project_path.resolve()
        self.project_id = project_id
        self.settings = settings or Settings()
        self.config = load_coderag_config(project_path)
        self.stats = IndexerStats()
        self.delta_manager = DeltaQueueManager(project_id)

    def index_full(
        self,
        force: bool = False,
        progress_callback: Optional[Callable[[int, int, int], None]] = None
    ) -> IndexerStats:
        """Perform full index of the codebase.

        Args:
            force: If True, re-index all files. If False, use incremental indexing.
            progress_callback: Optional callback invoked at progress points.
                Signature: (files_done: int, files_total: int, chunks: int) -> None
                Called at: file discovery, per-file completion, embedding generation.

        Returns:
            IndexerStats with results
        """
        logger.info(f"Starting full index of project {self.project_id} at {self.project_path}")

        def _report_progress(files_done: int, files_total: int, chunks: int):
            """Safely invoke progress callback if provided."""
            if progress_callback:
                try:
                    progress_callback(files_done, files_total, chunks)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

        try:
            # Step 1: Discover files
            files = self._discover_files()
            self.stats.files_discovered = len(files)
            logger.info(f"Discovered {len(files)} files")

            # Report file discovery
            _report_progress(0, len(files), 0)

            if not files:
                logger.warning("No files to index")
                self.stats.finish()
                return self.stats

            # Step 2: Filter by content hash (incremental)
            if not force:
                files = self._filter_unchanged_files(files)
                logger.info(f"After incremental filtering: {len(files)} files need indexing")

            if not files:
                logger.info("All files are up-to-date")
                _report_progress(0, 0, 0)
                self.stats.finish()
                return self.stats

            files_total = len(files)

            # Step 3: Parse and chunk files
            all_chunks = []
            parsed_files = {}  # For graph building
            for idx, file_path in enumerate(files):
                try:
                    chunks, tree, source, language = self._index_file(file_path)
                    if chunks:
                        all_chunks.extend(chunks)
                        parsed_files[str(file_path)] = (tree, source, language)
                        self.stats.files_indexed += 1
                    else:
                        self.stats.files_skipped += 1
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    self.stats.files_failed += 1
                    self.stats.errors.append(f"{file_path}: {str(e)}")

                # Report per-file completion
                _report_progress(idx + 1, files_total, len(all_chunks))

            self.stats.chunks_created = len(all_chunks)
            logger.info(f"Created {len(all_chunks)} chunks from {self.stats.files_indexed} files")

            if not all_chunks:
                logger.warning("No chunks created")
                self.stats.finish()
                return self.stats

            # Step 4: Generate embeddings (async batch)
            asyncio.run(self._generate_embeddings(all_chunks))

            # Report after embedding generation
            _report_progress(files_total, files_total, len(all_chunks))

            # Step 5: Store chunks in database
            self._store_chunks(all_chunks)

            # Step 6: Build code graph
            if parsed_files:
                self._build_graph(parsed_files)

            # Step 7: Generate ctags
            self._generate_ctags_index()

            # Step 8: Generate repo map
            self._generate_repo_map()

            self.stats.finish()
            logger.info(f"Indexing complete in {self.stats.duration_seconds:.2f}s")
            return self.stats

        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            self.stats.errors.append(f"Fatal: {str(e)}")
            self.stats.finish()
            raise

    def index_changed_files(self) -> IndexerStats:
        """Index only files that have changed since last index (T030).

        Uses content hash comparison to detect changes.

        Returns:
            IndexerStats with results
        """
        logger.info(f"Starting incremental index for project {self.project_id}")
        return self.index_full(force=False)

    def get_index_status(self) -> Dict[str, Any]:
        """Get current index status and statistics (T054 - includes delta queue).

        Returns:
            Dictionary with index health information including delta queue details
        """
        with Session(engine) as session:
            # Count chunks
            chunks_count = session.scalar(
                select(func.count(CodeChunk.id)).where(CodeChunk.project_id == self.project_id)
            ) or 0

            # Count nodes
            nodes_count = session.scalar(
                select(func.count(CodeNode.id)).where(CodeNode.project_id == self.project_id)
            ) or 0

            # Count edges
            edges_count = session.scalar(
                select(func.count(CodeEdge.id)).where(CodeEdge.project_id == self.project_id)
            ) or 0

            # Count symbols
            symbols_count = session.scalar(
                select(func.count(SymbolDefinition.id)).where(SymbolDefinition.project_id == self.project_id)
            ) or 0

            # Get distinct file count
            files_count = session.scalar(
                select(func.count(func.distinct(CodeChunk.file_path))).where(CodeChunk.project_id == self.project_id)
            ) or 0

            # Get last indexed time
            last_indexed = session.scalar(
                select(func.max(CodeChunk.updated_at)).where(CodeChunk.project_id == self.project_id)
            )

            # Get repo map info
            repo_map = session.scalar(
                select(RepoMap).where(RepoMap.project_id == self.project_id).order_by(RepoMap.created_at.desc())
            )

            # Get delta queue count (T030)
            queue_count = session.scalar(
                select(func.count(IndexDeltaQueue.id)).where(
                    IndexDeltaQueue.project_id == self.project_id,
                    IndexDeltaQueue.status == QueueStatus.PENDING
                )
            ) or 0

            # Get detailed delta queue status (T054)
            delta_queue_status = self.delta_manager.get_queue_status()

            return {
                "project_id": self.project_id,
                "files_count": files_count,
                "chunks_count": chunks_count,
                "symbols_count": symbols_count,
                "graph_nodes": nodes_count,
                "graph_edges": edges_count,
                "last_indexed": last_indexed.isoformat() if last_indexed else None,
                "repo_map": {
                    "token_count": repo_map.token_count if repo_map else 0,
                    "symbols_included": repo_map.symbols_included if repo_map else 0,
                    "symbols_total": repo_map.symbols_total if repo_map else 0,
                } if repo_map else None,
                "delta_queue_count": queue_count,
                "delta_queue": delta_queue_status,  # Enhanced with T054
            }

    # ========================================================================
    # Private methods
    # ========================================================================

    def _discover_files(self) -> List[Path]:
        """Discover files matching include/exclude patterns from config."""
        from pathlib import Path
        import fnmatch

        files = []

        # Get patterns from config
        include_patterns = self.config.include if self.config else ["**/*.py"]
        exclude_patterns = self.config.exclude if self.config else []

        # Collect all files matching include patterns
        for pattern in include_patterns:
            for file_path in self.project_path.glob(pattern):
                if file_path.is_file():
                    # Check exclude patterns
                    relative_path = str(file_path.relative_to(self.project_path))
                    excluded = False
                    for exclude_pattern in exclude_patterns:
                        if fnmatch.fnmatch(relative_path, exclude_pattern.replace("**", "*")):
                            excluded = True
                            break

                    if not excluded:
                        files.append(file_path)

        return sorted(set(files))  # Deduplicate and sort

    def _filter_unchanged_files(self, files: List[Path]) -> List[Path]:
        """Filter out files that haven't changed since last index.

        Compares MD5 hash of file content with stored hash in database.

        Args:
            files: List of file paths to check

        Returns:
            List of files that have changed or are new
        """
        changed_files = []

        with Session(engine) as session:
            for file_path in files:
                try:
                    # Calculate current hash
                    current_hash = self._calculate_file_hash(file_path)

                    # Check if file exists in index
                    relative_path = str(file_path.relative_to(self.project_path))
                    existing = session.scalar(
                        select(CodeChunk.file_hash)
                        .where(
                            CodeChunk.project_id == self.project_id,
                            CodeChunk.file_path == relative_path
                        )
                        .limit(1)
                    )

                    # Include if new or hash changed
                    if existing is None or existing != current_hash:
                        changed_files.append(file_path)
                    else:
                        logger.debug(f"Skipping unchanged file: {relative_path}")

                except Exception as e:
                    logger.warning(f"Error checking hash for {file_path}: {e}")
                    # Include file if we can't check hash
                    changed_files.append(file_path)

        return changed_files

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content."""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _index_file(self, file_path: Path) -> Tuple[List[Dict[str, Any]], Any, str, str]:
        """Parse and chunk a single file.

        Returns:
            Tuple of (chunks, tree, source, language)
        """
        if not chunker_available():
            raise RuntimeError("Chunker not available. Install dependencies: pip install llama-index-core tree-sitter tree-sitter-languages")

        # Detect language
        language = detect_language(str(file_path))
        if not language or language not in SUPPORTED_LANGUAGES:
            logger.debug(f"Skipping unsupported language: {file_path}")
            return [], None, "", ""

        # Read file
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return [], None, "", ""

        # Parse with tree-sitter
        from .parser import parse_file
        tree = parse_file(content, language)
        if not tree:
            logger.warning(f"Failed to parse {file_path}")
            return [], None, "", ""

        # Chunk the file
        relative_path = str(file_path.relative_to(self.project_path))
        chunks = chunk_file(content, language, relative_path)

        # Add metadata to each chunk
        file_hash = self._calculate_file_hash(file_path)
        for chunk in chunks:
            chunk['project_id'] = self.project_id
            chunk['file_hash'] = file_hash
            chunk['chunk_id'] = str(uuid4())

        return chunks, tree, content, language

    async def _generate_embeddings(self, chunks: List[Dict[str, Any]]):
        """Generate embeddings for all chunks (async batch)."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")

        # Extract texts to embed
        texts = [chunk['chunk_text'] for chunk in chunks]

        # Generate embeddings in batches
        try:
            batch_size = self.config.embedding.batch_size if self.config else 10
            embeddings = await get_embeddings_batch(texts, batch_size=batch_size, settings=self.settings)

            # Attach embeddings to chunks
            for chunk, embedding in zip(chunks, embeddings):
                if embedding:
                    # Store as bytes (numpy array serialized)
                    import numpy as np
                    chunk['embedding'] = np.array(embedding, dtype=np.float32).tobytes()
                    self.stats.embeddings_generated += 1
                else:
                    chunk['embedding'] = None
                    logger.warning(f"Failed to generate embedding for chunk {chunk['chunk_id']}")

        except EmbeddingError as e:
            logger.error(f"Embedding generation failed: {e}")
            # Continue without embeddings (vector search will be unavailable)
            for chunk in chunks:
                chunk['embedding'] = None

    def _store_chunks(self, chunks: List[Dict[str, Any]]):
        """Store chunks in database and BM25 index."""
        logger.info(f"Storing {len(chunks)} chunks in database...")

        with Session(engine) as session:
            for chunk_dict in chunks:
                try:
                    # Map chunk_type string to enum
                    chunk_type_str = chunk_dict.get('chunk_type', 'code')
                    if chunk_type_str == 'function':
                        chunk_type = ChunkType.FUNCTION
                    elif chunk_type_str == 'class':
                        chunk_type = ChunkType.CLASS
                    elif chunk_type_str == 'method':
                        chunk_type = ChunkType.METHOD
                    else:
                        chunk_type = ChunkType.MODULE

                    # Extract name from qualified_name if name is missing
                    name = chunk_dict.get('name')
                    if not name:
                        qualified_name = chunk_dict.get('qualified_name', '')
                        name = qualified_name.split('.')[-1] if '.' in qualified_name else qualified_name

                    # Convert decorators list to string
                    decorators = chunk_dict.get('decorators', [])
                    decorators_str = '\n'.join(decorators) if decorators else None

                    # Create CodeChunk model
                    chunk = CodeChunk(
                        id=chunk_dict['chunk_id'],
                        project_id=chunk_dict['project_id'],
                        file_path=chunk_dict['file_path'],
                        file_hash=chunk_dict['file_hash'],
                        chunk_type=chunk_type,
                        name=name or 'unknown',
                        qualified_name=chunk_dict.get('qualified_name', ''),
                        language=chunk_dict['language'],
                        lineno=chunk_dict.get('lineno', 1),
                        end_lineno=chunk_dict.get('end_lineno', 1),
                        imports=chunk_dict.get('imports'),
                        class_context=chunk_dict.get('class_context'),
                        signature=chunk_dict.get('signature'),
                        decorators=decorators_str,
                        docstring=chunk_dict.get('docstring'),
                        body=chunk_dict.get('body', ''),
                        embedding=chunk_dict.get('embedding'),
                        embedding_text=chunk_dict.get('chunk_text'),
                        token_count=len(chunk_dict.get('chunk_text', '').split()) // 4,  # Rough estimate
                    )

                    session.add(chunk)

                    # Add to BM25 index
                    bm25_index_chunk(
                        chunk_id=chunk_dict['chunk_id'],
                        name=name or '',
                        qualified_name=chunk_dict.get('qualified_name', ''),
                        signature=chunk_dict.get('signature', ''),
                        docstring=chunk_dict.get('docstring', ''),
                        body=chunk_dict.get('body', ''),
                    )

                except Exception as e:
                    logger.error(f"Error storing chunk: {e}")
                    self.stats.errors.append(f"Store chunk: {str(e)}")

            session.commit()

    def _build_graph(self, parsed_files: Dict[str, Tuple[Any, str, str]]):
        """Build code dependency graph from parsed files."""
        logger.info("Building code graph...")

        try:
            nodes, edges = build_graph(parsed_files, self.project_id)

            with Session(engine) as session:
                # Store nodes
                for node_dict in nodes:
                    try:
                        # Map node_type string to enum
                        node_type_str = node_dict.get('node_type', 'function')
                        node_type_map = {
                            'module': NodeType.MODULE,
                            'class': NodeType.CLASS,
                            'function': NodeType.FUNCTION,
                            'method': NodeType.METHOD,
                            'variable': NodeType.VARIABLE,
                        }
                        node_type = node_type_map.get(node_type_str, NodeType.FUNCTION)

                        node = CodeNode(
                            id=node_dict['id'],
                            project_id=node_dict['project_id'],
                            file_path=node_dict['file_path'],
                            node_type=node_type,
                            name=node_dict['name'],
                            signature=node_dict.get('signature'),
                            lineno=node_dict.get('lineno'),
                            docstring=node_dict.get('docstring'),
                            centrality_score=None,  # Will be calculated later
                        )
                        session.add(node)
                        self.stats.graph_nodes += 1
                    except Exception as e:
                        logger.error(f"Error storing node: {e}")

                # Store edges
                for edge_dict in edges:
                    try:
                        # Map edge_type string to enum
                        edge_type_str = edge_dict.get('edge_type', 'calls')
                        edge_type_map = {
                            'calls': EdgeType.CALLS,
                            'imports': EdgeType.IMPORTS,
                            'inherits': EdgeType.INHERITS,
                            'uses': EdgeType.USES,
                            'decorates': EdgeType.DECORATES,
                        }
                        edge_type = edge_type_map.get(edge_type_str, EdgeType.CALLS)

                        edge = CodeEdge(
                            id=str(uuid4()),
                            project_id=edge_dict['project_id'],
                            source_id=edge_dict['source_id'],
                            target_id=edge_dict['target_id'],
                            edge_type=edge_type,
                            lineno=edge_dict.get('lineno'),
                            count=edge_dict.get('count', 1),
                        )
                        session.add(edge)
                        self.stats.graph_edges += 1
                    except Exception as e:
                        logger.error(f"Error storing edge: {e}")

                session.commit()

        except Exception as e:
            logger.error(f"Graph building failed: {e}")
            self.stats.errors.append(f"Graph: {str(e)}")

    def _generate_ctags_index(self):
        """Generate ctags symbol definitions."""
        logger.info("Generating ctags index...")

        try:
            tags_path = generate_ctags(str(self.project_path))
            if not tags_path:
                logger.warning("ctags not available, skipping symbol indexing")
                return

            symbols = parse_ctags(tags_path)

            with Session(engine) as session:
                for symbol in symbols:
                    try:
                        # Make file path relative to project root
                        abs_path = (self.project_path / symbol.file_path).resolve()
                        if abs_path.is_relative_to(self.project_path):
                            relative_path = str(abs_path.relative_to(self.project_path))
                        else:
                            relative_path = symbol.file_path

                        sym_def = SymbolDefinition(
                            id=str(uuid4()),
                            project_id=self.project_id,
                            name=symbol.name,
                            file_path=relative_path,
                            lineno=symbol.lineno,
                            kind=symbol.kind,
                            scope=symbol.scope,
                            signature=symbol.signature,
                            language=symbol.language,
                        )
                        session.add(sym_def)
                        self.stats.symbols_indexed += 1
                    except Exception as e:
                        logger.error(f"Error storing symbol: {e}")

                session.commit()

        except Exception as e:
            logger.error(f"ctags indexing failed: {e}")
            self.stats.errors.append(f"ctags: {str(e)}")

    def _generate_repo_map(self, scope: Optional[str] = None):
        """Generate repository structure map (Aider-style) with centrality ranking.

        Args:
            scope: Optional subdirectory to focus on (e.g., "src/api/")
        """
        logger.info("Generating repository map...")

        try:
            from .repomap import (
                Symbol,
                build_reference_graph,
                calculate_centrality,
                generate_repo_map
            )

            with Session(engine) as session:
                # Get all nodes
                nodes = session.scalars(
                    select(CodeNode)
                    .where(CodeNode.project_id == self.project_id)
                ).all()

                if not nodes:
                    logger.warning("No nodes to build repo map")
                    return

                # Get all edges
                edges = session.scalars(
                    select(CodeEdge)
                    .where(CodeEdge.project_id == self.project_id)
                ).all()

                # Convert to Symbol objects for repomap module
                symbols = []
                for node in nodes:
                    symbol = Symbol(
                        name=node.name,
                        qualified_name=node.id,
                        file_path=node.file_path,
                        symbol_type=node.node_type.value,
                        signature=node.signature,
                        lineno=node.lineno,
                        docstring=node.docstring
                    )
                    symbols.append(symbol)

                # Build reference graph
                edge_tuples = [(edge.source_id, edge.target_id) for edge in edges]
                graph = build_reference_graph(symbols, edge_tuples)

                # Calculate centrality scores (PageRank)
                centrality_scores = calculate_centrality(graph)

                # Update centrality scores in database
                for node in nodes:
                    if node.id in centrality_scores:
                        node.centrality_score = centrality_scores[node.id]

                session.commit()

                # Generate map with token budget
                max_tokens = self.config.repomap.max_tokens if self.config else 4000
                include_signatures = self.config.repomap.include_signatures if self.config else True

                repo_map_data = generate_repo_map(
                    symbols=symbols,
                    graph=graph,
                    centrality_scores=centrality_scores,
                    max_tokens=max_tokens,
                    scope=scope,
                    include_signatures=include_signatures,
                    include_docstrings=False  # Too expensive for default map
                )

                # Store repo map
                repo_map = RepoMap(
                    id=str(uuid4()),
                    project_id=self.project_id,
                    scope=repo_map_data['scope'],
                    map_text=repo_map_data['map_text'],
                    token_count=repo_map_data['token_count'],
                    max_tokens=repo_map_data['max_tokens'],
                    files_included=repo_map_data['files_included'],
                    symbols_included=repo_map_data['symbols_included'],
                    symbols_total=repo_map_data['symbols_total'],
                )
                session.add(repo_map)
                session.commit()

                logger.info(
                    f"Generated repo map: {repo_map_data['symbols_included']}/{repo_map_data['symbols_total']} symbols, "
                    f"{repo_map_data['token_count']}/{max_tokens} tokens"
                )

        except Exception as e:
            logger.error(f"Repo map generation failed: {e}")
            self.stats.errors.append(f"Repo map: {str(e)}")

    # ========================================================================
    # T050-T055: Delta-Based Index Commits
    # ========================================================================

    def scan_for_changes(self) -> int:
        """Scan project directory for changed files and queue them.

        This scans the project directory and adds any changed files to the
        delta queue without immediately indexing them.

        Returns:
            Number of files queued
        """
        logger.info("Scanning for file changes...")

        # Get include/exclude patterns from config
        include_patterns = self.config.include if self.config else ["**/*.py"]
        exclude_patterns = self.config.exclude if self.config else []

        # Scan for changes
        changes = scan_directory_for_changes(
            self.project_path,
            self.project_id,
            include_patterns,
            exclude_patterns
        )

        # Queue changes
        queued = 0
        for file_path, change_type, old_hash, new_hash in changes:
            if self.delta_manager.queue_file_change(
                file_path,
                self.project_path,
                change_type,
                old_hash,
                new_hash
            ):
                queued += 1

        logger.info(f"Queued {queued} changed files")
        return queued

    def batch_commit_if_needed(self) -> Optional[IndexerStats]:
        """Check thresholds and batch commit if needed (T052).

        This checks if the delta queue has reached any threshold:
        - FILES_THRESHOLD: 5 files queued
        - LINES_THRESHOLD: 1000 lines changed
        - TIMEOUT: 5 minutes since last change

        If any threshold is exceeded, performs batch commit.

        Returns:
            IndexerStats if commit was performed, None otherwise
        """
        if not self.delta_manager.check_thresholds():
            return None

        logger.info("Delta threshold reached, starting batch commit...")
        return self.batch_commit_delta_queue()

    def batch_commit_delta_queue(self, force: bool = False) -> IndexerStats:
        """Commit all queued changes to indexes (T052, T054).

        Args:
            force: If True, commit even if no thresholds are met

        Returns:
            IndexerStats with results
        """
        logger.info("Starting batch commit of delta queue...")

        # Get pending files from queue
        pending_file_paths = self.delta_manager.get_pending_files()

        if not pending_file_paths:
            logger.info("No files in delta queue")
            self.stats.finish()
            return self.stats

        logger.info(f"Committing {len(pending_file_paths)} files from delta queue")

        # Convert relative paths to absolute
        files_to_index = []
        for rel_path in pending_file_paths:
            abs_path = self.project_path / rel_path
            if abs_path.exists():
                files_to_index.append(abs_path)
            else:
                logger.warning(f"File in queue no longer exists: {rel_path}")

        if not files_to_index:
            logger.warning("No files to index")
            self.delta_manager.clear_queue()
            self.stats.finish()
            return self.stats

        # Reuse existing indexing logic
        try:
            # Parse and chunk files
            all_chunks = []
            parsed_files = {}
            for file_path in files_to_index:
                try:
                    chunks, tree, source, language = self._index_file(file_path)
                    if chunks:
                        all_chunks.extend(chunks)
                        parsed_files[str(file_path)] = (tree, source, language)
                        self.stats.files_indexed += 1
                    else:
                        self.stats.files_skipped += 1
                except Exception as e:
                    logger.error(f"Error indexing {file_path}: {e}")
                    self.stats.files_failed += 1
                    self.stats.errors.append(f"{file_path}: {str(e)}")

            self.stats.chunks_created = len(all_chunks)

            if all_chunks:
                # Generate embeddings
                asyncio.run(self._generate_embeddings(all_chunks))

                # Store chunks
                self._store_chunks(all_chunks)

                # Update graph (incremental - only for these files)
                if parsed_files:
                    self._build_graph(parsed_files)

                # Update ctags
                self._generate_ctags_index()

                # Regenerate repo map (since graph changed)
                self._generate_repo_map()

            # Mark files as indexed in queue
            self.delta_manager.mark_as_indexed(pending_file_paths)

            self.stats.finish()
            logger.info(f"Batch commit complete: {self.stats.files_indexed} files indexed in {self.stats.duration_seconds:.2f}s")
            return self.stats

        except Exception as e:
            logger.error(f"Batch commit failed: {e}")
            self.stats.errors.append(f"Fatal: {str(e)}")
            self.stats.finish()
            raise

    def index_files_just_in_time(self, query: str) -> List[str]:
        """Index pending files that match a query (T053).

        This is called before oracle queries to ensure relevant uncommitted
        files are indexed.

        Args:
            query: Natural language query from user

        Returns:
            List of file paths that were indexed
        """
        from .delta import get_files_matching_query

        # Get pending files
        pending_files = self.delta_manager.get_pending_files()

        if not pending_files:
            return []

        # Find files matching query
        matching_files = get_files_matching_query(query, self.project_id, pending_files)

        if not matching_files:
            return []

        logger.info(f"Just-in-time indexing {len(matching_files)} files for query")

        # Index only matching files
        files_to_index = []
        for rel_path in matching_files:
            abs_path = self.project_path / rel_path
            if abs_path.exists():
                files_to_index.append(abs_path)

        if not files_to_index:
            return []

        # Index the files
        try:
            all_chunks = []
            parsed_files = {}
            indexed_paths = []

            for file_path in files_to_index:
                try:
                    chunks, tree, source, language = self._index_file(file_path)
                    if chunks:
                        all_chunks.extend(chunks)
                        parsed_files[str(file_path)] = (tree, source, language)
                        indexed_paths.append(str(file_path.relative_to(self.project_path)))
                except Exception as e:
                    logger.error(f"Error in JIT indexing {file_path}: {e}")

            if all_chunks:
                asyncio.run(self._generate_embeddings(all_chunks))
                self._store_chunks(all_chunks)

                if parsed_files:
                    self._build_graph(parsed_files)

            # Mark as indexed
            if indexed_paths:
                self.delta_manager.mark_as_indexed(indexed_paths)

            logger.info(f"JIT indexed {len(indexed_paths)} files")
            return indexed_paths

        except Exception as e:
            logger.error(f"JIT indexing failed: {e}")
            return []
