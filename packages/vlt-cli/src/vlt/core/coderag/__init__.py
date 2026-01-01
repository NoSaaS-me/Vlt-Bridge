"""CodeRAG subsystem for intelligent code retrieval and indexing.

This module provides components for building a production-grade code intelligence system:

1. Parser (parser.py):
   - Tree-sitter wrapper for language-agnostic code parsing
   - Supports Python, TypeScript, JavaScript, Go, Rust
   - Provides language detection from file extensions

2. Chunker (chunker.py):
   - LlamaIndex CodeSplitter integration (Sweep's chunker)
   - Context-enriched semantic code chunks
   - Extracts imports, class context, signatures, docstrings

3. Graph Builder (graph.py):
   - Extracts code relationships (imports, calls, inheritance) using tree-sitter
   - Supports Python, TypeScript, JavaScript
   - Returns CodeNode and CodeEdge dicts for database insertion

4. Ctags Wrapper (ctags.py):
   - Generates and parses Universal Ctags symbol indexes
   - Provides fast symbol definition lookup
   - Falls back gracefully if ctags is not installed

Example Usage:

    # Parsing and chunking
    from vlt.core.coderag import detect_language, chunk_file

    language = detect_language("src/main.py")
    with open("src/main.py") as f:
        content = f.read()

    chunks = chunk_file(content, language, "src/main.py")
    for chunk in chunks:
        print(f"{chunk['qualified_name']}: {chunk['chunk_type']}")

    # Graph building
    from tree_sitter_language_pack import get_parser
    from vlt.core.coderag.graph import build_graph

    parsed_files = {}
    for file_path in source_files:
        with open(file_path) as f:
            source = f.read()
        parser = get_parser("python")
        tree = parser.parse(bytes(source, "utf8"))
        parsed_files[file_path] = (tree, source, "python")

    nodes, edges = build_graph(parsed_files, project_id="my-project")

    # Ctags indexing
    from vlt.core.coderag.ctags import generate_ctags, parse_ctags, lookup_definition

    tags_path = generate_ctags("/path/to/project")
    if tags_path:
        symbols = parse_ctags(tags_path)
        definition = lookup_definition("UserService", symbols)
        if definition:
            print(f"Found at {definition.file_path}:{definition.lineno}")
"""

# Optional imports with graceful degradation
try:
    from vlt.core.coderag.graph import build_graph, CodeNode, CodeEdge
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False
    build_graph = None
    CodeNode = None
    CodeEdge = None

try:
    from vlt.core.coderag.ctags import (
        generate_ctags,
        parse_ctags,
        lookup_definition,
        lookup_all_definitions,
        SymbolDefinition
    )
    _CTAGS_AVAILABLE = True
except ImportError:
    _CTAGS_AVAILABLE = False
    generate_ctags = None
    parse_ctags = None
    lookup_definition = None
    lookup_all_definitions = None
    SymbolDefinition = None

try:
    from vlt.core.coderag.store import CodeRAGStore, CodeRAGStoreError
    from vlt.core.coderag.indexer import CodeRAGIndexer, IndexerStats
    _STORE_AVAILABLE = True
except ImportError:
    _STORE_AVAILABLE = False
    CodeRAGStore = None
    CodeRAGStoreError = None
    CodeRAGIndexer = None
    IndexerStats = None

__all__ = [
    # Embedding (always available)
    "get_embedding",
    "get_embeddings_batch",
    "get_embedding_sync",
    "get_embeddings_batch_sync",
    "EmbeddingError",
    # BM25 (always available)
    "BM25Indexer",
    "index_chunk",
    "search_bm25",
    "delete_chunk",
    # Parser (optional - tree-sitter)
    "parse_file",
    "detect_language",
    "get_node_text",
    "SUPPORTED_LANGUAGES",
    # Chunker (optional - llama-index)
    "chunk_file",
    "DEFAULT_CHUNK_LINES",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_MAX_CHARS",
    # Graph builder (optional - tree-sitter)
    "build_graph",
    "CodeNode",
    "CodeEdge",
    # Ctags (optional - ctags binary)
    "generate_ctags",
    "parse_ctags",
    "lookup_definition",
    "lookup_all_definitions",
    "SymbolDefinition",
    # Store (database interface)
    "CodeRAGStore",
    "CodeRAGStoreError",
    # Indexer (orchestrator)
    "CodeRAGIndexer",
    "IndexerStats",
]

from .embedder import (
    get_embedding,
    get_embeddings_batch,
    get_embedding_sync,
    get_embeddings_batch_sync,
    EmbeddingError,
)

from .bm25 import (
    BM25Indexer,
    index_chunk,
    search_bm25,
    delete_chunk,
)

from .parser import (
    parse_file,
    detect_language,
    get_node_text,
    SUPPORTED_LANGUAGES,
)

from .chunker import (
    chunk_file,
    DEFAULT_CHUNK_LINES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_MAX_CHARS,
)
