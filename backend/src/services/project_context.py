"""ProjectContext — Lazy-loading project wrapper for RLM Oracle REPL namespace.

Implements FR-007 (ProjectContext in REPL namespace), FR-008 (file manifest),
FR-009 (text search), FR-010 (lazy file handles), FR-011 (symbol extraction),
FR-012 (chunking), FR-013 (vlt thread access), FR-014 (large-file guard).

Components:
    FileEntry      — Single file metadata record (no content loaded).
    FileManifest   — Precomputed listing of all project files.
    GrepMatch      — A regex match with surrounding context lines.
    SearchMatch    — A search result from BM25/CodeRAG search.
    SymbolInfo     — A symbol extracted from source code via tree-sitter.
    TextHandle     — Lazy reference to a text resource (file/thread/note/chunk).
    ProjectContext — Root REPL object exposing the full project.

Part of 022-rlm-oracle (RLM Oracle replacement).
"""
from __future__ import annotations
