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

import logging
import mimetypes
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Extensions that are always treated as binary (no text content)
BINARY_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg',
    '.pdf', '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z',
    '.exe', '.dll', '.so', '.dylib', '.a', '.o',
    '.pyc', '.pyo', '.pyd',
    '.db', '.sqlite', '.sqlite3',
    '.mp3', '.mp4', '.wav', '.avi', '.mov',
    '.woff', '.woff2', '.ttf', '.eot',
}

# Directories to always skip when building manifest
SKIP_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv',
    '.env', 'dist', 'build', '.next', '.nuxt', 'target',
    '.pytest_cache', '.mypy_cache', '.ruff_cache',
    'vendor', '.idea', '.vscode',
}

# Language detection by extension
EXTENSION_TO_LANGUAGE = {
    '.py': 'python', '.pyi': 'python',
    '.ts': 'typescript', '.tsx': 'typescript',
    '.js': 'javascript', '.jsx': 'javascript', '.mjs': 'javascript',
    '.go': 'go',
    '.rs': 'rust',
    '.java': 'java',
    '.kt': 'kotlin',
    '.cpp': 'cpp', '.cc': 'cpp', '.cxx': 'cpp', '.c': 'c', '.h': 'c',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.cs': 'csharp',
    '.lua': 'lua',
    '.sh': 'bash', '.bash': 'bash',
    '.yaml': 'yaml', '.yml': 'yaml',
    '.toml': 'toml',
    '.json': 'json',
    '.md': 'markdown', '.mdx': 'markdown',
    '.html': 'html', '.htm': 'html',
    '.css': 'css', '.scss': 'scss',
    '.sql': 'sql',
    '.r': 'r', '.R': 'r',
}

MAX_MANIFEST_FILES = 10_000  # Warn and truncate above this


@dataclass
class FileEntry:
    """Single file metadata record. No content loaded."""
    path: str                     # Relative path from project root
    size_bytes: int
    language: Optional[str]       # Detected from extension (None if unknown)
    last_modified: datetime       # UTC
    is_binary: bool               # True for binary files (no text content)

    def __repr__(self) -> str:
        lang = f", {self.language}" if self.language else ""
        binary = ", binary" if self.is_binary else ""
        return f"FileEntry({self.path}, {self.size_bytes}B{lang}{binary})"


@dataclass
class FileManifest:
    """Precomputed listing of all project files. Built once at session start."""
    files: list[FileEntry] = field(default_factory=list)
    root_path: Optional[str] = None   # Absolute project root path
    truncated: bool = False            # True if >MAX_MANIFEST_FILES files found

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_size_bytes(self) -> int:
        return sum(f.size_bytes for f in self.files)

    def summary(self) -> str:
        """Short summary for LLM prompt context."""
        langs: dict[str, int] = {}
        for f in self.files:
            if f.language:
                langs[f.language] = langs.get(f.language, 0) + 1
        top_langs = sorted(langs.items(), key=lambda x: -x[1])[:5]
        lang_str = ", ".join(f"{l}:{n}" for l, n in top_langs)
        size_kb = self.total_size_bytes // 1024
        trunc = " [TRUNCATED]" if self.truncated else ""
        return f"{self.file_count} files, {size_kb}KB{trunc}. Languages: {lang_str}"

    def __repr__(self) -> str:
        return f"FileManifest({self.file_count} files, root={self.root_path})"


def build_manifest(project_path: Path) -> FileManifest:
    """Build a FileManifest by walking the filesystem.

    Args:
        project_path: Absolute path to project root directory.

    Returns:
        FileManifest with all non-skipped files catalogued.

    Notes:
        - Skips SKIP_DIRS directories
        - Marks BINARY_EXTENSIONS files as is_binary=True
        - Truncates at MAX_MANIFEST_FILES with a log warning
        - Catches and logs errors for individual files (keeps going)
    """
    manifest = FileManifest(root_path=str(project_path))

    for root, dirs, files in os.walk(project_path):
        # Prune skipped directories in-place (modifies dirs for os.walk)
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]

        for filename in sorted(files):
            if len(manifest.files) >= MAX_MANIFEST_FILES:
                manifest.truncated = True
                logger.warning(
                    "FileManifest truncated at %d files for %s",
                    MAX_MANIFEST_FILES, project_path
                )
                return manifest

            filepath = Path(root) / filename
            try:
                stat = filepath.stat()
                rel_path = str(filepath.relative_to(project_path))
                suffix = filepath.suffix.lower()

                entry = FileEntry(
                    path=rel_path,
                    size_bytes=stat.st_size,
                    language=EXTENSION_TO_LANGUAGE.get(suffix),
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    is_binary=(suffix in BINARY_EXTENSIONS),
                )
                manifest.files.append(entry)
            except (OSError, ValueError) as e:
                logger.debug("Skipping %s: %s", filepath, e)

    return manifest


@dataclass
class GrepMatch:
    """A regex match within a file."""
    path: str                                           # File path (relative to project root)
    line_number: int                                    # 1-indexed
    line_content: str                                   # The matching line
    context_before: list[str] = field(default_factory=list)  # Up to 2 lines before
    context_after: list[str] = field(default_factory=list)   # Up to 2 lines after


@dataclass
class SearchMatch:
    """A search result from BM25 or semantic search."""
    path: str                       # File path
    line_number: Optional[int]
    snippet: str                    # Relevant excerpt
    score: float = 0.0
    language: Optional[str] = None


@dataclass
class SymbolInfo:
    """A symbol extracted from source code."""
    name: str
    kind: str                       # "class" | "function" | "method"
    line_number: int                # 1-indexed
    end_line: int                   # 1-indexed end line
    signature: str                  # Full signature with type hints
    qualified_name: str             # e.g., "MyClass.my_method"
    docstring: Optional[str] = None
    parent_class: Optional[str] = None


# ---------------------------------------------------------------------------
# T012 — TextHandle
# ---------------------------------------------------------------------------

@dataclass
class TextHandle:
    """Lazy reference to a text resource — file, thread, or note.

    Content is not loaded until .read() is called.  Thread and note
    resource types are reserved for Phase 5 (T022-T023).
    """

    path: str               # Relative path (files) or ID (threads/notes)
    size_bytes: int         # 0 if unknown
    language: Optional[str]
    resource_type: str = "file"   # "file" | "thread" | "note" | "chunk"
    _project_root: Optional[str] = field(default=None, repr=False)
    _chunk_start: Optional[int] = field(default=None, repr=False)
    _chunk_end: Optional[int] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _full_path(self) -> Path:
        if self._project_root:
            return Path(self._project_root) / self.path
        return Path(self.path)

    def _is_binary_file(self) -> bool:
        suffix = Path(self.path).suffix.lower()
        return suffix in BINARY_EXTENSIONS

    def _read_raw_lines(self) -> list[str]:
        """Read file content as a list of lines (no slicing)."""
        return self._full_path().read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self, start_line: Optional[int] = None, end_line: Optional[int] = None) -> "str | dict":
        """Return file content as a string, or an error dict for binary/large files.

        Args:
            start_line: 1-indexed start (inclusive).  None means from the top.
            end_line:   1-indexed end (inclusive).    None means to the bottom.

        Returns:
            str on success; dict with "notice" key on failure.
        """
        if self.resource_type not in ("file", "chunk"):
            return {"notice": f"read() not supported for resource_type={self.resource_type!r}"}

        if self._is_binary_file():
            return {"notice": "file too large or binary", "size_bytes": self.size_bytes, "path": self.path}

        if self.size_bytes > 1_048_576:
            return {"notice": "file too large or binary", "size_bytes": self.size_bytes, "path": self.path}

        try:
            lines = self._read_raw_lines()
        except Exception as exc:
            logger.debug("TextHandle.read() failed for %s: %s", self.path, exc)
            return {"notice": f"could not read file: {exc}", "path": self.path}

        # Determine effective slice bounds (1-indexed → 0-indexed)
        if self._chunk_start is not None:
            # This handle represents a sub-chunk — honour chunk bounds
            lo = (self._chunk_start - 1)
            hi = self._chunk_end  # _chunk_end is 1-indexed inclusive → slice [lo:hi]
        else:
            lo = 0
            hi = len(lines)

        # Apply caller-supplied overrides (within chunk bounds)
        if start_line is not None:
            lo = max(lo, start_line - 1)
        if end_line is not None:
            hi = min(hi, end_line)

        return "".join(lines[lo:hi])

    def symbols(self) -> "list[SymbolInfo]":
        """Extract symbol definitions from the file.

        Tries vlt CodeRAG repomap first; falls back to regex for Python.
        Returns [] on any error or for unsupported languages.
        """
        SUPPORTED = {"python", "typescript", "javascript", "go", "rust"}
        if self.resource_type not in ("file",) or self.language not in SUPPORTED:
            return []

        # --- Try vlt CodeRAG integration ----------------------------------
        try:
            from vlt.core.coderag.repomap import extract_symbols  # type: ignore
            full = str(self._full_path())
            return extract_symbols(full)
        except ImportError:
            pass  # vlt-cli not on path — use fallback
        except Exception as exc:
            logger.debug("extract_symbols() failed for %s: %s", self.path, exc)

        # --- Python regex fallback ----------------------------------------
        if self.language != "python":
            return []

        import re
        results: list[SymbolInfo] = []
        try:
            lines = self._read_raw_lines()
        except Exception:
            return []

        pattern = re.compile(r'^(class|def)\s+(\w+)')
        for idx, line in enumerate(lines):
            m = pattern.match(line)
            if m:
                kind = "class" if m.group(1) == "class" else "function"
                name = m.group(2)
                results.append(SymbolInfo(
                    name=name,
                    kind=kind,
                    line_number=idx + 1,
                    end_line=idx + 1,
                    signature=line.rstrip(),
                    qualified_name=name,
                ))
        return results

    def grep(self, pattern: str) -> "list[GrepMatch]":
        """Search file content for a regex pattern.

        Returns one GrepMatch per matching line with up to 2 lines of context.
        Returns [] on any error.
        """
        import re
        try:
            lines = self._read_raw_lines()
        except Exception as exc:
            logger.debug("TextHandle.grep() read failed for %s: %s", self.path, exc)
            return []

        results: list[GrepMatch] = []
        try:
            rx = re.compile(pattern)
        except re.error as exc:
            logger.debug("TextHandle.grep() bad pattern %r: %s", pattern, exc)
            return []

        for idx, line in enumerate(lines):
            if rx.search(line):
                before = [l.rstrip('\n') for l in lines[max(0, idx - 2):idx]]
                after = [l.rstrip('\n') for l in lines[idx + 1:idx + 3]]
                results.append(GrepMatch(
                    path=self.path,
                    line_number=idx + 1,
                    line_content=line.rstrip('\n'),
                    context_before=before,
                    context_after=after,
                ))
        return results

    def chunks(self, max_lines: int = 200) -> "list[TextHandle]":
        """Split file into sub-handles.

        Tries to split at symbol (function/class) boundaries first.
        Falls back to fixed-line-count chunks when no symbols are found.

        Returns a list of TextHandle objects with resource_type="chunk".
        """
        try:
            lines = self._read_raw_lines()
        except Exception as exc:
            logger.debug("TextHandle.chunks() read failed for %s: %s", self.path, exc)
            return []

        total = len(lines)
        if total == 0:
            return []

        # Try symbol-boundary splitting
        syms = self.symbols()
        boundaries: list[int] = []
        if syms:
            # Collect symbol start lines that are far enough apart
            for sym in sorted(syms, key=lambda s: s.line_number):
                if not boundaries or (sym.line_number - boundaries[-1]) >= max_lines // 2:
                    boundaries.append(sym.line_number)

        def _make_chunk(start: int, end: int) -> TextHandle:
            return TextHandle(
                path=self.path,
                size_bytes=sum(len(l.encode()) for l in lines[start - 1:end]),
                language=self.language,
                resource_type="chunk",
                _project_root=self._project_root,
                _chunk_start=start,
                _chunk_end=end,
            )

        if boundaries:
            result: list[TextHandle] = []
            starts = boundaries + [total + 1]
            # Chunk [1 .. first_boundary - 1] if there's content before first symbol
            if boundaries[0] > 1:
                result.append(_make_chunk(1, boundaries[0] - 1))
            for i, sym_start in enumerate(boundaries):
                chunk_end = starts[i + 1] - 1
                result.append(_make_chunk(sym_start, chunk_end))
            return result

        # Fallback: fixed line-count chunks
        result = []
        start = 1
        while start <= total:
            end = min(start + max_lines - 1, total)
            result.append(_make_chunk(start, end))
            start = end + 1
        return result

    def __repr__(self) -> str:
        if self.resource_type == "chunk":
            content_desc = f"lines {self._chunk_start}-{self._chunk_end}"
        else:
            content_desc = f"{self.size_bytes}B"
        lang = f", {self.language}" if self.language else ""
        return f"TextHandle({self.path}, {content_desc}{lang})"


# ---------------------------------------------------------------------------
# T013 — ProjectContext
# ---------------------------------------------------------------------------

class ProjectContext:
    """Root REPL object — exposes the full project to LLM code.

    Injected as ``project`` in the restricted RLM namespace.
    All methods are defensive — exceptions are caught and logged rather
    than propagated.
    """

    def __init__(self, project_id: str, user_id: str, project_path: str) -> None:
        self.project_id = project_id
        self.user_id = user_id
        self.project_path = Path(project_path)
        self._manifest: Optional[FileManifest] = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def get_manifest(self) -> FileManifest:
        """Lazy-load and cache the file manifest for this project."""
        if self._manifest is None:
            self._manifest = build_manifest(self.project_path)
        return self._manifest

    # ------------------------------------------------------------------
    # File access
    # ------------------------------------------------------------------

    def file(self, path: str) -> TextHandle:
        """Return a lazy TextHandle for *path* (relative to project root).

        Does not raise even if the file doesn't exist — content errors
        surface only when .read() is called.
        """
        manifest = self.get_manifest()
        # Quick lookup by path
        entry_map = {e.path: e for e in manifest.files}
        entry = entry_map.get(path)
        if entry:
            return TextHandle(
                path=path,
                size_bytes=entry.size_bytes,
                language=entry.language,
                resource_type="file",
                _project_root=str(self.project_path),
            )
        # Not in manifest — return handle anyway (lazy)
        suffix = Path(path).suffix.lower()
        return TextHandle(
            path=path,
            size_bytes=0,
            language=EXTENSION_TO_LANGUAGE.get(suffix),
            resource_type="file",
            _project_root=str(self.project_path),
        )

    def files(self, pattern: str = "**/*") -> "list[TextHandle]":
        """Return TextHandles for all files matching *pattern*.

        Uses pathlib.Path.glob — pattern is relative to the project root.
        Skips directories in SKIP_DIRS and binary files.  Truncates at 500
        results with a warning.
        """
        MAX_RESULTS = 500
        results: list[TextHandle] = []

        try:
            manifest = self.get_manifest()
            entry_map = {e.path: e for e in manifest.files}

            for abs_path in self.project_path.glob(pattern):
                if not abs_path.is_file():
                    continue
                # Skip anything inside SKIP_DIRS
                parts = abs_path.relative_to(self.project_path).parts
                if any(part in SKIP_DIRS for part in parts):
                    continue

                rel = str(abs_path.relative_to(self.project_path))
                entry = entry_map.get(rel)
                if entry:
                    handle = TextHandle(
                        path=rel,
                        size_bytes=entry.size_bytes,
                        language=entry.language,
                        resource_type="file",
                        _project_root=str(self.project_path),
                    )
                else:
                    suffix = abs_path.suffix.lower()
                    handle = TextHandle(
                        path=rel,
                        size_bytes=0,
                        language=EXTENSION_TO_LANGUAGE.get(suffix),
                        resource_type="file",
                        _project_root=str(self.project_path),
                    )
                results.append(handle)
                if len(results) >= MAX_RESULTS:
                    logger.warning(
                        "ProjectContext.files() truncated at %d results (pattern=%r, project=%s)",
                        MAX_RESULTS, pattern, self.project_id,
                    )
                    break
        except Exception as exc:
            logger.debug("ProjectContext.files() failed: %s", exc)

        return results

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> "list[SearchMatch]":
        """Search the project for *query*.

        Tries CodeRAG first; falls back to grep on the full manifest.
        """
        # --- Try CodeRAG -------------------------------------------------
        try:
            from vlt.core.service import VltService  # type: ignore
            svc = VltService()
            raw = svc.code_search(query, project_id=self.project_id, limit=limit)
            if raw:
                results: list[SearchMatch] = []
                for item in raw:
                    results.append(SearchMatch(
                        path=item.get("file_path", ""),
                        line_number=item.get("lineno"),
                        snippet=item.get("snippet", ""),
                        score=float(item.get("score", 0.0)),
                        language=None,
                    ))
                return results
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("ProjectContext.search() CodeRAG failed: %s", exc)

        # --- Fallback: grep ----------------------------------------------
        return self._grep_search(query, limit)

    def _grep_search(self, query: str, limit: int) -> "list[SearchMatch]":
        """Convert grep results to SearchMatch objects."""
        matches = self.grep(query)
        results: list[SearchMatch] = []
        for m in matches[:limit]:
            results.append(SearchMatch(
                path=m.path,
                line_number=m.line_number,
                snippet=m.line_content,
                score=0.0,
                language=EXTENSION_TO_LANGUAGE.get(Path(m.path).suffix.lower()),
            ))
        return results

    def grep(self, pattern: str) -> "list[GrepMatch]":
        """Regex search across all non-binary text files in the manifest.

        Returns up to 500 matches total, each with up to 2 lines of
        before/after context.
        """
        import re
        MAX_TOTAL = 500
        results: list[GrepMatch] = []

        try:
            rx = re.compile(pattern)
        except re.error as exc:
            logger.debug("ProjectContext.grep() bad pattern %r: %s", pattern, exc)
            return []

        manifest = self.get_manifest()
        for entry in manifest.files:
            if entry.is_binary:
                continue
            if len(results) >= MAX_TOTAL:
                break

            try:
                full = self.project_path / entry.path
                lines = full.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                for idx, line in enumerate(lines):
                    if rx.search(line):
                        before = [l.rstrip('\n') for l in lines[max(0, idx - 2):idx]]
                        after = [l.rstrip('\n') for l in lines[idx + 1:idx + 3]]
                        results.append(GrepMatch(
                            path=entry.path,
                            line_number=idx + 1,
                            line_content=line.rstrip('\n'),
                            context_before=before,
                            context_after=after,
                        ))
                        if len(results) >= MAX_TOTAL:
                            break
            except Exception as exc:
                logger.debug("ProjectContext.grep() skip %s: %s", entry.path, exc)

        return results

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = self._manifest.file_count if self._manifest else "?"
        return f"ProjectContext(project_id={self.project_id!r}, files={n})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_project_context(project_id: str, user_id: str) -> ProjectContext:
    """Build a ProjectContext for *project_id*.

    Looks up the project path from the vlt project database.
    Falls back to the current working directory when the project is not
    found or vlt is unavailable.
    """
    try:
        from vlt.core.service import VltService  # type: ignore
        svc = VltService()
        project = svc.get_project(project_id)
        if project and getattr(project, "path", None):
            return ProjectContext(project_id, user_id, project.path)
    except Exception:
        logger.debug("Could not load project from vlt DB, falling back to cwd")

    return ProjectContext(project_id, user_id, os.getcwd())
