"""Unit tests for ProjectContext and TextHandle — T017 (022-rlm-oracle)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from backend.src.services.project_context import (
    BINARY_EXTENSIONS,
    SKIP_DIRS,
    FileEntry,
    FileManifest,
    GrepMatch,
    ProjectContext,
    SearchMatch,
    SymbolInfo,
    TextHandle,
    build_manifest,
    build_project_context,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal temp project tree for testing."""
    (tmp_path / "main.py").write_text("def hello():\n    print('hi')\n\nx = 1\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "README.md").write_text("# Project\n\nThis is a test project.\n")
    (tmp_path / "data.json").write_text('{"key": "value"}\n')
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "module.py").write_text("class Foo:\n    pass\n")
    return tmp_path


@pytest.fixture
def ctx(tmp_project: Path) -> ProjectContext:
    return ProjectContext("test-proj", "user1", str(tmp_project))


# ---------------------------------------------------------------------------
# build_manifest
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_manifest_lists_files(self, tmp_project: Path):
        m = build_manifest(tmp_project)
        paths = {f.path for f in m.files}
        assert "main.py" in paths
        assert "utils.py" in paths
        assert "README.md" in paths

    def test_manifest_skips_skip_dirs(self, tmp_project: Path):
        node = tmp_project / "node_modules"
        node.mkdir()
        (node / "pkg.js").write_text("module.exports = {}")
        m = build_manifest(tmp_project)
        paths = {f.path for f in m.files}
        assert not any("node_modules" in p for p in paths)

    def test_manifest_detects_python_language(self, tmp_project: Path):
        m = build_manifest(tmp_project)
        py_entries = [f for f in m.files if f.path == "main.py"]
        assert py_entries, "main.py not found in manifest"
        assert py_entries[0].language == "python"

    def test_manifest_summary_has_file_count(self, tmp_project: Path):
        m = build_manifest(tmp_project)
        s = m.summary()
        # Should mention file count and python language
        assert str(m.file_count) in s
        assert "python" in s


# ---------------------------------------------------------------------------
# ProjectContext
# ---------------------------------------------------------------------------

class TestProjectContextManifest:
    def test_get_manifest_returns_manifest(self, ctx: ProjectContext):
        m = ctx.get_manifest()
        assert isinstance(m, FileManifest)
        assert m.file_count > 0

    def test_manifest_is_cached(self, ctx: ProjectContext):
        m1 = ctx.get_manifest()
        m2 = ctx.get_manifest()
        assert m1 is m2

    def test_repr(self, ctx: ProjectContext):
        r = repr(ctx)
        assert "test-proj" in r


class TestProjectContextFile:
    def test_file_returns_text_handle(self, ctx: ProjectContext):
        h = ctx.file("main.py")
        assert isinstance(h, TextHandle)
        assert h.path == "main.py"
        assert h.language == "python"

    def test_file_non_manifest_path_still_returns_handle(self, ctx: ProjectContext):
        h = ctx.file("does_not_exist.py")
        assert isinstance(h, TextHandle)
        assert h.path == "does_not_exist.py"
        assert h.language == "python"  # Detected from extension

    def test_file_size_bytes_populated(self, ctx: ProjectContext):
        h = ctx.file("main.py")
        assert h.size_bytes > 0


class TestProjectContextFiles:
    def test_files_returns_list(self, ctx: ProjectContext):
        handles = ctx.files("**/*.py")
        assert len(handles) > 0
        assert all(isinstance(h, TextHandle) for h in handles)

    def test_files_filters_by_pattern(self, ctx: ProjectContext):
        py_handles = ctx.files("**/*.py")
        md_handles = ctx.files("**/*.md")
        py_paths = {h.path for h in py_handles}
        md_paths = {h.path for h in md_handles}
        assert all(p.endswith(".py") for p in py_paths)
        assert all(p.endswith(".md") for p in md_paths)
        # No overlap
        assert py_paths.isdisjoint(md_paths)

    def test_files_all_returns_multiple(self, ctx: ProjectContext):
        all_handles = ctx.files("**/*")
        assert len(all_handles) >= 4  # main.py, utils.py, README.md, data.json, subdir/module.py


class TestProjectContextGrep:
    def test_grep_finds_matches(self, ctx: ProjectContext):
        matches = ctx.grep("def ")
        assert len(matches) > 0
        assert all(isinstance(m, GrepMatch) for m in matches)

    def test_grep_returns_correct_paths(self, ctx: ProjectContext):
        matches = ctx.grep("def hello")
        paths = {m.path for m in matches}
        assert "main.py" in paths

    def test_grep_empty_on_no_match(self, ctx: ProjectContext):
        matches = ctx.grep("THIS_PATTERN_DOES_NOT_EXIST_XYZ_UNIQUE")
        assert matches == []

    def test_grep_includes_context_lines(self, ctx: ProjectContext):
        matches = ctx.grep("def hello")
        assert len(matches) > 0
        m = matches[0]
        # context_before and context_after are lists
        assert isinstance(m.context_before, list)
        assert isinstance(m.context_after, list)


# ---------------------------------------------------------------------------
# TextHandle
# ---------------------------------------------------------------------------

class TestTextHandleRead:
    def test_read_returns_content(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        content = h.read()
        assert isinstance(content, str)
        assert "def hello" in content

    def test_read_with_line_range(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        content = h.read(start_line=1, end_line=1)
        assert "def hello" in content

    def test_read_binary_returns_notice_dict(self, tmp_project: Path):
        png = tmp_project / "image.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        h = TextHandle(
            path="image.png",
            size_bytes=8,
            language=None,
            _project_root=str(tmp_project),
        )
        result = h.read()
        assert isinstance(result, dict)
        assert "notice" in result

    def test_read_oversized_returns_notice_dict(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=2_000_000,  # > 1 MB threshold
            language="python",
            _project_root=str(tmp_project),
        )
        result = h.read()
        assert isinstance(result, dict)
        assert "notice" in result

    def test_read_missing_file_returns_notice_dict(self, tmp_project: Path):
        h = TextHandle(
            path="does_not_exist.py",
            size_bytes=0,
            language="python",
            _project_root=str(tmp_project),
        )
        result = h.read()
        assert isinstance(result, dict)
        assert "notice" in result


class TestTextHandleSymbols:
    def test_symbols_returns_list_for_python(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        syms = h.symbols()
        assert isinstance(syms, list)
        # Should find at least `hello` function
        names = [s.name for s in syms]
        assert "hello" in names

    def test_symbols_returns_empty_for_unsupported_language(self, tmp_project: Path):
        h = TextHandle(
            path="README.md",
            size_bytes=100,
            language="markdown",
            _project_root=str(tmp_project),
        )
        assert h.symbols() == []

    def test_symbol_has_required_fields(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        syms = h.symbols()
        if syms:
            s = syms[0]
            assert hasattr(s, "name")
            assert hasattr(s, "kind")
            assert hasattr(s, "line_number")


class TestTextHandleGrep:
    def test_grep_finds_pattern_in_file(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        matches = h.grep("def ")
        assert len(matches) > 0
        assert all("def " in m.line_content for m in matches)

    def test_grep_returns_empty_on_no_match(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        assert h.grep("ZZZNOTHERE") == []


class TestTextHandleChunks:
    def test_chunks_returns_list(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        chunks = h.chunks(max_lines=2)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1

    def test_chunks_are_text_handles(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        chunks = h.chunks(max_lines=2)
        assert all(isinstance(c, TextHandle) for c in chunks)

    def test_chunks_have_chunk_resource_type(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        chunks = h.chunks(max_lines=2)
        if chunks:
            assert all(c.resource_type == "chunk" for c in chunks)

    def test_chunk_read_returns_content(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        chunks = h.chunks(max_lines=200)
        if chunks:
            content = chunks[0].read()
            assert isinstance(content, str)
            assert len(content) > 0


class TestTextHandleRepr:
    def test_repr_includes_path(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            _project_root=str(tmp_project),
        )
        r = repr(h)
        assert "main.py" in r

    def test_chunk_repr_shows_lines(self, tmp_project: Path):
        h = TextHandle(
            path="main.py",
            size_bytes=100,
            language="python",
            resource_type="chunk",
            _project_root=str(tmp_project),
            _chunk_start=1,
            _chunk_end=5,
        )
        r = repr(h)
        # Should mention line range
        assert "1" in r and "5" in r


# ---------------------------------------------------------------------------
# build_project_context factory
# ---------------------------------------------------------------------------

class TestBuildProjectContext:
    def test_falls_back_to_cwd(self):
        # vlt DB lookup will fail in tests — should fall back to cwd
        ctx = build_project_context("nonexistent-project", "user1")
        assert isinstance(ctx, ProjectContext)
        assert ctx.project_id == "nonexistent-project"
