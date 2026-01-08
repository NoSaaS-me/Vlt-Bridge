"""
Unit tests for TreeRegistry.

Tests the TreeRegistry class from lua/registry.py:
- Tree loading and registration
- Tree listing and retrieval
- Hot reload with file watching
- Reload policies (LET_FINISH_THEN_SWAP, CANCEL_AND_RESTART)
- Debounced file change handling
- Security (path traversal)

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.8.1-2.8.8 from tasks.md
"""

import pytest
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

from backend.src.bt.lua.registry import (
    ReloadPolicy,
    TreeRegistry,
    TreeLoadError,
    TreeValidationError,
    TreeInUseError,
    SecurityError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_tree_dir():
    """Create a temporary directory for tree files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_tree_dir: Path) -> TreeRegistry:
    """Create a test registry with temp directory."""
    return TreeRegistry(
        tree_dir=temp_tree_dir,
        default_reload_policy=ReloadPolicy.LET_FINISH_THEN_SWAP,
        debounce_ms=100,
        validate_on_load=False,
    )


@pytest.fixture
def mock_tree():
    """Create a mock BehaviorTree."""
    from backend.src.bt.core.tree import TreeStatus

    tree = MagicMock()
    tree.id = "test-tree"
    tree.name = "test-tree"
    tree.status = TreeStatus.IDLE
    tree.node_count = 5
    tree.source_hash = "abc123"
    return tree


def create_lua_file(tree_dir: Path, name: str, content: str = "") -> Path:
    """Helper to create a .lua file in the tree directory."""
    path = tree_dir / f"{name}.lua"
    path.write_text(content or f"-- {name} tree\n")
    return path


# =============================================================================
# ReloadPolicy Tests
# =============================================================================


class TestReloadPolicy:
    """Tests for ReloadPolicy enum."""

    def test_let_finish_then_swap(self) -> None:
        """Test LET_FINISH_THEN_SWAP policy value."""
        assert ReloadPolicy.LET_FINISH_THEN_SWAP == "let_finish_then_swap"

    def test_cancel_and_restart(self) -> None:
        """Test CANCEL_AND_RESTART policy value."""
        assert ReloadPolicy.CANCEL_AND_RESTART == "cancel_and_restart"


# =============================================================================
# Basic Registry Tests
# =============================================================================


class TestRegistryInit:
    """Tests for registry initialization."""

    def test_init_with_valid_dir(self, temp_tree_dir: Path) -> None:
        """Test registry initialization with valid directory."""
        registry = TreeRegistry(tree_dir=temp_tree_dir)

        assert registry.tree_dir == temp_tree_dir
        assert registry._default_reload_policy == ReloadPolicy.LET_FINISH_THEN_SWAP
        assert registry._debounce_ms == 500
        assert registry._validate_on_load is True

    def test_init_with_custom_options(self, temp_tree_dir: Path) -> None:
        """Test registry initialization with custom options."""
        registry = TreeRegistry(
            tree_dir=temp_tree_dir,
            default_reload_policy=ReloadPolicy.CANCEL_AND_RESTART,
            debounce_ms=1000,
            validate_on_load=False,
        )

        assert registry._default_reload_policy == ReloadPolicy.CANCEL_AND_RESTART
        assert registry._debounce_ms == 1000
        assert registry._validate_on_load is False

    def test_init_with_nonexistent_dir(self) -> None:
        """Test registry initialization with non-existent directory."""
        with pytest.raises(FileNotFoundError):
            TreeRegistry(tree_dir=Path("/nonexistent/directory"))

    def test_init_with_file_instead_of_dir(self, temp_tree_dir: Path) -> None:
        """Test registry initialization with a file instead of directory."""
        file_path = temp_tree_dir / "not_a_dir.txt"
        file_path.write_text("test")

        with pytest.raises(NotADirectoryError):
            TreeRegistry(tree_dir=file_path)


# =============================================================================
# Tree Loading Tests
# =============================================================================


class TestTreeLoading:
    """Tests for tree loading."""

    def test_load_creates_tree(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test that load creates a tree in the registry."""
        lua_file = create_lua_file(temp_tree_dir, "my-tree")

        # Mock the definition loading to return a valid definition
        with patch.object(registry, "_load_definition") as mock_load:
            from backend.src.bt.lua.definitions import NodeDefinition, TreeDefinition

            mock_load.return_value = TreeDefinition(
                name="my-tree",
                root=NodeDefinition(
                    type="sequence",
                    id="root",
                    children=[
                        NodeDefinition(type="action", id="action", config={"fn": "os.getcwd"}),
                    ],
                ),
                source_path=str(lua_file),
            )

            tree = registry.load(lua_file)

        assert tree is not None
        assert "my-tree" in registry.list_trees()

    def test_load_nonexistent_file(self, registry: TreeRegistry) -> None:
        """Test loading a non-existent file raises TreeLoadError."""
        with pytest.raises(TreeLoadError) as exc_info:
            registry.load(Path("/nonexistent/file.lua"))

        assert exc_info.value.code == "E4001"

    def test_load_wrong_extension(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test loading a non-.lua file raises TreeLoadError."""
        txt_file = temp_tree_dir / "test.txt"
        txt_file.write_text("not lua")

        with pytest.raises(TreeLoadError) as exc_info:
            registry.load(txt_file)

        assert exc_info.value.code == "E4001"
        assert ".lua" in str(exc_info.value)

    def test_load_path_traversal_blocked(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test that path traversal is blocked (SecurityError)."""
        # Create a file outside tree_dir
        with tempfile.NamedTemporaryFile(suffix=".lua", delete=False) as f:
            outside_file = Path(f.name)

        try:
            with pytest.raises(SecurityError) as exc_info:
                registry.load(outside_file)

            assert exc_info.value.code == "E7002"
        finally:
            outside_file.unlink()


# =============================================================================
# Tree Retrieval Tests
# =============================================================================


class TestTreeRetrieval:
    """Tests for tree retrieval."""

    def test_get_existing_tree(self, registry: TreeRegistry, mock_tree) -> None:
        """Test getting an existing tree."""
        registry._trees["test-tree"] = mock_tree

        tree = registry.get("test-tree")

        assert tree is mock_tree

    def test_get_nonexistent_tree(self, registry: TreeRegistry) -> None:
        """Test getting a non-existent tree returns None."""
        tree = registry.get("nonexistent")

        assert tree is None

    def test_list_trees(self, registry: TreeRegistry, mock_tree) -> None:
        """Test listing all tree IDs."""
        registry._trees["tree-a"] = mock_tree
        registry._trees["tree-b"] = mock_tree

        tree_ids = registry.list_trees()

        assert set(tree_ids) == {"tree-a", "tree-b"}

    def test_list_trees_empty(self, registry: TreeRegistry) -> None:
        """Test listing trees when registry is empty."""
        tree_ids = registry.list_trees()

        assert tree_ids == []


# =============================================================================
# Tree Unload Tests
# =============================================================================


class TestTreeUnload:
    """Tests for tree unloading."""

    def test_unload_idle_tree(self, registry: TreeRegistry, mock_tree) -> None:
        """Test unloading an idle tree."""
        from backend.src.bt.core.tree import TreeStatus

        mock_tree.status = TreeStatus.IDLE
        registry._trees["test-tree"] = mock_tree

        registry.unload("test-tree")

        assert "test-tree" not in registry._trees

    def test_unload_running_tree_raises(self, registry: TreeRegistry) -> None:
        """Test unloading a running tree raises TreeInUseError."""
        from backend.src.bt.core.tree import TreeStatus

        mock_tree = MagicMock()
        mock_tree.status = TreeStatus.RUNNING
        registry._trees["test-tree"] = mock_tree

        with pytest.raises(TreeInUseError) as exc_info:
            registry.unload("test-tree")

        assert "test-tree" in str(exc_info.value)

    def test_unload_nonexistent_noop(self, registry: TreeRegistry) -> None:
        """Test unloading a non-existent tree is a no-op."""
        registry.unload("nonexistent")  # Should not raise


# =============================================================================
# Hot Reload Tests
# =============================================================================


class TestHotReload:
    """Tests for hot reload functionality."""

    def test_reload_idle_tree(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test reloading an idle tree."""
        from backend.src.bt.core.tree import TreeStatus

        mock_tree = MagicMock()
        mock_tree.status = TreeStatus.IDLE
        mock_tree.id = "test-tree"
        mock_tree.node_count = 3
        mock_tree.source_hash = "old"

        lua_file = create_lua_file(temp_tree_dir, "test-tree")

        registry._trees["test-tree"] = mock_tree
        registry._source_paths["test-tree"] = lua_file

        # Mock apply_reload
        with patch.object(registry, "_apply_reload") as mock_apply:
            registry.reload("test-tree")

            mock_apply.assert_called_once_with("test-tree", lua_file)

    def test_reload_running_tree_let_finish(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test reloading a running tree with LET_FINISH_THEN_SWAP policy."""
        from backend.src.bt.core.tree import TreeStatus

        mock_tree = MagicMock()
        mock_tree.status = TreeStatus.RUNNING
        mock_tree.id = "test-tree"

        lua_file = create_lua_file(temp_tree_dir, "test-tree")

        registry._trees["test-tree"] = mock_tree
        registry._source_paths["test-tree"] = lua_file

        registry.reload("test-tree", policy=ReloadPolicy.LET_FINISH_THEN_SWAP)

        # Should queue reload
        assert "test-tree" in registry._reload_queue
        mock_tree.queue_reload.assert_called_once()

    def test_reload_running_tree_cancel(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test reloading a running tree with CANCEL_AND_RESTART policy."""
        from backend.src.bt.core.tree import TreeStatus

        mock_tree = MagicMock()
        mock_tree.status = TreeStatus.RUNNING
        mock_tree.id = "test-tree"

        lua_file = create_lua_file(temp_tree_dir, "test-tree")

        registry._trees["test-tree"] = mock_tree
        registry._source_paths["test-tree"] = lua_file

        with patch.object(registry, "_apply_reload") as mock_apply:
            registry.reload("test-tree", policy=ReloadPolicy.CANCEL_AND_RESTART)

            # Should cancel tree and apply reload
            mock_tree.cancel.assert_called_once()
            mock_apply.assert_called_once()

    def test_check_pending_reloads(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test checking pending reloads for completed trees."""
        from backend.src.bt.core.tree import TreeStatus
        from datetime import datetime, timezone

        mock_tree = MagicMock()
        mock_tree.status = TreeStatus.IDLE  # No longer running
        mock_tree.id = "test-tree"

        lua_file = create_lua_file(temp_tree_dir, "test-tree")

        registry._trees["test-tree"] = mock_tree
        registry._source_paths["test-tree"] = lua_file

        # Add pending reload
        from backend.src.bt.lua.registry import PendingReload

        registry._reload_queue["test-tree"] = PendingReload(
            tree_id="test-tree",
            source_path=lua_file,
            queued_at=datetime.now(timezone.utc),
        )

        with patch.object(registry, "_apply_reload") as mock_apply:
            registry.check_pending_reloads()

            mock_apply.assert_called_once_with("test-tree", lua_file)


# =============================================================================
# File Watching Tests
# =============================================================================


class TestFileWatching:
    """Tests for file watching functionality."""

    def test_start_watching(self, registry: TreeRegistry) -> None:
        """Test starting file watcher."""
        with patch("watchdog.observers.Observer") as MockObserver:
            mock_observer = MagicMock()
            MockObserver.return_value = mock_observer

            registry.start_watching()

            assert registry.is_watching
            mock_observer.start.assert_called_once()

    def test_stop_watching(self, registry: TreeRegistry) -> None:
        """Test stopping file watcher."""
        mock_observer = MagicMock()
        registry._observer = mock_observer
        registry._watching = True

        registry.stop_watching()

        assert not registry.is_watching
        mock_observer.stop.assert_called_once()

    def test_debounced_file_change(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test that file changes are debounced."""
        lua_file = create_lua_file(temp_tree_dir, "test-tree")

        with patch.object(registry, "_apply_file_change") as mock_apply:
            # Trigger multiple rapid changes
            registry.on_file_changed(lua_file)
            registry.on_file_changed(lua_file)
            registry.on_file_changed(lua_file)

            # Should have only one pending timer
            assert len(registry._pending_changes) == 1

            # Wait for debounce
            time.sleep(0.2)

            # Should have called apply once
            mock_apply.assert_called_once_with(lua_file.resolve())


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for registry callbacks."""

    def test_on_tree_loaded_callback(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test on_tree_loaded callback is called."""
        callback_called = []

        def on_loaded(tree_id, tree):
            callback_called.append((tree_id, tree))

        registry.on_tree_loaded(on_loaded)

        lua_file = create_lua_file(temp_tree_dir, "test-tree")

        with patch.object(registry, "_load_definition") as mock_load:
            from backend.src.bt.lua.definitions import NodeDefinition, TreeDefinition

            mock_load.return_value = TreeDefinition(
                name="test-tree",
                root=NodeDefinition(
                    type="sequence",
                    id="root",
                    children=[
                        NodeDefinition(type="action", id="action", config={"fn": "os.getcwd"}),
                    ],
                ),
                source_path=str(lua_file),
            )

            tree = registry.load(lua_file)

        assert len(callback_called) == 1
        assert callback_called[0][0] == "test-tree"

    def test_on_tree_unloaded_callback(self, registry: TreeRegistry, mock_tree) -> None:
        """Test on_tree_unloaded callback is called."""
        from backend.src.bt.core.tree import TreeStatus

        callback_called = []

        def on_unloaded(tree_id):
            callback_called.append(tree_id)

        registry.on_tree_unloaded(on_unloaded)

        mock_tree.status = TreeStatus.IDLE
        registry._trees["test-tree"] = mock_tree

        registry.unload("test-tree")

        assert callback_called == ["test-tree"]


# =============================================================================
# Security Tests
# =============================================================================


class TestSecurity:
    """Tests for security features."""

    def test_path_within_tree_dir(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test that paths within tree_dir are safe."""
        safe_path = temp_tree_dir / "subdir" / "tree.lua"
        assert registry._is_safe_path(safe_path)

    def test_path_outside_tree_dir(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test that paths outside tree_dir are blocked."""
        unsafe_path = temp_tree_dir / ".." / "outside.lua"
        assert not registry._is_safe_path(unsafe_path)

    def test_symlink_escape_blocked(self, registry: TreeRegistry, temp_tree_dir: Path) -> None:
        """Test that symlink escape is blocked."""
        # Create a symlink pointing outside
        with tempfile.NamedTemporaryFile(suffix=".lua", delete=False) as f:
            outside_file = Path(f.name)

        try:
            symlink_path = temp_tree_dir / "escape.lua"
            try:
                symlink_path.symlink_to(outside_file)

                # The resolved path should be outside tree_dir
                assert not registry._is_safe_path(symlink_path)
            except OSError:
                # Skip on systems where symlink creation fails
                pytest.skip("Cannot create symlink")
        finally:
            outside_file.unlink()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_reload_nonexistent_tree(self, registry: TreeRegistry) -> None:
        """Test reloading a non-existent tree raises TreeLoadError."""
        with pytest.raises(TreeLoadError) as exc_info:
            registry.reload("nonexistent")

        assert exc_info.value.code == "E3001"

    def test_reload_no_source_path(self, registry: TreeRegistry, mock_tree) -> None:
        """Test reloading tree without source path raises TreeLoadError."""
        from backend.src.bt.core.tree import TreeStatus

        mock_tree.status = TreeStatus.IDLE
        registry._trees["test-tree"] = mock_tree
        # No source path registered

        with pytest.raises(TreeLoadError) as exc_info:
            registry.reload("test-tree")

        assert exc_info.value.code == "E3005"


# =============================================================================
# TreeLoadError Tests
# =============================================================================


class TestTreeLoadErrorClass:
    """Tests for TreeLoadError class."""

    def test_error_message_format(self) -> None:
        """Test error message formatting."""
        error = TreeLoadError(
            code="E4001",
            message="File not found",
            path=Path("/path/to/file.lua"),
        )

        assert "[E4001]" in str(error)
        assert "File not found" in str(error)

    def test_error_attributes(self) -> None:
        """Test error attributes."""
        path = Path("/test/path.lua")
        error = TreeLoadError(
            code="E4001",
            message="Test error",
            path=path,
        )

        assert error.code == "E4001"
        assert error.message == "Test error"
        assert error.path == path


# =============================================================================
# Exports
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
