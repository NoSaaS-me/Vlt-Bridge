"""
TreeRegistry - Central registry for all loaded behavior trees with hot reload.

Handles:
- Loading trees from Lua files
- Validation before building
- Hot reload with file watching (watchdog)
- Reload policies (LET_FINISH_THEN_SWAP, CANCEL_AND_RESTART)
- Debounced file change handling

From tree-loader.yaml TreeRegistry interface.

Part of the BT Universal Runtime (spec 019).
Tasks covered: 2.8.1-2.8.8 from tasks.md
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

from .definitions import TreeDefinition, ValidationError
from .validator import TreeValidator
from .builder import TreeBuilder, TreeBuildError

if TYPE_CHECKING:
    from ..core.tree import BehaviorTree

logger = logging.getLogger(__name__)


# =============================================================================
# Reload Policy
# =============================================================================


class ReloadPolicy(str, Enum):
    """How to handle tree reload when tree is running.

    From tree-loader.yaml ReloadPolicy enum:
    - LET_FINISH_THEN_SWAP: Queue reload, apply after current execution completes
    - CANCEL_AND_RESTART: Cancel current execution, apply reload, restart

    Note: IMMEDIATE was removed per footgun audit (UB-08) as too dangerous.
    """

    LET_FINISH_THEN_SWAP = "let_finish_then_swap"
    CANCEL_AND_RESTART = "cancel_and_restart"


# =============================================================================
# Error Classes
# =============================================================================


class TreeLoadError(Exception):
    """Error loading a tree file."""

    def __init__(
        self,
        code: str,
        message: str,
        path: Optional[Path] = None,
    ) -> None:
        self.code = code
        self.message = message
        self.path = path

        full_msg = f"[{code}] {message}"
        if path:
            full_msg += f" (path: {path})"

        super().__init__(full_msg)


class TreeValidationError(Exception):
    """Error validating a tree definition."""

    def __init__(
        self,
        code: str,
        errors: List[ValidationError],
        path: Optional[Path] = None,
    ) -> None:
        self.code = code
        self.errors = errors
        self.path = path

        error_summary = "; ".join(e.message[:100] for e in errors[:3])
        if len(errors) > 3:
            error_summary += f" (+{len(errors) - 3} more)"

        super().__init__(f"[{code}] Validation failed: {error_summary}")


class TreeInUseError(Exception):
    """Error when trying to unload a running tree."""

    def __init__(self, tree_id: str) -> None:
        self.tree_id = tree_id
        super().__init__(f"Cannot unload tree '{tree_id}' while it is running")


class SecurityError(Exception):
    """Security violation (path traversal, etc.)."""

    def __init__(self, code: str, message: str, path: Optional[Path] = None) -> None:
        self.code = code
        self.path = path
        super().__init__(f"[{code}] {message}")


# =============================================================================
# Pending Reload Info
# =============================================================================


class PendingReload:
    """Information about a pending reload."""

    def __init__(
        self,
        tree_id: str,
        source_path: Path,
        queued_at: datetime,
    ) -> None:
        self.tree_id = tree_id
        self.source_path = source_path
        self.queued_at = queued_at


# =============================================================================
# TreeRegistry
# =============================================================================


class TreeRegistry:
    """Central registry for all loaded behavior trees with hot reload.

    From tree-loader.yaml TreeRegistry specification:
    - Loads trees from Lua files
    - Validates before building
    - Provides hot reload with file watching
    - Supports reload policies for running trees

    Invariants:
    - All tree IDs in _trees are unique
    - All loaded trees have valid root nodes
    - Hot reload queue depth <= 1 (latest wins)
    - No tree in registry has circular references

    Example:
        >>> registry = TreeRegistry(Path("./trees"))
        >>> tree = registry.load(Path("./trees/oracle-agent.lua"))
        >>> registry.start_watching()  # Enable hot reload
        >>> # ... tree is automatically reloaded on file changes
        >>> registry.stop_watching()
    """

    def __init__(
        self,
        tree_dir: Path,
        default_reload_policy: ReloadPolicy = ReloadPolicy.LET_FINISH_THEN_SWAP,
        debounce_ms: int = 500,
        validate_on_load: bool = True,
    ) -> None:
        """Initialize the tree registry.

        Args:
            tree_dir: Root directory for tree .lua files.
            default_reload_policy: Default policy for handling reloads.
            debounce_ms: Debounce rapid file changes (ms).
            validate_on_load: Run TreeValidator on every load.

        Raises:
            FileNotFoundError: If tree_dir doesn't exist.
            PermissionError: If tree_dir is not readable.
        """
        self._tree_dir = Path(tree_dir).resolve()

        # Validate tree_dir
        if not self._tree_dir.exists():
            raise FileNotFoundError(f"Tree directory not found: {self._tree_dir}")
        if not self._tree_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self._tree_dir}")

        self._default_reload_policy = default_reload_policy
        self._debounce_ms = debounce_ms
        self._validate_on_load = validate_on_load

        # Tree storage
        self._trees: Dict[str, "BehaviorTree"] = {}
        self._source_paths: Dict[str, Path] = {}  # tree_id -> source path

        # Reload state
        self._reload_queue: Dict[str, PendingReload] = {}
        self._pending_changes: Dict[Path, threading.Timer] = {}

        # File watching
        self._observer = None
        self._watching = False
        self._watch_lock = threading.Lock()

        # Callbacks
        self._on_tree_loaded: Optional[Callable[[str, "BehaviorTree"], None]] = None
        self._on_tree_reloaded: Optional[Callable[[str, "BehaviorTree"], None]] = None
        self._on_tree_unloaded: Optional[Callable[[str], None]] = None
        self._on_reload_failed: Optional[Callable[[str, Exception], None]] = None

        # Components
        self._validator = TreeValidator(
            resolve_functions=True,
            check_subtrees=True,
        )
        self._builder = TreeBuilder(registry=self)

        # Pending definitions (for subtree resolution during batch load)
        self._pending_definitions: Dict[str, TreeDefinition] = {}

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def tree_dir(self) -> Path:
        """Root directory for tree files."""
        return self._tree_dir

    @property
    def is_watching(self) -> bool:
        """True if file watcher is active."""
        return self._watching

    # =========================================================================
    # Loading Methods
    # =========================================================================

    def load(self, path: Path) -> "BehaviorTree":
        """Load tree from Lua file and register.

        From tree-loader.yaml TreeRegistry.load:
        - Validates path exists and is .lua file
        - Checks for path traversal
        - Loads, validates, and builds tree
        - Registers tree in registry

        Args:
            path: Path to .lua file.

        Returns:
            Loaded BehaviorTree instance.

        Raises:
            TreeLoadError: If file not found or not .lua.
            SecurityError: If path traversal attempted.
            TreeValidationError: If tree definition invalid.
            TreeBuildError: If tree build fails.
        """
        path = Path(path).resolve()

        # Validate path exists
        if not path.exists():
            raise TreeLoadError(
                code="E4001",
                message=f"Tree file not found: {path}",
                path=path,
            )

        # Check file extension
        if path.suffix != ".lua":
            raise TreeLoadError(
                code="E4001",
                message=f"Expected .lua file, got: {path.suffix}",
                path=path,
            )

        # Security: Check for path traversal
        if not self._is_safe_path(path):
            raise SecurityError(
                code="E7002",
                message=f"Path traversal attempt detected: {path}",
                path=path,
            )

        # Load definition from Lua file
        definition = self._load_definition(path)

        # Validate if enabled
        if self._validate_on_load:
            errors = self._validator.validate(definition, self)
            if errors:
                raise TreeValidationError(
                    code="E4002",
                    errors=errors,
                    path=path,
                )

        # Build tree
        tree = self._builder.build(definition)

        # Register
        self._trees[tree.id] = tree
        self._source_paths[tree.id] = path

        logger.info(f"Loaded tree '{tree.id}' from {path}")

        # Callback
        if self._on_tree_loaded:
            try:
                self._on_tree_loaded(tree.id, tree)
            except Exception as e:
                logger.error(f"on_tree_loaded callback failed: {e}")

        return tree

    def load_all(self) -> Dict[str, "BehaviorTree"]:
        """Load all .lua files in tree_dir.

        From tree-loader.yaml TreeRegistry.load_all:
        - Loads all valid trees
        - Invalid trees logged but don't block others

        Returns:
            Map of tree_id -> tree for successfully loaded trees.
        """
        loaded: Dict[str, "BehaviorTree"] = {}
        errors: List[tuple] = []

        # Find all .lua files
        lua_files = list(self._tree_dir.glob("**/*.lua"))

        # First pass: load definitions for subtree resolution
        self._pending_definitions = {}
        for path in lua_files:
            try:
                definition = self._load_definition(path)
                self._pending_definitions[definition.name] = definition
            except Exception as e:
                errors.append((path, e))
                logger.warning(f"Failed to parse {path}: {e}")

        # Second pass: validate and build
        for path in lua_files:
            if path.stem not in [d.name for d in self._pending_definitions.values()
                                  if d.source_path == str(path)]:
                continue  # Already failed in first pass

            try:
                tree = self.load(path)
                loaded[tree.id] = tree
            except Exception as e:
                errors.append((path, e))
                logger.warning(f"Failed to load tree from {path}: {e}")

        # Clear pending
        self._pending_definitions = {}

        if errors:
            logger.warning(
                f"Loaded {len(loaded)} trees, {len(errors)} failed"
            )

        return loaded

    def get(self, tree_id: str) -> Optional["BehaviorTree"]:
        """Get tree by ID.

        Args:
            tree_id: Tree identifier.

        Returns:
            Tree instance if found, None otherwise.
        """
        return self._trees.get(tree_id)

    def unload(self, tree_id: str) -> None:
        """Remove tree from registry.

        From tree-loader.yaml TreeRegistry.unload:
        - Cannot unload running tree

        Args:
            tree_id: Tree to unload.

        Raises:
            TreeInUseError: If tree is currently running.
        """
        tree = self._trees.get(tree_id)
        if tree is None:
            return

        # Check if running
        from ..core.tree import TreeStatus
        if tree.status == TreeStatus.RUNNING:
            raise TreeInUseError(tree_id)

        # Remove from registry
        del self._trees[tree_id]
        self._source_paths.pop(tree_id, None)
        self._reload_queue.pop(tree_id, None)

        logger.info(f"Unloaded tree '{tree_id}'")

        # Callback
        if self._on_tree_unloaded:
            try:
                self._on_tree_unloaded(tree_id)
            except Exception as e:
                logger.error(f"on_tree_unloaded callback failed: {e}")

    def list_trees(self) -> List[str]:
        """Return all registered tree IDs.

        Returns:
            List of tree IDs.
        """
        return list(self._trees.keys())

    # =========================================================================
    # Hot Reload Methods
    # =========================================================================

    def reload(
        self,
        tree_id: str,
        policy: Optional[ReloadPolicy] = None,
    ) -> None:
        """Reload tree from source file.

        From tree-loader.yaml TreeRegistry.reload:
        - If tree IDLE: reload applied immediately
        - If tree RUNNING: reload queued or cancelled per policy

        Args:
            tree_id: Tree to reload.
            policy: Reload policy (uses default if not specified).

        Raises:
            TreeLoadError: If tree not found.
            TreeValidationError: If new definition invalid.
        """
        from ..core.tree import TreeStatus

        if tree_id not in self._trees:
            raise TreeLoadError(
                code="E3001",
                message=f"Tree '{tree_id}' not found",
            )

        tree = self._trees[tree_id]
        source_path = self._source_paths.get(tree_id)

        if source_path is None:
            raise TreeLoadError(
                code="E3005",
                message=f"No source path for tree '{tree_id}'",
            )

        policy = policy or self._default_reload_policy

        # Check tree status
        if tree.status == TreeStatus.RUNNING:
            if policy == ReloadPolicy.LET_FINISH_THEN_SWAP:
                # Queue reload for later
                self._reload_queue[tree_id] = PendingReload(
                    tree_id=tree_id,
                    source_path=source_path,
                    queued_at=datetime.now(timezone.utc),
                )
                tree.queue_reload()
                logger.info(
                    f"Queued reload for running tree '{tree_id}' "
                    f"(policy: let_finish_then_swap)"
                )
                return

            elif policy == ReloadPolicy.CANCEL_AND_RESTART:
                # Cancel and reload immediately
                tree.cancel(reason="reload")
                logger.info(
                    f"Cancelled tree '{tree_id}' for reload "
                    f"(policy: cancel_and_restart)"
                )

        # Apply reload immediately
        self._apply_reload(tree_id, source_path)

    def _apply_reload(self, tree_id: str, source_path: Path) -> None:
        """Apply reload for a tree.

        Args:
            tree_id: Tree to reload.
            source_path: Path to source file.
        """
        try:
            # Keep old tree for rollback
            old_tree = self._trees.get(tree_id)

            # Load new tree
            new_definition = self._load_definition(source_path)

            # Validate
            if self._validate_on_load:
                errors = self._validator.validate(new_definition, self)
                if errors:
                    raise TreeValidationError(
                        code="E3005",
                        errors=errors,
                        path=source_path,
                    )

            # Build new tree
            new_tree = self._builder.build(new_definition)

            # Swap
            self._trees[tree_id] = new_tree
            new_tree.clear_reload_pending()

            # Clear queue
            self._reload_queue.pop(tree_id, None)

            # Log changes
            self._log_tree_diff(old_tree, new_tree)

            logger.info(f"Reloaded tree '{tree_id}' from {source_path}")

            # Callback
            if self._on_tree_reloaded:
                try:
                    self._on_tree_reloaded(tree_id, new_tree)
                except Exception as e:
                    logger.error(f"on_tree_reloaded callback failed: {e}")

        except Exception as e:
            logger.error(f"Failed to reload tree '{tree_id}': {e}")

            # Callback
            if self._on_reload_failed:
                try:
                    self._on_reload_failed(tree_id, e)
                except Exception as cb_e:
                    logger.error(f"on_reload_failed callback failed: {cb_e}")

            raise

    def check_pending_reloads(self) -> None:
        """Apply any pending reloads for completed trees.

        Called after tree tick completes. Checks if any trees
        have pending reloads and are no longer running.
        """
        from ..core.tree import TreeStatus

        for tree_id, pending in list(self._reload_queue.items()):
            tree = self._trees.get(tree_id)
            if tree is None:
                self._reload_queue.pop(tree_id, None)
                continue

            if tree.status != TreeStatus.RUNNING:
                logger.info(
                    f"Applying pending reload for '{tree_id}' "
                    f"(queued {(datetime.now(timezone.utc) - pending.queued_at).seconds}s ago)"
                )
                try:
                    self._apply_reload(tree_id, pending.source_path)
                except Exception as e:
                    logger.error(f"Pending reload for '{tree_id}' failed: {e}")

    # =========================================================================
    # File Watching
    # =========================================================================

    def start_watching(self) -> None:
        """Start file watcher for tree_dir.

        From tree-loader.yaml TreeRegistry.start_watching:
        - File changes trigger reload
        - Multiple rapid changes batched (debounce)
        - Only .lua files trigger reload
        """
        with self._watch_lock:
            if self._watching:
                return

            try:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler, FileModifiedEvent

                class TreeFileHandler(FileSystemEventHandler):
                    def __init__(handler_self, registry: "TreeRegistry"):
                        handler_self.registry = registry

                    def on_modified(handler_self, event):
                        if not event.is_directory:
                            path = Path(event.src_path)
                            if path.suffix == ".lua":
                                handler_self.registry.on_file_changed(path)

                    def on_created(handler_self, event):
                        if not event.is_directory:
                            path = Path(event.src_path)
                            if path.suffix == ".lua":
                                handler_self.registry.on_file_changed(path)

                self._observer = Observer()
                handler = TreeFileHandler(self)
                self._observer.schedule(
                    handler,
                    str(self._tree_dir),
                    recursive=True,
                )
                self._observer.start()
                self._watching = True

                logger.info(f"Started watching {self._tree_dir} for changes")

            except ImportError:
                logger.warning(
                    "watchdog not installed. Hot reload disabled. "
                    "Install with: pip install watchdog"
                )

    def stop_watching(self) -> None:
        """Stop file watcher."""
        with self._watch_lock:
            if not self._watching:
                return

            if self._observer:
                self._observer.stop()
                self._observer.join(timeout=5.0)
                self._observer = None

            # Cancel any pending debounced changes
            for timer in self._pending_changes.values():
                timer.cancel()
            self._pending_changes.clear()

            self._watching = False
            logger.info("Stopped watching for changes")

    def on_file_changed(self, path: Path) -> None:
        """Internal callback for file changes.

        Debounces rapid changes before triggering reload.

        Args:
            path: Path to changed file.
        """
        path = path.resolve()

        # Cancel any pending timer for this file
        if path in self._pending_changes:
            self._pending_changes[path].cancel()

        # Schedule debounced reload
        def apply_change():
            try:
                self._apply_file_change(path)
            finally:
                self._pending_changes.pop(path, None)

        timer = threading.Timer(
            self._debounce_ms / 1000.0,
            apply_change,
        )
        self._pending_changes[path] = timer
        timer.start()

    def _apply_file_change(self, path: Path) -> None:
        """Apply a file change after debounce.

        Args:
            path: Path to changed file.
        """
        # Find which tree this file belongs to
        for tree_id, source_path in self._source_paths.items():
            if source_path.resolve() == path:
                logger.info(f"[HOT RELOAD] Detected change: {path}")
                try:
                    # Validate first
                    definition = self._load_definition(path)
                    if self._validate_on_load:
                        errors = self._validator.validate(definition, self)
                        if errors:
                            logger.error(
                                f"[HOT RELOAD] Validation failed for {path}: "
                                f"{len(errors)} errors"
                            )
                            for error in errors[:3]:
                                logger.error(f"  - {error.message}")
                            return

                    logger.info(f"[HOT RELOAD] Validating... OK")

                    # Reload
                    self.reload(tree_id)

                except Exception as e:
                    logger.error(f"[HOT RELOAD] Reload failed: {e}")

                return

        # New file - try to load
        if path.suffix == ".lua" and self._is_safe_path(path):
            try:
                tree = self.load(path)
                logger.info(f"[HOT RELOAD] Loaded new tree: {tree.id}")
            except Exception as e:
                logger.warning(f"[HOT RELOAD] Failed to load new file {path}: {e}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_tree_loaded(
        self,
        callback: Callable[[str, "BehaviorTree"], None],
    ) -> None:
        """Register callback for tree load events."""
        self._on_tree_loaded = callback

    def on_tree_reloaded(
        self,
        callback: Callable[[str, "BehaviorTree"], None],
    ) -> None:
        """Register callback for tree reload events."""
        self._on_tree_reloaded = callback

    def on_tree_unloaded(
        self,
        callback: Callable[[str], None],
    ) -> None:
        """Register callback for tree unload events."""
        self._on_tree_unloaded = callback

    def on_reload_failed(
        self,
        callback: Callable[[str, Exception], None],
    ) -> None:
        """Register callback for reload failure events."""
        self._on_reload_failed = callback

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe (no traversal outside tree_dir).

        Args:
            path: Path to check.

        Returns:
            True if path is within tree_dir.
        """
        try:
            path = path.resolve()
            return path.is_relative_to(self._tree_dir)
        except (ValueError, RuntimeError):
            return False

    def _load_definition(self, path: Path) -> TreeDefinition:
        """Load tree definition from Lua file.

        This is a placeholder - actual Lua parsing would be done
        by LuaTreeLoader. For now, we'll parse a simplified format.

        Args:
            path: Path to .lua file.

        Returns:
            TreeDefinition instance.
        """
        # TODO: Integrate with actual LuaTreeLoader when available
        # For now, we'll use a simple approach - assume the tree name
        # is derived from the filename

        tree_name = path.stem

        # Check for pending definition (from batch load)
        if tree_name in self._pending_definitions:
            return self._pending_definitions[tree_name]

        # Create minimal definition from file
        # In the full implementation, LuaTreeLoader would parse the Lua DSL
        from .definitions import NodeDefinition

        root = NodeDefinition(
            type="sequence",
            id="root",
            source_line=1,
        )

        return TreeDefinition(
            name=tree_name,
            root=root,
            source_path=str(path),
        )

    def _log_tree_diff(
        self,
        old_tree: Optional["BehaviorTree"],
        new_tree: "BehaviorTree",
    ) -> None:
        """Log differences between old and new tree.

        From footgun-addendum.md E.6 hot reload feedback.

        Args:
            old_tree: Previous tree version (may be None).
            new_tree: New tree version.
        """
        if old_tree is None:
            return

        # Compare node counts
        old_count = old_tree.node_count
        new_count = new_tree.node_count

        if old_count != new_count:
            logger.info(
                f"  Tree '{new_tree.id}' node count: {old_count} -> {new_count}"
            )

        # Compare source hashes
        if old_tree.source_hash != new_tree.source_hash:
            logger.info(
                f"  Tree '{new_tree.id}' source changed"
            )


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "ReloadPolicy",
    "TreeRegistry",
    "TreeLoadError",
    "TreeValidationError",
    "TreeInUseError",
    "SecurityError",
]
