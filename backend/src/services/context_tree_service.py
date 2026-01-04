"""ContextTreeService - Manages tree-based Oracle context persistence.

This service handles:
- Context tree CRUD operations
- Node management (create, update, checkout)
- Tree pruning (remove old non-checkpoint nodes)
- Path-to-HEAD calculation for context building
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from ..models.oracle_context import (
    ContextNode,
    ContextTree,
    ToolCall,
    ToolCallStatus,
)
from .database import DatabaseService

# Type alias for system messages (list of strings or message objects)
SystemMessages = List[str]

logger = logging.getLogger(__name__)


class ContextTreeServiceError(Exception):
    """Raised when context tree service operations fail."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ContextTreeService:
    """Service for managing Oracle conversation context trees.

    Provides CRUD operations for context trees and nodes, tree navigation,
    and pruning functionality.
    """

    def __init__(self, db: Optional[DatabaseService] = None):
        """Initialize the context tree service.

        Args:
            db: Database service instance. Creates new one if not provided.
        """
        self.db = db or DatabaseService()

    # ========================================
    # Tree Operations
    # ========================================

    def get_trees(
        self,
        user_id: str,
        project_id: str = "default",
    ) -> List[ContextTree]:
        """Get all context trees for a user/project.

        Args:
            user_id: User identifier
            project_id: Project identifier (default: "default")

        Returns:
            List of ContextTree objects
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT root_id, user_id, project_id, current_node_id,
                       created_at, last_activity, node_count, max_nodes, label
                FROM context_trees
                WHERE user_id = ? AND project_id = ?
                ORDER BY last_activity DESC
                """,
                (user_id, project_id)
            )
            return [self._row_to_tree(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get trees for {user_id}/{project_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to get trees: {str(e)}",
                {"user_id": user_id, "project_id": project_id}
            )
        finally:
            conn.close()

    def get_tree(
        self,
        user_id: str,
        root_id: str,
    ) -> Optional[ContextTree]:
        """Get a specific context tree by root ID.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree

        Returns:
            ContextTree if found, None otherwise
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT root_id, user_id, project_id, current_node_id,
                       created_at, last_activity, node_count, max_nodes, label
                FROM context_trees
                WHERE user_id = ? AND root_id = ?
                """,
                (user_id, root_id)
            )
            row = cursor.fetchone()
            return self._row_to_tree(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get tree {root_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to get tree: {str(e)}",
                {"root_id": root_id}
            )
        finally:
            conn.close()

    def create_tree(
        self,
        user_id: str,
        project_id: str = "default",
        label: Optional[str] = None,
        max_nodes: int = 30,
    ) -> ContextTree:
        """Create a new context tree with an empty root node.

        Args:
            user_id: User identifier
            project_id: Project identifier
            label: Optional tree label
            max_nodes: Maximum nodes before pruning (default: 30)

        Returns:
            Newly created ContextTree
        """
        root_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        conn = self.db.connect()
        try:
            # Create root node (empty placeholder)
            conn.execute(
                """
                INSERT INTO context_nodes (
                    id, root_id, parent_id, user_id, project_id, created_at,
                    question, answer, tool_calls_json, tokens_used, model_used,
                    label, is_checkpoint, is_root
                ) VALUES (?, ?, NULL, ?, ?, ?, '', '', '[]', 0, NULL, NULL, 0, 1)
                """,
                (root_id, root_id, user_id, project_id, now.isoformat())
            )

            # Create tree metadata
            conn.execute(
                """
                INSERT INTO context_trees (
                    root_id, user_id, project_id, current_node_id,
                    created_at, last_activity, node_count, max_nodes, label
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (root_id, user_id, project_id, root_id, now.isoformat(), now.isoformat(), max_nodes, label)
            )
            conn.commit()

            logger.info(f"Created tree {root_id} for {user_id}/{project_id}")

            return ContextTree(
                root_id=root_id,
                user_id=user_id,
                project_id=project_id,
                current_node_id=root_id,
                created_at=now,
                last_activity=now,
                node_count=1,
                max_nodes=max_nodes,
                label=label,
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create tree: {e}")
            raise ContextTreeServiceError(
                f"Failed to create tree: {str(e)}",
                {"user_id": user_id, "project_id": project_id}
            )
        finally:
            conn.close()

    def delete_tree(
        self,
        user_id: str,
        root_id: str,
    ) -> bool:
        """Delete a context tree and all its nodes.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree

        Returns:
            True if deleted, False if not found
        """
        conn = self.db.connect()
        try:
            # Delete all nodes in the tree
            conn.execute(
                "DELETE FROM context_nodes WHERE user_id = ? AND root_id = ?",
                (user_id, root_id)
            )
            # Delete tree metadata
            cursor = conn.execute(
                "DELETE FROM context_trees WHERE user_id = ? AND root_id = ?",
                (user_id, root_id)
            )
            conn.commit()

            deleted = cursor.rowcount > 0
            if deleted:
                logger.info(f"Deleted tree {root_id}")
            return deleted
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to delete tree {root_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to delete tree: {str(e)}",
                {"root_id": root_id}
            )
        finally:
            conn.close()

    def update_tree(
        self,
        user_id: str,
        root_id: str,
        label: Optional[str] = None,
        max_nodes: Optional[int] = None,
        current_node_id: Optional[str] = None,
    ) -> Optional[ContextTree]:
        """Update tree metadata.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree
            label: New label (None to keep current)
            max_nodes: New max nodes (None to keep current)
            current_node_id: New HEAD position (None to keep current)

        Returns:
            Updated ContextTree or None if not found
        """
        tree = self.get_tree(user_id, root_id)
        if not tree:
            return None

        conn = self.db.connect()
        try:
            updates = []
            params = []

            if label is not None:
                updates.append("label = ?")
                params.append(label)
            if max_nodes is not None:
                updates.append("max_nodes = ?")
                params.append(max_nodes)
            if current_node_id is not None:
                updates.append("current_node_id = ?")
                params.append(current_node_id)

            if not updates:
                return tree

            updates.append("last_activity = ?")
            params.append(datetime.now(timezone.utc).isoformat())
            params.extend([user_id, root_id])

            conn.execute(
                f"UPDATE context_trees SET {', '.join(updates)} WHERE user_id = ? AND root_id = ?",
                params
            )
            conn.commit()

            return self.get_tree(user_id, root_id)
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update tree {root_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to update tree: {str(e)}",
                {"root_id": root_id}
            )
        finally:
            conn.close()

    # ========================================
    # Node Operations
    # ========================================

    def get_nodes(
        self,
        user_id: str,
        root_id: str,
    ) -> List[ContextNode]:
        """Get all nodes in a tree.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree

        Returns:
            List of ContextNode objects
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT id, root_id, parent_id, user_id, project_id, created_at,
                       question, answer, tool_calls_json, tokens_used, model_used,
                       label, is_checkpoint, is_root, system_messages_json
                FROM context_nodes
                WHERE user_id = ? AND root_id = ?
                ORDER BY created_at ASC
                """,
                (user_id, root_id)
            )
            return [self._row_to_node(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get nodes for tree {root_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to get nodes: {str(e)}",
                {"root_id": root_id}
            )
        finally:
            conn.close()

    def get_node(
        self,
        user_id: str,
        node_id: str,
    ) -> Optional[ContextNode]:
        """Get a specific node by ID.

        Args:
            user_id: User identifier
            node_id: Node ID

        Returns:
            ContextNode if found, None otherwise
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT id, root_id, parent_id, user_id, project_id, created_at,
                       question, answer, tool_calls_json, tokens_used, model_used,
                       label, is_checkpoint, is_root, system_messages_json
                FROM context_nodes
                WHERE user_id = ? AND id = ?
                """,
                (user_id, node_id)
            )
            row = cursor.fetchone()
            return self._row_to_node(row) if row else None
        except Exception as e:
            logger.error(f"Failed to get node {node_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to get node: {str(e)}",
                {"node_id": node_id}
            )
        finally:
            conn.close()

    def create_node(
        self,
        user_id: str,
        root_id: str,
        parent_id: str,
        question: str,
        answer: str,
        tool_calls: Optional[List[ToolCall]] = None,
        tokens_used: int = 0,
        model_used: Optional[str] = None,
        label: Optional[str] = None,
        is_checkpoint: bool = False,
        system_messages: Optional[SystemMessages] = None,
    ) -> ContextNode:
        """Create a new node in the tree.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree
            parent_id: Parent node ID
            question: User's question
            answer: Oracle's answer
            tool_calls: List of tool calls
            tokens_used: Tokens consumed
            model_used: Model that generated the answer
            label: Optional node label
            is_checkpoint: Whether this node is protected from pruning
            system_messages: List of system messages to store with this node

        Returns:
            Newly created ContextNode
        """
        node_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Serialize tool calls
        tool_calls_json = json.dumps([
            {
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
                "result": tc.result,
                "status": tc.status.value,
                "duration_ms": tc.duration_ms,
            }
            for tc in (tool_calls or [])
        ])

        # Serialize system messages
        system_messages_json = json.dumps(system_messages or [])

        conn = self.db.connect()
        try:
            # Get tree's project_id
            cursor = conn.execute(
                "SELECT project_id FROM context_trees WHERE root_id = ?",
                (root_id,)
            )
            row = cursor.fetchone()
            if not row:
                raise ContextTreeServiceError(
                    f"Tree not found: {root_id}",
                    {"root_id": root_id}
                )
            project_id = row["project_id"]

            # Create node
            conn.execute(
                """
                INSERT INTO context_nodes (
                    id, root_id, parent_id, user_id, project_id, created_at,
                    question, answer, tool_calls_json, tokens_used, model_used,
                    label, is_checkpoint, is_root, system_messages_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    node_id, root_id, parent_id, user_id, project_id, now.isoformat(),
                    question, answer, tool_calls_json, tokens_used, model_used,
                    label, int(is_checkpoint), system_messages_json
                )
            )

            # Update tree: increment node count, update HEAD, update last_activity
            conn.execute(
                """
                UPDATE context_trees
                SET node_count = node_count + 1,
                    current_node_id = ?,
                    last_activity = ?
                WHERE root_id = ?
                """,
                (node_id, now.isoformat(), root_id)
            )
            conn.commit()

            logger.debug(f"Created node {node_id} in tree {root_id} with {len(system_messages or [])} system messages")

            return ContextNode(
                id=node_id,
                root_id=root_id,
                parent_id=parent_id,
                user_id=user_id,
                project_id=project_id,
                created_at=now,
                question=question,
                answer=answer,
                tool_calls=tool_calls or [],
                tokens_used=tokens_used,
                model_used=model_used,
                label=label,
                is_checkpoint=is_checkpoint,
                is_root=False,
                child_count=0,
                system_messages=system_messages or [],
            )
        except ContextTreeServiceError:
            conn.rollback()
            raise
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to create node: {e}")
            raise ContextTreeServiceError(
                f"Failed to create node: {str(e)}",
                {"root_id": root_id, "parent_id": parent_id}
            )
        finally:
            conn.close()

    def update_node(
        self,
        user_id: str,
        node_id: str,
        label: Optional[str] = None,
        is_checkpoint: Optional[bool] = None,
    ) -> Optional[ContextNode]:
        """Update a node's metadata.

        Args:
            user_id: User identifier
            node_id: Node ID
            label: New label (None to keep current)
            is_checkpoint: New checkpoint status (None to keep current)

        Returns:
            Updated ContextNode or None if not found
        """
        node = self.get_node(user_id, node_id)
        if not node:
            return None

        conn = self.db.connect()
        try:
            updates = []
            params = []

            if label is not None:
                updates.append("label = ?")
                params.append(label)
            if is_checkpoint is not None:
                updates.append("is_checkpoint = ?")
                params.append(int(is_checkpoint))

            if not updates:
                return node

            params.extend([user_id, node_id])

            conn.execute(
                f"UPDATE context_nodes SET {', '.join(updates)} WHERE user_id = ? AND id = ?",
                params
            )
            conn.commit()

            return self.get_node(user_id, node_id)
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to update node {node_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to update node: {str(e)}",
                {"node_id": node_id}
            )
        finally:
            conn.close()

    def checkout_node(
        self,
        user_id: str,
        node_id: str,
    ) -> Optional[ContextTree]:
        """Set a node as the current HEAD of its tree.

        Args:
            user_id: User identifier
            node_id: Node ID to checkout

        Returns:
            Updated ContextTree or None if node not found
        """
        node = self.get_node(user_id, node_id)
        if not node:
            return None

        return self.update_tree(
            user_id=user_id,
            root_id=node.root_id,
            current_node_id=node_id,
        )

    # ========================================
    # Path Operations
    # ========================================

    def get_path_to_head(
        self,
        user_id: str,
        root_id: str,
    ) -> List[str]:
        """Get the path from root to current HEAD.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree

        Returns:
            List of node IDs from root to HEAD
        """
        tree = self.get_tree(user_id, root_id)
        if not tree:
            return []

        # Build path by walking up from HEAD to root
        path = []
        current_id = tree.current_node_id

        conn = self.db.connect()
        try:
            while current_id:
                path.append(current_id)
                cursor = conn.execute(
                    "SELECT parent_id FROM context_nodes WHERE id = ?",
                    (current_id,)
                )
                row = cursor.fetchone()
                if row and row["parent_id"]:
                    current_id = row["parent_id"]
                else:
                    break

            # Reverse to get root-to-HEAD order
            path.reverse()
            return path
        except Exception as e:
            logger.error(f"Failed to get path for tree {root_id}: {e}")
            return []
        finally:
            conn.close()

    # ========================================
    # Pruning Operations
    # ========================================

    def prune_tree(
        self,
        user_id: str,
        root_id: str,
    ) -> Tuple[int, int]:
        """Prune old non-checkpoint nodes from the tree.

        Removes nodes that are:
        - Not checkpoints
        - Not on the path to HEAD
        - Not the root node

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree

        Returns:
            Tuple of (nodes_pruned, nodes_remaining)
        """
        tree = self.get_tree(user_id, root_id)
        if not tree:
            return (0, 0)

        # Get path to HEAD (these nodes must be preserved)
        path_to_head = set(self.get_path_to_head(user_id, root_id))

        conn = self.db.connect()
        try:
            # Get all non-protected nodes (not checkpoints, not root, not on path)
            cursor = conn.execute(
                """
                SELECT id FROM context_nodes
                WHERE user_id = ? AND root_id = ?
                  AND is_checkpoint = 0
                  AND is_root = 0
                """,
                (user_id, root_id)
            )

            nodes_to_prune = [
                row["id"] for row in cursor.fetchall()
                if row["id"] not in path_to_head
            ]

            if nodes_to_prune:
                # Delete nodes
                placeholders = ",".join("?" * len(nodes_to_prune))
                conn.execute(
                    f"DELETE FROM context_nodes WHERE id IN ({placeholders})",
                    nodes_to_prune
                )

                # Update node count
                cursor = conn.execute(
                    "SELECT COUNT(*) as count FROM context_nodes WHERE root_id = ?",
                    (root_id,)
                )
                new_count = cursor.fetchone()["count"]

                conn.execute(
                    "UPDATE context_trees SET node_count = ? WHERE root_id = ?",
                    (new_count, root_id)
                )
                conn.commit()

                logger.info(f"Pruned {len(nodes_to_prune)} nodes from tree {root_id}")
                return (len(nodes_to_prune), new_count)

            return (0, tree.node_count)

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to prune tree {root_id}: {e}")
            raise ContextTreeServiceError(
                f"Failed to prune tree: {str(e)}",
                {"root_id": root_id}
            )
        finally:
            conn.close()

    # ========================================
    # Active Tree Management
    # ========================================

    def get_active_tree_id(
        self,
        user_id: str,
        project_id: str = "default",
    ) -> Optional[str]:
        """Get the active tree ID for a user/project (most recently used).

        Args:
            user_id: User identifier
            project_id: Project identifier

        Returns:
            Root ID of active tree or None
        """
        conn = self.db.connect()
        try:
            cursor = conn.execute(
                """
                SELECT root_id FROM context_trees
                WHERE user_id = ? AND project_id = ?
                ORDER BY last_activity DESC
                LIMIT 1
                """,
                (user_id, project_id)
            )
            row = cursor.fetchone()
            return row["root_id"] if row else None
        except Exception as e:
            logger.error(f"Failed to get active tree: {e}")
            return None
        finally:
            conn.close()

    def set_active_tree(
        self,
        user_id: str,
        root_id: str,
    ) -> Optional[ContextTree]:
        """Set a tree as active by updating its last_activity.

        Args:
            user_id: User identifier
            root_id: Root node ID of the tree

        Returns:
            Updated ContextTree or None if not found
        """
        conn = self.db.connect()
        try:
            now = datetime.now(timezone.utc).isoformat()
            cursor = conn.execute(
                """
                UPDATE context_trees
                SET last_activity = ?
                WHERE user_id = ? AND root_id = ?
                """,
                (now, user_id, root_id)
            )
            conn.commit()

            if cursor.rowcount > 0:
                return self.get_tree(user_id, root_id)
            return None
        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to set active tree: {e}")
            raise ContextTreeServiceError(
                f"Failed to set active tree: {str(e)}",
                {"root_id": root_id}
            )
        finally:
            conn.close()

    # ========================================
    # Helper Methods
    # ========================================

    def _row_to_tree(self, row) -> ContextTree:
        """Convert a database row to ContextTree."""
        return ContextTree(
            root_id=row["root_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            current_node_id=row["current_node_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_activity=datetime.fromisoformat(row["last_activity"]),
            node_count=row["node_count"],
            max_nodes=row["max_nodes"],
            label=row["label"],
        )

    def _row_to_node(self, row) -> ContextNode:
        """Convert a database row to ContextNode."""
        # Parse tool calls JSON
        tool_calls_json = row["tool_calls_json"] or "[]"
        try:
            tool_calls_data = json.loads(tool_calls_json)
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                    result=tc.get("result"),
                    status=ToolCallStatus(tc.get("status", "pending")),
                    duration_ms=tc.get("duration_ms"),
                )
                for tc in tool_calls_data
            ]
        except (json.JSONDecodeError, KeyError):
            tool_calls = []

        # Parse system messages JSON
        system_messages_json = row["system_messages_json"] if "system_messages_json" in row.keys() else "[]"
        try:
            system_messages = json.loads(system_messages_json or "[]")
            if not isinstance(system_messages, list):
                system_messages = []
        except json.JSONDecodeError:
            system_messages = []

        return ContextNode(
            id=row["id"],
            root_id=row["root_id"],
            parent_id=row["parent_id"],
            user_id=row["user_id"],
            project_id=row["project_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            question=row["question"],
            answer=row["answer"],
            tool_calls=tool_calls,
            tokens_used=row["tokens_used"],
            model_used=row["model_used"],
            label=row["label"],
            is_checkpoint=bool(row["is_checkpoint"]),
            is_root=bool(row["is_root"]),
            child_count=0,  # Could be computed if needed
            system_messages=system_messages,
        )


# ========================================
# Singleton Pattern
# ========================================

_tree_service: Optional[ContextTreeService] = None


def get_context_tree_service() -> ContextTreeService:
    """Get or create the ContextTreeService singleton.

    Returns:
        ContextTreeService instance
    """
    global _tree_service
    if _tree_service is None:
        _tree_service = ContextTreeService()
    return _tree_service


__all__ = [
    "ContextTreeService",
    "ContextTreeServiceError",
    "get_context_tree_service",
]
