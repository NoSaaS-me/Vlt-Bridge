"""Oracle Context API endpoints - Tree-based conversation context management.

This module provides API routes for managing Oracle conversation context trees.
The frontend uses these endpoints to:
- List and manage context trees
- Navigate and checkout nodes
- Label nodes and set checkpoints
- Prune old context
- Manage context settings
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from ..middleware import AuthContext, require_auth_context
from ...models.oracle_context import ContextNode, ContextTree
from ...services.context_tree_service import (
    ContextTreeService,
    ContextTreeServiceError,
    get_context_tree_service,
)
from ...services.user_settings import UserSettingsService, get_user_settings_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/oracle/context", tags=["oracle-context"])


# ========================================
# Request/Response Models
# ========================================


class ContextNodeResponse(BaseModel):
    """API response for a context node (subset of ContextNode for frontend)."""
    id: str
    root_id: str
    parent_id: Optional[str]
    created_at: str
    question: str
    answer: str
    tokens_used: int
    label: Optional[str]
    is_checkpoint: bool
    is_root: bool


class ContextTreeResponse(BaseModel):
    """API response for a context tree (subset of ContextTree for frontend)."""
    root_id: str
    current_node_id: str
    node_count: int
    max_nodes: int
    label: Optional[str]


class ContextTreeDataResponse(BaseModel):
    """Full tree data with nodes."""
    tree: ContextTreeResponse
    nodes: List[ContextNodeResponse]


class ContextTreesListResponse(BaseModel):
    """Response for listing all trees."""
    trees: List[ContextTreeDataResponse]
    active_tree_id: Optional[str]


class CreateTreeRequest(BaseModel):
    """Request to create a new context tree."""
    label: Optional[str] = None
    project_id: Optional[str] = "default"


class LabelNodeRequest(BaseModel):
    """Request to label a node."""
    label: str


class SetCheckpointRequest(BaseModel):
    """Request to set checkpoint status."""
    is_checkpoint: bool


class PruneResponse(BaseModel):
    """Response from pruning a tree."""
    pruned: int
    remaining: int


class ContextSettingsResponse(BaseModel):
    """Context settings for the user."""
    max_context_nodes: int = Field(default=30, ge=5, le=100)


class UpdateContextSettingsRequest(BaseModel):
    """Request to update context settings."""
    max_context_nodes: Optional[int] = Field(default=None, ge=5, le=100)


# ========================================
# Helper Functions
# ========================================


def node_to_response(node: ContextNode) -> ContextNodeResponse:
    """Convert ContextNode to API response."""
    return ContextNodeResponse(
        id=node.id,
        root_id=node.root_id,
        parent_id=node.parent_id,
        created_at=node.created_at.isoformat(),
        question=node.question,
        answer=node.answer,
        tokens_used=node.tokens_used,
        label=node.label,
        is_checkpoint=node.is_checkpoint,
        is_root=node.is_root,
    )


def tree_to_response(tree: ContextTree) -> ContextTreeResponse:
    """Convert ContextTree to API response."""
    return ContextTreeResponse(
        root_id=tree.root_id,
        current_node_id=tree.current_node_id,
        node_count=tree.node_count,
        max_nodes=tree.max_nodes,
        label=tree.label,
    )


# ========================================
# Tree Endpoints
# ========================================


@router.get("/trees", response_model=ContextTreesListResponse)
async def get_context_trees(
    project_id: Optional[str] = Query("default", description="Project ID to scope trees"),
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Get all context trees for the current user.

    Returns all trees with their nodes, plus the active tree ID.

    **Response:**
    - `trees`: List of tree data objects with nodes
    - `active_tree_id`: Root ID of the most recently used tree
    """
    try:
        # Get all trees for user within the specified project
        trees = tree_service.get_trees(auth.user_id, project_id)
        active_tree_id = tree_service.get_active_tree_id(auth.user_id, project_id)

        # Build response with nodes for each tree
        tree_data_list = []
        for tree in trees:
            nodes = tree_service.get_nodes(auth.user_id, tree.root_id)
            tree_data_list.append(
                ContextTreeDataResponse(
                    tree=tree_to_response(tree),
                    nodes=[node_to_response(n) for n in nodes],
                )
            )

        return ContextTreesListResponse(
            trees=tree_data_list,
            active_tree_id=active_tree_id,
        )

    except ContextTreeServiceError as e:
        logger.error(f"Failed to get trees: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trees: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to get context trees")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get trees: {str(e)}",
        )


@router.get("/trees/{root_id}", response_model=ContextTreeDataResponse)
async def get_context_tree(
    root_id: str,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Get a specific context tree by root ID.

    Returns the tree metadata and all nodes in the tree.

    **Path Parameters:**
    - `root_id`: Root node ID of the tree

    **Response:**
    - `tree`: Tree metadata
    - `nodes`: All nodes in the tree
    """
    try:
        tree = tree_service.get_tree(auth.user_id, root_id)
        if not tree:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tree not found: {root_id}",
            )

        nodes = tree_service.get_nodes(auth.user_id, root_id)

        return ContextTreeDataResponse(
            tree=tree_to_response(tree),
            nodes=[node_to_response(n) for n in nodes],
        )

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to get tree: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get tree: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to get context tree")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get tree: {str(e)}",
        )


@router.post("/trees", response_model=ContextTreeResponse)
async def create_context_tree(
    request: CreateTreeRequest,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """
    Create a new context tree.

    Creates a new tree with an empty root node. The max_nodes limit
    is taken from the user's settings.

    **Request Body:**
    - `label`: Optional tree label

    **Response:**
    - Created tree metadata
    """
    try:
        # Get user's max_nodes setting
        max_nodes = settings_service.get_max_context_nodes(auth.user_id)

        tree = tree_service.create_tree(
            user_id=auth.user_id,
            project_id=request.project_id or "default",
            label=request.label,
            max_nodes=max_nodes,
        )

        logger.info(f"Created tree {tree.root_id} for user {auth.user_id}")
        return tree_to_response(tree)

    except ContextTreeServiceError as e:
        logger.error(f"Failed to create tree: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tree: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to create context tree")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tree: {str(e)}",
        )


@router.delete("/trees/{root_id}")
async def delete_context_tree(
    root_id: str,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Delete a context tree.

    Deletes the tree and all its nodes.

    **Path Parameters:**
    - `root_id`: Root node ID of the tree to delete

    **Response:**
    - `{"status": "deleted"}` on success
    """
    try:
        deleted = tree_service.delete_tree(auth.user_id, root_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tree not found: {root_id}",
            )

        logger.info(f"Deleted tree {root_id} for user {auth.user_id}")
        return {"status": "deleted"}

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to delete tree: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete tree: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to delete context tree")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete tree: {str(e)}",
        )


@router.post("/trees/{root_id}/activate")
async def activate_context_tree(
    root_id: str,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Set a tree as the active tree.

    Updates the tree's last_activity timestamp to make it the most recent.

    **Path Parameters:**
    - `root_id`: Root node ID of the tree to activate

    **Response:**
    - `{"status": "activated"}` on success
    """
    try:
        tree = tree_service.set_active_tree(auth.user_id, root_id)
        if not tree:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tree not found: {root_id}",
            )

        logger.info(f"Activated tree {root_id} for user {auth.user_id}")
        return {"status": "activated"}

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to activate tree: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate tree: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to activate context tree")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to activate tree: {str(e)}",
        )


@router.post("/trees/{root_id}/prune", response_model=PruneResponse)
async def prune_context_tree(
    root_id: str,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Prune old nodes from a tree.

    Removes nodes that are not checkpoints, not on the path to HEAD,
    and not the root node.

    **Path Parameters:**
    - `root_id`: Root node ID of the tree to prune

    **Response:**
    - `pruned`: Number of nodes removed
    - `remaining`: Number of nodes after pruning
    """
    try:
        tree = tree_service.get_tree(auth.user_id, root_id)
        if not tree:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tree not found: {root_id}",
            )

        pruned, remaining = tree_service.prune_tree(auth.user_id, root_id)

        logger.info(f"Pruned {pruned} nodes from tree {root_id}")
        return PruneResponse(pruned=pruned, remaining=remaining)

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to prune tree: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prune tree: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to prune context tree")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to prune tree: {str(e)}",
        )


# ========================================
# Node Endpoints
# ========================================


@router.post("/nodes/{node_id}/checkout", response_model=ContextTreeResponse)
async def checkout_node(
    node_id: str,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Checkout a node (set as current HEAD).

    Sets the specified node as the current position in its tree.
    New messages will be added as children of this node.

    **Path Parameters:**
    - `node_id`: Node ID to checkout

    **Response:**
    - Updated tree metadata with new current_node_id
    """
    try:
        tree = tree_service.checkout_node(auth.user_id, node_id)
        if not tree:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node not found: {node_id}",
            )

        logger.info(f"Checked out node {node_id} in tree {tree.root_id}")
        return tree_to_response(tree)

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to checkout node: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to checkout node: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to checkout node")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to checkout node: {str(e)}",
        )


@router.put("/nodes/{node_id}/label", response_model=ContextNodeResponse)
async def label_node(
    node_id: str,
    request: LabelNodeRequest,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Label a node.

    Sets a user-defined label on the node for easy reference.

    **Path Parameters:**
    - `node_id`: Node ID to label

    **Request Body:**
    - `label`: New label text

    **Response:**
    - Updated node metadata
    """
    try:
        node = tree_service.update_node_label(auth.user_id, node_id, request.label)
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node not found: {node_id}",
            )

        logger.info(f"Labeled node {node_id}")
        return node_to_response(node)

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to label node: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to label node: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to label node")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to label node: {str(e)}",
        )


@router.put("/nodes/{node_id}/checkpoint", response_model=ContextNodeResponse)
async def set_checkpoint(
    node_id: str,
    request: SetCheckpointRequest,
    auth: AuthContext = Depends(require_auth_context),
    tree_service: ContextTreeService = Depends(get_context_tree_service),
):
    """
    Set checkpoint status on a node.

    Mark a node as a checkpoint so it won't be pruned during cleanup.

    **Path Parameters:**
    - `node_id`: Node ID to update

    **Request Body:**
    - `is_checkpoint`: Whether this node is a checkpoint

    **Response:**
    - Updated node metadata
    """
    try:
        node = tree_service.set_node_checkpoint(auth.user_id, node_id, request.is_checkpoint)
        if not node:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Node not found: {node_id}",
            )

        logger.info(f"Set checkpoint on node {node_id} to {request.is_checkpoint}")
        return node_to_response(node)

    except HTTPException:
        raise
    except ContextTreeServiceError as e:
        logger.error(f"Failed to set checkpoint: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set checkpoint: {e.message}",
        )
    except Exception as e:
        logger.exception("Failed to set checkpoint")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set checkpoint: {str(e)}",
        )


# ========================================
# Settings Endpoints
# ========================================


@router.get("/settings", response_model=ContextSettingsResponse)
async def get_context_settings(
    auth: AuthContext = Depends(require_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """
    Get context settings for the current user.

    **Response:**
    - `max_context_nodes`: Maximum nodes allowed per tree
    """
    try:
        max_nodes = settings_service.get_max_context_nodes(auth.user_id)
        return ContextSettingsResponse(max_context_nodes=max_nodes)

    except Exception as e:
        logger.exception("Failed to get context settings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get settings: {str(e)}",
        )


@router.put("/settings", response_model=ContextSettingsResponse)
async def update_context_settings(
    request: UpdateContextSettingsRequest,
    auth: AuthContext = Depends(require_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """
    Update context settings for the current user.

    **Request Body:**
    - `max_context_nodes`: Maximum nodes allowed per tree (5-100)

    **Response:**
    - Updated settings
    """
    try:
        if request.max_context_nodes is not None:
            settings_service.set_max_context_nodes(auth.user_id, request.max_context_nodes)
            max_nodes = request.max_context_nodes
        else:
            max_nodes = settings_service.get_max_context_nodes(auth.user_id)

        logger.info(f"Updated context settings for user {auth.user_id}")
        return ContextSettingsResponse(max_context_nodes=max_nodes)

    except Exception as e:
        logger.exception("Failed to update context settings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update settings: {str(e)}",
        )