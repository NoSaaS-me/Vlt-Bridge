import os
import tomllib
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


# ============================================================================
# Project Configuration
# ============================================================================

class ProjectConfig(BaseModel):
    name: str
    id: str
    description: Optional[str] = None


# ============================================================================
# CodeRAG Configuration
# ============================================================================

class CodeRAGEmbeddingConfig(BaseModel):
    """Embedding model configuration for CodeRAG."""
    model: str = Field(
        default="qwen/qwen3-embedding-8b",
        description="Embedding model for code semantic search"
    )
    batch_size: int = Field(
        default=10,
        description="Number of chunks to embed in one batch"
    )


class CodeRAGRepoMapConfig(BaseModel):
    """Repository map generation configuration."""
    max_tokens: int = Field(
        default=4000,
        description="Maximum tokens for generated repo map"
    )
    include_signatures: bool = Field(
        default=True,
        description="Include function signatures in repo map"
    )
    include_docstrings: bool = Field(
        default=False,
        description="Include docstrings in repo map"
    )


class CodeRAGDeltaConfig(BaseModel):
    """Delta-based indexing configuration."""
    file_threshold: int = Field(
        default=5,
        description="Commit index after N files changed"
    )
    line_threshold: int = Field(
        default=1000,
        description="Commit index after N total lines changed"
    )
    timeout_seconds: int = Field(
        default=300,
        description="Commit index after N seconds of inactivity"
    )
    jit_indexing: bool = Field(
        default=True,
        description="Index queued files on-demand if they match query"
    )


class CodeRAGConfig(BaseModel):
    """CodeRAG configuration section."""
    include: List[str] = Field(
        default_factory=lambda: ["src/**/*.py", "lib/**/*.py", "tests/**/*.py"],
        description="File patterns to include in indexing"
    )
    exclude: List[str] = Field(
        default_factory=lambda: ["**/node_modules/**", "**/__pycache__/**", "**/.git/**"],
        description="File patterns to exclude from indexing"
    )
    languages: List[str] = Field(
        default_factory=lambda: ["python", "typescript", "javascript"],
        description="Programming languages to index"
    )
    embedding: CodeRAGEmbeddingConfig = Field(
        default_factory=CodeRAGEmbeddingConfig,
        description="Embedding configuration"
    )
    repomap: CodeRAGRepoMapConfig = Field(
        default_factory=CodeRAGRepoMapConfig,
        description="Repository map configuration"
    )
    delta: CodeRAGDeltaConfig = Field(
        default_factory=CodeRAGDeltaConfig,
        description="Delta-based indexing configuration"
    )
    # Indexed root tracking (set by vlt init)
    indexed_root: Optional[str] = Field(
        default=None,
        description="Absolute path of the indexed project root"
    )
    indexed_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp of when the project was indexed"
    )


# ============================================================================
# Oracle Configuration
# ============================================================================

class OracleConfig(BaseModel):
    """Oracle configuration section."""
    vault_url: str = Field(
        default="http://localhost:8000",
        description="Document-MCP vault URL for markdown notes"
    )
    synthesis_model: str = Field(
        default="anthropic/claude-sonnet-4",
        description="LLM model for answer synthesis"
    )
    rerank_model: str = Field(
        default="openai/gpt-4o-mini",
        description="LLM model for result reranking"
    )
    max_context_tokens: int = Field(
        default=16000,
        description="Maximum tokens for assembled context"
    )


# ============================================================================
# Complete VLT Configuration
# ============================================================================

class VltConfig(BaseModel):
    """Complete vlt.toml configuration."""
    project: ProjectConfig
    coderag: Optional[CodeRAGConfig] = None
    oracle: Optional[OracleConfig] = None

def find_vlt_toml(start_path: Path = Path(".")) -> Optional[Path]:
    """Recursively search for vlt.toml in parent directories."""
    current = start_path.resolve()
    for _ in range(len(current.parts)):
        check_path = current / "vlt.toml"
        if check_path.exists():
            return check_path
        if current == current.parent: # Root reached
            break
        current = current.parent
    return None


def find_parent_indexed_root(start_path: Path = Path(".")) -> Optional[tuple[Path, str]]:
    """Find if a parent directory has a vlt.toml with coderag indexed_root set.

    Walks UP the directory tree looking for existing vlt.toml files that have
    the coderag.indexed_root field set, indicating this directory is already
    covered by an existing index.

    Args:
        start_path: Starting directory to search from

    Returns:
        Tuple of (vlt_toml_path, indexed_root) if found, None otherwise
    """
    current = start_path.resolve().parent  # Start from parent, not current

    for _ in range(len(current.parts)):
        check_path = current / "vlt.toml"
        if check_path.exists():
            try:
                with open(check_path, "rb") as f:
                    data = tomllib.load(f)
                coderag = data.get("coderag", {})
                indexed_root = coderag.get("indexed_root")
                if indexed_root:
                    return (check_path, indexed_root)
            except Exception:
                pass  # Malformed TOML, continue searching

        if current == current.parent:  # Root reached
            break
        current = current.parent

    return None


def update_vlt_toml_coderag_indexed(toml_path: Path, indexed_root: str, indexed_at: str) -> None:
    """Update vlt.toml with coderag indexed_root and indexed_at fields.

    This preserves all existing content while adding/updating the indexed tracking fields.

    Args:
        toml_path: Path to vlt.toml file
        indexed_root: Absolute path of the indexed project root
        indexed_at: ISO timestamp of when the project was indexed
    """
    import re

    with open(toml_path, "r") as f:
        content = f.read()

    # Check if [coderag] section exists
    if "[coderag]" not in content:
        # Add coderag section at the end
        content += f"\n[coderag]\nindexed_root = \"{indexed_root}\"\nindexed_at = \"{indexed_at}\"\n"
    else:
        # Find the coderag section and add/update the fields
        # First, remove existing indexed_root and indexed_at if present
        content = re.sub(r'\nindexed_root\s*=\s*"[^"]*"', '', content)
        content = re.sub(r'\nindexed_at\s*=\s*"[^"]*"', '', content)

        # Find the position right after [coderag] line
        match = re.search(r'(\[coderag\]\n)', content)
        if match:
            insert_pos = match.end()
            content = (
                content[:insert_pos] +
                f'indexed_root = "{indexed_root}"\n' +
                f'indexed_at = "{indexed_at}"\n' +
                content[insert_pos:]
            )

    with open(toml_path, "w") as f:
        f.write(content)

def load_vlt_config(start_path: Path = Path(".")) -> Optional[VltConfig]:
    """Load complete vlt.toml configuration."""
    toml_path = find_vlt_toml(start_path)
    if not toml_path:
        return None

    try:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        return VltConfig(**data)
    except Exception as e:
        # Malformed TOML
        return None


def load_project_identity(start_path: Path = Path(".")) -> Optional[ProjectConfig]:
    """Load project identity from the nearest vlt.toml."""
    config = load_vlt_config(start_path)
    return config.project if config else None


def load_coderag_config(start_path: Path = Path(".")) -> Optional[CodeRAGConfig]:
    """Load CodeRAG configuration from vlt.toml, returning defaults if not present."""
    config = load_vlt_config(start_path)
    if config and config.coderag:
        return config.coderag
    # Return defaults if section missing
    return CodeRAGConfig()


def load_oracle_config(start_path: Path = Path(".")) -> Optional[OracleConfig]:
    """Load Oracle configuration from vlt.toml, returning defaults if not present."""
    config = load_vlt_config(start_path)
    if config and config.oracle:
        return config.oracle
    # Return defaults if section missing
    return OracleConfig()

def create_vlt_toml(path: Path, name: str, id: str, description: str = ""):
    """Create a vlt.toml file with sensible defaults."""
    content = f"""[project]
name = "{name}"
id = "{id}"
description = "{description}"

[coderag]
include = ["src/**/*.py", "lib/**/*.py", "tests/**/*.py"]
exclude = ["**/node_modules/**", "**/__pycache__/**", "**/.git/**"]
languages = ["python", "typescript", "javascript"]

[coderag.embedding]
model = "qwen/qwen3-embedding-8b"
batch_size = 10

[coderag.repomap]
max_tokens = 4000
include_signatures = true
include_docstrings = false

[coderag.delta]
file_threshold = 5
line_threshold = 1000
timeout_seconds = 300
jit_indexing = true

[oracle]
vault_url = "http://localhost:8000"
synthesis_model = "anthropic/claude-sonnet-4"
rerank_model = "openai/gpt-4o-mini"
max_context_tokens = 16000
"""
    with open(path / "vlt.toml", "w") as f:
        f.write(content)
