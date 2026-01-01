"""Tree-sitter parser wrapper for language-agnostic code parsing.

This module provides a simple interface to tree-sitter for parsing source code
files across multiple languages. It handles parser initialization, language
detection, and graceful error handling.

Supported languages: Python, TypeScript, JavaScript, Go, Rust
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Optional dependencies - graceful degradation if not installed
try:
    from tree_sitter import Parser, Tree, Language
    # tree-sitter-language-pack is the maintained replacement for tree-sitter-languages
    # It's compatible with tree-sitter 0.25+ (the old package is not maintained)
    try:
        from tree_sitter_language_pack import get_language, get_parser
    except ImportError:
        # Fallback to deprecated tree-sitter-languages (works with tree-sitter < 0.23)
        from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter not available. Install with: pip install tree-sitter tree-sitter-language-pack")
    Tree = Any  # Type hint placeholder


# Language detection mapping: file extension -> tree-sitter language name
EXTENSION_TO_LANGUAGE: Dict[str, str] = {
    # Python
    '.py': 'python',
    '.pyi': 'python',
    # JavaScript/TypeScript
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    # Systems languages
    '.go': 'go',
    '.rs': 'rust',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    # JVM languages
    '.java': 'java',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
    # .NET
    '.cs': 'c_sharp',
    # Ruby
    '.rb': 'ruby',
    '.rake': 'ruby',
    # PHP
    '.php': 'php',
    # Swift/Objective-C
    '.swift': 'swift',
    '.m': 'objc',
    '.mm': 'objc',
    # Shell
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'bash',
    # Markup/Config (useful for monorepos)
    '.json': 'json',
    '.yaml': 'yaml',
    '.yml': 'yaml',
    '.toml': 'toml',
    '.md': 'markdown',
    # SQL
    '.sql': 'sql',
    # Elixir/Erlang
    '.ex': 'elixir',
    '.exs': 'elixir',
    '.erl': 'erlang',
    # Haskell
    '.hs': 'haskell',
    # Lua
    '.lua': 'lua',
    # R
    '.r': 'r',
    '.R': 'r',
}

# Supported languages (all languages with tree-sitter-languages support)
SUPPORTED_LANGUAGES = {
    'python', 'typescript', 'tsx', 'javascript', 'go', 'rust',
    'c', 'cpp', 'java', 'kotlin', 'scala', 'c_sharp',
    'ruby', 'php', 'swift', 'objc',
    'bash', 'json', 'yaml', 'toml', 'markdown', 'sql',
    'elixir', 'erlang', 'haskell', 'lua', 'r',
}


def detect_language(file_path: str) -> Optional[str]:
    """Detect programming language from file extension.

    Args:
        file_path: Path to the source file

    Returns:
        Language name (e.g., 'python', 'typescript') or None if unsupported

    Examples:
        >>> detect_language("src/main.py")
        'python'
        >>> detect_language("app.tsx")
        'tsx'
        >>> detect_language("unknown.txt")
        None
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    return EXTENSION_TO_LANGUAGE.get(ext)


def parse_file(content: str, language: str) -> Optional[Tree]:
    """Parse source code file using tree-sitter.

    This function parses the given source code content using tree-sitter's
    language-specific parser. It handles parser initialization and provides
    graceful error handling.

    Args:
        content: Source code content as a string
        language: Language name (must be in SUPPORTED_LANGUAGES)

    Returns:
        Tree-sitter Tree object representing the parsed AST, or None if parsing fails

    Raises:
        ValueError: If language is not supported
        RuntimeError: If tree-sitter is not available

    Examples:
        >>> code = "def hello():\n    print('world')"
        >>> tree = parse_file(code, "python")
        >>> tree.root_node.type
        'module'
    """
    if not TREE_SITTER_AVAILABLE:
        raise RuntimeError(
            "tree-sitter not available. Install with: pip install tree-sitter tree-sitter-languages"
        )

    if language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language: {language}. Supported: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    try:
        # Get language-specific parser
        parser = get_parser(language)

        # Parse content (tree-sitter expects bytes)
        content_bytes = content.encode('utf-8')
        tree = parser.parse(content_bytes)

        # Check for parse errors
        if tree.root_node.has_error:
            logger.warning(f"Parse errors detected in {language} code")
            # Still return the tree - partial parsing may be useful

        return tree

    except Exception as e:
        logger.error(f"Failed to parse {language} code: {e}")
        return None


def get_node_text(node: Any, source: bytes) -> str:
    """Extract text content from a tree-sitter node.

    Args:
        node: Tree-sitter Node object
        source: Original source code as bytes

    Returns:
        The text content of the node as a string
    """
    if not TREE_SITTER_AVAILABLE:
        return ""

    return source[node.start_byte:node.end_byte].decode('utf-8')


def is_available() -> bool:
    """Check if tree-sitter is available.

    Returns:
        True if tree-sitter and tree-sitter-languages are installed
    """
    return TREE_SITTER_AVAILABLE
