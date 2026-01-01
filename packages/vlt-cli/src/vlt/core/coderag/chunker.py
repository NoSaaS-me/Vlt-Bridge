"""LlamaIndex CodeSplitter integration with context enrichment.

This module implements semantic code chunking using LlamaIndex's CodeSplitter
(Sweep's chunker) with tree-sitter backend. Each chunk is enriched with context
(imports, class definitions, signatures, docstrings) to make it self-contained.

Based on research from:
- Sweep's chunker (LlamaIndex CodeSplitter)
- Qodo's context enrichment pattern
- Section 1 and 10 of research.md
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import logging

from .parser import parse_file, detect_language, get_node_text, is_available as parser_available

logger = logging.getLogger(__name__)

# Optional dependencies - graceful degradation
try:
    from llama_index.core.node_parser import CodeSplitter
    from llama_index.core.schema import Document
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    logger.warning("LlamaIndex not available. Install with: pip install llama-index-core")


# Chunking configuration (from research.md Section 1)
DEFAULT_CHUNK_LINES = 40
DEFAULT_CHUNK_OVERLAP = 15
DEFAULT_MAX_CHARS = 1500


def chunk_file(
    content: str,
    language: str,
    file_path: str,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    chunk_lines_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> List[Dict[str, Any]]:
    """Chunk source code file using LlamaIndex CodeSplitter with context enrichment.

    This function splits source code into semantic chunks at function/class boundaries,
    then enriches each chunk with contextual information (imports, class context,
    signatures, docstrings) to make each chunk self-contained.

    Args:
        content: Source code content as a string
        language: Programming language (e.g., 'python', 'typescript')
        file_path: Path to the source file (for metadata)
        chunk_lines: Target lines per chunk (default: 40)
        chunk_lines_overlap: Lines of overlap between chunks (default: 15)
        max_chars: Maximum characters per chunk (default: 1500)

    Returns:
        List of chunk dictionaries with the following structure:
        {
            'file_path': str,
            'chunk_type': str,  # e.g., 'function', 'method', 'class'
            'qualified_name': str,  # e.g., 'ClassName.method_name'
            'imports': str,  # Import statements from file
            'class_context': str,  # Enclosing class header (for methods)
            'signature': str,  # Full function/method signature
            'decorators': List[str],  # Decorator lines
            'docstring': str,  # Documentation string
            'body': str,  # Function/method body
            'lineno': int,  # Start line number
            'end_lineno': int,  # End line number
            'language': str,
            'chunk_text': str,  # Full enriched chunk text for embedding
        }

    Raises:
        RuntimeError: If LlamaIndex or tree-sitter is not available
        ValueError: If language is not supported

    Examples:
        >>> code = '''
        ... def hello(name: str) -> str:
        ...     \"\"\"Greet someone.\"\"\"
        ...     return f"Hello, {name}"
        ... '''
        >>> chunks = chunk_file(code, 'python', 'example.py')
        >>> len(chunks)
        1
        >>> chunks[0]['qualified_name']
        'hello'
    """
    if not LLAMA_INDEX_AVAILABLE:
        raise RuntimeError(
            "LlamaIndex not available. Install with: pip install llama-index-core"
        )

    if not parser_available():
        raise RuntimeError(
            "tree-sitter not available. Install with: pip install tree-sitter tree-sitter-languages"
        )

    try:
        # Step 1: Parse file with tree-sitter to extract context
        tree = parse_file(content, language)
        if tree is None:
            logger.warning(f"Failed to parse {file_path}, returning empty chunks")
            return []

        content_bytes = content.encode('utf-8')

        # Step 2: Extract file-level context
        imports = _extract_imports(tree.root_node, content_bytes, language)

        # Step 3: Use LlamaIndex CodeSplitter for semantic chunking
        splitter = CodeSplitter(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )

        # Convert to LlamaIndex Document format
        doc = Document(text=content, metadata={'file_path': file_path})
        nodes = splitter.get_nodes_from_documents([doc])

        # Step 4: Enrich each chunk with context
        enriched_chunks = []
        for node in nodes:
            chunk_dict = _enrich_chunk(
                chunk_text=node.text,
                tree=tree,
                content_bytes=content_bytes,
                file_path=file_path,
                language=language,
                imports=imports,
            )
            enriched_chunks.append(chunk_dict)

        logger.info(f"Chunked {file_path}: {len(enriched_chunks)} chunks created")
        return enriched_chunks

    except Exception as e:
        logger.error(f"Error chunking {file_path}: {e}")
        return []


def _extract_imports(root_node: Any, source: bytes, language: str) -> str:
    """Extract import statements from the file.

    Args:
        root_node: Tree-sitter root node
        source: Source code as bytes
        language: Programming language

    Returns:
        Concatenated import statements
    """
    imports = []

    # Query patterns for different languages
    if language == 'python':
        # Find import_statement and import_from_statement nodes
        for child in root_node.children:
            if child.type in ('import_statement', 'import_from_statement'):
                imports.append(get_node_text(child, source))

    elif language in ('typescript', 'tsx', 'javascript'):
        # Find import_statement nodes
        for child in root_node.children:
            if child.type == 'import_statement':
                imports.append(get_node_text(child, source))

    elif language == 'go':
        # Find import_declaration nodes
        for child in root_node.children:
            if child.type == 'import_declaration':
                imports.append(get_node_text(child, source))

    elif language == 'rust':
        # Find use_declaration nodes
        for child in root_node.children:
            if child.type == 'use_declaration':
                imports.append(get_node_text(child, source))

    return '\n'.join(imports)


def _enrich_chunk(
    chunk_text: str,
    tree: Any,
    content_bytes: bytes,
    file_path: str,
    language: str,
    imports: str,
) -> Dict[str, Any]:
    """Enrich a code chunk with contextual information.

    Extracts and adds context like class definitions, function signatures,
    decorators, and docstrings to make the chunk self-contained.

    Args:
        chunk_text: The raw chunk text from CodeSplitter
        tree: Tree-sitter Tree object
        content_bytes: Source code as bytes
        file_path: Path to source file
        language: Programming language
        imports: Pre-extracted import statements

    Returns:
        Dictionary with enriched chunk information
    """
    # Find the node in the tree that corresponds to this chunk
    # This is a simplified approach - in production, you'd want more sophisticated matching
    chunk_lines = chunk_text.strip().split('\n')
    first_line = chunk_lines[0] if chunk_lines else ""

    # Default chunk structure
    chunk_dict = {
        'file_path': file_path,
        'chunk_type': 'code',
        'qualified_name': _extract_qualified_name(first_line, language),
        'imports': imports,
        'class_context': '',
        'signature': '',
        'decorators': [],
        'docstring': '',
        'body': chunk_text,
        'lineno': 1,  # Will be updated if we can find the node
        'end_lineno': len(chunk_text.split('\n')),
        'language': language,
        'chunk_text': '',  # Will be assembled below
    }

    # Try to find and analyze the corresponding AST node
    node = _find_chunk_node(tree.root_node, chunk_text, content_bytes, language)
    if node:
        _enhance_from_node(chunk_dict, node, content_bytes, language)

    # Assemble full enriched chunk text for embedding
    chunk_dict['chunk_text'] = _assemble_chunk_text(chunk_dict)

    return chunk_dict


def _find_chunk_node(root_node: Any, chunk_text: str, source: bytes, language: str) -> Optional[Any]:
    """Find the AST node corresponding to a chunk.

    This is a simplified implementation that searches for nodes containing
    the chunk text. A production version would use more sophisticated matching.

    Args:
        root_node: Tree-sitter root node
        chunk_text: The chunk text to find
        source: Source code as bytes
        language: Programming language

    Returns:
        The matching node or None
    """
    chunk_start = chunk_text.strip()[:50]  # Use first 50 chars as signature

    # Define function/class node types per language
    target_types = {
        'python': ('function_definition', 'class_definition'),
        'typescript': ('function_declaration', 'method_definition', 'class_declaration', 'arrow_function'),
        'tsx': ('function_declaration', 'method_definition', 'class_declaration', 'arrow_function'),
        'javascript': ('function_declaration', 'method_definition', 'class_declaration', 'arrow_function'),
        'go': ('function_declaration', 'method_declaration', 'type_declaration'),
        'rust': ('function_item', 'impl_item', 'struct_item', 'enum_item', 'trait_item'),
        # Systems languages
        'c': ('function_definition', 'struct_specifier'),
        'cpp': ('function_definition', 'class_specifier', 'struct_specifier', 'namespace_definition'),
        # JVM languages
        'java': ('method_declaration', 'class_declaration', 'interface_declaration', 'enum_declaration'),
        'kotlin': ('function_declaration', 'class_declaration', 'object_declaration'),
        'scala': ('function_definition', 'class_definition', 'object_definition', 'trait_definition'),
        # .NET
        'c_sharp': ('method_declaration', 'class_declaration', 'interface_declaration', 'struct_declaration'),
        # Ruby
        'ruby': ('method', 'class', 'module', 'singleton_method'),
        # PHP
        'php': ('function_definition', 'method_declaration', 'class_declaration', 'interface_declaration'),
        # Swift/Objective-C
        'swift': ('function_declaration', 'class_declaration', 'struct_declaration', 'protocol_declaration'),
        'objc': ('function_definition', 'method_definition', 'class_interface', 'protocol_declaration'),
        # Shell (functions only)
        'bash': ('function_definition',),
        # Elixir/Erlang
        'elixir': ('call',),  # def/defp are calls in elixir grammar
        'erlang': ('function_clause',),
        # Haskell
        'haskell': ('function', 'type_declaration'),
        # Lua
        'lua': ('function_declaration', 'local_function_declaration'),
        # R
        'r': ('function_definition',),
    }

    node_types = target_types.get(language, ())

    def traverse(node):
        """Recursively traverse tree to find matching node."""
        if node.type in node_types:
            node_text = get_node_text(node, source).strip()
            if chunk_start in node_text:
                return node

        for child in node.children:
            result = traverse(child)
            if result:
                return result

        return None

    return traverse(root_node)


def _enhance_from_node(chunk_dict: Dict[str, Any], node: Any, source: bytes, language: str) -> None:
    """Enhance chunk dictionary with information extracted from AST node.

    Args:
        chunk_dict: Chunk dictionary to enhance (modified in-place)
        node: Tree-sitter node
        source: Source code as bytes
        language: Programming language
    """
    # Update line numbers
    chunk_dict['lineno'] = node.start_point[0] + 1
    chunk_dict['end_lineno'] = node.end_point[0] + 1

    # Extract chunk type
    chunk_dict['chunk_type'] = _get_chunk_type(node.type, language)

    # Extract decorators (Python)
    if language == 'python':
        decorators = _extract_decorators(node, source)
        chunk_dict['decorators'] = decorators

    # Extract signature
    signature = _extract_signature(node, source, language)
    if signature:
        chunk_dict['signature'] = signature

    # Extract docstring
    docstring = _extract_docstring(node, source, language)
    if docstring:
        chunk_dict['docstring'] = docstring

    # Extract class context (if this is a method)
    class_context = _extract_class_context(node, source, language)
    if class_context:
        chunk_dict['class_context'] = class_context


def _get_chunk_type(node_type: str, language: str) -> str:
    """Map tree-sitter node type to chunk type.

    Args:
        node_type: Tree-sitter node type
        language: Programming language

    Returns:
        Chunk type string ('function', 'method', 'class', etc.)
    """
    type_map = {
        # Functions
        'function_definition': 'function',
        'function_declaration': 'function',
        'function_item': 'function',
        'local_function_declaration': 'function',
        'arrow_function': 'function',
        'function_clause': 'function',
        'function': 'function',
        # Methods
        'method_definition': 'method',
        'method_declaration': 'method',
        'method': 'method',
        'singleton_method': 'method',
        # Classes
        'class_definition': 'class',
        'class_declaration': 'class',
        'class_specifier': 'class',
        'class_interface': 'class',
        # Structs/Types
        'struct_specifier': 'struct',
        'struct_declaration': 'struct',
        'struct_item': 'struct',
        'type_declaration': 'type',
        # Interfaces/Protocols/Traits
        'interface_declaration': 'interface',
        'protocol_declaration': 'protocol',
        'trait_item': 'trait',
        'trait_definition': 'trait',
        # Enums
        'enum_declaration': 'enum',
        'enum_item': 'enum',
        # Modules/Namespaces
        'module': 'module',
        'namespace_definition': 'namespace',
        'object_declaration': 'object',
        'object_definition': 'object',
        # Rust-specific
        'impl_item': 'impl',
        # Elixir - def/defp are parsed as calls
        'call': 'function',
    }
    return type_map.get(node_type, 'code')


def _extract_decorators(node: Any, source: bytes) -> List[str]:
    """Extract decorator lines (Python only).

    Args:
        node: Tree-sitter node
        source: Source code as bytes

    Returns:
        List of decorator strings
    """
    decorators = []
    for child in node.children:
        if child.type == 'decorator':
            decorators.append(get_node_text(child, source))
    return decorators


def _extract_signature(node: Any, source: bytes, language: str) -> str:
    """Extract function/method signature.

    Args:
        node: Tree-sitter node
        source: Source code as bytes
        language: Programming language

    Returns:
        Signature string
    """
    # For functions/methods, extract the first line (signature)
    if language == 'python':
        # Find the identifier and parameters
        for child in node.children:
            if child.type == 'identifier':
                name = get_node_text(child, source)
            elif child.type == 'parameters':
                params = get_node_text(child, source)
                # Also look for return type
                return_type = ''
                for sibling in node.children:
                    if sibling.type == 'type':
                        return_type = f" -> {get_node_text(sibling, source)}"
                return f"def {name}{params}{return_type}:"

    # For other languages, use the first line
    text = get_node_text(node, source)
    first_line = text.split('\n')[0]
    return first_line


def _extract_docstring(node: Any, source: bytes, language: str) -> str:
    """Extract documentation string.

    Args:
        node: Tree-sitter node
        source: Source code as bytes
        language: Programming language

    Returns:
        Docstring text
    """
    if language == 'python':
        # In Python, docstring is usually the first expression_statement containing a string
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type == 'expression_statement':
                        for expr_child in stmt.children:
                            if expr_child.type == 'string':
                                return get_node_text(expr_child, source).strip('"\'')
                        break
                break

    # For other languages, look for comment blocks
    # This is a simplified approach
    return ''


def _extract_class_context(node: Any, source: bytes, language: str) -> str:
    """Extract enclosing class header (for methods).

    Args:
        node: Tree-sitter node
        source: Source code as bytes
        language: Programming language

    Returns:
        Class context string
    """
    # Walk up the tree to find parent class
    parent = node.parent
    while parent:
        if language == 'python' and parent.type == 'class_definition':
            # Extract class header (name and inheritance)
            class_text = get_node_text(parent, source)
            # Get just the class declaration line
            class_header = class_text.split('\n')[0]
            # Also get docstring if available
            docstring = _extract_docstring(parent, source, language)
            if docstring:
                return f"{class_header}\n    \"\"\"{docstring}\"\"\""
            return class_header

        elif language in ('typescript', 'tsx', 'javascript') and parent.type == 'class_declaration':
            class_text = get_node_text(parent, source)
            class_header = class_text.split('\n')[0]
            return class_header

        parent = parent.parent

    return ''


def _extract_qualified_name(first_line: str, language: str) -> str:
    """Extract qualified name from chunk's first line.

    Args:
        first_line: First line of chunk
        language: Programming language

    Returns:
        Qualified name (e.g., 'ClassName.method_name')
    """
    # Simplified extraction - just grab the function/class name
    # A production version would use proper AST analysis
    if language == 'python':
        if 'def ' in first_line:
            name = first_line.split('def ')[1].split('(')[0].strip()
            return name
        elif 'class ' in first_line:
            name = first_line.split('class ')[1].split('(')[0].split(':')[0].strip()
            return name

    # Generic fallback
    return first_line.strip()[:50]


def _assemble_chunk_text(chunk_dict: Dict[str, Any]) -> str:
    """Assemble the full enriched chunk text for embedding.

    This combines all context elements into a single text suitable for
    generating embeddings, following the pattern from research.md.

    Args:
        chunk_dict: Chunk dictionary with context fields

    Returns:
        Assembled chunk text
    """
    parts = []

    # Add imports
    if chunk_dict['imports']:
        parts.append(f"# Imports:\n{chunk_dict['imports']}")

    # Add class context
    if chunk_dict['class_context']:
        parts.append(f"# Class context:\n{chunk_dict['class_context']}")

    # Add decorators
    if chunk_dict['decorators']:
        parts.append('\n'.join(chunk_dict['decorators']))

    # Add signature
    if chunk_dict['signature']:
        parts.append(chunk_dict['signature'])

    # Add docstring
    if chunk_dict['docstring']:
        parts.append(f'    """{chunk_dict["docstring"]}"""')

    # Add body
    if chunk_dict['body']:
        parts.append(chunk_dict['body'])

    return '\n\n'.join(parts)


def is_available() -> bool:
    """Check if chunker dependencies are available.

    Returns:
        True if both LlamaIndex and tree-sitter are installed
    """
    return LLAMA_INDEX_AVAILABLE and parser_available()
