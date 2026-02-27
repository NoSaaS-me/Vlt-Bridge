"""Repository map generator following Aider pattern.

Generates condensed codebase structure maps with PageRank-based symbol importance.
This enables LLMs to navigate codebases efficiently without exhausting context.

Tasks: T031-T037 (Phase 4)
- T031: Symbol extraction from tree-sitter AST
- T032: Reference graph construction
- T033: PageRank/centrality calculation
- T034: Token-budgeted map generation
- T035: Map scope filtering
- T036: CLI command (handled in main.py)
- T037: Cache repo maps (handled by store.py)
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from collections import defaultdict

from tree_sitter import Tree, Node as TSNode

logger = logging.getLogger(__name__)


# ============================================================================
# Symbol Extraction (T031)
# ============================================================================

class Symbol:
    """Represents a top-level symbol (class, function, method) in the codebase."""

    def __init__(
        self,
        name: str,
        qualified_name: str,
        file_path: str,
        symbol_type: str,  # 'class', 'function', 'method'
        signature: Optional[str] = None,
        lineno: Optional[int] = None,
        end_line: Optional[int] = None,
        parent: Optional[str] = None,  # For methods: parent class name
        docstring: Optional[str] = None,
    ):
        self.name = name
        self.qualified_name = qualified_name
        self.file_path = file_path
        self.symbol_type = symbol_type
        self.signature = signature
        self.lineno = lineno
        self.end_line = end_line
        self.parent = parent
        self.docstring = docstring

    def __repr__(self):
        return f"Symbol({self.qualified_name}, {self.symbol_type})"


def extract_symbols_from_ast(
    tree: Tree,
    source_code: str,
    file_path: str,
    language: str
) -> List[Symbol]:
    """Extract top-level symbols (classes, functions, methods) from tree-sitter AST.

    Supports Python, TypeScript, JavaScript.
    Includes signatures with type annotations where available.

    Args:
        tree: Parsed tree-sitter AST
        source_code: Original source code
        file_path: Path to the source file
        language: Programming language ('python', 'typescript', 'javascript')

    Returns:
        List of Symbol objects
    """
    symbols: List[Symbol] = []

    if language == "python":
        symbols.extend(_extract_python_symbols(tree, source_code, file_path))
    elif language in ("typescript", "javascript"):
        symbols.extend(_extract_ts_js_symbols(tree, source_code, file_path, language))
    elif language == "go":
        symbols.extend(_extract_go_symbols(tree, source_code, file_path))

    return symbols


def _extract_python_symbols(tree: Tree, source_code: str, file_path: str) -> List[Symbol]:
    """Extract Python symbols with type annotations."""
    symbols: List[Symbol] = []
    root = tree.root_node
    module_name = _file_path_to_module_name(file_path)

    for node in _traverse(root):
        # Classes
        if node.type == "class_definition":
            name_node = _find_child_by_field(node, "name")
            if name_node:
                class_name = _get_text(name_node, source_code)
                qualified_name = f"{module_name}.{class_name}"

                # Get class signature with inheritance
                signature = _extract_python_class_signature(node, source_code)
                docstring = _extract_docstring(node, source_code)

                symbols.append(Symbol(
                    name=class_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    symbol_type='class',
                    signature=signature,
                    lineno=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=docstring
                ))

                # Extract methods within class
                for child in _traverse(node):
                    if child.type == "function_definition" and _is_direct_child_of_class(child, node):
                        method_name_node = _find_child_by_field(child, "name")
                        if method_name_node:
                            method_name = _get_text(method_name_node, source_code)
                            method_qualified = f"{qualified_name}.{method_name}"

                            # Get method signature with type hints
                            method_sig = _extract_python_function_signature(child, source_code)
                            method_doc = _extract_docstring(child, source_code)

                            symbols.append(Symbol(
                                name=method_name,
                                qualified_name=method_qualified,
                                file_path=file_path,
                                symbol_type='method',
                                signature=method_sig,
                                lineno=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                parent=class_name,
                                docstring=method_doc
                            ))

        # Top-level functions
        elif node.type == "function_definition" and node.parent.type == "module":
            name_node = _find_child_by_field(node, "name")
            if name_node:
                func_name = _get_text(name_node, source_code)
                qualified_name = f"{module_name}.{func_name}"

                signature = _extract_python_function_signature(node, source_code)
                docstring = _extract_docstring(node, source_code)

                symbols.append(Symbol(
                    name=func_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    symbol_type='function',
                    signature=signature,
                    lineno=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    docstring=docstring
                ))

    return symbols


def _extract_ts_js_symbols(tree: Tree, source_code: str, file_path: str, language: str) -> List[Symbol]:
    """Extract TypeScript/JavaScript symbols with type annotations."""
    symbols: List[Symbol] = []
    root = tree.root_node
    module_name = _file_path_to_module_name(file_path)

    for node in _traverse(root):
        # Classes
        if node.type == "class_declaration":
            name_node = _find_child_by_field(node, "name")
            if name_node:
                class_name = _get_text(name_node, source_code)
                qualified_name = f"{module_name}.{class_name}"

                signature = _extract_ts_class_signature(node, source_code)

                symbols.append(Symbol(
                    name=class_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    symbol_type='class',
                    signature=signature,
                    lineno=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))

                # Extract methods
                for child in node.children:
                    if child.type == "method_definition":
                        method_name_node = _find_child_by_field(child, "name")
                        if method_name_node:
                            method_name = _get_text(method_name_node, source_code)
                            method_qualified = f"{qualified_name}.{method_name}"

                            method_sig = _extract_ts_function_signature(child, source_code)

                            symbols.append(Symbol(
                                name=method_name,
                                qualified_name=method_qualified,
                                file_path=file_path,
                                symbol_type='method',
                                signature=method_sig,
                                lineno=child.start_point[0] + 1,
                                end_line=child.end_point[0] + 1,
                                parent=class_name
                            ))

        # Function declarations
        elif node.type == "function_declaration":
            name_node = _find_child_by_field(node, "name")
            if name_node:
                func_name = _get_text(name_node, source_code)
                qualified_name = f"{module_name}.{func_name}"

                signature = _extract_ts_function_signature(node, source_code)

                symbols.append(Symbol(
                    name=func_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    symbol_type='function',
                    signature=signature,
                    lineno=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))

    return symbols


def _extract_go_symbols(tree: Tree, source_code: str, file_path: str) -> List[Symbol]:
    """Extract Go symbols: functions, methods, and type declarations (structs/interfaces).

    Note: Go's tree-sitter grammar does not use field names, so we use
    _find_child_by_type() rather than _find_child_by_field().
    """
    symbols: List[Symbol] = []
    root = tree.root_node
    module_name = _file_path_to_module_name(file_path)

    for node in _traverse(root):
        # Top-level function declarations: func Name(params) result { ... }
        if node.type == "function_declaration":
            # Name is an 'identifier' child (no field name in Go grammar)
            name_node = _find_child_by_type(node, "identifier")
            if name_node:
                func_name = _get_text(name_node, source_code)
                qualified_name = f"{module_name}.{func_name}"
                signature = _extract_go_function_signature(node, source_code)
                symbols.append(Symbol(
                    name=func_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    symbol_type='function',
                    signature=signature,
                    lineno=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                ))

        # Method declarations: func (recv T) Name(params) result { ... }
        elif node.type == "method_declaration":
            # Method name is a 'field_identifier' child
            name_node = _find_child_by_type(node, "field_identifier")
            if name_node:
                method_name = _get_text(name_node, source_code)
                # Receiver is the first 'parameter_list' child
                receiver_node = _find_child_by_type(node, "parameter_list")
                receiver_type = _extract_go_receiver_type(receiver_node, source_code) if receiver_node else None
                if receiver_type:
                    qualified_name = f"{module_name}.{receiver_type}.{method_name}"
                else:
                    qualified_name = f"{module_name}.{method_name}"
                signature = _extract_go_method_signature(node, source_code)
                symbols.append(Symbol(
                    name=method_name,
                    qualified_name=qualified_name,
                    file_path=file_path,
                    symbol_type='method',
                    signature=signature,
                    lineno=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parent=receiver_type,
                ))

        # Type declarations: type Name struct/interface/alias
        elif node.type == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    # Name is a 'type_identifier' child of type_spec
                    name_node = _find_child_by_type(child, "type_identifier")
                    if name_node:
                        type_name = _get_text(name_node, source_code)
                        qualified_name = f"{module_name}.{type_name}"
                        # Determine struct vs interface vs alias
                        struct_node = _find_child_by_type(child, "struct_type")
                        iface_node = _find_child_by_type(child, "interface_type")
                        if struct_node:
                            sig = f"type {type_name} struct"
                        elif iface_node:
                            sig = f"type {type_name} interface"
                        else:
                            sig = f"type {type_name}"
                        symbols.append(Symbol(
                            name=type_name,
                            qualified_name=qualified_name,
                            file_path=file_path,
                            symbol_type='class',  # Map Go types to 'class' for display
                            signature=sig,
                            lineno=node.start_point[0] + 1,
                            end_line=node.end_point[0] + 1,
                        ))

    return symbols


def _extract_go_function_signature(node: TSNode, source_code: str) -> str:
    """Extract Go function signature: func Name(params) result."""
    name_node = _find_child_by_type(node, "identifier")
    if not name_node:
        return "func <unknown>()"
    func_name = _get_text(name_node, source_code)
    # Parameters: first (and only) parameter_list
    params_node = _find_child_by_type(node, "parameter_list")
    params_text = _get_text(params_node, source_code) if params_node else "()"
    # Result: everything between params and block
    result_text = _extract_go_result(node, source_code, after_node=params_node)
    if result_text:
        return f"func {func_name}{params_text} {result_text}"
    return f"func {func_name}{params_text}"


def _extract_go_method_signature(node: TSNode, source_code: str) -> str:
    """Extract Go method signature including receiver: func (r T) Name(params) result."""
    name_node = _find_child_by_type(node, "field_identifier")
    if not name_node:
        return "func <unknown>()"
    method_name = _get_text(name_node, source_code)
    # First parameter_list is receiver, second is params
    param_lists = [c for c in node.children if c.type == "parameter_list"]
    receiver_text = _get_text(param_lists[0], source_code) if param_lists else ""
    params_node = param_lists[1] if len(param_lists) > 1 else None
    params_text = _get_text(params_node, source_code) if params_node else "()"
    result_text = _extract_go_result(node, source_code, after_node=params_node)
    if result_text:
        return f"func {receiver_text} {method_name}{params_text} {result_text}"
    return f"func {receiver_text} {method_name}{params_text}"


def _extract_go_result(node: TSNode, source_code: str, after_node: Optional[TSNode]) -> Optional[str]:
    """Extract Go function result type (everything between the last params and block)."""
    found = after_node is None
    for child in node.children:
        if not found:
            if child == after_node:
                found = True
            continue
        # Stop at the block
        if child.type == "block":
            break
        # Result type: named node that isn't another parameter_list
        if child.is_named and child.type != "parameter_list":
            return _get_text(child, source_code).strip()
    return None


def _extract_go_receiver_type(receiver_node: TSNode, source_code: str) -> Optional[str]:
    """Extract the bare receiver type name from a Go method receiver (e.g., `(s *Foo)` → `Foo`)."""
    for child in _traverse(receiver_node):
        if child.type == "type_identifier":
            return _get_text(child, source_code)
    return None


# ============================================================================
# Reference Graph Construction (T032)
# ============================================================================

class ReferenceGraph:
    """Graph of who-calls-who and who-imports-what relationships."""

    def __init__(self):
        self.nodes: Set[str] = set()  # Qualified names
        self.edges: List[Tuple[str, str]] = []  # (source, target) tuples
        self.adjacency_list: Dict[str, List[str]] = defaultdict(list)
        self.reverse_adjacency_list: Dict[str, List[str]] = defaultdict(list)

    def add_node(self, qualified_name: str):
        """Add a node (symbol) to the graph."""
        self.nodes.add(qualified_name)

    def add_edge(self, source: str, target: str):
        """Add a directed edge (source calls/imports target)."""
        self.edges.append((source, target))
        self.adjacency_list[source].append(target)
        self.reverse_adjacency_list[target].append(source)

    def get_callers(self, symbol: str) -> List[str]:
        """Get all symbols that call/import this symbol."""
        return self.reverse_adjacency_list.get(symbol, [])

    def get_callees(self, symbol: str) -> List[str]:
        """Get all symbols that this symbol calls/imports."""
        return self.adjacency_list.get(symbol, [])


def build_reference_graph(
    symbols: List[Symbol],
    edges: List[Tuple[str, str]]
) -> ReferenceGraph:
    """Build reference graph from symbols and edges.

    Args:
        symbols: List of extracted symbols
        edges: List of (source_qualified_name, target_qualified_name) tuples from graph builder

    Returns:
        ReferenceGraph with adjacency lists
    """
    graph = ReferenceGraph()

    # Add all symbols as nodes
    for symbol in symbols:
        graph.add_node(symbol.qualified_name)

    # Add all edges
    for source, target in edges:
        # Only add edges where both nodes exist
        if source in graph.nodes and target in graph.nodes:
            graph.add_edge(source, target)

    logger.debug(f"Built reference graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph


# ============================================================================
# PageRank/Centrality Calculation (T033)
# ============================================================================

def calculate_centrality(graph: ReferenceGraph, max_iterations: int = 100, damping: float = 0.85) -> Dict[str, float]:
    """Calculate PageRank centrality scores for each symbol in the graph.

    Higher scores indicate more referenced/important symbols.
    Uses simple PageRank algorithm without external dependencies.

    Args:
        graph: Reference graph with adjacency lists
        max_iterations: Maximum iterations for convergence
        damping: Damping factor (default 0.85, standard PageRank value)

    Returns:
        Dict mapping qualified names to centrality scores
    """
    if not graph.nodes:
        return {}

    # Initialize scores
    num_nodes = len(graph.nodes)
    scores = {node: 1.0 / num_nodes for node in graph.nodes}

    # Iterative PageRank
    for iteration in range(max_iterations):
        new_scores = {}
        max_diff = 0.0

        for node in graph.nodes:
            # PageRank formula: PR(A) = (1-d)/N + d * sum(PR(Ti)/C(Ti))
            # where Ti are nodes that link to A, C(Ti) is out-degree of Ti
            rank_sum = 0.0

            for caller in graph.get_callers(node):
                out_degree = len(graph.get_callees(caller))
                if out_degree > 0:
                    rank_sum += scores[caller] / out_degree

            new_score = (1 - damping) / num_nodes + damping * rank_sum
            new_scores[node] = new_score

            # Track convergence
            max_diff = max(max_diff, abs(new_score - scores[node]))

        scores = new_scores

        # Check convergence
        if max_diff < 1e-6:
            logger.debug(f"PageRank converged after {iteration + 1} iterations")
            break

    return scores


# ============================================================================
# Token-Budgeted Map Generation (T034)
# ============================================================================

def generate_repo_map(
    symbols: List[Symbol],
    graph: ReferenceGraph,
    centrality_scores: Dict[str, float],
    max_tokens: int = 4000,
    scope: Optional[str] = None,
    include_signatures: bool = True,
    include_docstrings: bool = False
) -> Dict[str, Any]:
    """Generate Aider-style repository map with token budget.

    Prunes low-centrality symbols to fit within token budget.

    Args:
        symbols: List of extracted symbols
        graph: Reference graph
        centrality_scores: PageRank scores for each symbol
        max_tokens: Maximum tokens for the map (default 4000)
        scope: Optional subdirectory to filter by (e.g., "src/api/")
        include_signatures: Include function/method signatures
        include_docstrings: Include docstrings in map (uses more tokens)

    Returns:
        Dict with:
            - map_text: Generated map string
            - token_count: Actual token count
            - files_included: Number of files
            - symbols_included: Number of symbols included
            - symbols_total: Total symbols before pruning
            - max_tokens: Budget used
            - scope: Scope filter applied
    """
    # Filter symbols by scope if specified
    filtered_symbols = symbols
    if scope:
        filtered_symbols = [s for s in symbols if s.file_path.startswith(scope)]

    if not filtered_symbols:
        logger.warning(f"No symbols found in scope '{scope}'")
        return {
            "map_text": "# No symbols found\n",
            "token_count": 0,
            "files_included": 0,
            "symbols_included": 0,
            "symbols_total": len(symbols),
            "max_tokens": max_tokens,
            "scope": scope
        }

    # Sort symbols by centrality (highest first)
    sorted_symbols = sorted(
        filtered_symbols,
        key=lambda s: centrality_scores.get(s.qualified_name, 0.0),
        reverse=True
    )

    # Group by file
    files_map: Dict[str, List[Symbol]] = defaultdict(list)
    for symbol in sorted_symbols:
        files_map[symbol.file_path].append(symbol)

    # Generate map text with token budget
    lines = []
    included_symbols = 0
    included_files = set()

    # Build tree structure
    for file_path in sorted(files_map.keys()):
        file_line = f"├── {file_path}"

        # Estimate tokens (rough: 1 token ≈ 4 characters)
        if _estimate_tokens('\n'.join(lines + [file_line])) > max_tokens:
            logger.debug(f"Reached token budget, stopping at {len(included_files)} files")
            break

        lines.append(file_line)
        included_files.add(file_path)

        # Sort symbols within file by centrality
        file_symbols = sorted(
            files_map[file_path],
            key=lambda s: centrality_scores.get(s.qualified_name, 0.0),
            reverse=True
        )

        # Add symbols with indentation
        for symbol in file_symbols:
            symbol_lines = _format_symbol(
                symbol,
                centrality_scores.get(symbol.qualified_name, 0.0),
                include_signatures=include_signatures,
                include_docstrings=include_docstrings
            )

            # Check token budget before adding
            test_text = '\n'.join(lines + symbol_lines)
            if _estimate_tokens(test_text) > max_tokens:
                logger.debug(f"Token budget reached at {included_symbols} symbols")
                break

            lines.extend(symbol_lines)
            included_symbols += 1

    map_text = '\n'.join(lines)
    token_count = _estimate_tokens(map_text)

    logger.info(
        f"Generated repo map: {included_symbols}/{len(filtered_symbols)} symbols, "
        f"{len(included_files)} files, {token_count}/{max_tokens} tokens"
    )

    return {
        "map_text": map_text,
        "token_count": token_count,
        "files_included": len(included_files),
        "symbols_included": included_symbols,
        "symbols_total": len(filtered_symbols),
        "max_tokens": max_tokens,
        "scope": scope
    }


def _format_symbol(
    symbol: Symbol,
    centrality: float,
    include_signatures: bool = True,
    include_docstrings: bool = False
) -> List[str]:
    """Format a symbol for display in the repo map.

    Returns list of lines (may be multiple if including docstring).
    """
    lines = []

    # Indentation based on symbol type
    if symbol.symbol_type == 'class':
        indent = "│   ├── "
        if include_signatures and symbol.signature:
            lines.append(f"{indent}{symbol.signature}")
        else:
            lines.append(f"{indent}class {symbol.name}")

    elif symbol.symbol_type == 'method':
        indent = "│   │   ├── "
        if include_signatures and symbol.signature:
            lines.append(f"{indent}{symbol.signature}")
        else:
            lines.append(f"{indent}{symbol.name}()")

    else:  # function
        indent = "│   ├── "
        if include_signatures and symbol.signature:
            lines.append(f"{indent}{symbol.signature}")
        else:
            lines.append(f"{indent}{symbol.name}()")

    # Optionally include docstring (expensive in tokens)
    if include_docstrings and symbol.docstring:
        doc_indent = "│   │       "
        # Truncate long docstrings
        doc = symbol.docstring.split('\n')[0][:80]
        lines.append(f"{doc_indent}# {doc}")

    return lines


def _estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses rough heuristic: 1 token ≈ 4 characters.
    This is conservative for code (which typically has more tokens per char).
    """
    return len(text) // 4


# ============================================================================
# Map Scope Filtering (T035)
# ============================================================================

def filter_symbols_by_scope(symbols: List[Symbol], scope: str) -> List[Symbol]:
    """Filter symbols to only those in a specific subdirectory.

    Args:
        symbols: List of all symbols
        scope: Subdirectory path (e.g., "src/api/")

    Returns:
        Filtered list of symbols
    """
    return [s for s in symbols if s.file_path.startswith(scope)]


# ============================================================================
# Helper Functions
# ============================================================================

def _traverse(node: TSNode) -> List[TSNode]:
    """Depth-first traversal of tree-sitter AST."""
    nodes = [node]
    for child in node.children:
        nodes.extend(_traverse(child))
    return nodes


def _get_text(node: TSNode, source_code: str) -> str:
    """Extract text for a tree-sitter node."""
    return source_code[node.start_byte:node.end_byte]


def _find_child_by_field(node: TSNode, field_name: str) -> Optional[TSNode]:
    """Find child node by field name."""
    return node.child_by_field_name(field_name)


def _find_child_by_type(node: TSNode, node_type: str) -> Optional[TSNode]:
    """Find first child node of given type."""
    for child in node.children:
        if child.type == node_type:
            return child
    return None


def _extract_docstring(node: TSNode, source_code: str) -> Optional[str]:
    """Extract docstring from Python function/class."""
    for child in node.children:
        if child.type == "block":
            # First statement in block
            for stmt in child.children:
                if stmt.type == "expression_statement":
                    for expr in stmt.children:
                        if expr.type == "string":
                            text = _get_text(expr, source_code)
                            # Remove quotes and clean up
                            text = text.strip('"').strip("'").strip()
                            return text
    return None


def _extract_python_class_signature(node: TSNode, source_code: str) -> str:
    """Extract Python class signature including inheritance.

    Examples:
        class Foo
        class Foo(Bar)
        class Foo(Bar, Baz)
    """
    name_node = _find_child_by_field(node, "name")
    if not name_node:
        return "class <unknown>"

    class_name = _get_text(name_node, source_code)

    # Check for base classes
    bases_node = _find_child_by_type(node, "argument_list")
    if bases_node:
        bases_text = _get_text(bases_node, source_code)
        return f"class {class_name}{bases_text}"

    return f"class {class_name}"


def _extract_python_function_signature(node: TSNode, source_code: str) -> str:
    """Extract Python function signature with type hints.

    Examples:
        def foo(x: int, y: str) -> bool
        def bar(a, b)
    """
    name_node = _find_child_by_field(node, "name")
    if not name_node:
        return "def <unknown>()"

    func_name = _get_text(name_node, source_code)

    # Get parameters
    params_node = _find_child_by_field(node, "parameters")
    params_text = _get_text(params_node, source_code) if params_node else "()"

    # Get return type
    return_type_node = _find_child_by_field(node, "return_type")
    if return_type_node:
        return_type = _get_text(return_type_node, source_code)
        # remove the arrow
        return_type = return_type.replace("->", "").strip()
        return f"def {func_name}{params_text} → {return_type}"

    return f"def {func_name}{params_text}"


def _extract_ts_class_signature(node: TSNode, source_code: str) -> str:
    """Extract TypeScript class signature including extends/implements."""
    name_node = _find_child_by_field(node, "name")
    if not name_node:
        return "class <unknown>"

    class_name = _get_text(name_node, source_code)

    # Check for extends/implements
    heritage_node = _find_child_by_type(node, "class_heritage")
    if heritage_node:
        heritage_text = _get_text(heritage_node, source_code)
        return f"class {class_name} {heritage_text}"

    return f"class {class_name}"


def _extract_ts_function_signature(node: TSNode, source_code: str) -> str:
    """Extract TypeScript function signature with types."""
    name_node = _find_child_by_field(node, "name")
    if not name_node:
        return "function <unknown>()"

    func_name = _get_text(name_node, source_code)

    # Get parameters
    params_node = _find_child_by_field(node, "parameters")
    params_text = _get_text(params_node, source_code) if params_node else "()"

    # Get return type
    return_type_node = _find_child_by_field(node, "return_type")
    if return_type_node:
        return_type = _get_text(return_type_node, source_code)
        return_type = return_type.replace(":", "").strip()
        return f"{func_name}{params_text} → {return_type}"

    return f"{func_name}{params_text}"


def _file_path_to_module_name(file_path: str) -> str:
    """Convert file path to module name.

    Examples:
        src/api/routes.py -> src.api.routes
        lib/utils.ts -> lib.utils
    """
    # Remove extension
    path = file_path
    for ext in ['.py', '.ts', '.js', '.tsx', '.jsx', '.go']:
        if path.endswith(ext):
            path = path[:-len(ext)]
            break

    # Replace slashes with dots
    return path.replace('/', '.').replace('\\', '.').lstrip('.')


def _is_direct_child_of_class(method_node: TSNode, class_node: TSNode) -> bool:
    """Check if a method is a direct child of a class (not nested)."""
    # Walk up to find first class parent
    current = method_node.parent
    while current and current.type != "module":
        if current.type in ("class_definition", "class_declaration"):
            return current == class_node
        current = current.parent
    return False
