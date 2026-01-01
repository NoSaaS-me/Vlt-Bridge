"""Tests for coderag graph builder module."""

import pytest

# Skip all tests if tree-sitter is not available
pytest.importorskip("tree_sitter")
# tree-sitter-language-pack is the maintained replacement for tree-sitter-languages
pytest.importorskip("tree_sitter_language_pack")

from tree_sitter_language_pack import get_parser
from vlt.core.coderag.graph import (
    build_graph,
    extract_nodes,
    extract_edges,
    CodeNode,
    CodeEdge,
    _get_text,
    _traverse,
)
from vlt.core.models import NodeType, EdgeType


@pytest.fixture
def python_parser():
    """Get Python tree-sitter parser."""
    return get_parser("python")


@pytest.fixture
def typescript_parser():
    """Get TypeScript tree-sitter parser."""
    return get_parser("typescript")


def test_build_graph_empty():
    """Test graph building with no files."""
    nodes, edges = build_graph({}, project_id="test-project")
    assert nodes == []
    assert edges == []


def test_extract_python_class(python_parser):
    """Test extracting Python class node."""
    source = """
class UserService:
    '''User management service.'''
    def __init__(self):
        pass
"""

    tree = python_parser.parse(bytes(source, "utf8"))
    nodes = extract_nodes(tree, source, "python", "src/user.py", "test-project")

    # Should find class
    class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS.value]
    assert len(class_nodes) == 1

    user_class = class_nodes[0]
    assert user_class.name == "UserService"
    assert "src.user.UserService" in user_class.id
    assert user_class.signature == "class UserService"
    assert "User management service" in user_class.docstring


def test_extract_python_function(python_parser):
    """Test extracting Python function node."""
    source = """
def authenticate_user(username, password):
    '''Authenticate a user.'''
    return True
"""

    tree = python_parser.parse(bytes(source, "utf8"))
    nodes = extract_nodes(tree, source, "python", "auth.py", "test-project")

    func_nodes = [n for n in nodes if n.node_type == NodeType.FUNCTION.value]
    assert len(func_nodes) == 1

    func = func_nodes[0]
    assert func.name == "authenticate_user"
    assert "username, password" in func.signature
    assert func.lineno > 0


def test_extract_python_method(python_parser):
    """Test extracting Python method node."""
    source = """
class UserService:
    def create_user(self, email):
        '''Create a new user.'''
        pass
"""

    tree = python_parser.parse(bytes(source, "utf8"))
    nodes = extract_nodes(tree, source, "python", "user.py", "test-project")

    method_nodes = [n for n in nodes if n.node_type == NodeType.METHOD.value]
    assert len(method_nodes) == 1

    method = method_nodes[0]
    assert method.name == "create_user"
    assert "UserService.create_user" in method.id


def test_extract_python_imports(python_parser):
    """Test extracting Python import edges."""
    source = """
import os
from typing import List
from src.models import User

def process():
    pass
"""

    tree = python_parser.parse(bytes(source, "utf8"))

    # Need symbol registry for edge extraction
    symbol_registry = {}

    edges = extract_edges(tree, source, "python", "main.py", "test-project", symbol_registry)

    import_edges = [e for e in edges if e.edge_type == EdgeType.IMPORTS.value]
    assert len(import_edges) >= 2  # At least 'typing' and 'src.models'

    target_ids = [e.target_id for e in import_edges]
    assert any("typing" in t for t in target_ids)
    assert any("src.models" in t for t in target_ids)


def test_extract_python_calls(python_parser):
    """Test extracting Python function call edges."""
    source = """
def helper():
    pass

def main():
    helper()
    print("test")
"""

    tree = python_parser.parse(bytes(source, "utf8"))

    # Build symbol registry
    nodes = extract_nodes(tree, source, "python", "main.py", "test-project")
    symbol_registry = {n.name: n.id for n in nodes}

    edges = extract_edges(tree, source, "python", "main.py", "test-project", symbol_registry)

    call_edges = [e for e in edges if e.edge_type == EdgeType.CALLS.value]
    assert len(call_edges) >= 1

    # Should have call from main to helper or print
    assert any(e.target_id in ["helper", "print"] or "helper" in e.target_id for e in call_edges)


def test_extract_python_inheritance(python_parser):
    """Test extracting Python class inheritance edges."""
    source = """
class BaseService:
    pass

class UserService(BaseService):
    pass
"""

    tree = python_parser.parse(bytes(source, "utf8"))

    nodes = extract_nodes(tree, source, "python", "services.py", "test-project")
    symbol_registry = {n.name: n.id for n in nodes}

    edges = extract_edges(tree, source, "python", "services.py", "test-project", symbol_registry)

    inherit_edges = [e for e in edges if e.edge_type == EdgeType.INHERITS.value]
    assert len(inherit_edges) == 1

    edge = inherit_edges[0]
    assert "UserService" in edge.source_id
    assert "BaseService" in edge.target_id


def test_extract_typescript_class(typescript_parser):
    """Test extracting TypeScript class node."""
    source = """
class UserService {
    constructor() {}

    createUser(email: string) {
        return true;
    }
}
"""

    tree = typescript_parser.parse(bytes(source, "utf8"))
    nodes = extract_nodes(tree, source, "typescript", "user.ts", "test-project")

    class_nodes = [n for n in nodes if n.node_type == NodeType.CLASS.value]
    assert len(class_nodes) == 1

    user_class = class_nodes[0]
    assert user_class.name == "UserService"

    # Should also find methods
    method_nodes = [n for n in nodes if n.node_type == NodeType.METHOD.value]
    assert len(method_nodes) >= 1


def test_extract_typescript_function(typescript_parser):
    """Test extracting TypeScript function node."""
    source = """
function authenticate(username: string, password: string): boolean {
    return true;
}
"""

    tree = typescript_parser.parse(bytes(source, "utf8"))
    nodes = extract_nodes(tree, source, "typescript", "auth.ts", "test-project")

    func_nodes = [n for n in nodes if n.node_type == NodeType.FUNCTION.value]
    assert len(func_nodes) == 1

    func = func_nodes[0]
    assert func.name == "authenticate"


def test_build_graph_integration(python_parser):
    """Integration test for full graph building."""
    source1 = """
class UserService:
    def create_user(self):
        pass
"""

    source2 = """
from models import UserService

def main():
    service = UserService()
    service.create_user()
"""

    tree1 = python_parser.parse(bytes(source1, "utf8"))
    tree2 = python_parser.parse(bytes(source2, "utf8"))

    parsed_files = {
        "models.py": (tree1, source1, "python"),
        "main.py": (tree2, source2, "python"),
    }

    nodes, edges = build_graph(parsed_files, project_id="test-project")

    # Should have nodes from both files
    assert len(nodes) >= 2  # At least UserService class and create_user method

    # Should have edges
    assert len(edges) >= 1  # At least import or call edges

    # Verify node structure
    for node in nodes:
        assert isinstance(node, CodeNode)
        assert node.project_id == "test-project"
        assert node.file_path in ["models.py", "main.py"]
        assert node.lineno is not None

    # Verify edge structure
    for edge in edges:
        assert isinstance(edge, CodeEdge)
        assert edge.project_id == "test-project"
        assert edge.edge_type in [e.value for e in EdgeType]


def test_get_text_helper(python_parser):
    """Test _get_text helper function."""
    source = "def test(): pass"
    tree = python_parser.parse(bytes(source, "utf8"))
    root = tree.root_node

    text = _get_text(root, source)
    assert text == source


def test_traverse_helper(python_parser):
    """Test _traverse helper function."""
    source = """
class Test:
    def method(self):
        pass
"""
    tree = python_parser.parse(bytes(source, "utf8"))
    root = tree.root_node

    nodes = _traverse(root)

    # Should traverse entire tree
    assert len(nodes) > 1
    assert root in nodes


def test_code_node_dataclass():
    """Test CodeNode dataclass structure."""
    node = CodeNode(
        id="module.Class.method",
        project_id="test-project",
        file_path="src/test.py",
        node_type=NodeType.METHOD.value,
        name="method",
        signature="def method(self)",
        lineno=42,
        docstring="Test method"
    )

    assert node.id == "module.Class.method"
    assert node.project_id == "test-project"
    assert node.file_path == "src/test.py"
    assert node.node_type == NodeType.METHOD.value
    assert node.name == "method"
    assert node.lineno == 42


def test_code_edge_dataclass():
    """Test CodeEdge dataclass structure."""
    edge = CodeEdge(
        source_id="module.func1",
        target_id="module.func2",
        edge_type=EdgeType.CALLS.value,
        lineno=10,
        project_id="test-project"
    )

    assert edge.source_id == "module.func1"
    assert edge.target_id == "module.func2"
    assert edge.edge_type == EdgeType.CALLS.value
    assert edge.lineno == 10
    assert edge.project_id == "test-project"
