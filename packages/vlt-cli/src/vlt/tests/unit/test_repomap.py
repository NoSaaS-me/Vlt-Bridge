"""Tests for repository map generation with PageRank centrality.

Tests tasks T031-T035:
- T031: Symbol extraction from AST
- T032: Reference graph construction
- T033: PageRank/centrality calculation
- T034: Token-budgeted map generation
- T035: Map scope filtering
"""

import pytest
from vlt.core.coderag.repomap import (
    Symbol,
    ReferenceGraph,
    build_reference_graph,
    calculate_centrality,
    generate_repo_map,
    filter_symbols_by_scope,
    extract_symbols_from_ast,
    _estimate_tokens,
    _file_path_to_module_name,
)


class TestSymbolExtraction:
    """Test T031 - Symbol extraction from tree-sitter AST."""

    def test_extract_python_symbols(self):
        """Test extracting Python classes, functions, and methods."""
        source_code = '''
class UserService:
    """Service for user operations."""

    def __init__(self, db):
        self.db = db

    def get_user(self, user_id: int) -> dict:
        """Get user by ID."""
        return self.db.query(user_id)

def authenticate(username: str, password: str) -> bool:
    """Authenticate user credentials."""
    return True
'''

        from vlt.core.coderag.parser import parse_file
        tree = parse_file(source_code, "python")
        assert tree is not None

        symbols = extract_symbols_from_ast(tree, source_code, "services/user.py", "python")

        # Should extract class, 2 methods, and 1 function
        assert len(symbols) == 4

        # Check class
        class_sym = next(s for s in symbols if s.symbol_type == 'class')
        assert class_sym.name == "UserService"
        assert class_sym.qualified_name == "services.user.UserService"
        assert "Service for user operations" in class_sym.docstring

        # Check method
        method_sym = next(s for s in symbols if s.name == 'get_user')
        assert method_sym.symbol_type == 'method'
        assert method_sym.parent == "UserService"
        assert "user_id: int" in method_sym.signature

        # Check function
        func_sym = next(s for s in symbols if s.symbol_type == 'function')
        assert func_sym.name == "authenticate"
        assert "username: str" in func_sym.signature

    def test_extract_typescript_symbols(self):
        """Test extracting TypeScript classes and functions."""
        source_code = '''
class AuthService {
    login(username: string, password: string): Promise<Token> {
        return Promise.resolve({token: "abc"});
    }
}

function validateToken(token: string): boolean {
    return token.length > 0;
}
'''

        from vlt.core.coderag.parser import parse_file
        tree = parse_file(source_code, "typescript")
        assert tree is not None

        symbols = extract_symbols_from_ast(tree, source_code, "services/auth.ts", "typescript")

        # Should extract class, 1 method, and 1 function
        assert len(symbols) == 3

        # Check class
        class_sym = next(s for s in symbols if s.symbol_type == 'class')
        assert class_sym.name == "AuthService"

        # Check method
        method_sym = next(s for s in symbols if s.name == 'login')
        assert method_sym.symbol_type == 'method'
        assert method_sym.parent == "AuthService"

    def test_extract_go_symbols(self):
        """Test T039 — extracting Go functions, methods, and type declarations."""
        source_code = '''package main

type Server struct {
    port int
}

type Handler interface {
    Handle() error
}

func (s *Server) Run() error {
    return nil
}

func NewServer(port int) *Server {
    return &Server{port: port}
}
'''
        from vlt.core.coderag.parser import parse_file
        tree = parse_file(source_code, "go")
        assert tree is not None

        symbols = extract_symbols_from_ast(tree, source_code, "pkg/server.go", "go")

        # Should extract 2 types, 1 method, 1 function
        assert len(symbols) == 4

        # Check struct type (maps to 'class')
        struct_sym = next(s for s in symbols if s.name == "Server")
        assert struct_sym.symbol_type == 'class'
        assert "struct" in struct_sym.signature
        assert struct_sym.qualified_name == "pkg.server.Server"
        assert struct_sym.lineno is not None
        assert struct_sym.end_line is not None
        assert struct_sym.end_line >= struct_sym.lineno

        # Check interface type
        iface_sym = next(s for s in symbols if s.name == "Handler")
        assert iface_sym.symbol_type == 'class'
        assert "interface" in iface_sym.signature

        # Check method (receiver-bound)
        method_sym = next(s for s in symbols if s.name == "Run")
        assert method_sym.symbol_type == 'method'
        assert method_sym.parent == "Server"
        assert method_sym.qualified_name == "pkg.server.Server.Run"
        assert method_sym.end_line is not None

        # Check top-level function
        func_sym = next(s for s in symbols if s.name == "NewServer")
        assert func_sym.symbol_type == 'function'
        assert func_sym.qualified_name == "pkg.server.NewServer"
        assert "NewServer" in func_sym.signature

    def test_symbol_end_line_populated(self):
        """Test T038 — Symbol.end_line is set for all extracted symbols."""
        source_code = '''
class Foo:
    def bar(self):
        pass

def baz():
    pass
'''
        from vlt.core.coderag.parser import parse_file
        tree = parse_file(source_code, "python")
        assert tree is not None

        symbols = extract_symbols_from_ast(tree, source_code, "mod.py", "python")
        assert len(symbols) >= 1
        for sym in symbols:
            assert sym.end_line is not None, f"{sym.qualified_name} missing end_line"
            assert sym.lineno is not None
            assert sym.end_line >= sym.lineno


class TestReferenceGraph:
    """Test T032 - Reference graph construction."""

    def test_build_reference_graph(self):
        """Test building a graph from symbols and edges."""
        symbols = [
            Symbol("foo", "module.foo", "file.py", "function", "def foo()"),
            Symbol("bar", "module.bar", "file.py", "function", "def bar()"),
            Symbol("baz", "module.baz", "file.py", "function", "def baz()"),
        ]

        edges = [
            ("module.foo", "module.bar"),  # foo calls bar
            ("module.foo", "module.baz"),  # foo calls baz
            ("module.bar", "module.baz"),  # bar calls baz
        ]

        graph = build_reference_graph(symbols, edges)

        # Check nodes
        assert len(graph.nodes) == 3
        assert "module.foo" in graph.nodes

        # Check edges
        assert len(graph.edges) == 3

        # Check adjacency lists
        assert "module.bar" in graph.get_callees("module.foo")
        assert "module.baz" in graph.get_callees("module.foo")
        assert "module.foo" in graph.get_callers("module.bar")

    def test_reference_graph_operations(self):
        """Test graph query operations."""
        graph = ReferenceGraph()

        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")

        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")

        # A calls B and C
        assert set(graph.get_callees("A")) == {"B", "C"}

        # C is called by A and B
        assert set(graph.get_callers("C")) == {"A", "B"}

        # B is only called by A
        assert graph.get_callers("B") == ["A"]


class TestCentralityCalculation:
    """Test T033 - PageRank/centrality calculation."""

    def test_calculate_centrality_simple(self):
        """Test PageRank on a simple graph."""
        graph = ReferenceGraph()

        # A -> B -> C
        # A -> C
        # C is most central (referenced by both A and B)
        graph.add_node("A")
        graph.add_node("B")
        graph.add_node("C")

        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "C")

        centrality = calculate_centrality(graph)

        # C should have highest score (most referenced)
        assert centrality["C"] > centrality["B"]
        assert centrality["C"] > centrality["A"]

        # All scores should sum to approximately 1.0
        total = sum(centrality.values())
        assert 0.95 < total < 1.05

    def test_calculate_centrality_hub(self):
        """Test centrality with a hub node."""
        graph = ReferenceGraph()

        # Hub pattern: A calls everyone
        # Hub node: Everyone calls D
        nodes = ["A", "B", "C", "D"]
        for node in nodes:
            graph.add_node(node)

        # A calls everyone
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("A", "D")

        # Everyone calls D
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")

        centrality = calculate_centrality(graph)

        # D should be most central (5 incoming edges)
        assert centrality["D"] > centrality["A"]
        assert centrality["D"] > centrality["B"]
        assert centrality["D"] > centrality["C"]

    def test_calculate_centrality_empty(self):
        """Test centrality on empty graph."""
        graph = ReferenceGraph()
        centrality = calculate_centrality(graph)
        assert centrality == {}

    def test_calculate_centrality_singleton(self):
        """Test centrality with single node."""
        graph = ReferenceGraph()
        graph.add_node("A")

        centrality = calculate_centrality(graph)
        assert centrality["A"] == 1.0


class TestTokenBudgetedMapGeneration:
    """Test T034 - Token-budgeted map generation."""

    def test_generate_repo_map_basic(self):
        """Test basic map generation."""
        symbols = [
            Symbol("UserService", "module.UserService", "services/user.py", "class", "class UserService", lineno=1),
            Symbol("get_user", "module.UserService.get_user", "services/user.py", "method", "def get_user(id)", lineno=5, parent="UserService"),
            Symbol("authenticate", "module.authenticate", "services/auth.py", "function", "def authenticate()", lineno=1),
        ]

        # Simple graph: no edges (all equal centrality)
        graph = ReferenceGraph()
        for sym in symbols:
            graph.add_node(sym.qualified_name)

        centrality = calculate_centrality(graph)

        map_data = generate_repo_map(
            symbols=symbols,
            graph=graph,
            centrality_scores=centrality,
            max_tokens=1000,
            include_signatures=True
        )

        assert "map_text" in map_data
        assert "token_count" in map_data
        assert "symbols_included" in map_data
        assert "files_included" in map_data

        # Should include both files
        assert "services/user.py" in map_data["map_text"]
        assert "services/auth.py" in map_data["map_text"]

        # Should include class and method
        assert "UserService" in map_data["map_text"]
        assert "get_user" in map_data["map_text"]

    def test_generate_repo_map_pruning(self):
        """Test token budget pruning."""
        # Create many symbols to force pruning
        symbols = []
        for i in range(100):
            symbols.append(Symbol(
                f"func_{i}",
                f"module.func_{i}",
                f"file_{i % 10}.py",
                "function",
                f"def func_{i}()",
                lineno=i
            ))

        graph = ReferenceGraph()
        for sym in symbols:
            graph.add_node(sym.qualified_name)

        # Make first symbol highly central
        for sym in symbols[1:]:
            graph.add_edge(sym.qualified_name, symbols[0].qualified_name)

        centrality = calculate_centrality(graph)

        # Generate with small budget
        map_data = generate_repo_map(
            symbols=symbols,
            graph=graph,
            centrality_scores=centrality,
            max_tokens=200,  # Very small budget
            include_signatures=True
        )

        # Should be pruned
        assert map_data["symbols_included"] < 100
        assert map_data["token_count"] <= 200

        # Most central symbol should be included
        assert "func_0" in map_data["map_text"]

    def test_generate_repo_map_centrality_ordering(self):
        """Test that high-centrality symbols appear first."""
        symbols = [
            Symbol("low_centrality", "module.low", "file.py", "function", "def low()", lineno=1),
            Symbol("high_centrality", "module.high", "file.py", "function", "def high()", lineno=2),
        ]

        graph = ReferenceGraph()
        graph.add_node("module.low")
        graph.add_node("module.high")

        # high is called by low (making high more central)
        graph.add_edge("module.low", "module.high")

        centrality = calculate_centrality(graph)

        map_data = generate_repo_map(
            symbols=symbols,
            graph=graph,
            centrality_scores=centrality,
            max_tokens=1000
        )

        map_text = map_data["map_text"]

        # high_centrality should appear before low_centrality
        high_pos = map_text.index("high_centrality")
        low_pos = map_text.index("low_centrality")
        assert high_pos < low_pos


class TestMapScopeFiltering:
    """Test T035 - Map scope filtering."""

    def test_filter_symbols_by_scope(self):
        """Test filtering symbols to a subdirectory."""
        symbols = [
            Symbol("Foo", "module.Foo", "src/api/routes.py", "class", "class Foo"),
            Symbol("Bar", "module.Bar", "src/services/user.py", "class", "class Bar"),
            Symbol("Baz", "module.Baz", "src/api/middleware.py", "class", "class Baz"),
            Symbol("Qux", "module.Qux", "tests/test_api.py", "function", "def qux()"),
        ]

        filtered = filter_symbols_by_scope(symbols, "src/api/")

        # Should only include symbols from src/api/
        assert len(filtered) == 2
        assert all(s.file_path.startswith("src/api/") for s in filtered)

        names = [s.name for s in filtered]
        assert "Foo" in names
        assert "Baz" in names
        assert "Bar" not in names

    def test_generate_repo_map_with_scope(self):
        """Test map generation with scope parameter."""
        symbols = [
            Symbol("ApiRoute", "module.ApiRoute", "src/api/routes.py", "class", "class ApiRoute"),
            Symbol("Service", "module.Service", "src/services/user.py", "class", "class Service"),
        ]

        graph = ReferenceGraph()
        for sym in symbols:
            graph.add_node(sym.qualified_name)

        centrality = calculate_centrality(graph)

        map_data = generate_repo_map(
            symbols=symbols,
            graph=graph,
            centrality_scores=centrality,
            max_tokens=1000,
            scope="src/api/"
        )

        # Should only include API symbols
        assert "ApiRoute" in map_data["map_text"]
        assert "Service" not in map_data["map_text"]
        assert map_data["symbols_included"] == 1
        assert map_data["scope"] == "src/api/"


class TestHelperFunctions:
    """Test utility functions."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "x" * 400  # 400 characters
        tokens = _estimate_tokens(text)
        assert tokens == 100  # 400 / 4

    def test_file_path_to_module_name(self):
        """Test file path to module name conversion."""
        assert _file_path_to_module_name("src/api/routes.py") == "src.api.routes"
        assert _file_path_to_module_name("lib/utils.ts") == "lib.utils"
        assert _file_path_to_module_name("services/auth.js") == "services.auth"
        assert _file_path_to_module_name("foo.py") == "foo"


class TestIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow(self):
        """Test complete workflow from symbols to map."""
        # Create realistic symbol set
        symbols = [
            # Core service (high centrality expected)
            Symbol("UserService", "module.UserService", "services/user.py", "class", "class UserService", lineno=1),
            Symbol("get_user", "module.UserService.get_user", "services/user.py", "method", "def get_user(id: int) → User", lineno=5, parent="UserService"),
            Symbol("create_user", "module.UserService.create_user", "services/user.py", "method", "def create_user(data: dict) → User", lineno=10, parent="UserService"),

            # API routes (call service)
            Symbol("users_router", "module.users_router", "api/routes.py", "function", "def users_router()", lineno=1),
            Symbol("get_user_endpoint", "module.get_user_endpoint", "api/routes.py", "function", "def get_user_endpoint(id: int)", lineno=5),

            # Utility function
            Symbol("validate_id", "module.validate_id", "utils/validation.py", "function", "def validate_id(id: int) → bool", lineno=1),
        ]

        # Build realistic graph
        graph = ReferenceGraph()
        for sym in symbols:
            graph.add_node(sym.qualified_name)

        # Routes call service methods
        graph.add_edge("module.users_router", "module.UserService.get_user")
        graph.add_edge("module.get_user_endpoint", "module.UserService.get_user")

        # Service uses validation
        graph.add_edge("module.UserService.get_user", "module.validate_id")
        graph.add_edge("module.UserService.create_user", "module.validate_id")

        # Calculate centrality
        centrality = calculate_centrality(graph)

        # UserService methods should be highly central
        assert centrality["module.UserService.get_user"] > centrality["module.users_router"]
        assert centrality["module.validate_id"] > 0

        # Generate map
        map_data = generate_repo_map(
            symbols=symbols,
            graph=graph,
            centrality_scores=centrality,
            max_tokens=2000,
            include_signatures=True
        )

        # Verify map content
        assert map_data["symbols_included"] > 0
        assert map_data["files_included"] == 3  # user.py, routes.py, validation.py
        assert "UserService" in map_data["map_text"]
        assert "def get_user" in map_data["map_text"]

        # Verify structure
        lines = map_data["map_text"].split('\n')
        assert any("services/user.py" in line for line in lines)
        assert any("api/routes.py" in line for line in lines)
