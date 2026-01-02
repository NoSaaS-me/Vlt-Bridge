"""
Comprehensive tests for BM25 FTS5 query sanitization.

These tests ensure that all special characters that can cause FTS5 syntax errors
are properly handled by the _sanitize_query() method.
"""

import pytest
from vlt.core.coderag.bm25 import BM25Indexer


class TestBM25QuerySanitization:
    """Test suite for FTS5 query sanitization."""

    @pytest.fixture
    def indexer(self):
        """Create a BM25Indexer instance for testing."""
        # We don't need a real database for sanitization tests
        # The _sanitize_query method is pure string processing
        return BM25Indexer()

    def test_simple_query(self, indexer):
        """Test that simple alphanumeric queries work."""
        result = indexer._sanitize_query("VaultService")
        assert result == '"VaultService"'

    def test_multiword_query(self, indexer):
        """Test that multi-word queries are tokenized correctly."""
        result = indexer._sanitize_query("from fastapi import FastAPI")
        assert result == '"from" "fastapi" "import" "FastAPI"'

    def test_equals_sign(self, indexer):
        """Test that = character is handled (reported bug)."""
        result = indexer._sanitize_query("from fastapi import FastAPI app = FastAPI")
        assert "=" not in result
        assert result == '"from" "fastapi" "import" "FastAPI" "app" "FastAPI"'

    def test_comparison_operators(self, indexer):
        """Test that comparison operators are removed."""
        result = indexer._sanitize_query("x < y")
        assert "<" not in result
        assert result == '"x" "y"'

        result = indexer._sanitize_query("a > b")
        assert ">" not in result
        assert result == '"a" "b"'

        result = indexer._sanitize_query("value <= max")
        assert "<=" not in result
        assert result == '"value" "max"'

    def test_arithmetic_operators(self, indexer):
        """Test that arithmetic operators are removed."""
        result = indexer._sanitize_query("a / b")
        assert "/" not in result
        assert result == '"a" "b"'

        result = indexer._sanitize_query("x - y")
        assert result == '"x" "y"'

        result = indexer._sanitize_query("a + b")
        assert result == '"a" "b"'

    def test_special_symbols(self, indexer):
        """Test that special symbols are removed."""
        test_cases = [
            ("@user", '"user"'),
            ("#tag", '"tag"'),
            ("$variable", '"variable"'),
            ("%modulo", '"modulo"'),
            ("&ampersand", '"ampersand"'),
            ("!negation", '"negation"'),
        ]

        for input_query, expected in test_cases:
            result = indexer._sanitize_query(input_query)
            assert result == expected, f"Failed for input: {input_query}"

    def test_punctuation(self, indexer):
        """Test that punctuation is removed."""
        test_cases = [
            ("test;semicolon", '"test" "semicolon"'),
            ("a,b,c", '"a" "b" "c"'),
            ("one.two.three", '"one" "two" "three"'),
            ("tilde~test", '"tilde" "test"'),
            ("`backtick`", '"backtick"'),
        ]

        for input_query, expected in test_cases:
            result = indexer._sanitize_query(input_query)
            assert result == expected, f"Failed for input: {input_query}"

    def test_pipe_and_backslash(self, indexer):
        """Test that pipe and backslash are removed."""
        result = indexer._sanitize_query("a | b")
        assert "|" not in result
        assert result == '"a" "b"'

        result = indexer._sanitize_query("path\\to\\file")
        assert "\\" not in result
        assert result == '"path" "to" "file"'

    def test_brackets_and_parens(self, indexer):
        """Test that brackets and parentheses are removed."""
        result = indexer._sanitize_query("function(arg)")
        assert "(" not in result
        assert ")" not in result
        assert result == '"function" "arg"'

        result = indexer._sanitize_query("array[index]")
        assert "[" not in result
        assert "]" not in result
        assert result == '"array" "index"'

        result = indexer._sanitize_query("{key: value}")
        assert "{" not in result
        assert "}" not in result
        assert result == '"key" "value"'

    def test_quotes(self, indexer):
        """Test that quotes are removed."""
        result = indexer._sanitize_query('"quoted string"')
        # The outer quotes are removed, inner words are preserved
        assert result == '"quoted" "string"'

    def test_underscores_preserved(self, indexer):
        """Test that underscores in identifiers are preserved."""
        result = indexer._sanitize_query("my_function_name")
        assert result == '"my_function_name"'

        result = indexer._sanitize_query("__init__")
        assert result == '"__init__"'

    def test_prefix_matching(self, indexer):
        """Test that trailing * for prefix matching is preserved."""
        result = indexer._sanitize_query("find_definition*")
        assert result == '"find_definition"*'

        result = indexer._sanitize_query("vault*")
        assert result == '"vault"*'

    def test_empty_query(self, indexer):
        """Test that empty queries return empty string."""
        assert indexer._sanitize_query("") == ""
        assert indexer._sanitize_query("   ") == ""
        assert indexer._sanitize_query(None) == ""

    def test_only_special_chars(self, indexer):
        """Test that queries with only special characters return empty."""
        result = indexer._sanitize_query("!@#$%^&*()")
        assert result == ""

        result = indexer._sanitize_query("===")
        assert result == ""

    def test_mixed_content(self, indexer):
        """Test realistic code search queries."""
        # Python function definition
        result = indexer._sanitize_query("def authenticate(user: str) -> bool:")
        assert ":" not in result
        assert "(" not in result
        assert ")" not in result
        assert result == '"def" "authenticate" "user" "str" "bool"'

        # JavaScript/TypeScript
        result = indexer._sanitize_query("const app = new FastAPI();")
        assert "=" not in result
        assert "(" not in result
        assert ";" not in result
        assert result == '"const" "app" "new" "FastAPI"'

        # Import statement
        result = indexer._sanitize_query("from vlt.core.coderag import BM25Indexer")
        assert "." not in result
        assert result == '"from" "vlt" "core" "coderag" "import" "BM25Indexer"'

    def test_case_sensitivity(self, indexer):
        """Test that case is preserved in tokens."""
        result = indexer._sanitize_query("VaultService FastAPI my_function")
        assert result == '"VaultService" "FastAPI" "my_function"'

    def test_numbers(self, indexer):
        """Test that numbers are preserved."""
        result = indexer._sanitize_query("version 3.11")
        assert result == '"version" "3" "11"'

        result = indexer._sanitize_query("Python3")
        assert result == '"Python3"'

    def test_url_like_strings(self, indexer):
        """Test that URLs are tokenized into parts."""
        result = indexer._sanitize_query("https://example.com/path")
        assert result == '"https" "example" "com" "path"'

    def test_multiple_spaces(self, indexer):
        """Test that multiple spaces are handled correctly."""
        result = indexer._sanitize_query("one    two     three")
        assert result == '"one" "two" "three"'

    def test_leading_trailing_spaces(self, indexer):
        """Test that leading/trailing whitespace is handled."""
        result = indexer._sanitize_query("  test  ")
        assert result == '"test"'

    def test_fts5_operators_not_special(self, indexer):
        """
        Test that FTS5 operators are treated as regular tokens.

        When wrapped in quotes, AND/OR/NOT/NEAR become literal search terms.
        """
        # These should be wrapped in quotes and treated as literal search terms
        result = indexer._sanitize_query("AND")
        assert result == '"AND"'

        result = indexer._sanitize_query("search AND query")
        assert result == '"search" "AND" "query"'

        result = indexer._sanitize_query("NOT OR NEAR")
        assert result == '"NOT" "OR" "NEAR"'

    def test_real_world_code_patterns(self, indexer):
        """Test with real code patterns that previously failed."""
        # The original reported bug
        result = indexer._sanitize_query("from fastapi import FastAPI app = FastAPI")
        assert "=" not in result
        expected = '"from" "fastapi" "import" "FastAPI" "app" "FastAPI"'
        assert result == expected

        # Class definition
        result = indexer._sanitize_query("class VaultService:")
        assert result == '"class" "VaultService"'

        # Function call
        result = indexer._sanitize_query("authenticate(username, password)")
        assert result == '"authenticate" "username" "password"'

        # Conditional
        result = indexer._sanitize_query("if x > 10:")
        assert result == '"if" "x" "10"'


class TestBM25Integration:
    """Integration tests to verify sanitization doesn't break actual searches."""

    @pytest.fixture
    def indexer(self):
        """Create a BM25Indexer instance for testing."""
        return BM25Indexer()

    def test_sanitization_produces_valid_fts5_query(self, indexer):
        """
        Test that sanitized queries are valid FTS5 syntax.

        We can't test with a real database here, but we can verify
        the output format is correct for FTS5 MATCH queries.
        """
        # Valid FTS5 queries should be space-separated quoted tokens
        test_cases = [
            "VaultService",
            "from fastapi import FastAPI app = FastAPI",
            "class VaultService:",
            "def authenticate():",
            "x = y",
        ]

        for query in test_cases:
            result = indexer._sanitize_query(query)
            # Should not be empty
            assert result, f"Empty result for: {query}"
            # Should only contain alphanumeric, spaces, quotes, underscores, and *
            assert all(
                c.isalnum() or c in ' "_*' for c in result
            ), f"Invalid characters in sanitized query: {result}"
