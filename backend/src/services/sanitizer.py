"""HTML sanitization utilities for search snippets."""

from __future__ import annotations

import html
import logging

logger = logging.getLogger(__name__)


def sanitize_snippet(snippet: str) -> str:
    """
    Sanitize a search snippet by escaping all HTML except FTS5-generated <mark> tags.

    SQLite FTS5's snippet() function generates snippets with <mark> tags to highlight
    matches. This function:
    1. HTML-escapes the entire snippet (converting <, >, &, ", ' to entities)
    2. Restores only the legitimate FTS5-generated <mark> and </mark> tags
    3. Handles edge cases like nested/malformed tags

    This prevents XSS attacks from user-generated content while preserving
    search result highlighting.

    Args:
        snippet: Raw snippet text from FTS5 snippet() function

    Returns:
        Sanitized snippet with only <mark> tags preserved

    Examples:
        >>> sanitize_snippet("Hello <mark>world</mark>")
        'Hello <mark>world</mark>'

        >>> sanitize_snippet("<script>alert('xss')</script> <mark>test</mark>")
        '&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt; <mark>test</mark>'

        >>> sanitize_snippet("<img onerror=alert(1) src=x> <mark>text</mark>")
        '&lt;img onerror=alert(1) src=x&gt; <mark>text</mark>'
    """
    if not snippet:
        return ""

    # Step 1: HTML-escape everything (including <mark> tags)
    # This converts < to &lt;, > to &gt;, & to &amp;, etc.
    escaped = html.escape(snippet, quote=True)

    # Step 2: Restore only the legitimate FTS5-generated <mark> and </mark> tags
    # The FTS5 snippet() function only generates simple <mark> and </mark> tags,
    # never with attributes or nested tags.
    # We restore these specific escaped sequences back to their tag form.
    sanitized = escaped.replace("&lt;mark&gt;", "<mark>")
    sanitized = sanitized.replace("&lt;/mark&gt;", "</mark>")

    # Note: We intentionally do NOT restore any other escaped HTML tags.
    # Even if there are malformed/nested <mark> tags in the original content,
    # only the exact &lt;mark&gt; and &lt;/mark&gt; sequences will be restored.
    # Any malicious content like <mark onclick="..."> will remain escaped as
    # &lt;mark onclick=&quot;...&quot;&gt;

    return sanitized
