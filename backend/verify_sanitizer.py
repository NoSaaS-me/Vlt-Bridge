#!/usr/bin/env python3
"""Simple verification of sanitizer functionality."""

import html


def sanitize_snippet(snippet: str) -> str:
    """Copy of sanitizer function for verification."""
    if not snippet:
        return ""

    escaped = html.escape(snippet, quote=True)
    sanitized = escaped.replace("&lt;mark&gt;", "<mark>")
    sanitized = sanitized.replace("&lt;/mark&gt;", "</mark>")

    return sanitized


def test():
    """Run verification tests."""
    tests_passed = 0
    tests_total = 0

    # Test 1: Normal text
    tests_total += 1
    result = sanitize_snippet("This is simple text")
    if result == "This is simple text":
        print("✓ Test 1: Normal text passes")
        tests_passed += 1
    else:
        print(f"✗ Test 1 FAILED: {result}")

    # Test 2: Mark tags preserved
    tests_total += 1
    result = sanitize_snippet("This is a <mark>highlighted</mark> word")
    if "<mark>highlighted</mark>" in result:
        print("✓ Test 2: Mark tags preserved")
        tests_passed += 1
    else:
        print(f"✗ Test 2 FAILED: {result}")

    # Test 3: Script tags escaped
    tests_total += 1
    result = sanitize_snippet("<script>alert('xss')</script> text")
    if "&lt;script&gt;" in result and "<script>" not in result:
        print("✓ Test 3: Script tags escaped")
        tests_passed += 1
    else:
        print(f"✗ Test 3 FAILED: {result}")

    # Test 4: Script + Mark combination
    tests_total += 1
    result = sanitize_snippet("<script>evil</script> <mark>safe</mark>")
    if "&lt;script&gt;" in result and "<mark>safe</mark>" in result and "<script>" not in result:
        print("✓ Test 4: Script escaped, mark preserved")
        tests_passed += 1
    else:
        print(f"✗ Test 4 FAILED: {result}")

    # Test 5: Event handlers
    tests_total += 1
    result = sanitize_snippet("<img onerror=alert(1)> <mark>text</mark>")
    if "&lt;img" in result and "<mark>text</mark>" in result:
        print("✓ Test 5: Event handlers escaped")
        tests_passed += 1
    else:
        print(f"✗ Test 5 FAILED: {result}")

    # Test 6: Special characters
    tests_total += 1
    result = sanitize_snippet("Use & and < and > symbols")
    if "&amp;" in result and "&lt;" in result and "&gt;" in result:
        print("✓ Test 6: Special characters escaped")
        tests_passed += 1
    else:
        print(f"✗ Test 6 FAILED: {result}")

    # Test 7: Multiple mark tags
    tests_total += 1
    result = sanitize_snippet("<mark>First</mark> and <mark>Second</mark>")
    if result.count("<mark>") == 2 and result.count("</mark>") == 2:
        print("✓ Test 7: Multiple mark tags preserved")
        tests_passed += 1
    else:
        print(f"✗ Test 7 FAILED: {result}")

    # Test 8: Empty string
    tests_total += 1
    result = sanitize_snippet("")
    if result == "":
        print("✓ Test 8: Empty string handled")
        tests_passed += 1
    else:
        print(f"✗ Test 8 FAILED: {result}")

    # Test 9: Complex XSS
    tests_total += 1
    result = sanitize_snippet("<script>fetch('evil.com')</script> <mark>search</mark>")
    if "&lt;script&gt;" in result and "<mark>search</mark>" in result and "<script>" not in result:
        print("✓ Test 9: Complex XSS escaped")
        tests_passed += 1
    else:
        print(f"✗ Test 9 FAILED: {result}")

    # Test 10: Real-world example
    tests_total += 1
    result = sanitize_snippet("Function: <mark>async</mark> def foo() -> None:")
    if "<mark>async</mark>" in result and "-&gt;" in result:
        print("✓ Test 10: Real-world snippet works")
        tests_passed += 1
    else:
        print(f"✗ Test 10 FAILED: {result}")

    print(f"\n{'='*60}")
    print(f"Results: {tests_passed}/{tests_total} tests passed")

    if tests_passed == tests_total:
        print("✓ ALL TESTS PASSED - Sanitizer is working correctly!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(test())
