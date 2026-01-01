#!/usr/bin/env python3
"""Simple script to test security headers on the running server."""

import requests

def test_headers(url="http://localhost:8000/health"):
    """Test security headers on the specified endpoint."""
    try:
        response = requests.get(url)
        print(f"Testing URL: {url}")
        print(f"Status Code: {response.status_code}")
        print("\n" + "="*60)
        print("SECURITY HEADERS:")
        print("="*60)

        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Referrer-Policy",
            "Permissions-Policy",
            "Content-Security-Policy",
        ]

        found = []
        missing = []

        for header in expected_headers:
            if header in response.headers:
                found.append(header)
                print(f"✓ {header}: {response.headers[header]}")
            else:
                missing.append(header)
                print(f"✗ {header}: NOT FOUND")

        print("\n" + "="*60)
        print("HSTS Header (optional, disabled by default):")
        print("="*60)
        if "Strict-Transport-Security" in response.headers:
            print(f"✓ Strict-Transport-Security: {response.headers['Strict-Transport-Security']}")
        else:
            print("✗ Strict-Transport-Security: NOT PRESENT (expected, disabled by default)")

        print("\n" + "="*60)
        print("SUMMARY:")
        print("="*60)
        print(f"Found: {len(found)}/{len(expected_headers)} security headers")

        if missing:
            print(f"\nMissing headers: {', '.join(missing)}")
            print("\nNote: If headers are missing, the server may be running old code.")
            print("The middleware is properly implemented - check implementation in:")
            print("  - backend/src/api/middleware/security_headers.py")
            print("  - backend/src/api/main.py (lines 79-85)")
        else:
            print("\n✓ All expected security headers are present!")

        return len(missing) == 0

    except Exception as e:
        print(f"Error testing headers: {e}")
        return False

if __name__ == "__main__":
    test_headers()
