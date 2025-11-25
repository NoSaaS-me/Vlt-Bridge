#!/usr/bin/env python3
"""Debug script to check which endpoints are working."""

import requests
import json
from urllib.parse import urljoin


def test_endpoint(url, method="GET", headers=None, data=None, timeout=5):
    """Test a single endpoint and return detailed results."""
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method.upper() == "POST":
            response = requests.post(url, headers=headers, json=data, timeout=timeout)

        return {
            "status": "SUCCESS",
            "status_code": response.status_code,
            "response": response.text[:500],  # Limit response length
            "headers": dict(response.headers),
        }
    except requests.exceptions.ConnectionError:
        return {"status": "CONNECTION_ERROR", "error": "Cannot connect to server"}
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "error": "Request timed out"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def check_all_endpoints():
    """Check all possible endpoints systematically."""

    print("🔍 Debugging Endpoint Availability")
    print("=" * 60)

    # Define test cases
    test_cases = [
        # FastAPI Server (Port 8000)
        {
            "name": "FastAPI Health",
            "url": "http://localhost:8000/health",
            "method": "GET",
        },
        {"name": "FastAPI Root", "url": "http://localhost:8000/", "method": "GET"},
        {
            "name": "FastAPI API Root",
            "url": "http://localhost:8000/api/",
            "method": "GET",
        },
        # MCP Server (Port 8001)
        {"name": "MCP Root", "url": "http://localhost:8001/", "method": "GET"},
        {"name": "MCP Endpoint", "url": "http://localhost:8001/mcp", "method": "GET"},
        {
            "name": "MCP Health (if exists)",
            "url": "http://localhost:8001/health",
            "method": "GET",
        },
        # MCP Protocol Tests
        {
            "name": "MCP Initialize",
            "url": "http://localhost:8001/mcp",
            "method": "POST",
            "headers": {
                "Authorization": "Bearer local-dev-token",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            "data": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "debug-client", "version": "1.0.0"},
                },
            },
        },
        {
            "name": "MCP Tools List",
            "url": "http://localhost:8001/mcp",
            "method": "POST",
            "headers": {
                "Authorization": "Bearer local-dev-token",
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            "data": {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        },
    ]

    results = {}

    for test in test_cases:
        print(f"\n🧪 Testing: {test['name']}")
        print(f"   URL: {test['url']}")
        print(f"   Method: {test['method']}")

        result = test_endpoint(
            url=test["url"],
            method=test["method"],
            headers=test.get("headers"),
            data=test.get("data"),
        )

        results[test["name"]] = result

        if result["status"] == "SUCCESS":
            print(f"   ✅ Status: {result['status_code']}")
            if result["response"]:
                # Try to pretty print JSON
                try:
                    json_resp = json.loads(result["response"])
                    print(f"   📄 Response: {json.dumps(json_resp, indent=2)[:200]}...")
                except:
                    print(f"   📄 Response: {result['response'][:100]}...")
        else:
            print(f"   ❌ {result['status']}: {result.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    working = []
    not_working = []

    for name, result in results.items():
        if result["status"] == "SUCCESS" and result.get("status_code", 0) < 400:
            working.append(name)
        else:
            not_working.append(name)

    print(f"\n✅ WORKING ({len(working)}):")
    for name in working:
        status_code = results[name].get("status_code", "N/A")
        print(f"   - {name} (HTTP {status_code})")

    print(f"\n❌ NOT WORKING ({len(not_working)}):")
    for name in not_working:
        result = results[name]
        if result["status"] == "SUCCESS":
            print(f"   - {name} (HTTP {result.get('status_code', 'N/A')})")
        else:
            print(f"   - {name} ({result['status']})")

    print(f"\n🔧 RECOMMENDATIONS:")

    # Check if any server is running
    fastapi_working = any("FastAPI" in name for name in working)
    mcp_working = any("MCP" in name for name in working)

    if not fastapi_working:
        print("   - Start FastAPI server: cd backend && python main.py")

    if not mcp_working:
        print(
            "   - Start MCP server: cd backend && MCP_TRANSPORT=http MCP_PORT=8001 python -m src.mcp.server"
        )

    if not working:
        print("   - Check if any servers are running: netstat -ano | findstr :8000")
        print("   - Check if any servers are running: netstat -ano | findstr :8001")

    return results


if __name__ == "__main__":
    check_all_endpoints()
