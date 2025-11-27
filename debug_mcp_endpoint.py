
import asyncio
import logging
from fastapi.testclient import TestClient
from backend.src.api.main import app

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def debug_mcp():
    client = TestClient(app)
    
    print("--- Test 1: GET /mcp (No Auth) ---")
    try:
        response = client.get("/mcp")
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Crash: {e}")

    print("\n--- Test 2: POST /mcp (No Auth) - List Tools ---")
    # JSON-RPC payload
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/list",
        "id": 1
    }
    try:
        response = client.post("/mcp", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Crash: {e}")

    print("\n--- Test 3: POST /mcp (With Auth) - List Tools ---")
    try:
        # Assuming local-dev-token is valid
        response = client.post("/mcp", json=payload, headers={"Authorization": "Bearer local-dev-token"})
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Crash: {e}")

if __name__ == "__main__":
    debug_mcp()
