
import requests
import json
import sys

# Use the public HF Space URL
BASE_URL = "https://bigwolfe-document-mcp.hf.space/mcp"

def test_search_notes():
    print("--- Testing tools/call search_notes ---")
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": "search_notes",
            "arguments": {"query": "API"}
        },
        "id": 1
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        try:
            data = response.json()
            # print(json.dumps(data, indent=2))
            
            res = data.get("result", {})
            if "structuredContent" in res:
                print("✅ structuredContent found in result")
                print("Keys:", res["structuredContent"].keys())
            else:
                print("❌ structuredContent NOT found in result")
                print(json.dumps(res, indent=2))
                
            if "_meta" in res:
                print("✅ _meta found in result")
            else:
                print("❌ _meta NOT found in result")

        except json.JSONDecodeError:
            print("Response not JSON:", response.text)
            
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    # test_read_note()
    test_search_notes()
    # test_read_resource()
