
from mcp.types import CallToolResult
import inspect

def inspect_call_tool_result():
    print("Fields:", CallToolResult.model_fields.keys())

if __name__ == "__main__":
    inspect_call_tool_result()
