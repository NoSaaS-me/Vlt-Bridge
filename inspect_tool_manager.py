
from backend.src.mcp.server import mcp
import inspect

def inspect_tool_manager():
    tm = mcp._tool_manager
    print("Tool Manager:", tm)
    print("Dir TM:", dir(tm))
    
    if hasattr(tm, '_tools'):
        print("Tools dict keys:", tm._tools.keys())
        # Inspect one tool
        if 'read_note' in tm._tools:
            tool = tm._tools['read_note']
            print("Tool object:", tool)
            print("Dir Tool:", dir(tool))
            # Check if it has a description or attributes we can patch

if __name__ == "__main__":
    try:
        inspect_tool_manager()
    except Exception as e:
        print(f"Error: {e}")
