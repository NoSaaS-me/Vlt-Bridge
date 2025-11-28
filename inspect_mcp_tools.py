
from backend.src.mcp.server import mcp
import inspect

def inspect_fastmcp():
    print("MCP Object:", mcp)
    print("Dir MCP:", dir(mcp))
    
    # Try to find tools
    if hasattr(mcp, 'tools'):
        print("Tools attr:", mcp.tools)
    
    if hasattr(mcp, '_tools'):
        print("_tools attr:", mcp._tools)
        
    if hasattr(mcp, '_mcp_server'):
        print("_mcp_server:", mcp._mcp_server)
        print("Dir _mcp_server:", dir(mcp._mcp_server))

if __name__ == "__main__":
    try:
        inspect_fastmcp()
    except Exception as e:
        print(f"Error: {e}")
