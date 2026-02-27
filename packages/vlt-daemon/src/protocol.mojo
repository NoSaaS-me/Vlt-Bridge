from python import Python

struct JsonRpcRequest:
    var jsonrpc: String
    var method: String
    var id: String
    # Params handled as dynamic object via Python for flexibility

    fn __init__(inout self, method: String, id: String):
        self.jsonrpc = "2.0"
        self.method = method
        self.id = id

def parse_message(message: String) -> String:
    """
    Parses a JSON string and validates it against JSON-RPC 2.0.
    Returns method name or error.
    """
    try:
        var json = Python.import_module("json")
        var data = json.loads(message)
        
        # Simple validation
        if str(data["jsonrpc"]) != "2.0":
            return "error: invalid version"
            
        return str(data["method"])
    except:
        return "error: parse failed"
