from python import Python
from .process_pool import spawn_process

def handle_execute_task(params: String) -> String:
    """
    Handles 'execute_task' RPC.
    Params: {"command": "ls -la"}
    """
    try:
        var json = Python.import_module("json")
        var p = json.loads(params)
        var cmd = str(p["command"])
        
        print("Executing: " + cmd)
        var proc = spawn_process(cmd)
        
        return json.dumps({"status": "started", "pid": str(proc.pid)})
    except:
        return "{'error': 'failed'}"
