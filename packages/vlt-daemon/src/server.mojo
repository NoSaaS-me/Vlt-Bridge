from python import Python

def start_server(port: Int):
    """
    Starts a WebSocket server using Python's asyncio/websockets for now.
    Mojo native networking will replace this in future phases.
    """
    try:
        print("Starting Vlt Daemon on port " + String(port))
        
        # Add current directory to python path so we can import local modules
        var sys = Python.import_module("sys")
        var os = Python.import_module("os")
        sys.path.append(os.getcwd() + "/src")
        
        var server_impl = Python.import_module("server_impl")
        server_impl.run(port)
        
    except e:
        print("Failed to start server:")
        print(e)

def main():
    start_server(9000)