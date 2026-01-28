from python import Python

def log_latency(operation: String, duration_ms: Float64):
    """
    Logs latency metrics.
    """
    try:
        var time = Python.import_module("time")
        print(f"METRIC: {operation} took {duration_ms}ms")
    except:
        pass
