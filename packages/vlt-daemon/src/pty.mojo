from python import Python

def create_pty() -> PythonObject:
    """
    Creates a pseudo-terminal pair.
    Returns (master_fd, slave_fd).
    """
    try:
        var pty = Python.import_module("pty")
        var os = Python.import_module("os")
        
        var pair = pty.openpty()
        return pair
    except:
        print("Failed to open PTY")
        return Python.none()

def fork_pty() -> PythonObject:
    """
    Forks and connects child to PTY.
    Returns (pid, master_fd).
    """
    try:
        var pty = Python.import_module("pty")
        var pair = pty.fork()
        return pair
    except:
        return Python.none()
