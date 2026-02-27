from python import Python
from .process_pool import get_warm_process

var active_sessions = Python.dict()

def create_session(session_id: String) -> PythonObject:
    """
    Creates a new session using a warm process.
    """
    var proc = get_warm_process()
    active_sessions[session_id] = proc
    print("Session created: " + session_id)
    return proc

def get_session(session_id: String) -> PythonObject:
    return active_sessions.get(session_id)
