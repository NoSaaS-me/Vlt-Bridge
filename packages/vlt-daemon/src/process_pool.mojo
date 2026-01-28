from python import Python

var warm_pool = Python.list()

def pre_fork(count: Int):
    """
    Pre-forks 'count' python processes and keeps them in a list.
    """
    for i in range(count):
        # Start a generic python shell ready to accept code
        var proc = spawn_process("python3 -i")
        warm_pool.append(proc)
        print("Pre-forked worker " + str(i))

def get_warm_process() -> PythonObject:
    """Gets a process from the warm pool or spawns new."""
    if len(warm_pool) > 0:
        return warm_pool.pop(0)
    return spawn_process("python3 -i")