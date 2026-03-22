#!/usr/bin/env python3
"""Artifact backend process harness.

Thin wrapper that the daemon runs as a subprocess for each artifact backend.
Reads JSON lines from stdin, imports the artifact's main.py, calls handle(),
and writes JSON responses to stdout.

Usage: python artifact_harness.py /path/to/artifact/backend/

Bidirectional Protocol
----------------------
Inbound (daemon → harness stdin):
  {"id": "req_1", "action": "generate", "params": {...}}
  {"_id": "out_1", "_type": "connector_response", "result": {...}}
  {"_id": "out_2", "_type": "storage_response", "result": {...}}

Outbound (harness stdout → daemon):
  {"id": "req_1", "result": {...}}           # response to inbound request
  {"_type": "connector_call", "_id": "out_1", "connector": "openrouter",
   "action": "generate_text", "params": {...}}   # backend-initiated call
  {"_type": "storage", "_id": "out_2", "action": "set", "key": "k", "value": 42}
  {"_type": "event", "event_type": "content.approved", "payload": {...}}
  {"_type": "notification", "severity": "info", "message": "..."}
  {"_type": "log", "level": "info", "message": "..."}

harness_call() — Backend-initiated synchronous call
-----------------------------------------------------
When handle() is executing, the main loop is blocked waiting for it to return.
This means stdin is exclusively available to the backend during that window.

    from artifact_harness import harness_call

    def handle(action, params):
        result = harness_call("connector_call", {
            "connector": "openrouter",
            "action": "generate_text",
            "params": {"model": "anthropic/claude-sonnet-4", "prompt": "Hello"},
        })
        return {"text": result["result"]}

Flow:
  1. harness_call writes {"_type": ..., "_id": ..., ...} to stdout
  2. Daemon reads it, performs the operation, writes response to harness stdin
  3. harness_call reads the next line from stdin and returns the parsed dict
"""

import importlib.util
import json
import os
import sys
import traceback
import uuid


# ---------------------------------------------------------------------------
# Global stdin reference — set in main() so harness_call() can use it.
# We keep a direct reference because sys.stdin may be line-buffered.
# ---------------------------------------------------------------------------
_stdin = None
_stdout = None


def harness_call(call_type: str, payload: dict) -> dict:
    """Make a synchronous outbound call to the daemon from inside handle().

    This is safe to call *only* during handle() execution (i.e. while the main
    loop is blocked waiting for handle() to return).  It writes a _type message
    to stdout and reads the daemon's synchronous reply from stdin.

    Args:
        call_type: One of "connector_call", "storage", or any registered type.
        payload:   Dict of additional fields to include (e.g. connector, action,
                   params for connector_call; action/key/value for storage).

    Returns:
        Parsed dict from the daemon's response line.

    Raises:
        RuntimeError: If called outside of handle() (stdin unavailable).
        ValueError:   If the daemon returns an error field.
    """
    if _stdin is None or _stdout is None:
        raise RuntimeError("harness_call() called before main loop initialised")

    call_id = uuid.uuid4().hex[:8]
    message = {"_type": call_type, "_id": call_id, **payload}
    _stdout.write(json.dumps(message) + "\n")
    _stdout.flush()

    # Read the reply synchronously — the daemon must respond with {"_id": call_id, ...}
    while True:
        line = _stdin.readline()
        if not line:
            raise RuntimeError("stdin closed while waiting for harness_call response")
        line = line.strip()
        if not line:
            continue
        try:
            response = json.loads(line)
        except json.JSONDecodeError:
            # Non-JSON line during outbound call — skip it
            continue

        # Accept any response bearing our call_id
        if response.get("_id") == call_id:
            if "error" in response and "_type" not in response:
                raise ValueError(f"harness_call error: {response['error']}")
            return response

        # Lines from the daemon that aren't our response should not appear here
        # (daemon sends only one reply per harness_call), but be defensive.
        sys.stderr.write(
            f"[harness] Unexpected line while waiting for _id={call_id}: {line}\n"
        )


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

def load_artifact_module(backend_dir: str):
    """Import the artifact's main.py as a module."""
    main_path = os.path.join(backend_dir, "main.py")
    if not os.path.exists(main_path):
        raise FileNotFoundError(f"No main.py found in {backend_dir}")

    spec = importlib.util.spec_from_file_location("artifact_main", main_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["artifact_main"] = module
    spec.loader.exec_module(module)
    return module


def _apply_resource_limits(manifest_path: str):
    """Apply resource limits from manifest quotas."""
    try:
        import resource
        manifest_file = os.path.join(os.path.dirname(manifest_path), "manifest.json")
        if os.path.exists(manifest_file):
            with open(manifest_file) as f:
                manifest = json.load(f)
            quotas = manifest.get("quotas", {})
            max_memory_mb = quotas.get("max_memory_mb", 512)
            # Set virtual memory limit
            mem_bytes = max_memory_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, resource.error):
                pass  # Some systems don't support RLIMIT_AS
    except Exception:
        pass  # Non-critical


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    global _stdin, _stdout

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: artifact_harness.py <backend_dir>"}))
        sys.exit(1)

    backend_dir = sys.argv[1]

    # Apply resource quotas
    _apply_resource_limits(backend_dir)

    # Add backend dir to path for local imports
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    # Expose stdin/stdout for harness_call()
    global _stdin, _stdout
    _stdin = sys.stdin
    _stdout = sys.stdout

    # Register this module under "artifact_harness" so backend code can
    # `from artifact_harness import harness_call` and get the SAME module
    # instance (with globals set) rather than a fresh import.
    sys.modules["artifact_harness"] = sys.modules[__name__]

    try:
        module = load_artifact_module(backend_dir)
    except Exception as e:
        print(json.dumps({"error": f"Failed to load main.py: {e}"}))
        sys.stdout.flush()
        sys.exit(1)

    # Check for required handle function
    if not hasattr(module, "handle"):
        print(json.dumps({"error": "main.py must define a handle(action, params) function"}))
        sys.stdout.flush()
        sys.exit(1)

    # Load hot state if available
    hot_state_path = os.path.join(backend_dir, "..", ".vlt", "hot_state.json")
    if os.path.exists(hot_state_path) and hasattr(module, "load_state"):
        try:
            with open(hot_state_path) as f:
                state = json.load(f)
            module.load_state(state)
            os.remove(hot_state_path)  # Consumed
        except Exception as e:
            sys.stderr.write(f"Warning: Failed to load hot state: {e}\n")

    # Signal ready
    print(json.dumps({"status": "ready"}))
    sys.stdout.flush()

    # -----------------------------------------------------------------------
    # Main loop: read JSON lines from stdin, call handle(), write response.
    #
    # Lines arriving here are one of:
    #   (a) A normal request:  {"id": "...", "action": "...", "params": {...}}
    #   (b) A harness_call reply: {"_id": "...", "_type": "connector_response", ...}
    #       — these only arrive while handle() is blocked inside harness_call(),
    #         so they are consumed there, not here.
    # -----------------------------------------------------------------------
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON: {e}"}))
            sys.stdout.flush()
            continue

        # If this is a harness_call reply that somehow reached the main loop
        # (shouldn't happen during normal operation), discard it with a warning.
        if "_type" in request and request.get("_id"):
            reply_type = request["_type"]
            if reply_type in ("connector_response", "storage_response"):
                sys.stderr.write(
                    f"[harness] Unexpected {reply_type} in main loop (dropped): {line}\n"
                )
                continue

        # Extract request id for echoing back (backwards-compatible: id may be absent)
        req_id = request.get("id")
        action = request.get("action", "")
        params = request.get("params", {})

        try:
            # Handle special harness actions
            if action == "__save_state":
                if hasattr(module, "save_state"):
                    state = module.save_state()
                    response = {"state": state}
                else:
                    response = {"state": None}
            elif action == "__load_state":
                if hasattr(module, "load_state"):
                    module.load_state(params)
                    response = {"result": "ok"}
                else:
                    response = {"result": "no load_state defined"}
            elif action == "__event":
                # IPC event from another artifact
                if hasattr(module, "on_event"):
                    result = module.on_event(
                        params.get("event_type", ""),
                        params.get("source", ""),
                        params.get("payload", {}),
                    )
                    response = {"result": result}
                else:
                    response = {"result": "no on_event handler"}
            else:
                result = module.handle(action, params)
                response = {"result": result}
        except Exception as e:
            response = {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        # Echo request id if present (backwards-compatible)
        if req_id is not None:
            response["id"] = req_id

        print(json.dumps(response))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
