#!/usr/bin/env python3
"""Artifact backend process harness.

Thin wrapper that the daemon runs as a subprocess for each artifact backend.
Reads JSON lines from stdin, imports the artifact's main.py, calls handle(),
and writes JSON responses to stdout.

Usage: python artifact_harness.py /path/to/artifact/backend/
"""

import importlib.util
import json
import os
import sys
import traceback


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


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: artifact_harness.py <backend_dir>"}))
        sys.exit(1)

    backend_dir = sys.argv[1]

    # Apply resource quotas
    _apply_resource_limits(backend_dir)

    # Add backend dir to path for local imports
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

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

    # Main loop: read JSON lines from stdin, call handle(), write response
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

        action = request.get("action", "")
        params = request.get("params", {})

        try:
            # Handle special harness actions
            if action == "__save_state":
                if hasattr(module, "save_state"):
                    state = module.save_state()
                    print(json.dumps({"state": state}))
                else:
                    print(json.dumps({"state": None}))
            elif action == "__load_state":
                if hasattr(module, "load_state"):
                    module.load_state(params)
                    print(json.dumps({"result": "ok"}))
                else:
                    print(json.dumps({"result": "no load_state defined"}))
            elif action == "__event":
                # IPC event from another artifact
                if hasattr(module, "on_event"):
                    result = module.on_event(
                        params.get("event_type", ""),
                        params.get("source", ""),
                        params.get("payload", {}),
                    )
                    print(json.dumps({"result": result}))
                else:
                    print(json.dumps({"result": "no on_event handler"}))
            else:
                result = module.handle(action, params)
                print(json.dumps({"result": result}))
        except Exception as e:
            print(json.dumps({
                "error": str(e),
                "traceback": traceback.format_exc(),
            }))

        sys.stdout.flush()


if __name__ == "__main__":
    main()
