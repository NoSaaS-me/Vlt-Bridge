"""
vlt session-relay — Transparent PTY bridge for Claude Code sessions.

Wraps the `claude` CLI so the vlt daemon can relay terminal I/O to the web UI.
The user experience is identical to running `claude` directly.

Platform support:
  Linux/macOS: os.openpty() + select + termios (stdlib only)
  Windows:     pywinpty ConPTY + WebSocket streaming (pywinpty required)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import struct
import sys
import uuid
from pathlib import Path
from typing import Optional

# POSIX-only imports — deferred to avoid ImportError on Windows
if platform.system() != "Windows":
    import termios
    import tty


def find_real_claude() -> str:
    """Find the real claude binary, not a shell function wrapper."""
    # On Linux/Mac, shutil.which follows PATH — since we're a subprocess
    # (not a shell function), there's no fish function shadowing us here.
    # But be defensive: skip any path that looks like a vlt wrapper.
    for candidate in _which_all("claude"):
        # Skip if it's a symlink to vlt or contains "vlt"
        try:
            real = os.path.realpath(candidate)
            if "vlt" not in real.lower():
                return candidate
        except OSError:
            continue
    raise RuntimeError("claude binary not found in PATH")


def _which_all(name: str) -> list[str]:
    """Return all PATH entries for a command name (Windows-aware)."""
    results = []
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    # On Windows, check common executable extensions
    if platform.system() == "Windows":
        pathext = os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";")
        suffixes = [""] + [ext.lower() for ext in pathext]
    else:
        suffixes = [""]
    for d in path_dirs:
        for suffix in suffixes:
            full = os.path.join(d, name + suffix)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                results.append(full)
    return results


def detect_vlt_project(cwd: str) -> Optional[str]:
    """Walk up from cwd looking for vlt.toml to identify project_id."""
    path = Path(cwd)
    for parent in [path] + list(path.parents):
        toml = parent / "vlt.toml"
        if toml.exists():
            try:
                import tomllib  # Python 3.11+
            except ImportError:
                try:
                    import tomli as tomllib
                except ImportError:
                    return None
            with open(toml, "rb") as f:
                data = tomllib.load(f)
            return data.get("project", {}).get("id") or data.get("project_id")
    return None


def get_daemon_url() -> str:
    """Get the daemon base URL from settings."""
    try:
        from vlt.config import get_settings
        settings = get_settings()
        port = getattr(settings, "daemon_port", 8765)
        return f"http://localhost:{port}"
    except Exception:
        return "http://localhost:8765"


def register_session(session_id: str, cwd: str, pid: int, args: list[str]) -> None:
    """Register this session with the daemon (best-effort, non-blocking)."""
    try:
        import httpx
        project_id = detect_vlt_project(cwd)
        httpx.post(
            f"{get_daemon_url()}/api/sessions/register",
            json={
                "session_id": session_id,
                "cwd": cwd,
                "project_id": project_id,
                "pid": pid,
                "args": args,
            },
            timeout=1.0,
        )
    except Exception:
        pass  # Daemon not running — relay still works for the user


def deregister_session(session_id: str) -> None:
    """Deregister session from daemon on exit (best-effort)."""
    try:
        import httpx
        httpx.post(
            f"{get_daemon_url()}/api/sessions/deregister",
            json={"session_id": session_id},
            timeout=1.0,
        )
    except Exception:
        pass


# ── Linux/macOS PTY relay ────────────────────────────────────────────────────

def _relay_posix(claude_bin: str, args: list[str], session_id: str) -> int:
    """
    POSIX (Linux/macOS) relay using os.openpty().

    Creates a PTY pair:
      master_fd <- daemon reads this (terminal output stream)
      slave_fd  -> attached to claude's stdin/stdout/stderr

    The user's terminal is bridged:
      user stdin  -> master_fd  -> claude's stdin
      claude's stdout/stderr -> master_fd -> user stdout + daemon stream
    """
    import select
    import subprocess
    import threading

    master_fd, slave_fd = os.openpty()

    # Match terminal size
    try:
        import fcntl
        winsize = struct.pack("HHHH", 0, 0, 0, 0)
        winsize = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, winsize)
        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
    except Exception:
        pass

    env = os.environ.copy()
    env["VLT_SESSION_ID"] = session_id
    env["TERM"] = os.environ.get("TERM", "xterm-256color")

    # Inject --session-id so Claude's hook session ID matches our relay registration UUID.
    # Without this, the relay registers with UUID-X but Claude fires hooks with UUID-Y,
    # so the inject queue never connects to the right session.
    relay_args = args[:]
    if "--session-id" not in relay_args and "--resume" not in relay_args:
        relay_args = ["--session-id", session_id] + relay_args

    proc = subprocess.Popen(
        [claude_bin] + relay_args,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        env=env,
        preexec_fn=os.setsid,
    )
    os.close(slave_fd)

    # Register with daemon (includes our real PID, claude's PID is proc.pid)
    register_session(session_id, os.getcwd(), proc.pid, args)

    # Pre-allocate write queue NOW so PTY output is captured from the very first byte.
    # Claude renders its TUI immediately; if the queue is None we'd drop all of it.
    import queue as _relay_queue
    ws_write_queue: _relay_queue.Queue = _relay_queue.Queue(maxsize=8192)

    # Relay-side scrollback: captures all PTY output so we can replay it on daemon
    # reconnect. The daemon's in-memory scrollback is lost on restart; the relay's
    # buffer (in this process) survives, so we resend it whenever we reconnect.
    _MAX_RELAY_SCROLLBACK = 2 * 1024 * 1024  # 2 MiB cap
    relay_scrollback: bytearray = bytearray()
    relay_scrollback_lock: threading.Lock = threading.Lock()

    _relay_active = threading.Event()
    _relay_active.set()

    def start_ws_relay():
        """Background thread: opens WebSocket to daemon and streams output.
        Reconnects automatically if the daemon restarts or the connection drops.
        """
        nonlocal ws_write_queue
        try:
            import queue as _queue
            import time
            import websocket  # websocket-client
        except ImportError:
            return  # websocket-client not installed — relay works without streaming

        daemon_base = get_daemon_url().replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{daemon_base}/ws/relay/{session_id}"
        backoff = 1.0
        first_connection = True
        is_reconnect = False

        while _relay_active.is_set():
            if first_connection:
                # Reuse the pre-initialized queue so we drain buffered initial output
                q = ws_write_queue
                first_connection = False
                is_reconnect = False
            else:
                # On reconnect, start fresh (stale data from a dead WS is irrelevant)
                q = _queue.Queue(maxsize=8192)
                ws_write_queue = q
                is_reconnect = True

            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    if data.get("type") == "inject":
                        os.write(master_fd, data.get("data", "").encode())
                    elif data.get("type") == "resize":
                        cols, rows = data.get("cols", 80), data.get("rows", 24)
                        win = struct.pack("HHHH", rows, cols, 0, 0)
                        import fcntl
                        fcntl.ioctl(master_fd, termios.TIOCSWINSZ, win)
                except Exception:
                    pass

            def ws_send_loop(ws):
                while True:
                    chunk = q.get()
                    if chunk is None:
                        break
                    try:
                        ws.send_bytes(chunk)
                    except Exception:
                        break

            try:
                ws = websocket.WebSocketApp(ws_url, on_message=on_message)

                def on_open(ws):
                    nonlocal backoff
                    backoff = 1.0  # reset backoff on successful connect
                    # Re-register session each time we reconnect (daemon may have restarted)
                    register_session(session_id, os.getcwd(), proc.pid, args)
                    # On reconnect: replay our scrollback so the daemon can restore the
                    # browser's terminal view. First connect uses the pre-buffered queue.
                    if is_reconnect:
                        with relay_scrollback_lock:
                            snapshot = bytes(relay_scrollback) if relay_scrollback else None
                        if snapshot:
                            try:
                                ws.send_bytes(snapshot)
                            except Exception:
                                pass
                    # Start send thread AFTER socket is open so ws.send_binary() succeeds
                    send_thread = threading.Thread(target=ws_send_loop, args=(ws,), daemon=True)
                    send_thread.start()

                ws.on_open = on_open
                ws.run_forever(ping_interval=30)
                q.put(None)  # stop send thread
            except Exception:
                pass

            if not _relay_active.is_set():
                break
            # Exponential backoff before reconnect (cap at 30s)
            time.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2, 30.0)

    ws_thread = threading.Thread(target=start_ws_relay, daemon=True)
    ws_thread.start()

    # Save terminal settings, switch to raw mode
    old_settings = termios.tcgetattr(sys.stdin.fileno())
    tty.setraw(sys.stdin.fileno())

    # SIGWINCH: propagate terminal resize to PTY
    def handle_sigwinch(sig, frame):
        try:
            import fcntl
            winsize = struct.pack("HHHH", 0, 0, 0, 0)
            winsize = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, winsize)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass

    signal.signal(signal.SIGWINCH, handle_sigwinch)

    try:
        while True:
            try:
                r, _, _ = select.select([sys.stdin.fileno(), master_fd], [], [], 0.1)
            except (ValueError, OSError):
                break

            if sys.stdin.fileno() in r:
                try:
                    data = os.read(sys.stdin.fileno(), 4096)
                    if data:
                        os.write(master_fd, data)
                except OSError:
                    break

            if master_fd in r:
                try:
                    data = os.read(master_fd, 4096)
                    if not data:
                        break
                    os.write(sys.stdout.fileno(), data)
                    # Maintain relay-side scrollback (survives daemon restarts)
                    with relay_scrollback_lock:
                        relay_scrollback.extend(data)
                        if len(relay_scrollback) > _MAX_RELAY_SCROLLBACK:
                            del relay_scrollback[:len(relay_scrollback) - _MAX_RELAY_SCROLLBACK]
                    # Forward to WebSocket stream
                    if ws_write_queue is not None:
                        try:
                            ws_write_queue.put_nowait(data)
                        except Exception:
                            pass
                except OSError:
                    break

            if proc.poll() is not None:
                break

    finally:
        _relay_active.clear()  # stop reconnect loop
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        try:
            os.close(master_fd)
        except OSError:
            pass
        deregister_session(session_id)

    proc.wait()
    return proc.returncode


# ── Windows ConPTY relay ────────────────────────────────────────────────────

def _relay_windows(claude_bin: str, args: list[str], session_id: str) -> int:
    """
    Windows relay using pywinpty ConPTY.

    Creates a ConPTY pseudo-terminal pair:
      PtyProcess wraps claude's stdin/stdout/stderr

    The user's console is bridged:
      user stdin  -> PtyProcess.write() -> claude's stdin
      claude's stdout -> PtyProcess.read() -> user stdout + daemon stream

    WebSocket streaming, scrollback, inject, and resize all mirror the POSIX
    implementation. Falls back to plain subprocess (no relay) if pywinpty is
    not installed.
    """
    try:
        from winpty import PtyProcess
    except ImportError:
        print(
            "vlt session-relay: pywinpty not installed — running claude without relay.\n"
            "Install with: pip install pywinpty",
            file=sys.stderr,
        )
        import subprocess
        proc = subprocess.run([claude_bin] + args)
        return proc.returncode

    import ctypes
    import msvcrt
    import queue as _relay_queue
    import threading
    import time

    cols, rows = os.get_terminal_size()

    env = os.environ.copy()
    env["VLT_SESSION_ID"] = session_id

    relay_args = args[:]
    if "--session-id" not in relay_args and "--resume" not in relay_args:
        relay_args = ["--session-id", session_id] + relay_args

    proc = PtyProcess.spawn(
        [claude_bin] + relay_args,
        dimensions=(rows, cols),
        env=env,
    )

    register_session(session_id, os.getcwd(), os.getpid(), args)

    # Pre-allocate write queue so PTY output is captured from the first byte.
    ws_write_queue: _relay_queue.Queue = _relay_queue.Queue(maxsize=8192)

    # Relay-side scrollback (survives daemon restarts, replayed on reconnect)
    _MAX_RELAY_SCROLLBACK = 2 * 1024 * 1024  # 2 MiB
    relay_scrollback: bytearray = bytearray()
    relay_scrollback_lock: threading.Lock = threading.Lock()

    _relay_active = threading.Event()
    _relay_active.set()

    # ── WebSocket relay thread (mirrors POSIX start_ws_relay) ─────────

    def start_ws_relay():
        nonlocal ws_write_queue
        try:
            import websocket  # websocket-client
        except ImportError:
            return  # No streaming without websocket-client

        daemon_base = get_daemon_url().replace("http://", "ws://").replace("https://", "wss://")
        ws_url = f"{daemon_base}/ws/relay/{session_id}"
        backoff = 1.0
        first_connection = True
        is_reconnect = False

        while _relay_active.is_set():
            if first_connection:
                q = ws_write_queue
                first_connection = False
                is_reconnect = False
            else:
                q = _relay_queue.Queue(maxsize=8192)
                ws_write_queue = q
                is_reconnect = True

            def on_message(ws, message):
                """Handle inject (keystroke forwarding) and resize from browser."""
                try:
                    data = json.loads(message)
                    if data.get("type") == "inject":
                        proc.write(data.get("data", ""))
                    elif data.get("type") == "resize":
                        new_cols = data.get("cols", 80)
                        new_rows = data.get("rows", 24)
                        try:
                            proc.setwinsize(new_rows, new_cols)
                        except Exception:
                            pass
                except Exception:
                    pass

            def ws_send_loop(ws):
                while True:
                    chunk = q.get()
                    if chunk is None:
                        break
                    try:
                        ws.send_bytes(chunk)
                    except Exception:
                        break

            try:
                ws = websocket.WebSocketApp(ws_url, on_message=on_message)

                def on_open(ws):
                    nonlocal backoff
                    backoff = 1.0
                    register_session(session_id, os.getcwd(), os.getpid(), args)
                    if is_reconnect:
                        with relay_scrollback_lock:
                            snapshot = bytes(relay_scrollback) if relay_scrollback else None
                        if snapshot:
                            try:
                                ws.send_bytes(snapshot)
                            except Exception:
                                pass
                    send_thread = threading.Thread(
                        target=ws_send_loop, args=(ws,), daemon=True,
                    )
                    send_thread.start()

                ws.on_open = on_open
                ws.run_forever(ping_interval=30)
                q.put(None)  # stop send thread
            except Exception:
                pass

            if not _relay_active.is_set():
                break
            time.sleep(min(backoff, 30.0))
            backoff = min(backoff * 2, 30.0)

    ws_thread = threading.Thread(target=start_ws_relay, daemon=True)
    ws_thread.start()

    # ── Enable Windows virtual terminal processing for ANSI sequences ─

    _ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    _ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
    kernel32 = ctypes.windll.kernel32
    # stdout: enable ANSI escape rendering
    stdout_handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
    mode = ctypes.c_ulong()
    kernel32.GetConsoleMode(stdout_handle, ctypes.byref(mode))
    old_stdout_mode = mode.value
    kernel32.SetConsoleMode(stdout_handle, mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING)
    # stdin: enable virtual terminal input for raw key capture
    stdin_handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
    kernel32.GetConsoleMode(stdin_handle, ctypes.byref(mode))
    old_stdin_mode = mode.value
    kernel32.SetConsoleMode(stdin_handle, mode.value | _ENABLE_VIRTUAL_TERMINAL_INPUT)

    # ── I/O bridge threads ────────────────────────────────────────────

    def stdin_to_pty():
        """Read user keystrokes and forward to the ConPTY."""
        while _relay_active.is_set() and proc.isalive():
            try:
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    proc.write(ch)
                else:
                    time.sleep(0.01)  # Avoid busy-wait
            except Exception:
                break

    def pty_to_stdout():
        """Read ConPTY output, display locally, buffer for WebSocket."""
        nonlocal ws_write_queue
        while _relay_active.is_set() and proc.isalive():
            try:
                data = proc.read(4096)
                if not data:
                    time.sleep(0.01)
                    continue
                # Write to local console
                sys.stdout.write(data)
                sys.stdout.flush()
                # Encode for scrollback + WebSocket (ConPTY returns str)
                raw = data.encode("utf-8", errors="replace")
                # Maintain relay-side scrollback
                with relay_scrollback_lock:
                    relay_scrollback.extend(raw)
                    if len(relay_scrollback) > _MAX_RELAY_SCROLLBACK:
                        del relay_scrollback[:len(relay_scrollback) - _MAX_RELAY_SCROLLBACK]
                # Forward to WebSocket stream
                if ws_write_queue is not None:
                    try:
                        ws_write_queue.put_nowait(raw)
                    except Exception:
                        pass
            except Exception:
                break

    t_in = threading.Thread(target=stdin_to_pty, daemon=True)
    t_out = threading.Thread(target=pty_to_stdout, daemon=True)
    t_in.start()
    t_out.start()

    # ── Console resize polling (Windows has no SIGWINCH) ──────────────

    def resize_poll():
        """Poll terminal size every 500ms and propagate changes to ConPTY."""
        last_size = (cols, rows)
        while _relay_active.is_set() and proc.isalive():
            try:
                cur = os.get_terminal_size()
                if (cur.columns, cur.lines) != last_size:
                    last_size = (cur.columns, cur.lines)
                    proc.setwinsize(cur.lines, cur.columns)
            except Exception:
                pass
            time.sleep(0.5)

    t_resize = threading.Thread(target=resize_poll, daemon=True)
    t_resize.start()

    # ── Wait for process exit ─────────────────────────────────────────

    try:
        proc.wait()
    finally:
        _relay_active.clear()
        # Restore original console modes
        try:
            kernel32.SetConsoleMode(stdout_handle, old_stdout_mode)
            kernel32.SetConsoleMode(stdin_handle, old_stdin_mode)
        except Exception:
            pass
        deregister_session(session_id)

    return proc.exitstatus or 0


# ── Public entry point ───────────────────────────────────────────────────────

def run_relay(args: list[str]) -> int:
    """
    Main entry point. Called from the CLI command.
    Detects platform and runs the appropriate relay.
    """
    # Reuse the --resume session ID so the relay inject queue matches the
    # hook-reported session ID, enabling webchat ↔ CLI bidirectional messaging.
    _resume_idx = next((i for i, a in enumerate(args) if a == "--resume"), None)
    if _resume_idx is not None and _resume_idx + 1 < len(args):
        session_id = args[_resume_idx + 1]
    else:
        session_id = str(uuid.uuid4())

    try:
        claude_bin = find_real_claude()
    except RuntimeError as e:
        print(f"vlt session-relay: {e}", file=sys.stderr)
        return 1

    if platform.system() == "Windows":
        return _relay_windows(claude_bin, args, session_id)
    else:
        return _relay_posix(claude_bin, args, session_id)
