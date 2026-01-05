"""
VLT Daemon Manager - Process lifecycle management for the daemon.

Provides start/stop/status operations for the daemon process.
The daemon runs as a background process and writes its PID to a file for management.

Profile Support:
Each profile has its own daemon with:
- Unique port (hash-based from profile name)
- Separate PID file (~/.vlt/profiles/{profile}/daemon.pid)
- Separate log file (~/.vlt/profiles/{profile}/daemon.log)

This allows multiple profiles to run daemons simultaneously without conflicts.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from .client import DaemonClient, DaemonStatus

logger = logging.getLogger(__name__)


class DaemonManager:
    """
    Manages the vlt daemon process lifecycle.

    The daemon runs as a background process on localhost. This manager handles:
    - Starting the daemon (with or without foreground mode)
    - Stopping the daemon gracefully
    - Checking daemon status

    Each profile has its own daemon instance with isolated PID/log files and port.
    """

    def __init__(
        self,
        port: Optional[int] = None,
        profile_name: Optional[str] = None,
    ):
        """
        Initialize daemon manager.

        Args:
            port: Port for daemon to listen on. If None, uses profile-specific port.
            profile_name: Profile to manage. If None, uses active profile.
        """
        from vlt.profile import get_profile_manager

        self.profile_manager = get_profile_manager()
        self.profile_name = profile_name or self.profile_manager.get_active_profile()

        # Get profile-specific paths
        profile_dir = self.profile_manager.get_profile_dir(self.profile_name)
        self.pid_file = profile_dir / "daemon.pid"
        self.log_file = profile_dir / "daemon.log"

        # Get port - use provided or profile-specific
        if port is not None:
            self.port = port
        else:
            self.port = self.profile_manager.get_daemon_port(self.profile_name)

        self.client = DaemonClient(f"http://127.0.0.1:{self.port}")

    def _ensure_profile_dir(self) -> None:
        """Ensure profile directory exists."""
        profile_dir = self.profile_manager.get_profile_dir(self.profile_name)
        profile_dir.mkdir(parents=True, exist_ok=True)

    def _read_pid(self) -> Optional[int]:
        """Read PID from file, returns None if not found or invalid."""
        if not self.pid_file.exists():
            return None

        try:
            with open(self.pid_file, "r") as f:
                pid_str = f.read().strip()
                if pid_str:
                    return int(pid_str)
        except (ValueError, IOError) as e:
            logger.debug(f"Error reading PID file: {e}")

        return None

    def _write_pid(self, pid: int) -> None:
        """Write PID to file."""
        self._ensure_profile_dir()
        with open(self.pid_file, "w") as f:
            f.write(str(pid))

    def _remove_pid(self) -> None:
        """Remove PID file."""
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except IOError as e:
                logger.debug(f"Error removing PID file: {e}")

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            os.kill(pid, 0)  # Signal 0 doesn't kill, just checks
            return True
        except OSError:
            return False

    def start(self, foreground: bool = False) -> Dict[str, any]:
        """
        Start the daemon process.

        Args:
            foreground: If True, run in foreground (blocking). If False, run as background process.

        Returns:
            Dict with success, message, and pid (if background).
        """
        import asyncio

        # Check if already running
        if asyncio.run(self.client.is_running()):
            return {
                "success": False,
                "message": f"Daemon is already running on port {self.port}",
                "pid": self._read_pid(),
                "profile": self.profile_name,
            }

        # Check for stale PID file
        old_pid = self._read_pid()
        if old_pid and not self._is_process_running(old_pid):
            logger.info(f"Removing stale PID file (process {old_pid} not running)")
            self._remove_pid()

        self._ensure_profile_dir()

        if foreground:
            # Run in foreground (blocking)
            return self._run_foreground()
        else:
            # Run as background process
            return self._run_background()

    def _run_foreground(self) -> Dict[str, any]:
        """Run daemon in foreground mode (blocking)."""
        from .server import run_server

        try:
            # Write our own PID
            self._write_pid(os.getpid())

            # This blocks until the server is stopped
            run_server(
                host="127.0.0.1",
                port=self.port,
                profile_name=self.profile_name,
            )

            return {
                "success": True,
                "message": "Daemon stopped",
                "pid": None,
                "profile": self.profile_name,
            }
        except KeyboardInterrupt:
            return {
                "success": True,
                "message": "Daemon stopped by user",
                "pid": None,
                "profile": self.profile_name,
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Daemon error: {e}",
                "pid": None,
                "profile": self.profile_name,
            }
        finally:
            self._remove_pid()

    def _run_background(self) -> Dict[str, any]:
        """Run daemon as background process."""
        # Find the Python interpreter
        python_exe = sys.executable

        # Build the command to run the daemon server module with profile
        cmd = [
            python_exe,
            "-m", "vlt.daemon.server",
            "--port", str(self.port),
            "--profile", self.profile_name,
        ]

        # Open log file for output
        with open(self.log_file, "a") as log:
            log.write(f"\n{'=' * 60}\n")
            log.write(f"Starting daemon at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write(f"Profile: {self.profile_name}\n")
            log.write(f"Port: {self.port}\n")
            log.write(f"Command: {' '.join(cmd)}\n")
            log.write(f"{'=' * 60}\n")
            log.flush()

            # Start process
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=log,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,  # Detach from parent
                    cwd=str(Path.home()),  # Run from home directory
                )
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Failed to start daemon: {e}",
                    "pid": None,
                    "profile": self.profile_name,
                }

        # Write PID
        self._write_pid(process.pid)

        # Wait a moment for the server to start
        time.sleep(0.5)

        # Verify it's running
        import asyncio
        if asyncio.run(self.client.is_running()):
            return {
                "success": True,
                "message": f"Daemon started on port {self.port} (profile: {self.profile_name})",
                "pid": process.pid,
                "profile": self.profile_name,
            }
        else:
            # Check if process is still alive
            if self._is_process_running(process.pid):
                # Process is running but not responding yet, might just need more time
                time.sleep(1.0)
                if asyncio.run(self.client.is_running()):
                    return {
                        "success": True,
                        "message": f"Daemon started on port {self.port} (profile: {self.profile_name})",
                        "pid": process.pid,
                        "profile": self.profile_name,
                    }

            # Process failed to start properly
            self._remove_pid()
            return {
                "success": False,
                "message": f"Daemon started but not responding. Check {self.log_file}",
                "pid": process.pid,
                "profile": self.profile_name,
            }

    def stop(self) -> Dict[str, any]:
        """
        Stop the daemon process.

        Returns:
            Dict with success and message.
        """
        pid = self._read_pid()

        if not pid:
            return {
                "success": False,
                "message": "No PID file found - daemon may not be running",
                "profile": self.profile_name,
            }

        if not self._is_process_running(pid):
            self._remove_pid()
            return {
                "success": True,
                "message": "Daemon was not running (cleaned up stale PID file)",
                "profile": self.profile_name,
            }

        # Try graceful shutdown with SIGTERM
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            self._remove_pid()
            return {
                "success": False,
                "message": f"Failed to send SIGTERM: {e}",
                "profile": self.profile_name,
            }

        # Wait for process to exit
        for _ in range(30):  # Wait up to 3 seconds
            if not self._is_process_running(pid):
                self._remove_pid()
                return {
                    "success": True,
                    "message": "Daemon stopped gracefully",
                    "profile": self.profile_name,
                }
            time.sleep(0.1)

        # Force kill if still running
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.1)
            self._remove_pid()
            return {
                "success": True,
                "message": "Daemon force killed (SIGKILL)",
                "profile": self.profile_name,
            }
        except OSError as e:
            return {
                "success": False,
                "message": f"Failed to kill daemon: {e}",
                "profile": self.profile_name,
            }

    def status(self) -> Dict[str, any]:
        """
        Get daemon status.

        Returns:
            Dict with running state and details.
        """
        import asyncio

        pid = self._read_pid()
        daemon_status = asyncio.run(self.client.get_status())

        if daemon_status.running:
            return {
                "running": True,
                "pid": pid,
                "port": self.port,
                "profile": self.profile_name,
                "uptime_seconds": daemon_status.uptime_seconds,
                "backend_url": daemon_status.backend_url,
                "backend_connected": daemon_status.backend_connected,
                "queue_size": daemon_status.queue_size,
            }
        else:
            # Check if process exists but isn't responding
            if pid and self._is_process_running(pid):
                return {
                    "running": False,
                    "pid": pid,
                    "port": self.port,
                    "profile": self.profile_name,
                    "message": "Process exists but not responding",
                    "error": daemon_status.error,
                }
            else:
                if pid:
                    self._remove_pid()
                return {
                    "running": False,
                    "pid": None,
                    "port": self.port,
                    "profile": self.profile_name,
                    "message": "Daemon not running",
                }

    def restart(self) -> Dict[str, any]:
        """
        Restart the daemon.

        Returns:
            Dict with success and message.
        """
        stop_result = self.stop()
        if not stop_result.get("success", False) and "not running" not in stop_result.get("message", "").lower():
            return stop_result

        # Small delay before restart
        time.sleep(0.5)

        return self.start(foreground=False)
