import typer
from pathlib import Path
from typing import List
from rich import print
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
from rich.console import Console
from rich.prompt import Prompt, Confirm
import json
import os
import time
import logging
from vlt.core.migrations import init_db
from vlt.core.service import SqliteVaultService
from vlt.core.librarian import Librarian
from vlt.lib.llm import OpenRouterLLMProvider
from vlt.config import Settings
from uuid import uuid4
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

APP_HELP = """
vlt (Vault): Persistent Cognitive State & Semantic Threading for Agents.

'vlt' acts as your Long-Term Semantic Memory, allowing you to decouple your
reasoning state from your immediate context window. It helps you pick up exactly
where you left off, even across different sessions.

THE ARCHITECTURE:
1. STATE PERSISTENCE: Threads are stored permanently. You can retrieve them
   at any time to restore context.
2. COMPRESSED COGNITION: The 'Librarian' background process compresses raw
   thoughts into dense summaries (State Objects), so you don't have to re-read
   entire logs.
3. FAST LOGGING: 'thread push' is optimized for speed (<50ms). Log intermediate
   thoughts freely without slowing down.

PRIMITIVES:
- PROJECT: The bounded context (e.g., 'crypto-bot').
- THREAD:  A specific reasoning chain (e.g., 'optimization-strategy').
- NODE:    An atomic thought or event.
- STATE:   The computed, current truth of a thread (lossy compression).

CORE WORKFLOW:
1. WAKE UP: Run `vlt overview` to see active projects and states.
2. RESUME:  Run `vlt thread read <thread_id>` to load the semantic state.
3. THINK:   Run `vlt thread push <thread_id> "<thought>"` to log progress.
4. SEARCH:  Run `vlt thread seek "<concept>"` to find past solutions.

META-COGNITION STRATEGIES:
1. PROACTIVE INTROSPECTION: Before starting a complex task, ask yourself:
   "Is this a multi-step problem where I might lose my train of thought?"
   If YES, create a thread immediately and log your plan.

2. RELEVANCE PREDICTION: When you make a decision, ask:
   "Will I need to know WHY I did this in 3 days?"
   If YES, push the reasoning (e.g., "Chose SQLite over Postgres for portability").

3. CONTEXT OFFLOADING: If your context window is filling up,
   summarize your current state into `vlt`, then clear your context.
   Trust `vlt` to hold the state while you perform the execution.

4. PROJECT ORCHESTRATION: Do not just log code. Create a dedicated thread
   (e.g., 'planning' or 'meta') to track high-level milestones, architectural
   decisions, and blockers. Use this thread as the "Director" of your work.
"""

THREAD_HELP = """
The Cognitive Loop: Manage reasoning streams.

Use these commands to Create (new), Log (push), Resume (read), and Recall (seek)
your train of thought. This is your primary interface for interacting with the Vault.
"""

app = typer.Typer(name="vlt", help=APP_HELP, no_args_is_help=True)
thread_app = typer.Typer(name="thread", help=THREAD_HELP)
config_app = typer.Typer(name="config", help="Manage configuration and keys.")
sync_app = typer.Typer(name="sync", help="Sync commands for remote backend.")
daemon_app = typer.Typer(name="daemon", help="Background sync daemon management.")
app.add_typer(thread_app, name="thread")
app.add_typer(config_app, name="config")
app.add_typer(sync_app, name="sync")
app.add_typer(daemon_app, name="daemon")

service = SqliteVaultService()

@config_app.command("set-key")
def set_key(
    token: str = typer.Argument(..., help="Server sync token for authentication"),
    server_url: str = typer.Option(None, "--server", "-s", help="Backend server URL (e.g., https://your-server.com)")
):
    """
    Set the server sync token for backend authentication.

    This saves the token to ~/.vlt/.env as VLT_SYNC_TOKEN so you don't have to
    export it every time. The token authenticates vlt-cli with the backend server
    for syncing threads and using server-side features like summarization.

    Get your token from the backend server's settings page or via the /api/tokens endpoint.

    Examples:
        vlt config set-key sk-abc123xyz
        vlt config set-key sk-abc123xyz --server https://my-vault.example.com
    """
    env_path = os.path.expanduser("~/.vlt/.env")

    # Read existing lines to preserve other configs if any
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    # Remove existing sync token if present
    lines = [l for l in lines if not l.startswith("VLT_SYNC_TOKEN=")]

    # Also remove deprecated OpenRouter key reference (migration)
    lines = [l for l in lines if not l.startswith("VLT_OPENROUTER_API_KEY=")]

    # Append new sync token
    lines.append(f"VLT_SYNC_TOKEN={token}\n")

    # Optionally set server URL
    if server_url:
        lines = [l for l in lines if not l.startswith("VLT_VAULT_URL=")]
        lines.append(f"VLT_VAULT_URL={server_url}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    print(f"[green]Sync token saved to {env_path}[/green]")
    if server_url:
        print(f"[green]Server URL set to {server_url}[/green]")
    print("[dim]The CLI will now authenticate with the backend server for sync operations.[/dim]")

from vlt.core.identity import (
    create_vlt_toml,
    load_project_identity,
    load_vlt_config,
    find_vlt_toml,
    find_parent_indexed_root,
    update_vlt_toml_coderag_indexed,
)

# ============================================================================
# Sync Commands (T028-T029)
# ============================================================================

@sync_app.command("status")
def sync_status():
    """
    Show sync status and queue.

    Displays pending entries in the sync queue that failed to sync
    to the remote backend and are waiting for retry.
    """
    from vlt.core.sync import ThreadSyncClient

    client = ThreadSyncClient()
    status = client.get_queue_status()

    if status["pending"] == 0:
        print("[green]Sync queue is empty - all entries synced[/green]")
    else:
        print(f"[yellow]Pending entries: {status['pending']}[/yellow]")
        for item in status["items"]:
            entry_id = item['entry'].get('entry_id', 'unknown')[:8]
            print(f"  - {item['thread_id']}/{entry_id}... (attempts: {item['attempts']})")
            if item.get('error'):
                print(f"    [dim]Last error: {item['error'][:60]}...[/dim]")


@sync_app.command("retry")
def sync_retry():
    """
    Retry failed sync entries.

    Attempts to sync all pending entries in the queue to the remote backend.
    Entries that exceed max retries are skipped but kept for manual review.

    If the daemon is running, routes the retry through it for better connection
    management. Falls back to direct sync if daemon is not available.
    """
    from vlt.daemon.client import DaemonClient
    from vlt.core.sync import ThreadSyncClient
    from vlt.config import settings
    import asyncio

    async def do_retry():
        # Try daemon first if enabled
        if settings.daemon_enabled:
            client = DaemonClient(settings.daemon_url)
            if await client.is_running():
                result = await client.retry_sync()
                if result.success:
                    return {
                        "success": result.synced,
                        "failed": result.failed,
                        "skipped": result.skipped,
                        "via_daemon": True,
                    }
                # If daemon call failed, fall through to direct

        # Fallback to direct sync
        sync_client = ThreadSyncClient()
        result = await sync_client.retry_queue()
        result["via_daemon"] = False
        return result

    result = asyncio.run(do_retry())

    if result.get("via_daemon"):
        print("[dim](via daemon)[/dim]")

    print(f"[green]Success: {result['success']}[/green]")
    print(f"[red]Failed: {result['failed']}[/red]")
    print(f"[yellow]Skipped (max retries): {result['skipped']}[/yellow]")


# ============================================================================
# Daemon Commands
# ============================================================================

@daemon_app.command("start")
def daemon_start(
    port: int = typer.Option(8765, "--port", "-p", help="Port for daemon to listen on"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground (blocking)")
):
    """
    Start the background sync daemon.

    The daemon provides:
    - Persistent HTTP connection to backend (no connection overhead per CLI call)
    - Fast CLI responses (queue and return immediately)
    - Background sync with automatic retry

    By default, runs as a background process. Use --foreground for debugging.

    Examples:
        vlt daemon start                    # Start in background
        vlt daemon start --foreground       # Run in foreground (for debugging)
        vlt daemon start --port 9000        # Use custom port
    """
    from vlt.daemon.manager import DaemonManager

    manager = DaemonManager(port=port)
    result = manager.start(foreground=foreground)

    if result["success"]:
        if foreground:
            # Foreground mode - this will only print after server stops
            print(f"[green]{result['message']}[/green]")
        else:
            print(f"[green]{result['message']}[/green]")
            print(f"PID: {result.get('pid')}")
            print(f"[dim]Log file: ~/.vlt/daemon.log[/dim]")
    else:
        print(f"[red]{result['message']}[/red]")
        raise typer.Exit(code=1)


@daemon_app.command("stop")
def daemon_stop():
    """
    Stop the background sync daemon.

    Sends SIGTERM for graceful shutdown. If the daemon doesn't stop within
    3 seconds, it will be force killed with SIGKILL.
    """
    from vlt.daemon.manager import DaemonManager
    from vlt.config import settings

    manager = DaemonManager(port=settings.daemon_port)
    result = manager.stop()

    if result["success"]:
        print(f"[green]{result['message']}[/green]")
    else:
        print(f"[yellow]{result['message']}[/yellow]")


@daemon_app.command("status")
def daemon_status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """
    Show daemon status and statistics.

    Displays:
    - Running state and PID
    - Uptime
    - Backend connection status
    - Sync queue size
    """
    from vlt.daemon.manager import DaemonManager
    from vlt.config import settings

    manager = DaemonManager(port=settings.daemon_port)
    status = manager.status()

    if json_output:
        print(json.dumps(status, indent=2))
        return

    if status["running"]:
        print(f"[bold green]Daemon is running[/bold green]")
        print(f"  PID: {status.get('pid')}")
        print(f"  Port: {status.get('port')}")

        uptime = status.get("uptime_seconds", 0)
        if uptime > 3600:
            uptime_str = f"{uptime / 3600:.1f} hours"
        elif uptime > 60:
            uptime_str = f"{uptime / 60:.1f} minutes"
        else:
            uptime_str = f"{uptime:.0f} seconds"
        print(f"  Uptime: {uptime_str}")

        backend_status = "[green]connected[/green]" if status.get("backend_connected") else "[yellow]disconnected[/yellow]"
        print(f"  Backend: {status.get('backend_url')} ({backend_status})")
        print(f"  Queue size: {status.get('queue_size', 0)}")
    else:
        print(f"[dim]Daemon is not running[/dim]")
        if status.get("message"):
            print(f"  {status['message']}")
        if status.get("error"):
            print(f"  Error: {status['error']}")


@daemon_app.command("restart")
def daemon_restart():
    """
    Restart the daemon.

    Stops the daemon if running, then starts it again.
    """
    from vlt.daemon.manager import DaemonManager
    from vlt.config import settings

    manager = DaemonManager(port=settings.daemon_port)
    result = manager.restart()

    if result["success"]:
        print(f"[green]{result['message']}[/green]")
        if result.get("pid"):
            print(f"PID: {result['pid']}")
    else:
        print(f"[red]{result['message']}[/red]")
        raise typer.Exit(code=1)


@daemon_app.command("logs")
def daemon_logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    lines: int = typer.Option(20, "--lines", "-n", help="Number of lines to show")
):
    """
    Show daemon logs.

    Displays the daemon log file contents. Use --follow to watch for new entries.
    """
    import subprocess
    from pathlib import Path

    log_file = Path.home() / ".vlt" / "daemon.log"

    if not log_file.exists():
        print("[yellow]No daemon log file found[/yellow]")
        print(f"[dim]Expected at: {log_file}[/dim]")
        return

    if follow:
        # Use tail -f to follow
        try:
            subprocess.run(["tail", "-f", "-n", str(lines), str(log_file)])
        except KeyboardInterrupt:
            pass
    else:
        # Just show last N lines
        try:
            result = subprocess.run(
                ["tail", "-n", str(lines), str(log_file)],
                capture_output=True,
                text=True
            )
            print(result.stdout)
        except Exception as e:
            print(f"[red]Error reading log file: {e}[/red]")


# ...

state = {"author": "user", "show_hint": False}

@app.callback()
def main(
    author: str = typer.Option("user", "--author", help="Identify the speaker (e.g. 'Architect')."),
):
    """
    Vault CLI: Cognitive Hard Drive.
    """
    if author == "user" and not os.environ.get("VLT_AUTHOR"):
        state["show_hint"] = True
    else:
        state["author"] = author or os.environ.get("VLT_AUTHOR", "user")

def _get_platform_service_instructions(vlt_executable: str) -> str:
    """Generate platform-specific daemon startup service instructions.

    Args:
        vlt_executable: Path to the vlt executable

    Returns:
        String with platform-specific instructions
    """
    import platform

    system = platform.system().lower()

    if system == "linux":
        return f"""
[bold]To run vlt daemon on startup (Linux systemd):[/bold]

  mkdir -p ~/.config/systemd/user
  cat > ~/.config/systemd/user/vlt-daemon.service << 'EOF'
  [Unit]
  Description=VLT Daemon
  After=network.target

  [Service]
  ExecStart={vlt_executable} daemon start --foreground
  Restart=on-failure
  RestartSec=5

  [Install]
  WantedBy=default.target
  EOF

  systemctl --user daemon-reload
  systemctl --user enable vlt-daemon
  systemctl --user start vlt-daemon
"""
    elif system == "darwin":  # macOS
        return f"""
[bold]To run vlt daemon on startup (macOS launchd):[/bold]

  cat > ~/Library/LaunchAgents/com.vlt.daemon.plist << 'EOF'
  <?xml version="1.0" encoding="UTF-8"?>
  <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
  <plist version="1.0">
  <dict>
    <key>Label</key>
    <string>com.vlt.daemon</string>
    <key>ProgramArguments</key>
    <array>
      <string>{vlt_executable}</string>
      <string>daemon</string>
      <string>start</string>
      <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/vlt-daemon.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/vlt-daemon.err</string>
  </dict>
  </plist>
  EOF

  launchctl load ~/Library/LaunchAgents/com.vlt.daemon.plist
"""
    elif system == "windows":
        return f"""
[bold]To run vlt daemon on startup (Windows):[/bold]

Option 1: Task Scheduler (GUI)
  1. Open Task Scheduler (taskschd.msc)
  2. Create Basic Task > Name: "VLT Daemon"
  3. Trigger: "When I log on"
  4. Action: Start a program
     Program: {vlt_executable}
     Arguments: daemon start --foreground
  5. Finish

Option 2: Using NSSM (Non-Sucking Service Manager)
  # Install NSSM from https://nssm.cc/
  nssm install VLTDaemon "{vlt_executable}" daemon start --foreground
  nssm start VLTDaemon
"""
    else:
        return f"""
[bold]To run vlt daemon on startup:[/bold]

  Add this command to your system's startup scripts:
    {vlt_executable} daemon start

  Or run manually:
    vlt daemon start
"""


def _perform_health_check(console: Console, target_path: Path) -> dict:
    """Perform health check on existing vlt initialization.

    Verifies:
    - vlt.toml exists and is valid
    - Threads DB connection works
    - CodeRAG index status

    Args:
        console: Rich console for output
        target_path: Path to check

    Returns:
        Dict with health check results
    """
    from vlt.daemon.client import is_daemon_running
    from vlt.db import engine
    from sqlalchemy import text

    results = {
        "vlt_toml": {"status": "unknown", "message": ""},
        "threads_db": {"status": "unknown", "message": ""},
        "coderag_index": {"status": "unknown", "message": ""},
        "daemon": {"status": "unknown", "message": ""},
    }

    # Check vlt.toml
    toml_path = find_vlt_toml(target_path)
    if toml_path:
        try:
            config = load_vlt_config(target_path)
            if config:
                results["vlt_toml"] = {
                    "status": "ok",
                    "message": f"Project: {config.project.name} ({config.project.id})",
                    "path": str(toml_path),
                }
            else:
                results["vlt_toml"] = {
                    "status": "error",
                    "message": "Invalid vlt.toml format",
                    "path": str(toml_path),
                }
        except Exception as e:
            results["vlt_toml"] = {
                "status": "error",
                "message": f"Error reading vlt.toml: {e}",
            }
    else:
        results["vlt_toml"] = {
            "status": "missing",
            "message": "No vlt.toml found",
        }

    # Check threads DB
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM threads"))
            thread_count = result.scalar()
            results["threads_db"] = {
                "status": "ok",
                "message": f"{thread_count} threads in database",
            }
    except Exception as e:
        results["threads_db"] = {
            "status": "error",
            "message": f"Database error: {e}",
        }

    # Check CodeRAG index status
    if results["vlt_toml"]["status"] == "ok" and config:
        try:
            project_id = config.project.id
            if service.has_coderag_index(project_id):
                # Get chunk count
                from sqlalchemy import func, select
                from vlt.core.models import CodeChunk
                from sqlalchemy.orm import Session

                with Session(engine) as session:
                    count = session.scalar(
                        select(func.count()).select_from(CodeChunk)
                        .where(CodeChunk.project_id == project_id)
                    )
                results["coderag_index"] = {
                    "status": "ok",
                    "message": f"{count} code chunks indexed",
                }
            else:
                results["coderag_index"] = {
                    "status": "missing",
                    "message": "No CodeRAG index found",
                }
        except Exception as e:
            results["coderag_index"] = {
                "status": "error",
                "message": f"Error checking index: {e}",
            }
    else:
        results["coderag_index"] = {
            "status": "skipped",
            "message": "No project configured",
        }

    # Check daemon
    if is_daemon_running():
        results["daemon"] = {
            "status": "ok",
            "message": "Daemon is running",
        }
    else:
        results["daemon"] = {
            "status": "stopped",
            "message": "Daemon is not running",
        }

    return results


@app.command()
def init(
    path: Path = typer.Option(None, "--path", "-p", help="Directory to initialize (defaults to current directory)"),
    override_nesting: bool = typer.Option(False, "--override-directory-nesting", help="Allow nested initialization even if parent has index"),
    skip_coderag: bool = typer.Option(False, "--skip-coderag", help="Skip code indexing"),
    skip_daemon: bool = typer.Option(False, "--skip-daemon", help="Don't start daemon"),
):
    """
    Initialize vlt for a project directory.

    This unified command consolidates the initialization workflow:

    FIRST RUN (new project folder):
      1. Initialize threads database (ensure tables exist)
      2. Run coderag init wizard (interactive project selection)
      3. Start the daemon if not already running
      4. Print platform-specific instructions for running daemon as startup service

    SECOND RUN (already initialized):
      1. Perform health check (vlt.toml, DB, CodeRAG index, daemon)
      2. Ensure daemon is running, start if not
      3. Report health status

    CHILD DIRECTORY DETECTION:
      Before initializing, walks up the directory tree looking for existing
      vlt.toml files with coderag indexed. If found, prints a warning and
      exits (use --override-directory-nesting to bypass).

    Examples:
        vlt init                           # Initialize current directory
        vlt init --path /my/project        # Initialize specific directory
        vlt init --skip-coderag            # Skip code indexing
        vlt init --skip-daemon             # Don't start daemon
        vlt init --override-directory-nesting  # Force init in child of indexed dir
    """
    from vlt.daemon.client import is_daemon_running
    from vlt.daemon.manager import DaemonManager
    from vlt.config import settings
    import shutil

    console = Console()
    target_path = (path or Path(".")).resolve()

    # =========================================================================
    # Child Directory Detection
    # =========================================================================
    if not override_nesting:
        parent_result = find_parent_indexed_root(target_path)
        if parent_result:
            parent_toml, indexed_root = parent_result
            console.print()
            console.print("[bold yellow]Warning: This directory is already covered by an existing index.[/bold yellow]")
            console.print()
            console.print(f"  Parent vlt.toml: [cyan]{parent_toml}[/cyan]")
            console.print(f"  Indexed root: [cyan]{indexed_root}[/cyan]")
            console.print()
            console.print("[dim]Use --override-directory-nesting to bypass this check.[/dim]")
            raise typer.Exit(code=1)

    # =========================================================================
    # Check if already initialized (second run)
    # =========================================================================
    existing_toml = find_vlt_toml(target_path)
    is_second_run = existing_toml is not None and existing_toml.parent == target_path

    if is_second_run:
        # =====================================================================
        # Second Run: Health Check
        # =====================================================================
        console.print("[bold blue]VLT Health Check[/bold blue]")
        console.print()

        health = _perform_health_check(console, target_path)

        # Display health status
        status_icons = {
            "ok": "[green]\u2713[/green]",
            "error": "[red]\u2717[/red]",
            "missing": "[yellow]![/yellow]",
            "stopped": "[yellow]\u25cb[/yellow]",
            "skipped": "[dim]-[/dim]",
            "unknown": "[dim]?[/dim]",
        }

        table = Table(title="Health Status", show_header=True)
        table.add_column("Component", style="cyan")
        table.add_column("Status")
        table.add_column("Details")

        for component, info in health.items():
            icon = status_icons.get(info["status"], "[dim]?[/dim]")
            table.add_row(
                component.replace("_", " ").title(),
                f"{icon} {info['status']}",
                info["message"],
            )

        console.print(table)
        console.print()

        # Ensure daemon is running
        if not skip_daemon and health["daemon"]["status"] != "ok":
            console.print("[dim]Starting daemon...[/dim]")
            manager = DaemonManager(port=settings.daemon_port)
            result = manager.start(foreground=False)
            if result["success"]:
                console.print(f"[green]Daemon started (PID: {result.get('pid')})[/green]")
            else:
                console.print(f"[yellow]Warning: Could not start daemon: {result.get('message')}[/yellow]")

        # Offer to initialize CodeRAG if missing
        if not skip_coderag and health.get("coderag_index", {}).get("status") == "missing":
            console.print()
            if Confirm.ask("[yellow]CodeRAG index is missing. Would you like to initialize it now?[/yellow]"):
                console.print()
                # Get project from vlt.toml
                vlt_config = load_vlt_config(target_path)
                if vlt_config and vlt_config.project:
                    project_id = vlt_config.project.id
                    console.print(f"[dim]Starting CodeRAG indexing for project '{project_id}'...[/dim]")
                    console.print()

                    # Run foreground indexing with progress
                    _run_foreground_indexing(project_id, target_path, force=False, console=console)

                    # Update vlt.toml with indexed timestamp
                    from datetime import datetime, timezone
                    indexed_at = datetime.now(timezone.utc).isoformat()
                    update_vlt_toml_coderag_indexed(target_path / "vlt.toml", str(target_path), indexed_at)

                    console.print()
                    console.print(f"[green]✓ CodeRAG index created![/green]")
                    console.print(f"[dim]Check status anytime: vlt coderag status --project {project_id}[/dim]")
                else:
                    console.print("[red]Could not determine project from vlt.toml[/red]")
            else:
                console.print("[dim]Skipping CodeRAG initialization. Run 'vlt coderag init' later.[/dim]")
            console.print()

        # Overall status
        all_ok = all(
            info["status"] in ("ok", "skipped")
            for info in health.values()
        )
        if all_ok:
            console.print("[bold green]All systems healthy![/bold green]")
        else:
            # Re-check after potential fixes
            if health.get("coderag_index", {}).get("status") == "missing":
                console.print("[yellow]Some issues detected. See above for details.[/yellow]")
            else:
                console.print("[bold green]All systems healthy![/bold green]")

        return

    # =========================================================================
    # First Run: Full Initialization
    # =========================================================================
    console.print("[bold blue]Initializing VLT...[/bold blue]")
    console.print()

    # Step 1: Initialize threads database
    console.print("[dim]Step 1/4: Initializing database...[/dim]")
    try:
        init_db()
        console.print("[green]\u2713 Database initialized[/green]")
    except Exception as e:
        console.print(f"[red]Error initializing database: {e}[/red]")
        raise typer.Exit(code=1)

    # Step 2: Interactive project selection and vlt.toml creation
    console.print()
    console.print("[dim]Step 2/4: Project configuration...[/dim]")

    project_id = _interactive_project_selection(console, service)
    if not project_id:
        console.print("[yellow]No project selected. Exiting.[/yellow]")
        raise typer.Exit(code=1)

    # Create vlt.toml if it doesn't exist
    toml_path = target_path / "vlt.toml"
    if not toml_path.exists():
        # Get project name
        try:
            project = service.db.get(service.db.get.__self__.__class__.__bases__[0], project_id)
            project_name = project.name if project else project_id
        except Exception:
            project_name = project_id

        create_vlt_toml(target_path, name=project_name, id=project_id)
        console.print(f"[green]\u2713 Created vlt.toml for project '{project_id}'[/green]")
    else:
        console.print(f"[green]\u2713 Using existing vlt.toml[/green]")

    # Step 3: CodeRAG indexing
    console.print()
    if skip_coderag:
        console.print("[dim]Step 3/4: Skipping code indexing (--skip-coderag)[/dim]")
    else:
        console.print("[dim]Step 3/4: Initializing code index...[/dim]")

        # Check for daemon (needed for background indexing)
        daemon_running = is_daemon_running()

        if daemon_running:
            # Queue background indexing job
            job_id = _queue_background_indexing(project_id, target_path, force=False)
            console.print(f"[green]\u2713 Indexing job queued (ID: {job_id[:8]}...)[/green]")
            console.print(f"[dim]  Check status: vlt coderag status --project {project_id}[/dim]")
        else:
            # Run foreground indexing
            console.print("[dim]  (Running in foreground since daemon is not started yet)[/dim]")
            _run_foreground_indexing(project_id, target_path, force=False, console=console)

    # Update vlt.toml with indexed_root tracking
    indexed_at = datetime.now(timezone.utc).isoformat()
    update_vlt_toml_coderag_indexed(toml_path, str(target_path), indexed_at)

    # Step 4: Start daemon
    console.print()
    if skip_daemon:
        console.print("[dim]Step 4/4: Skipping daemon start (--skip-daemon)[/dim]")
    else:
        console.print("[dim]Step 4/4: Starting daemon...[/dim]")

        if is_daemon_running():
            console.print("[green]\u2713 Daemon already running[/green]")
        else:
            manager = DaemonManager(port=settings.daemon_port)
            result = manager.start(foreground=False)
            if result["success"]:
                console.print(f"[green]\u2713 Daemon started (PID: {result.get('pid')})[/green]")
            else:
                console.print(f"[yellow]Warning: Could not start daemon: {result.get('message')}[/yellow]")

    # Final summary
    console.print()
    console.print("[bold green]VLT initialization complete![/bold green]")
    console.print()
    console.print(f"  Project: [cyan]{project_id}[/cyan]")
    console.print(f"  Path: [cyan]{target_path}[/cyan]")
    console.print()

    # Platform-specific service instructions
    vlt_executable = shutil.which("vlt") or "vlt"
    instructions = _get_platform_service_instructions(vlt_executable)
    console.print(Panel(instructions, title="Daemon Startup Service", border_style="dim"))
# ...

@thread_app.command("new")
def new_thread(
    name: str = typer.Argument(..., help="Thread slug (e.g. 'optim-strategy')"),
    initial_thought: str = typer.Argument(..., help="Initial thought"),
    project: str = typer.Option(None, "--project", "-p", help="Project slug. Defaults to vlt.toml context."),
    author: str = typer.Option(None, "--author", help="Override the author for this thread.")
):
    """
    The Cognitive Loop: Start a new reasoning chain.
    
    Creates a dedicated stream. Links it to a Project context.
    If 'vlt.toml' is present, the project is auto-detected.
    """
    # Resolve Author
    effective_author = author or state["author"]

    # 1. Resolve Project
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id
        else:
            print("[red]Error: No project specified and no vlt.toml found.[/red]")
            print("Usage: vlt thread new <name> <thought> --project <project>")
            print("Or run: vlt init --name <name>")
            raise typer.Exit(code=1)

    print(f"DEBUG: Creating thread {project}/{name}")
    # Ensure project exists (auto-create for MVP)
    try:
        service.create_project(name=project, description="Auto-created project")
    except Exception:
        # Project might already exist, which is fine for now
        pass
        
    thread = service.create_thread(project_id=project, name=name, initial_thought=initial_thought, author=effective_author)
    print(f"[bold green]CREATED:[/bold green] {thread.project_id}/{thread.id}")
    print(f"STATUS: {thread.status}")
    
    if effective_author == "user" and not os.environ.get("VLT_AUTHOR"):
        print("[dim](Tip: Use --author to sign your thoughts)[/dim]")

@thread_app.command("push")
def push_thought(
    thread_id: str = typer.Argument(..., help="Thread slug or path"),
    content: str = typer.Argument(..., help="The thought to log"),
    author: str = typer.Option(None, "--author", help="Override the author for this thought.")
):
    """
    The Cognitive Loop: Commit a thought to permanent memory.

    Fire-and-forget logging. Use this to offload intermediate reasoning steps so you
    can free up context window space.

    If the daemon is running, sync is routed through it for better performance
    (persistent connection, immediate queue response). Falls back to direct sync
    if daemon is not available.
    """
    # Resolve Author
    effective_author = author or state["author"]

    # Assuming thread_id format is project/thread or just thread if unique?
    # For MVP assume we pass just thread slug or handle project/thread splitting if needed.
    # The spec examples show `vlt thread push crypto-bot/optim-strategy`.
    # Our DB stores thread_id as slug.

    # Simple parsing if composite ID is passed
    if "/" in thread_id:
        _, thread_slug = thread_id.split("/")
    else:
        thread_slug = thread_id

    node = service.add_thought(thread_id=thread_slug, content=content, author=effective_author)
    print(f"[bold green]OK:[/bold green] {node.thread_id}/{node.sequence_id}")

    # Sync to backend if configured
    from vlt.config import settings
    import asyncio
    from datetime import datetime

    # Only attempt sync if server is configured
    if settings.is_server_configured:
        try:
            thread_info = service.get_thread_state(thread_slug, limit=1)
            if thread_info:
                entry = {
                    "entry_id": node.id,
                    "sequence_id": node.sequence_id,
                    "content": content,
                    "author": effective_author,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Try daemon first if enabled
                synced = False
                via_daemon = False

                if settings.daemon_enabled:
                    from vlt.daemon.client import DaemonClient

                    async def try_daemon():
                        client = DaemonClient(settings.daemon_url)
                        if await client.is_running():
                            result = await client.enqueue_sync(
                                thread_id=thread_slug,
                                project_id=thread_info.project_id,
                                name=thread_info.thread_id,
                                entry=entry,
                            )
                            return result.success, not result.queued  # synced if not queued
                        return False, False  # Not running, didn't sync

                    daemon_ok, synced = asyncio.run(try_daemon())
                    via_daemon = daemon_ok

                # Fallback to direct sync if daemon not available
                if not via_daemon:
                    from vlt.core.sync import sync_thread_entry

                    synced = asyncio.run(sync_thread_entry(
                        thread_id=thread_slug,
                        project_id=thread_info.project_id,
                        name=thread_info.thread_id,
                        entry_id=node.id,
                        sequence_id=node.sequence_id,
                        content=content,
                        author=effective_author,
                    ))

                if synced:
                    msg = "[dim]Synced to server[/dim]"
                    if via_daemon:
                        msg += " [dim](via daemon)[/dim]"
                    print(msg)
                else:
                    msg = "[dim yellow]Queued for sync (will retry)[/dim yellow]"
                    if via_daemon:
                        msg += " [dim](via daemon)[/dim]"
                    print(msg)
        except Exception as e:
            # Don't fail push if sync fails
            logger.debug(f"Sync failed (non-fatal): {e}")
            print("[dim yellow]Sync pending (will retry later)[/dim yellow]")

    if effective_author == "user" and not os.environ.get("VLT_AUTHOR"):
        print("[dim](Tip: Use --author to sign your thoughts)[/dim]")
@app.command("overview")
def overview(project_id: str = typer.Argument(None, help="Project ID"), json_output: bool = typer.Option(False, "--json", help="Output as JSON")):
    """
    List active Projects and their Thread States.
    
    The 'Wake Up' command. Use this to orient yourself in the broader project context
    before diving into specific threads.
    """
    if not project_id:
        identity = load_project_identity()
        if identity:
            project_id = identity.id
        else:
            # Fallback to "default" or list all?
            # For now, require it or default.
            project_id = "default"

    view = service.get_project_overview(project_id)
    
    if json_output:
        print(json.dumps(view.model_dump(), default=str))
        return

    print(Panel(Markdown(f"# Project: {view.project_id}\n\n{view.summary}"), title="Project Overview", border_style="blue"))
    
    table = Table(title="Active Threads")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="magenta")
    
    for t in view.active_threads:
        table.add_row(t["id"], t["status"])
        
    print(table)
@thread_app.command("read")
def read_thread(
    thread_id: str, 
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show full thread history."),
    search_query: str = typer.Option(None, "--search", "-s", help="Semantic search within this thread.")
):
    """
    The Cognitive Loop: Load the Semantic State.
    
    Retrieves the compressed 'Truth' of a thread (State).
    By default, shows only the Summary and last 5 thoughts.
    Use --all to see everything, or --search to find specific details.
    """
    # 1. Search Mode
    if search_query:
        results = service.search_thread(thread_id, search_query)
        if json_output:
            print(json.dumps([r.model_dump() for r in results], default=str))
            return
            
        print(Panel(f"Search Results for '{search_query}' in {thread_id}", border_style="cyan"))
        for res in results:
            score_color = "green" if res.score > 0.8 else "yellow"
            print(f"[[{score_color}]{res.score:.2f}[/{score_color}]] {res.content}")
        return

    # 2. Read Mode
    limit = -1 if show_all else 5
    
    # Context for potential repair
    current_project = "orphaned"
    identity = load_project_identity()
    if identity:
        current_project = identity.id
        
    view = service.get_thread_state(thread_id, limit=limit, current_project_id=current_project)
    
    if json_output:
        print(json.dumps(view.model_dump(), default=str))
        return

    print(Panel(Markdown(f"# Thread: {view.thread_id}\n**Project:** {view.project_id}\n\n{view.summary}"), title="Thread State", border_style="green"))
    
    if view.meta:
         print(Panel(str(view.meta), title="Meta", border_style="yellow"))

    print(f"\n[bold]Recent Thoughts ({'All' if show_all else 'Last 5'}):[/bold]")
    for node in view.recent_nodes:
        author_str = f"[{node.author}]" if node.author != "user" else ""
        print(f"[dim]{node.sequence_id} | {node.timestamp.strftime('%H:%M:%S')}[/dim] [cyan]{author_str}[/cyan] {node.content}")
librarian_app = typer.Typer(name="librarian", help="Background daemon for summarization and embeddings.")
app.add_typer(librarian_app, name="librarian")

# CodeRAG subcommand group
coderag_app = typer.Typer(name="coderag", help="Code intelligence and indexing for hybrid retrieval.")
app.add_typer(coderag_app, name="coderag")

@librarian_app.command("run")
def run_librarian(
    daemon: bool = typer.Option(False, "--daemon", "-d", help="Run continuously in background"),
    interval: int = typer.Option(10, "--interval", "-i", help="Seconds between processing runs (daemon mode)"),
    legacy: bool = typer.Option(False, "--legacy", help="Use deprecated local LLM calls instead of server")
):
    """
    Process pending nodes into summaries using server-side LLM.

    By default, this command uses the backend server for summarization,
    which handles LLM API keys and billing centrally. Threads must be
    synced to the server first.

    To configure server access:
        vlt config set-key <your-sync-token>

    The --legacy flag enables the deprecated local LLM mode, which requires
    configuring your own OpenRouter API key. This mode will be removed in
    a future version.
    """
    from vlt.core.librarian import ServerLibrarian, Librarian

    if legacy:
        # Deprecated: Use local LLM provider
        print("[yellow]WARNING: Using deprecated local LLM mode.[/yellow]")
        print("[yellow]This mode requires your own OpenRouter API key and will be removed.[/yellow]")
        print("[dim]Consider using server-side summarization instead: vlt config set-key <token>[/dim]")
        print()

        llm = OpenRouterLLMProvider()
        librarian = Librarian(llm_provider=llm)

        print("[bold blue]Librarian started (legacy mode).[/bold blue]")

        while True:
            try:
                print("Processing pending nodes...")
                nodes_count = librarian.process_pending_nodes()
                if nodes_count > 0:
                    print(f"[green]Processed {nodes_count} nodes.[/green]")

                    print("Updating project overviews...")
                    proj_count = librarian.update_project_overviews()
                    print(f"[green]Updated {proj_count} projects.[/green]")
                else:
                    print("No new nodes.")

            except Exception as e:
                print(f"[red]Error:[/red] {e}")

            if not daemon:
                break

            time.sleep(interval)
    else:
        # New: Use server-side summarization
        librarian = ServerLibrarian()

        # Check for sync token
        if not librarian.sync_token:
            print("[red]Error: No sync token configured.[/red]")
            print("Run: vlt config set-key <your-sync-token>")
            print("[dim]Or use --legacy flag to use local LLM calls (deprecated)[/dim]")
            raise typer.Exit(code=1)

        print(f"[bold blue]Librarian started (server: {librarian.vault_url}).[/bold blue]")
        print()
        print("[dim]The librarian will:[/dim]")
        print("[dim]1. Sync all local threads to the server[/dim]")
        print("[dim]2. Request server-side summarization for each thread[/dim]")
        print()

        while True:
            try:
                print("Syncing and processing threads via server...")
                nodes_count = librarian.process_pending_nodes_via_server()
                if nodes_count > 0:
                    print(f"[green]Processed {nodes_count} nodes via server.[/green]")
                else:
                    print("[dim]No new nodes to summarize.[/dim]")

            except Exception as e:
                print(f"[red]Error:[/red] {e}")
                import traceback
                traceback.print_exc()

            if not daemon:
                break

            time.sleep(interval)
@thread_app.command("move")
def move_thread(
    thread_id: str = typer.Argument(..., help="Thread slug"),
    project_id: str = typer.Argument(..., help="Target Project ID")
):
    """
    Move a thread to a different project.
    
    Useful for reorganizing orphaned threads or correcting mistakes.
    """
    try:
        thread = service.move_thread(thread_id, project_id)
        print(f"[green]Moved thread '{thread.id}' to project '{thread.project_id}'[/green]")
    except Exception as e:
        print(f"[red]Error moving thread: {e}[/red]")

@thread_app.command("seek")
def seek(query: str, project: str = typer.Option(None, "--project", "-p", help="Filter by project")):
    """
    The Cognitive Loop: Semantic Search.
    
    Query your permanent memory for similar problems or solutions encountered in the past.
    """
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id

    results = service.search(query, project_id=project)
    
    if not results:
        print("[yellow]No matches found.[/yellow]")
        return
        
    for res in results:
        score_color = "green" if res.score > 0.8 else "yellow"
        print(f"[[{score_color}]{res.score:.2f}[/{score_color}]] [bold]{res.thread_id}[/bold] ({res.node_id[:8]}): {res.content}")

@app.command()
def tag(node_id: str, name: str):
    """
    Attach a semantic tag to a specific node (thought).
    
    Tags allow for cross-cutting taxonomy (e.g., #bug, #architecture).
    """
    try:
        tag = service.add_tag(node_id, name)
        print(f"[green]Tagged node {node_id[:8]} with #{tag.name}[/green]")
    except Exception as e:
        print(f"[red]Error tagging node: {e}[/red]")

@app.command()
def link(source_node_id: str, target_thread: str, note: str = "Relates to"):
    """
    Create a semantic link between a thought and another thread.

    Use this to connect reasoning chains (e.g., 'This bug relates to physics-engine').
    """
    try:
        ref = service.add_reference(source_node_id, target_thread, note)
        print(f"[green]Linked node {source_node_id[:8]} -> {target_thread} ({note})[/green]")
    except Exception as e:
        print(f"[red]Error linking node: {e}[/red]")


# ============================================================================
# CodeRAG Commands (T027-T030)
# ============================================================================


# ============================================================================
# CodeRAG Interactive Helpers (T011-T014)
# ============================================================================


def _fetch_server_projects(console: Console) -> list[dict] | None:
    """Fetch projects from the backend server.

    Returns:
        List of project dicts with 'id', 'name', 'description' keys, or None on error.
    """
    import httpx
    from vlt.config import settings

    vault_url = settings.vault_url
    sync_token = settings.sync_token

    if not vault_url:
        console.print("[yellow]No vault_url configured. Using local projects only.[/yellow]")
        return None

    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {"Authorization": f"Bearer {sync_token}"} if sync_token else {}
            response = client.get(f"{vault_url}/api/projects", headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("projects", [])
    except httpx.ConnectError:
        console.print(f"[yellow]Cannot connect to server at {vault_url}. Using local projects.[/yellow]")
        return None
    except Exception as e:
        console.print(f"[yellow]Error fetching projects from server: {e}[/yellow]")
        return None


def _create_server_project(console: Console, name: str, description: str = "") -> dict | None:
    """Create a project on the backend server.

    Returns:
        Created project dict with 'id', 'name' keys, or None on error.
    """
    import httpx
    from vlt.config import settings

    vault_url = settings.vault_url
    sync_token = settings.sync_token

    if not vault_url:
        return None

    try:
        with httpx.Client(timeout=10.0) as client:
            headers = {
                "Authorization": f"Bearer {sync_token}",
                "Content-Type": "application/json",
            } if sync_token else {"Content-Type": "application/json"}
            response = client.post(
                f"{vault_url}/api/projects",
                headers=headers,
                json={"name": name, "description": description},
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        console.print(f"[red]Error creating project on server: {e}[/red]")
        return None


def _interactive_project_selection(console: Console, svc: SqliteVaultService) -> str | None:
    """Interactive project selection with create option (T011-T014).

    Fetches projects from the backend server (with fallback to local projects),
    displays them with CodeRAG index status, and allows creating new projects.

    Args:
        console: Rich Console for styled output
        svc: SqliteVaultService for local project operations and index checking

    Returns:
        Selected project ID or None if user cancels
    """
    # Try to fetch projects from server first, fall back to local
    server_projects = _fetch_server_projects(console)

    if server_projects is not None:
        projects = server_projects
        using_server = True
        console.print("[dim]Fetched projects from server.[/dim]")
    else:
        # Fallback to local projects
        local_projects = svc.list_projects()
        projects = [{"id": p.id, "name": p.name, "description": getattr(p, 'description', '')} for p in local_projects]
        using_server = False

    # T012: Numbered project list display
    if not projects:
        # No projects exist - offer to create one
        console.print("[yellow]No projects found.[/yellow]")
        if Confirm.ask("Create a new project?"):
            name = Prompt.ask("Project name")
            if not name.strip():
                console.print("[red]Project name cannot be empty.[/red]")
                return None
            try:
                if using_server:
                    project = _create_server_project(console, name, "Created via vlt init")
                    if project:
                        # Also create locally for index tracking
                        try:
                            svc.create_project(name, "Created via vlt init", project_id=project["id"])
                        except Exception:
                            pass  # Local creation is optional
                        console.print(f"[green]Created project '{project['name']}' (id: {project['id']})[/green]")
                        return project["id"]
                    return None
                else:
                    project = svc.create_project(name, "Created via vlt init")
                    console.print(f"[green]Created project '{project.name}' (id: {project.id})[/green]")
                    return project.id
            except Exception as e:
                console.print(f"[red]Error creating project: {e}[/red]")
                return None
        return None

    # Display project list with index status
    console.print()
    console.print("[bold]Available Projects:[/bold]")
    if using_server:
        console.print("[dim](from server)[/dim]")
    console.print()

    for i, proj in enumerate(projects, 1):
        proj_id = proj["id"] if isinstance(proj, dict) else proj.id
        proj_name = proj["name"] if isinstance(proj, dict) else proj.name
        has_index = svc.has_coderag_index(proj_id)
        # T012: Mark projects with existing indexes
        index_marker = "[green]\u2713[/green]" if has_index else "[ ]"
        console.print(f"  {i}. [{index_marker}] {proj_name} [dim]({proj_id})[/dim]")

    # T013: Add "Create new project" option
    create_option = len(projects) + 1
    console.print(f"  {create_option}. [cyan]+ Create new project[/cyan]")
    console.print()

    # Get user selection
    valid_choices = [str(i) for i in range(1, create_option + 1)]
    try:
        choice = Prompt.ask(
            "Select project number",
            choices=valid_choices,
            default="1"
        )
    except KeyboardInterrupt:
        console.print("\n[dim]Cancelled.[/dim]")
        return None

    idx = int(choice)

    if idx <= len(projects):
        # User selected an existing project
        selected = projects[idx - 1]
        proj_id = selected["id"] if isinstance(selected, dict) else selected.id
        return proj_id
    else:
        # T014: User chose to create a new project
        try:
            name = Prompt.ask("New project name")
            if not name.strip():
                console.print("[red]Project name cannot be empty.[/red]")
                return None

            if using_server:
                project = _create_server_project(console, name, "Created via vlt init")
                if project:
                    # Also create locally for index tracking
                    try:
                        svc.create_project(name, "Created via vlt init", project_id=project["id"])
                    except Exception:
                        pass  # Local creation is optional
                    console.print(f"[green]Created project '{project['name']}' (id: {project['id']})[/green]")
                    return project["id"]
                return None
            else:
                project = svc.create_project(name, "Created via vlt init")
                console.print(f"[green]Created project '{project.name}' (id: {project.id})[/green]")
                return project.id
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            return None
        except Exception as e:
            console.print(f"[red]Error creating project: {e}[/red]")
            return None


def _queue_background_indexing(
    project_id: str,
    target_path: Path,
    force: bool = False,
    priority: int = 0,
) -> str:
    """Queue a background indexing job for daemon processing.

    Creates a CodeRAGIndexJob record in the database and returns the job ID.
    The daemon will pick up this job and execute the indexing asynchronously.

    Args:
        project_id: Project to index
        target_path: Directory to index
        force: If True, ignore incremental caching
        priority: Job priority (higher = processed first)

    Returns:
        Job ID (UUID string)
    """
    from sqlalchemy.orm import Session
    from vlt.db import engine
    from vlt.core.models import CodeRAGIndexJob, JobStatus

    job_id = str(uuid4())

    with Session(engine) as session:
        job = CodeRAGIndexJob(
            id=job_id,
            project_id=project_id,
            status=JobStatus.PENDING,
            target_path=str(target_path.resolve()),
            force=force,
            priority=priority,
            files_total=0,
            files_processed=0,
            chunks_created=0,
            progress_percent=0,
            created_at=datetime.now(timezone.utc),
        )
        session.add(job)
        session.commit()

    return job_id


def _run_foreground_indexing(
    project_id: str,
    target_path: Path,
    force: bool,
    console: Console,
):
    """Run indexing in foreground with rich progress display.

    Args:
        project_id: Project to index
        target_path: Directory to index
        force: If True, ignore incremental caching
        console: Rich console for output

    T063: Handles edge case when no indexable files are found by displaying
    a clear warning with supported file types and recovery suggestions.
    """
    from vlt.core.coderag.indexer import CodeRAGIndexer
    from vlt.core.coderag.parser import SUPPORTED_LANGUAGES
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TaskProgressColumn

    # Create indexer
    indexer = CodeRAGIndexer(target_path, project_id)

    # Run indexing with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Discovering files...", total=None)

        def on_progress(files_done: int, files_total: int, chunks: int):
            """Progress callback for indexer."""
            if files_total > 0:
                progress.update(
                    task,
                    description=f"Indexing files ({chunks} chunks)...",
                    total=files_total,
                    completed=files_done
                )
            else:
                progress.update(
                    task,
                    description=f"Discovering files... ({files_done} found)"
                )

        try:
            # Run index with progress callback
            stats = indexer.index_full(force=force, progress_callback=on_progress)

            # T063: Handle no files found edge case
            if stats.files_discovered == 0:
                progress.update(task, description="[yellow]No files found[/yellow]", total=1, completed=1)
                console.print()
                console.print("[yellow]Warning: No indexable files found.[/yellow]")
                console.print()
                console.print("[bold]Supported languages:[/bold]")
                console.print(f"  {', '.join(sorted(SUPPORTED_LANGUAGES))}")
                console.print()
                console.print("[bold]Recovery suggestions:[/bold]")
                console.print("  1. Check that the target directory contains source code files")
                console.print("  2. Verify include patterns in coderag.toml (default: **/*.py)")
                console.print("  3. Ensure files are not excluded by patterns in .gitignore")
                console.print("  4. Try specifying a different path: vlt coderag init --path <dir>")
                console.print()
                console.print("[dim]No indexing job was created.[/dim]")
                return

            progress.update(task, description="Indexing complete!", total=1, completed=1)

            # Display results
            console.print()
            console.print("[bold green]Indexing complete![/bold green]")
            console.print()

            # Stats table
            table = Table(title="Index Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Files discovered", str(stats.files_discovered))
            table.add_row("Files indexed", str(stats.files_indexed))
            table.add_row("Files skipped", str(stats.files_skipped))
            table.add_row("Files failed", str(stats.files_failed))
            table.add_row("Chunks created", str(stats.chunks_created))
            table.add_row("Embeddings generated", str(stats.embeddings_generated))
            table.add_row("Symbols indexed", str(stats.symbols_indexed))
            table.add_row("Graph nodes", str(stats.graph_nodes))
            table.add_row("Graph edges", str(stats.graph_edges))
            table.add_row("Time elapsed", f"{stats.duration_seconds:.2f}s")

            console.print(table)

            # Show errors if any
            if stats.errors:
                console.print()
                console.print("[bold red]Errors:[/bold red]")
                for error in stats.errors[:10]:  # Show first 10 errors
                    console.print(f"  - {error}")
                if len(stats.errors) > 10:
                    console.print(f"  ... and {len(stats.errors) - 10} more errors")

            # T019: Confirmation message with status check instructions
            console.print()
            console.print(f"[bold]Check status:[/bold] vlt coderag status --project {project_id}")

        except OSError as e:
            # T065: Handle disk space exhaustion with clear message
            progress.update(task, description="[red]Indexing failed![/red]")
            error_str = str(e)
            if "No space left on device" in error_str or getattr(e, 'errno', 0) == 28:
                console.print()
                console.print("[red]Error: Disk space exhausted during indexing.[/red]")
                console.print()
                console.print("[bold]Recovery:[/bold]")
                console.print("  1. Free up disk space")
                console.print(f"  2. Retry with: vlt coderag init --project {project_id} --force")
            else:
                console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)

        except Exception as e:
            # T066: Handle other errors with recovery suggestions
            progress.update(task, description="[red]Indexing failed![/red]")
            error_str = str(e)
            console.print(f"[red]Error: {error_str}[/red]")

            # Add recovery suggestions for common errors
            if "database is locked" in error_str.lower():
                console.print()
                console.print("[bold]Recovery:[/bold] Close other vlt processes and retry.")
            elif "permission denied" in error_str.lower():
                console.print()
                console.print("[bold]Recovery:[/bold] Check file permissions for the target directory.")

            raise typer.Exit(code=1)


@coderag_app.command("init")
def coderag_init(
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    path: Path = typer.Option(None, "--path", help="Directory to index (defaults to current directory)"),
    force: bool = typer.Option(False, "--force", help="Force re-index (overwrite existing index without confirmation)"),
    background: bool = typer.Option(True, "--background/--foreground", help="Run indexing in background via daemon (default) or foreground"),
):
    """
    Initialize CodeRAG index for a project.

    This command performs a full codebase index:
    - Parses files using tree-sitter
    - Generates context-enriched semantic chunks
    - Creates vector embeddings (qwen/qwen3-embedding-8b)
    - Builds BM25 keyword index
    - Constructs import/call graph
    - Generates repository map
    - Runs ctags for symbol index

    If --project is not specified and no vlt.toml is found, shows an interactive
    project selection menu. Projects with existing indexes are marked with a
    checkmark.

    By default, indexing runs in the background via the daemon so you can
    continue working. Use --foreground to run in the current terminal with
    progress display.

    Use --force to overwrite an existing index without confirmation prompt.

    Examples:
        vlt coderag init                        # Interactive project selection
        vlt coderag init --project myproj       # Specify project directly
        vlt coderag init --foreground           # Foreground with progress
        vlt coderag init --force                # Overwrite without confirmation
    """
    from vlt.core.identity import load_project_identity

    console = Console()

    # T018: Interactive project selection when --project not provided
    if not project:
        # First try vlt.toml
        identity = load_project_identity()
        if identity:
            project = identity.id
            console.print(f"[dim]Using project from vlt.toml: {project}[/dim]")
        else:
            # No vlt.toml - show interactive selection (T011-T014)
            project = _interactive_project_selection(console, service)
            if not project:
                raise typer.Exit(code=1)

    # T015-T017: Overwrite detection and protection
    if service.has_coderag_index(project) and not force:
        console.print()
        console.print(f"[yellow]Warning: Project '{project}' already has a code index.[/yellow]")
        console.print("[dim]Re-indexing will replace the existing index.[/dim]")
        console.print()

        if not Confirm.ask("Overwrite existing index?", default=False):
            console.print("[dim]Cancelled. Use --force to skip this prompt.[/dim]")
            raise typer.Exit(code=0)

        # User confirmed overwrite
        console.print()

    # Resolve path
    if not path:
        path = Path(".")

    console.print(f"[bold blue]Initializing CodeRAG index for project '{project}'[/bold blue]")
    console.print(f"Path: {path.resolve()}")
    console.print(f"Mode: {'Full re-index' if force else 'Incremental'}")
    console.print(f"Execution: {'Background (daemon)' if background else 'Foreground'}")
    console.print()

    # Run indexing
    if background:
        # T064: Check if daemon is running before queuing job
        from vlt.daemon.client import is_daemon_running
        if not is_daemon_running():
            console.print("[yellow]Warning: Daemon is not running.[/yellow]")
            console.print()
            console.print("[bold]Options:[/bold]")
            console.print("  1. Start the daemon: [bold]vlt daemon start[/bold]")
            console.print("  2. Run in foreground: [bold]vlt coderag init --foreground[/bold]")
            console.print()
            # Ask user if they want to run in foreground instead
            if Confirm.ask("Run indexing in foreground instead?", default=True):
                console.print()
                _run_foreground_indexing(project, path, force, console)
            else:
                console.print("[dim]Indexing cancelled. Start daemon first or use --foreground.[/dim]")
                raise typer.Exit(code=1)
        else:
            # Daemon is running - queue job for background processing
            job_id = _queue_background_indexing(project, path, force)
            console.print(f"[green]Indexing job queued.[/green] Job ID: {job_id[:8]}...")
            console.print()
            console.print("[dim]The daemon will process this job in the background.[/dim]")
            console.print(f"[dim]Check status: [bold]vlt coderag status --project {project}[/bold][/dim]")
    else:
        # Run in foreground with progress display
        _run_foreground_indexing(project, path, force, console)


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string (e.g., '2m 34s')."""
    if seconds < 0:
        return "0s"
    total_seconds = int(seconds)
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes = total_seconds // 60
    secs = total_seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


def _build_job_status_json(job, index_status: dict) -> dict:
    """Build JSON-compatible job status response matching JobStatusResponse schema."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    # Calculate duration
    duration_seconds = None
    if job.started_at:
        end_time = job.completed_at or now
        duration_seconds = (end_time - job.started_at).total_seconds()

    # Calculate ETA
    eta_seconds = None
    if job.started_at and job.progress_percent > 0 and job.status.value == "running":
        elapsed = (now - job.started_at).total_seconds()
        eta_seconds = (elapsed / job.progress_percent) * (100 - job.progress_percent)

    # Map job status to index status
    # The status field should reflect the INDEX state, not the job state
    job_status_value = job.status.value
    if job_status_value in ("completed", "cancelled"):
        # Completed or cancelled job - check if index has data
        index_status_value = "ready" if index_status.get("chunks_count", 0) > 0 else "not_initialized"
    elif job_status_value in ("running", "pending"):
        index_status_value = "indexing"
    elif job_status_value == "failed":
        index_status_value = "failed"
    else:
        # Unknown status, default based on chunks
        index_status_value = "ready" if index_status.get("chunks_count", 0) > 0 else "not_initialized"

    return {
        "job_id": job.id,
        "project_id": job.project_id,
        "status": index_status_value,  # Index status, not job status
        "progress_percent": job.progress_percent,
        "files_total": job.files_total,
        "files_processed": job.files_processed,
        "chunks_created": job.chunks_created,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "error_message": job.error_message,
        "duration_seconds": duration_seconds,
        "eta_seconds": eta_seconds,
        # Include index stats for full context
        "index_stats": {
            "files_count": index_status.get("files_count", 0),
            "chunks_count": index_status.get("chunks_count", 0),
            "symbols_count": index_status.get("symbols_count", 0),
            "graph_nodes": index_status.get("graph_nodes", 0),
            "graph_edges": index_status.get("graph_edges", 0),
        }
    }


@coderag_app.command("status")
def coderag_status(
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON (T036: machine-readable format)"),
):
    """
    Display index health, statistics, and job progress.

    Shows:
    - Active job progress with files processed, time elapsed, and ETA
    - Completion summary for finished jobs
    - Files count, chunks count, symbols count
    - Graph nodes/edges count
    - Last indexed time
    - Repository map statistics
    - Delta queue count (pending changes)

    Examples:
        vlt coderag status
        vlt coderag status --project my-project
        vlt coderag status --json
    """
    from vlt.core.identity import load_project_identity
    from vlt.core.coderag.indexer import CodeRAGIndexer
    from vlt.core.service import SqliteVaultService
    from vlt.core.models import JobStatus
    from rich.console import Console
    from rich.panel import Panel
    from datetime import datetime, timezone
    import json as json_lib

    console = Console()

    # Resolve project
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id
        else:
            console.print("[red]Error: No project specified and no vlt.toml found.[/red]")
            raise typer.Exit(code=1)

    # Get index status
    indexer = CodeRAGIndexer(Path("."), project)
    status = indexer.get_index_status()

    # T030-T031: Get job status from service
    service = SqliteVaultService()
    active_job = service.get_active_job_for_project(project)
    recent_job = service.get_most_recent_job_for_project(project) if not active_job else None

    # T036: JSON output mode
    if json_output:
        if active_job:
            output = _build_job_status_json(active_job, status)
        elif recent_job:
            output = _build_job_status_json(recent_job, status)
        else:
            # No job history, just return index stats
            output = {
                "job_id": None,
                "project_id": project,
                "status": "not_initialized" if status.get("chunks_count", 0) == 0 else "ready",
                "progress_percent": 100 if status.get("chunks_count", 0) > 0 else 0,
                "files_total": status.get("files_count", 0),
                "files_processed": status.get("files_count", 0),
                "chunks_created": status.get("chunks_count", 0),
                "started_at": None,
                "completed_at": status.get("last_indexed"),
                "error_message": None,
                "duration_seconds": None,
                "eta_seconds": None,
                "index_stats": {
                    "files_count": status.get("files_count", 0),
                    "chunks_count": status.get("chunks_count", 0),
                    "symbols_count": status.get("symbols_count", 0),
                    "graph_nodes": status.get("graph_nodes", 0),
                    "graph_edges": status.get("graph_edges", 0),
                }
            }
        console.print(json_lib.dumps(output, indent=2))
        return

    # T032-T035: Display job progress if there's an active or recent job
    now = datetime.now(timezone.utc)

    if active_job:
        # Show active job progress panel
        job = active_job
        status_emoji = "[yellow]running[/yellow]" if job.status == JobStatus.RUNNING else "[dim]pending[/dim]"

        # T033: Files processed / total with percentage
        progress_line = f"Progress: {job.files_processed}/{job.files_total} files ({job.progress_percent}%)"

        # T034: Time elapsed since started_at
        elapsed_line = ""
        if job.started_at:
            # Ensure both datetimes are timezone-aware for comparison
            started = job.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed_seconds = (now - started).total_seconds()
            elapsed_line = f"Elapsed: {_format_duration(elapsed_seconds)}"

        # T035: Estimated time remaining based on progress rate
        eta_line = ""
        if job.started_at and job.progress_percent > 0 and job.status == JobStatus.RUNNING:
            started = job.started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed_seconds = (now - started).total_seconds()
            eta_seconds = (elapsed_seconds / job.progress_percent) * (100 - job.progress_percent)
            eta_line = f"ETA: ~{_format_duration(eta_seconds)}"

        # Build progress panel content
        panel_lines = [
            f"Status: {status_emoji}",
            progress_line,
            f"Chunks: {job.chunks_created}",
        ]
        if elapsed_line:
            panel_lines.append(elapsed_line)
        if eta_line:
            panel_lines.append(eta_line)

        # Create visual progress bar
        bar_width = 34
        filled = int((job.progress_percent / 100) * bar_width)
        progress_bar = "[green]" + ("=" * filled) + "[/green]" + ("-" * (bar_width - filled))

        console.print(f"[bold blue]CodeRAG Index Status: {project}[/bold blue]")
        console.print("[dim]" + "=" * 36 + "[/dim]")
        for line in panel_lines:
            console.print(line)
        console.print(f"[{progress_bar}]")
        console.print()

    elif recent_job and recent_job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        # T037: Display completion summary when job status is COMPLETED
        job = recent_job

        if job.status == JobStatus.COMPLETED:
            console.print(f"[bold blue]CodeRAG Index Status: {project}[/bold blue]")
            console.print("[dim]" + "=" * 36 + "[/dim]")
            console.print(f"Status: [green]completed[/green]")

            # Show completion stats
            if job.completed_at and job.started_at:
                duration = (job.completed_at - job.started_at).total_seconds()
                console.print(f"Duration: {_format_duration(duration)}")

            console.print(f"Files indexed: {job.files_processed}/{job.files_total}")
            console.print(f"Chunks created: {job.chunks_created}")

            if job.completed_at:
                completed_str = job.completed_at.strftime("%Y-%m-%d %H:%M:%S UTC")
                console.print(f"Completed at: {completed_str}")
            console.print()

        elif job.status == JobStatus.FAILED:
            console.print(f"[bold blue]CodeRAG Index Status: {project}[/bold blue]")
            console.print("[dim]" + "=" * 36 + "[/dim]")
            console.print(f"Status: [red]failed[/red]")
            if job.error_message:
                console.print(f"[red]Error: {job.error_message}[/red]")
            console.print()

    else:
        # No active or recent job, just show header
        console.print(f"[bold blue]CodeRAG Index Status[/bold blue]")
        console.print(f"Project: {status['project_id']}")
        console.print()

    # Always show the index statistics table
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Files indexed", str(status['files_count']))
    table.add_row("Chunks", str(status['chunks_count']))
    table.add_row("Symbols", str(status['symbols_count']))
    table.add_row("Graph nodes", str(status['graph_nodes']))
    table.add_row("Graph edges", str(status['graph_edges']))
    table.add_row("Last indexed", status['last_indexed'] or "Never")

    if status['repo_map']:
        table.add_row("Repo map tokens", str(status['repo_map']['token_count']))
        table.add_row("Repo map symbols", f"{status['repo_map']['symbols_included']}/{status['repo_map']['symbols_total']}")

    # Delta queue details (T054)
    delta_queue = status.get('delta_queue', {})
    if delta_queue:
        queued_files = delta_queue.get('queued_files', 0)
        total_lines = delta_queue.get('total_lines', 0)
        should_commit = delta_queue.get('should_commit', False)

        delta_status = f"{queued_files} files, {total_lines} lines"
        if should_commit:
            delta_status += " [red](threshold reached!)[/red]"

        table.add_row("Delta queue", delta_status)

        # Show individual queued files if any
        if queued_files > 0:
            console.print()
            console.print("[bold]Queued Files:[/bold]")
            for entry in delta_queue.get('queued_entries', [])[:5]:  # Show first 5
                file_path = entry['file_path']
                change_type = entry['change_type']
                lines = entry['lines_changed']
                age_min = entry['age_seconds'] // 60
                console.print(f"  [bullet] {file_path} ({change_type}, +{lines} lines, {age_min}m ago)")

            if queued_files > 5:
                console.print(f"  ... and {queued_files - 5} more files")

            # Show auto-commit info
            timeout_min = delta_queue.get('timeout_seconds', 300) // 60
            oldest_age_min = delta_queue.get('oldest_age_seconds', 0) // 60
            remaining_min = timeout_min - oldest_age_min

            if remaining_min > 0:
                console.print(f"\n  Auto-commit in: {remaining_min} minutes")
            else:
                console.print("\n  [yellow]Auto-commit pending (run 'vlt coderag sync' to commit now)[/yellow]")
    else:
        table.add_row("Delta queue", str(status['delta_queue_count']))

    console.print(table)


@coderag_app.command("search")
def coderag_search(
    query: str = typer.Argument(..., help="Search query"),
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum results to return"),
    language: str = typer.Option(None, "--language", "-l", help="Filter by programming language"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Direct code search using hybrid retrieval.

    Uses both vector search (semantic) and BM25 (keyword) for best results.

    Examples:
        vlt coderag search "authentication function"
        vlt coderag search "retry logic" --limit 5
        vlt coderag search "UserService" --language python
    """
    from vlt.core.identity import load_project_identity
    from vlt.core.coderag.bm25 import search_bm25
    from rich.console import Console
    from rich.syntax import Syntax
    import json as json_lib

    console = Console()

    # Resolve project
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id
        else:
            console.print("[red]Error: No project specified and no vlt.toml found.[/red]")
            raise typer.Exit(code=1)

    # Perform BM25 search
    results = search_bm25(query, project_id=project, limit=limit)

    if not results:
        if json_output:
            # Write directly to stdout, bypassing Rich
            import sys
            sys.stdout.write("[]\n")
        else:
            console.print("[yellow]No results found.[/yellow]")
        return

    if json_output:
        # Format for JSON output
        json_results = []
        for result in results:
            json_results.append({
                "chunk_id": result['chunk_id'],
                "file_path": result['file_path'],
                "qualified_name": result['qualified_name'],
                "score": result['score'],
                "retrieval_method": "bm25",
                "snippet": result['body'][:200] + "..." if len(result['body']) > 200 else result['body']
            })
        # Write directly to stdout, bypassing Rich completely
        import sys
        sys.stdout.write(json_lib.dumps(json_results, indent=2) + "\n")
        return

    # Display results
    console.print(f"[bold blue]Search Results[/bold blue] ({len(results)} found)")
    console.print(f"Query: {query}")
    console.print()

    for i, result in enumerate(results, 1):
        # Header
        score_color = "green" if result['score'] > 10 else "yellow"
        console.print(f"[bold]{i}. {result['qualified_name']}[/bold] ([{score_color}]score: {result['score']:.2f}[/{score_color}])")
        console.print(f"   [dim]{result['file_path']}:{result['lineno']}[/dim]")

        # Show signature if available
        if result.get('signature'):
            console.print(f"   [cyan]{result['signature']}[/cyan]")

        # Show snippet
        snippet = result['body'][:200] + "..." if len(result['body']) > 200 else result['body']
        console.print(f"   {snippet}")
        console.print()


@coderag_app.command("map")
def coderag_map(
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    scope: str = typer.Option(None, "--scope", "-s", help="Subdirectory to focus on (e.g., 'src/api/')"),
    max_tokens: int = typer.Option(4000, "--max-tokens", "-t", help="Maximum tokens for the map"),
    regenerate: bool = typer.Option(False, "--regenerate", "-r", help="Force regeneration (ignore cached map)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Display or regenerate the repository structure map.

    Generates an Aider-style condensed view of the codebase with:
    - File tree structure
    - Classes, functions, methods with signatures
    - Symbols ranked by PageRank centrality (most important first)
    - Token-budgeted output (fits within context window)

    The map is cached and reused unless --regenerate is specified.

    Examples:
        vlt coderag map                           # Show cached map or generate new
        vlt coderag map --scope src/api/          # Focus on specific subdirectory
        vlt coderag map --max-tokens 8000         # Larger map
        vlt coderag map --regenerate              # Force regeneration with new centrality scores
    """
    from vlt.core.identity import load_project_identity
    from vlt.core.coderag.indexer import CodeRAGIndexer
    from vlt.core.coderag.store import CodeRAGStore
    from rich.console import Console
    import json as json_lib

    console = Console()

    # Resolve project
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id
        else:
            console.print("[red]Error: No project specified and no vlt.toml found.[/red]")
            raise typer.Exit(code=1)

    # Check for cached map (T037)
    if not regenerate:
        with CodeRAGStore() as store:
            cached_map = store.get_repo_map(project, scope=scope)
            if cached_map:
                if json_output:
                    map_data = {
                        "map_text": cached_map.map_text,
                        "token_count": cached_map.token_count,
                        "max_tokens": cached_map.max_tokens,
                        "files_included": cached_map.files_included,
                        "symbols_included": cached_map.symbols_included,
                        "symbols_total": cached_map.symbols_total,
                        "scope": cached_map.scope,
                        "created_at": cached_map.created_at.isoformat()
                    }
                    console.print(json_lib.dumps(map_data, indent=2))
                else:
                    console.print("[bold green]Repository Map[/bold green] (cached)")
                    console.print(f"Scope: {cached_map.scope or 'all'} | "
                                  f"Symbols: {cached_map.symbols_included}/{cached_map.symbols_total} | "
                                  f"Tokens: {cached_map.token_count}/{cached_map.max_tokens}")
                    console.print()
                    console.print(cached_map.map_text)
                    console.print()
                    console.print("[dim]Use --regenerate to force regeneration with updated centrality scores[/dim]")
                return

    # Generate new map
    console.print("[bold blue]Generating repository map...[/bold blue]")

    indexer = CodeRAGIndexer(Path("."), project)

    # Need to import repomap module and generate
    from vlt.core.coderag.repomap import (
        Symbol,
        build_reference_graph,
        calculate_centrality,
        generate_repo_map
    )
    from sqlalchemy import select
    from sqlalchemy.orm import Session
    from vlt.db import engine
    from vlt.core.models import CodeNode, CodeEdge
    import uuid

    try:
        with Session(engine) as session:
            # Get all nodes
            nodes = session.scalars(
                select(CodeNode).where(CodeNode.project_id == project)
            ).all()

            if not nodes:
                console.print("[yellow]No symbols found. Run 'vlt coderag init' first.[/yellow]")
                return

            # Get all edges
            edges = session.scalars(
                select(CodeEdge).where(CodeEdge.project_id == project)
            ).all()

            # Convert to Symbol objects
            symbols = []
            for node in nodes:
                symbol = Symbol(
                    name=node.name,
                    qualified_name=node.id,
                    file_path=node.file_path,
                    symbol_type=node.node_type.value,
                    signature=node.signature,
                    lineno=node.lineno,
                    docstring=node.docstring
                )
                symbols.append(symbol)

            # Build reference graph
            edge_tuples = [(edge.source_id, edge.target_id) for edge in edges]
            graph = build_reference_graph(symbols, edge_tuples)

            # Calculate centrality scores
            centrality_scores = calculate_centrality(graph)

            # Update centrality scores in database
            for node in nodes:
                if node.id in centrality_scores:
                    node.centrality_score = centrality_scores[node.id]
            session.commit()

            # Generate map
            repo_map_data = generate_repo_map(
                symbols=symbols,
                graph=graph,
                centrality_scores=centrality_scores,
                max_tokens=max_tokens,
                scope=scope,
                include_signatures=True,
                include_docstrings=False
            )

            # Store in database
            from vlt.core.models import RepoMap
            repo_map = RepoMap(
                id=str(uuid.uuid4()),
                project_id=project,
                scope=repo_map_data['scope'],
                map_text=repo_map_data['map_text'],
                token_count=repo_map_data['token_count'],
                max_tokens=repo_map_data['max_tokens'],
                files_included=repo_map_data['files_included'],
                symbols_included=repo_map_data['symbols_included'],
                symbols_total=repo_map_data['symbols_total'],
            )
            session.add(repo_map)
            session.commit()

            # Output
            if json_output:
                output_data = {
                    **repo_map_data,
                    "created_at": repo_map.created_at.isoformat()
                }
                console.print(json_lib.dumps(output_data, indent=2))
            else:
                console.print("[bold green]Repository Map[/bold green] (newly generated)")
                console.print(f"Scope: {repo_map_data['scope'] or 'all'} | "
                              f"Symbols: {repo_map_data['symbols_included']}/{repo_map_data['symbols_total']} | "
                              f"Tokens: {repo_map_data['token_count']}/{max_tokens}")
                console.print()
                console.print(repo_map_data['map_text'])

    except Exception as e:
        console.print(f"[red]Error generating map: {e}[/red]")
        raise typer.Exit(code=1)


@coderag_app.command("sync")
def coderag_sync(
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force commit even if thresholds not met"),
    scan: bool = typer.Option(False, "--scan", "-s", help="Scan for changes before committing"),
):
    """
    Commit pending delta queue changes to indexes (T055).

    This command commits all queued file changes to the indexes:
    - Vector embeddings (semantic search)
    - BM25 keyword index
    - Code graph
    - Symbol definitions (ctags)
    - Repository map

    By default, commits all pending changes regardless of thresholds.
    Use --scan to scan for new changes before committing.

    Examples:
        vlt coderag sync                    # Commit all pending changes
        vlt coderag sync --force            # Force commit (same as default)
        vlt coderag sync --scan             # Scan for changes first, then commit
    """
    from vlt.core.identity import load_project_identity
    from vlt.core.coderag.indexer import CodeRAGIndexer
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.console import Console

    console = Console()

    # Resolve project
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id
        else:
            console.print("[red]Error: No project specified and no vlt.toml found.[/red]")
            raise typer.Exit(code=1)

    console.print(f"[bold blue]Syncing delta queue for project '{project}'[/bold blue]")
    console.print()

    # Create indexer
    indexer = CodeRAGIndexer(Path("."), project)

    # Scan for changes if requested
    if scan:
        console.print("[bold]Scanning for file changes...[/bold]")
        queued = indexer.scan_for_changes()
        console.print(f"[green]Queued {queued} changed files[/green]")
        console.print()

    # Check queue status
    queue_status = indexer.delta_manager.get_queue_status()
    queued_files = queue_status.get('queued_files', 0)

    if queued_files == 0:
        console.print("[yellow]No files in delta queue. Nothing to commit.[/yellow]")
        console.print()
        console.print("[dim]Tip: Use --scan to check for changes, or run 'vlt coderag init' for full reindex[/dim]")
        return

    console.print(f"[bold]Delta Queue Status:[/bold]")
    console.print(f"  Files queued: {queued_files}")
    console.print(f"  Total lines: {queue_status.get('total_lines', 0)}")
    console.print()

    # Commit changes
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Committing changes...", total=None)

        try:
            # Batch commit
            stats = indexer.batch_commit_delta_queue(force=True)

            progress.update(task, description="Commit complete!", total=1, completed=1)

            # Display results
            console.print()
            console.print("[bold green]Commit complete![/bold green]")
            console.print()

            # Stats table
            from rich.table import Table
            table = Table(title="Sync Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Files indexed", str(stats.files_indexed))
            table.add_row("Files skipped", str(stats.files_skipped))
            table.add_row("Files failed", str(stats.files_failed))
            table.add_row("Chunks created", str(stats.chunks_created))
            table.add_row("Embeddings generated", str(stats.embeddings_generated))
            table.add_row("Symbols indexed", str(stats.symbols_indexed))
            table.add_row("Graph nodes", str(stats.graph_nodes))
            table.add_row("Graph edges", str(stats.graph_edges))
            table.add_row("Time elapsed", f"{stats.duration_seconds:.2f}s")

            console.print(table)

            # Show errors if any
            if stats.errors:
                console.print()
                console.print("[bold red]Errors:[/bold red]")
                for error in stats.errors[:10]:
                    console.print(f"  • {error}")
                if len(stats.errors) > 10:
                    console.print(f"  ... and {len(stats.errors) - 10} more errors")

        except Exception as e:
            progress.update(task, description="[red]Commit failed![/red]")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)


@coderag_app.command("delete")
def coderag_delete(
    project: str = typer.Option(..., "--project", "-p", help="Project ID to delete CodeRAG index for"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Delete CodeRAG index for a project (Phase 7: T055-T059).

    This command performs cascade deletion of all CodeRAG data:
    - Code chunks (semantic units)
    - Code nodes (graph nodes)
    - Code edges (graph edges)
    - Symbol definitions (ctags)
    - Indexing jobs (job history)
    - Repository maps (cached maps)
    - Delta queue (pending changes)

    Use --yes to skip the confirmation prompt for scripted/automated use.

    Examples:
        vlt coderag delete --project myproj      # Interactive confirmation
        vlt coderag delete --project myproj -y   # Skip confirmation
        vlt coderag delete -p myproj --json      # JSON output
    """
    from rich.console import Console
    from vlt.core.service import SqliteVaultService
    import json as json_lib

    console = Console()
    service = SqliteVaultService()

    # T059: Confirmation prompt unless --yes is passed
    if not yes:
        console.print(f"[yellow]Warning: This will delete all CodeRAG data for project '{project}'.[/yellow]")
        console.print("[dim]This action cannot be undone.[/dim]")
        console.print()

        if not typer.confirm(f"Delete CodeRAG index for project '{project}'?"):
            console.print("[dim]Cancelled.[/dim]")
            raise typer.Exit(code=0)

    # Check if project has any data
    if not service.has_coderag_index(project):
        if json_output:
            console.print(json_lib.dumps({
                "status": "no_data",
                "project_id": project,
                "message": "No CodeRAG index found for project"
            }))
        else:
            console.print(f"[yellow]No CodeRAG index found for project '{project}'.[/yellow]")
        raise typer.Exit(code=0)

    try:
        # Perform cascade delete
        deleted = service.delete_coderag_index(project)

        if json_output:
            console.print(json_lib.dumps({
                "status": "deleted",
                "project_id": project,
                "deleted": deleted
            }))
        else:
            console.print()
            console.print(f"[green]Deleted CodeRAG index for project '{project}':[/green]")
            console.print(f"  - {deleted['chunks']} chunks")
            console.print(f"  - {deleted['nodes']} nodes")
            console.print(f"  - {deleted['edges']} edges")
            console.print(f"  - {deleted['symbols']} symbols")
            console.print(f"  - {deleted['jobs']} jobs")
            if deleted.get('repo_maps', 0) > 0:
                console.print(f"  - {deleted['repo_maps']} repo maps")
            if deleted.get('delta_queue', 0) > 0:
                console.print(f"  - {deleted['delta_queue']} delta queue items")

    except Exception as e:
        if json_output:
            console.print(json_lib.dumps({
                "status": "error",
                "project_id": project,
                "error": str(e)
            }))
        else:
            console.print(f"[red]Error deleting CodeRAG index: {e}[/red]")
        raise typer.Exit(code=1)


# ============================================================================
# Oracle Commands (T076-T078) - Phase 10
# ============================================================================

@app.command("oracle")
def oracle_query(
    question: str = typer.Argument(..., help="Natural language question about the codebase"),
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    source: List[str] = typer.Option(None, "--source", "-s", help="Filter sources: 'code', 'vault', 'threads' (can be used multiple times)"),
    explain: bool = typer.Option(False, "--explain", help="Show detailed retrieval traces for debugging"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    max_tokens: int = typer.Option(16000, "--max-tokens", help="Maximum tokens for context assembly"),
    local: bool = typer.Option(False, "--local", "-l", help="Force local mode (skip backend check)"),
    model: str = typer.Option(None, "--model", "-m", help="Override LLM model (e.g., 'anthropic/claude-sonnet-4')"),
    thinking: bool = typer.Option(False, "--thinking", "-t", help="Enable extended thinking mode"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Enable/disable streaming output"),
):
    """
    Ask Oracle a question about the codebase.

    Oracle is a multi-source intelligent context retrieval system that:
    - Searches code index (vector + BM25 + graph)
    - Searches documentation vault (markdown notes)
    - Searches development threads (historical context)
    - Reranks results for relevance
    - Synthesizes a comprehensive answer with citations

    By default, Oracle uses the backend server when available (thin client mode),
    which shares context with the web UI. Use --local to force local processing.

    Examples:
        vlt oracle "How does authentication work?"
        vlt oracle "Where is UserService defined?" --source code
        vlt oracle "What calls the login function?" --explain
        vlt oracle "Why did we choose SQLite?" --source threads
        vlt oracle "Explain the architecture" --local
        vlt oracle "Complex question" --thinking --model anthropic/claude-sonnet-4

    The response includes:
    - A synthesized answer from an LLM
    - Source citations [file.py:42], [note.md], [thread:id#node]
    - Repository structure context
    - Cost and timing information
    """
    import asyncio
    from vlt.core.identity import load_project_identity
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    import json as json_lib

    console = Console()

    # Resolve project
    if not project:
        identity = load_project_identity()
        if identity:
            project = identity.id
        else:
            console.print("[red]Error: No project specified and no vlt.toml found.[/red]")
            console.print("Usage: vlt oracle <question> --project <project>")
            console.print("Or run: vlt init --project <name> to create vlt.toml")
            raise typer.Exit(code=1)

    # Resolve project path
    project_path = Path(".").resolve()

    # Load settings
    settings = Settings()

    # Display query header (skip in JSON mode)
    if not json_output:
        console.print()
        console.print(Panel(
            f"[bold cyan]Question:[/bold cyan] {question}",
            title="Oracle Query",
            border_style="blue"
        ))
        console.print()

    # Try thin client mode (backend API) first unless --local is specified
    client = OracleClient()
    use_backend = False

    if not local and settings.sync_token:
        if not json_output:
            with console.status("[bold blue]Checking backend availability...[/bold blue]"):
                use_backend = client.is_available()
        else:
            # Silent check for JSON mode
            use_backend = client.is_available()

        if use_backend and not json_output:
            console.print("[dim]Using backend server (thin client mode)[/dim]")
        elif not use_backend and not json_output:
            console.print("[dim yellow]Backend unavailable, using local mode[/dim yellow]")

    if use_backend:
        # =====================================================================
        # Thin Client Mode - Use Backend API
        # =====================================================================
        # Get active context for conversation continuity
        context_id = None
        try:
            context_id = asyncio.run(client.get_context_id())
            if context_id and not json_output:
                console.print(f"[dim]Continuing context: {context_id[:8]}...[/dim]")
        except Exception as e:
            logger.debug(f"Failed to get context_id (non-fatal): {e}")

        _oracle_via_backend(
            console=console,
            client=client,
            question=question,
            source=source,
            explain=explain,
            json_output=json_output,
            max_tokens=max_tokens,
            model=model,
            thinking=thinking,
            stream=stream,
            context_id=context_id,
        )
    else:
        # =====================================================================
        # Local Mode - Use Local OracleOrchestrator
        # =====================================================================
        _oracle_local(
            console=console,
            question=question,
            project=project,
            project_path=project_path,
            settings=settings,
            source=source,
            explain=explain,
            json_output=json_output,
            max_tokens=max_tokens,
        )


def _oracle_via_backend(
    console,
    client: "OracleClient",
    question: str,
    source: List[str],
    explain: bool,
    json_output: bool,
    max_tokens: int,
    model: str,
    thinking: bool,
    stream: bool,
    context_id: str = None,
):
    """Execute Oracle query via backend API (thin client mode)."""
    import asyncio
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.live import Live
    import json as json_lib

    async def run_streaming():
        """Run streaming query."""
        content_parts = []
        sources = []
        tokens_used = None
        model_used = None
        error_msg = None

        # Use Live for real-time updates
        current_content = ""

        if not json_output:
            with Live(console=console, refresh_per_second=4) as live:
                async for chunk in client.query_stream(
                    question=question,
                    sources=source if source else None,
                    explain=explain,
                    model=model,
                    thinking=thinking,
                    max_tokens=max_tokens,
                    context_id=context_id,
                ):
                    if chunk.type == "thinking":
                        if chunk.content:
                            live.update(Panel(
                                f"[dim italic]{chunk.content}[/dim italic]",
                                title="Thinking...",
                                border_style="dim"
                            ))
                    elif chunk.type == "tool_call":
                        if chunk.tool_call:
                            tool_name = chunk.tool_call.get("name", "unknown")
                            live.update(Panel(
                                f"[cyan]Calling: {tool_name}[/cyan]",
                                title="Tool Execution",
                                border_style="cyan"
                            ))
                    elif chunk.type == "content":
                        if chunk.content:
                            content_parts.append(chunk.content)
                            current_content = "".join(content_parts)
                            live.update(Panel(
                                Markdown(current_content),
                                title="Answer",
                                border_style="green",
                                padding=(1, 2)
                            ))
                    elif chunk.type == "source":
                        if chunk.source:
                            sources.append(chunk.source)
                    elif chunk.type == "done":
                        tokens_used = chunk.tokens_used
                        model_used = chunk.model_used
                    elif chunk.type == "error":
                        error_msg = chunk.error
                        break
        else:
            # JSON mode - collect all chunks without live display
            async for chunk in client.query_stream(
                question=question,
                sources=source if source else None,
                explain=explain,
                model=model,
                thinking=thinking,
                max_tokens=max_tokens,
                context_id=context_id,
            ):
                if chunk.type == "content" and chunk.content:
                    content_parts.append(chunk.content)
                elif chunk.type == "source" and chunk.source:
                    sources.append(chunk.source)
                elif chunk.type == "done":
                    tokens_used = chunk.tokens_used
                    model_used = chunk.model_used
                elif chunk.type == "error":
                    error_msg = chunk.error
                    break

        return {
            "answer": "".join(content_parts),
            "sources": sources,
            "tokens_used": tokens_used,
            "model_used": model_used,
            "error": error_msg,
        }

    async def run_non_streaming():
        """Run non-streaming query."""
        try:
            response = await client.query(
                question=question,
                sources=source if source else None,
                explain=explain,
                model=model,
                thinking=thinking,
                max_tokens=max_tokens,
                context_id=context_id,
            )
            return {
                "answer": response.answer,
                "sources": response.sources,
                "tokens_used": response.tokens_used,
                "model_used": response.model_used,
                "error": None,
            }
        except Exception as e:
            return {"answer": "", "sources": [], "tokens_used": None, "model_used": None, "error": str(e)}

    # Execute query
    if stream:
        result = asyncio.run(run_streaming())
    else:
        with console.status("[bold blue]Querying Oracle...[/bold blue]"):
            result = asyncio.run(run_non_streaming())

    # Handle error
    if result["error"]:
        console.print(f"[red]Error: {result['error']}[/red]")
        raise typer.Exit(code=1)

    # JSON output mode
    if json_output:
        output_data = {
            "question": question,
            "answer": result["answer"],
            "sources": [
                {
                    "path": s.path,
                    "type": s.source_type,
                    "snippet": s.snippet,
                    "score": s.score
                }
                for s in result["sources"]
            ],
            "tokens_used": result["tokens_used"],
            "model_used": result["model_used"],
            "mode": "backend",
        }
        # Write directly to stdout, bypassing Rich
        import sys
        sys.stdout.write(json_lib.dumps(output_data, indent=2) + "\n")
        return

    # If not streaming, display answer now
    if not stream:
        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title="Answer",
            border_style="green",
            padding=(1, 2)
        ))

    # Show sources
    if result["sources"]:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for i, src in enumerate(result["sources"][:5], 1):
            score = src.score or 0
            score_color = "green" if score >= 0.8 else "yellow"
            console.print(
                f"  {i}. [{score_color}]{src.path}[/{score_color}] "
                f"({src.source_type}, score: {score:.2f})"
            )

    # Show metadata
    console.print()
    console.print(
        f"[dim]Mode: backend | "
        f"Model: {result['model_used'] or 'unknown'} | "
        f"Tokens: {result['tokens_used'] or 'N/A'}[/dim]"
    )

    console.print()


def _oracle_local(
    console,
    question: str,
    project: str,
    project_path: Path,
    settings: "Settings",
    source: List[str],
    explain: bool,
    json_output: bool,
    max_tokens: int,
):
    """Execute Oracle query using local OracleOrchestrator."""
    import asyncio
    from vlt.core.oracle import OracleOrchestrator
    from rich.markdown import Markdown
    from rich.panel import Panel
    import json as json_lib

    # Check if API key is configured for local mode
    if not settings.openrouter_api_key and not settings.sync_token:
        console.print("[red]Error: No API credentials configured for local mode.[/red]")
        console.print()
        console.print("Option 1 (Recommended): Configure server sync token:")
        console.print("  vlt config set-key <your-sync-token>")
        console.print()
        console.print("Option 2 (Legacy): Set OpenRouter API key directly:")
        console.print("  export VLT_OPENROUTER_API_KEY=<your-api-key>")
        raise typer.Exit(code=1)

    # Show status while processing
    with console.status("[bold blue]Searching knowledge sources (local)...[/bold blue]") as status:
        try:
            # Create orchestrator
            orchestrator = OracleOrchestrator(
                project_id=project,
                project_path=str(project_path),
                settings=settings
            )

            # Execute query
            response = asyncio.run(orchestrator.query(
                question=question,
                sources=source if source else None,
                explain=explain,
                max_context_tokens=max_tokens,
                include_repo_map=True
            ))

        except Exception as e:
            console.print(f"[red]Error during oracle query: {e}[/red]")
            logger.error(f"Oracle query failed", exc_info=True)
            raise typer.Exit(code=1)

    # JSON output mode
    if json_output:
        output_data = {
            "question": question,
            "answer": response.answer,
            "sources": [
                {
                    "path": src.source_path,
                    "type": src.source_type.value,
                    "method": src.retrieval_method.value,
                    "score": src.score
                }
                for src in response.sources
            ],
            "query_type": response.query_type,
            "model": response.model,
            "tokens_used": response.tokens_used,
            "cost_cents": response.cost_cents,
            "duration_ms": response.duration_ms,
            "mode": "local",
        }

        if response.traces:
            output_data["traces"] = response.traces

        # Write directly to stdout, bypassing Rich
        import sys
        sys.stdout.write(json_lib.dumps(output_data, indent=2) + "\n")
        return

    # Rich formatted output
    console.print()
    console.print(Panel(
        Markdown(response.answer),
        title="Answer",
        border_style="green",
        padding=(1, 2)
    ))

    # Show sources
    if response.sources:
        console.print()
        console.print("[bold]Sources:[/bold]")
        for i, src in enumerate(response.sources[:5], 1):  # Show top 5
            score_color = "green" if src.score >= 0.8 else "yellow"
            console.print(
                f"  {i}. [{score_color}]{src.source_path}[/{score_color}] "
                f"({src.source_type.value} via {src.retrieval_method.value}, "
                f"score: {src.score:.2f})"
            )

    # Show metadata
    console.print()
    console.print(
        f"[dim]Mode: local | "
        f"Query type: {response.query_type} | "
        f"Model: {response.model} | "
        f"Tokens: {response.tokens_used} | "
        f"Cost: ${response.cost_cents/100:.4f} | "
        f"Time: {response.duration_ms}ms[/dim]"
    )

    # Show explain traces if requested
    if explain and response.traces:
        console.print()
        console.print(Panel(
            Markdown(f"""
## Query Analysis
- Type: {response.traces['query_analysis']['query_type']}
- Confidence: {response.traces['query_analysis']['confidence']:.2f}
- Symbols: {', '.join(response.traces['query_analysis']['extracted_symbols']) or 'none'}

## Retrieval Statistics
- Code: {response.traces['retrieval_stats']['code']['count']} results (avg: {response.traces['retrieval_stats']['code']['avg_score']:.2f})
- Vault: {response.traces['retrieval_stats']['vault']['count']} results (avg: {response.traces['retrieval_stats']['vault']['avg_score']:.2f})
- Threads: {response.traces['retrieval_stats']['threads']['count']} results (avg: {response.traces['retrieval_stats']['threads']['avg_score']:.2f})

## Context Assembly
- Tokens used: {response.traces['context_stats']['token_count']}/{response.traces['context_stats']['max_tokens']}
- Sources included: {response.traces['context_stats']['sources_included']}
- Sources excluded: {response.traces['context_stats']['sources_excluded']}

## Timing
- Query analysis: {response.traces['timings_ms'].get('query_analysis', 0)}ms
- Retrieval: {response.traces['timings_ms'].get('retrieval', 0)}ms
- Context assembly: {response.traces['timings_ms'].get('context_assembly', 0)}ms
- Synthesis: {response.traces['timings_ms'].get('synthesis', 0)}ms
            """),
            title="Debug Trace",
            border_style="yellow"
        ))

    console.print()


# ============================================================================
# Context Commands - Manage Oracle context tree
# ============================================================================

context_app = typer.Typer(name="context", help="Manage Oracle context tree (conversation history).")
app.add_typer(context_app, name="context")


@context_app.command("list")
def context_list(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    List all context trees (conversations).

    Shows all Oracle conversation trees with their current state.
    Each tree represents an independent conversation branch.

    Example:
        vlt context list
        vlt context list --json
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console
    from rich.table import Table
    import json as json_lib

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured. Context tree requires backend.[/yellow]")
        console.print("[dim]Run: vlt config set-key <your-sync-token>[/dim]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable. Context tree requires backend connection.[/yellow]")
        raise typer.Exit(code=1)

    response = asyncio.run(client.get_trees())
    trees = response.trees
    active_tree = response.active_tree

    if json_output:
        output = {
            "trees": [
                {
                    "root_id": t.root_id,
                    "current_node_id": t.current_node_id,
                    "label": t.label,
                    "node_count": t.node_count,
                    "created_at": t.created_at.isoformat(),
                    "last_activity": t.last_activity.isoformat(),
                    "is_active": active_tree and t.root_id == active_tree.root_id,
                }
                for t in trees
            ],
            "active_tree_id": active_tree.root_id if active_tree else None,
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    if not trees:
        console.print("[dim]No context trees found. Start a conversation with 'vlt oracle'.[/dim]")
        return

    table = Table(title="Oracle Context Trees")
    table.add_column("Active", style="green", width=6)
    table.add_column("Root ID", style="cyan")
    table.add_column("Label", style="magenta")
    table.add_column("Nodes", style="yellow")
    table.add_column("Last Activity", style="dim")

    for tree in trees:
        is_active = active_tree and tree.root_id == active_tree.root_id
        table.add_row(
            "*" if is_active else "",
            tree.root_id[:8] + "...",
            tree.label or "-",
            str(tree.node_count),
            tree.last_activity.strftime("%Y-%m-%d %H:%M"),
        )

    console.print(table)

    if active_tree:
        console.print(f"\n[dim]Active context: {active_tree.root_id[:8]}... (use 'vlt context activate <id>' to switch)[/dim]")


@context_app.command("new")
def context_new(
    label: str = typer.Option(None, "--label", "-l", help="Label for the new conversation"),
):
    """
    Start a new conversation (create new context tree).

    Creates a new conversation branch. The next oracle query will use
    this new context instead of continuing the previous conversation.

    Example:
        vlt context new
        vlt context new --label "Debugging auth"
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured. Context tree requires backend.[/yellow]")
        console.print("[dim]Run: vlt config set-key <your-sync-token>[/dim]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable. Context tree requires backend connection.[/yellow]")
        raise typer.Exit(code=1)

    tree = asyncio.run(client.create_tree(label=label))

    if tree:
        console.print(f"[green]Created new context tree: {tree.root_id[:8]}...[/green]")
        if label:
            console.print(f"[dim]Label: {label}[/dim]")
    else:
        console.print("[yellow]Backend doesn't support context tree management yet.[/yellow]")
        console.print("[dim]Conversation history is still available via 'vlt context history'[/dim]")


@context_app.command("checkout")
def context_checkout(
    node_id: str = typer.Argument(..., help="Node ID to switch to (use 'vlt context list' to find IDs)"),
):
    """
    Switch to a different node in the context tree.

    This allows you to branch off from a previous point in the conversation.
    Useful for exploring alternative lines of questioning.

    Example:
        vlt context checkout abc123
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    tree = asyncio.run(client.checkout(node_id))

    if tree:
        console.print(f"[green]Switched to node: {tree.current_node_id[:8]}...[/green]")
    else:
        console.print(f"[red]Failed to checkout node: {node_id}[/red]")
        console.print("[dim]The node may not exist or the backend doesn't support this feature.[/dim]")


@context_app.command("activate")
def context_activate(
    tree_id: str = typer.Argument(..., help="Tree root ID to activate (use 'vlt context list' to find IDs)"),
):
    """
    Set a context tree as the active conversation.

    The active tree is used for oracle queries. This allows you to switch
    between different conversation threads.

    Example:
        vlt context activate abc123
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    success = asyncio.run(client.activate_tree(tree_id))

    if success:
        console.print(f"[green]Activated context tree: {tree_id[:8]}...[/green]")
        console.print("[dim]Future oracle queries will use this context.[/dim]")
    else:
        console.print(f"[red]Failed to activate tree: {tree_id}[/red]")
        console.print("[dim]The tree may not exist or the backend doesn't support this feature.[/dim]")


@context_app.command("show")
def context_show(
    tree_id: str = typer.Argument(None, help="Tree root ID (defaults to active tree)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Show details of a context tree including all nodes.

    Displays the tree structure with questions, answers, and node hierarchy.

    Example:
        vlt context show              # Show active tree
        vlt context show abc123       # Show specific tree
        vlt context show --json
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console
    from rich.tree import Tree as RichTree
    import json as json_lib

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    # Get tree_id from active context if not provided
    if not tree_id:
        active = asyncio.run(client.get_active_context())
        if not active:
            console.print("[yellow]No active context. Specify a tree ID or activate one first.[/yellow]")
            raise typer.Exit(code=1)
        tree_id = active.root_id

    tree_data = asyncio.run(client.get_tree(tree_id))

    if not tree_data or not tree_data.active_tree:
        console.print(f"[red]Tree not found: {tree_id}[/red]")
        raise typer.Exit(code=1)

    if json_output:
        output = {
            "tree": {
                "root_id": tree_data.active_tree.root_id,
                "current_node_id": tree_data.active_tree.current_node_id,
                "label": tree_data.active_tree.label,
                "node_count": tree_data.active_tree.node_count,
            },
            "nodes": {
                node_id: {
                    "id": node.id,
                    "parent_id": node.parent_id,
                    "question": node.question[:100] + "..." if len(node.question) > 100 else node.question,
                    "answer_preview": node.answer[:100] + "..." if len(node.answer) > 100 else node.answer,
                    "is_checkpoint": node.is_checkpoint,
                    "label": node.label,
                }
                for node_id, node in tree_data.nodes.items()
            },
            "path_to_head": tree_data.path_to_head,
        }
        console.print(json_lib.dumps(output, indent=2))
        return

    # Build visual tree
    tree = tree_data.active_tree
    nodes = tree_data.nodes

    console.print(f"[bold]Context Tree: {tree.root_id[:8]}...[/bold]")
    if tree.label:
        console.print(f"[dim]Label: {tree.label}[/dim]")
    console.print(f"[dim]Nodes: {tree.node_count} | Current: {tree.current_node_id[:8]}...[/dim]")
    console.print()

    # Find root node
    root_node = None
    for node in nodes.values():
        if node.is_root:
            root_node = node
            break

    if not root_node:
        console.print("[yellow]No root node found in tree.[/yellow]")
        return

    # Build tree structure recursively
    def add_node_to_tree(rich_tree, node):
        is_current = node.id == tree.current_node_id
        is_checkpoint = node.is_checkpoint
        prefix = "[bold green]>> [/bold green]" if is_current else ""
        checkpoint = " [yellow](checkpoint)[/yellow]" if is_checkpoint else ""
        label_text = f" [magenta]({node.label})[/magenta]" if node.label else ""

        question_preview = node.question[:50] + "..." if len(node.question) > 50 else node.question
        answer_preview = node.answer[:50] + "..." if len(node.answer) > 50 else node.answer

        node_text = f"{prefix}{node.id[:8]}{label_text}{checkpoint}\n  Q: {question_preview}\n  A: {answer_preview}"
        branch = rich_tree.add(node_text)

        # Find children
        for child in nodes.values():
            if child.parent_id == node.id:
                add_node_to_tree(branch, child)

    rich_tree = RichTree(f"[bold cyan]{root_node.id[:8]}[/bold cyan] (root)")
    for child in nodes.values():
        if child.parent_id == root_node.id:
            add_node_to_tree(rich_tree, child)

    console.print(rich_tree)


@context_app.command("delete")
def context_delete(
    tree_id: str = typer.Argument(..., help="Tree root ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Delete a context tree and all its nodes.

    This action cannot be undone.

    Example:
        vlt context delete abc123
        vlt context delete abc123 --force
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Delete context tree {tree_id[:8]}...? This cannot be undone.")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            return

    success = asyncio.run(client.delete_tree(tree_id))

    if success:
        console.print(f"[green]Deleted context tree: {tree_id[:8]}...[/green]")
    else:
        console.print(f"[red]Failed to delete tree: {tree_id}[/red]")


@context_app.command("history")
def context_history(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum messages to show"),
):
    """
    Show conversation history.

    Displays the recent conversation history from the backend.

    Example:
        vlt context history
        vlt context history --limit 20
        vlt context history --json
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    import json as json_lib

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        console.print("[dim]Run: vlt config set-key <your-sync-token>[/dim]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    messages = asyncio.run(client.get_history())

    if json_output:
        output = [
            {
                "role": m.role,
                "content": m.content[:500] + "..." if len(m.content) > 500 else m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in messages[-limit:]
        ]
        console.print(json_lib.dumps(output, indent=2))
        return

    if not messages:
        console.print("[dim]No conversation history found.[/dim]")
        console.print("[dim]Start a conversation with 'vlt oracle <question>'.[/dim]")
        return

    console.print(f"[bold]Conversation History[/bold] (last {min(limit, len(messages))} messages)")
    console.print()

    for msg in messages[-limit:]:
        if msg.role == "user":
            console.print(Panel(
                msg.content[:500] + "..." if len(msg.content) > 500 else msg.content,
                title="[cyan]You[/cyan]",
                border_style="cyan",
            ))
        else:
            console.print(Panel(
                Markdown(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content),
                title="[green]Oracle[/green]",
                border_style="green",
            ))
        console.print()


@context_app.command("clear")
def context_clear(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Clear conversation history.

    Removes all conversation history from the backend.
    This action cannot be undone.

    Example:
        vlt context clear
        vlt context clear --force
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm("This will clear all conversation history. Continue?")
        if not confirm:
            console.print("[dim]Aborted.[/dim]")
            return

    success = asyncio.run(client.clear_history())

    if success:
        console.print("[green]Conversation history cleared.[/green]")
    else:
        console.print("[red]Failed to clear history.[/red]")


@context_app.command("cancel")
def context_cancel():
    """
    Cancel the active Oracle query.

    If an Oracle query is currently running, this will cancel it.

    Example:
        vlt context cancel
    """
    import asyncio
    from vlt.core.oracle_client import OracleClient
    from rich.console import Console

    console = Console()
    client = OracleClient()

    if not client.token:
        console.print("[yellow]No sync token configured.[/yellow]")
        raise typer.Exit(code=1)

    if not client.is_available():
        console.print("[yellow]Backend unavailable.[/yellow]")
        raise typer.Exit(code=1)

    cancelled = asyncio.run(client.cancel_query())

    if cancelled:
        console.print("[green]Active query cancelled.[/green]")
    else:
        console.print("[dim]No active query to cancel.[/dim]")


if __name__ == "__main__":
    app()
