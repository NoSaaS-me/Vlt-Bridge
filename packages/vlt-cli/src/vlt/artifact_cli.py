"""vlt artifact — CLI commands for artifact sandbox management.

Provides shell access to create, list, inspect, sync, and control artifact
pipelines.  All operations proxy through the daemon REST API at localhost:8765.

Sync model:
  vlt artifact sync  <id> <folder>   push local folder → artifact disk
  vlt artifact pull  <id> <folder>   download artifact disk → local folder
  vlt artifact files <id>            list files on disk with hashes
"""

from __future__ import annotations

import io
import json
import os
import sys
import zipfile
from pathlib import Path

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
artifact_app = typer.Typer(
    name="artifact",
    help="Manage artifact sandboxes — create, list, inspect, run pipelines.",
    no_args_is_help=True,
)


def _daemon_url() -> str:
    try:
        from vlt.config import get_settings
        return get_settings().daemon_url
    except Exception:
        return "http://127.0.0.1:8765"


def _api(method: str, path: str, **kwargs) -> dict:
    """Make an HTTP request to the daemon API. Raises on failure."""
    import httpx

    url = f"{_daemon_url()}{path}"
    timeout = kwargs.pop("timeout", 10.0)
    resp = httpx.request(method, url, timeout=timeout, **kwargs)
    if resp.status_code >= 400:
        detail = resp.text[:300]
        try:
            detail = resp.json().get("detail", detail)
        except Exception:
            pass
        console.print(f"[red]Error ({resp.status_code}):[/red] {detail}")
        raise typer.Exit(1)
    if resp.status_code == 204:
        return {}
    return resp.json()


# ============================================================================
# list
# ============================================================================

@artifact_app.command("list")
def artifact_list(
    project: str = typer.Option("", "--project", "-p", help="Filter by project ID"),
    state: str = typer.Option("", "--state", "-s", help="Filter by state"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List artifacts, optionally filtered by project or state."""
    params = {}
    if project:
        params["project_id"] = project
    if state:
        params["state"] = state

    items = _api("GET", "/api/artifacts", params=params)

    if json_output:
        rprint(json.dumps(items, indent=2))
        return

    if not items:
        console.print("[dim]No artifacts found.[/dim]")
        return

    table = Table(title="Artifacts", show_lines=False)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Name", style="bold")
    table.add_column("Project", style="dim")
    table.add_column("State", width=12)
    table.add_column("Type", width=10)
    table.add_column("Updated")

    state_colors = {
        "draft": "dim", "building": "yellow", "testing": "blue",
        "reviewing": "magenta", "approved": "green", "deployed": "green bold",
        "error": "red", "update_pending": "yellow",
    }

    for a in items:
        st = a.get("state", "?")
        color = state_colors.get(st, "")
        table.add_row(
            a["id"],
            a["name"],
            a.get("project_id", ""),
            f"[{color}]{st}[/{color}]" if color else st,
            a.get("type", ""),
            a.get("updated_at", "")[:19],
        )
    console.print(table)


# ============================================================================
# get
# ============================================================================

@artifact_app.command("get")
def artifact_get(
    artifact_id: str = typer.Argument(..., help="Artifact ID (8-char hex)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Show detailed info for a single artifact."""
    data = _api("GET", f"/api/artifacts/{artifact_id}")

    if json_output:
        rprint(json.dumps(data, indent=2))
        return

    manifest = data.get("manifest", {})
    connectors = manifest.get("connectors", [])
    pipeline = manifest.get("pipeline", {})
    stages = pipeline.get("stages", [])

    info = f"""[cyan]ID:[/cyan]       {data['id']}
[cyan]Name:[/cyan]     {data['name']}
[cyan]Project:[/cyan]  {data.get('project_id', '')}
[cyan]State:[/cyan]    {data['state']}
[cyan]Type:[/cyan]     {data.get('type', '')}
[cyan]Version:[/cyan]  {data.get('version', '')}
[cyan]Path:[/cyan]     {data.get('disk_path', '')}
[cyan]Created:[/cyan]  {data.get('created_at', '')[:19]}
[cyan]Updated:[/cyan]  {data.get('updated_at', '')[:19]}"""

    if connectors:
        names = [c if isinstance(c, str) else c.get("connector", "?") for c in connectors]
        info += f"\n[cyan]Connectors:[/cyan] {', '.join(names)}"

    if stages:
        stage_desc = ", ".join(f"{s.get('id', '?')}({s.get('type', '?')})" for s in stages)
        info += f"\n[cyan]Pipeline:[/cyan]  {stage_desc}"

    console.print(Panel(info, title=f"Artifact {data['id']}", border_style="blue"))


# ============================================================================
# create
# ============================================================================

@artifact_app.command("create")
def artifact_create(
    name: str = typer.Argument(..., help="Artifact name"),
    project: str = typer.Option("default", "--project", "-p", help="Project ID"),
    template: str = typer.Option(None, "--template", "-t", help="Template name (e.g. text_factory)"),
    persistent: bool = typer.Option(False, "--persistent", help="Create as persistent (default: ephemeral)"),
    description: str = typer.Option("", "--desc", "-d", help="Description"),
):
    """Create a new artifact sandbox."""
    body: dict = {
        "name": name,
        "project_id": project,
        "type": "persistent" if persistent else "ephemeral",
    }
    if description:
        body["description"] = description
    if template:
        body["template"] = template

    data = _api("POST", "/api/artifacts", json=body)
    console.print(f"[green]Created:[/green] {data['id']}  {data['name']}")
    console.print(f"  State: {data['state']}  Path: {data.get('disk_path', '')}")


# ============================================================================
# delete
# ============================================================================

@artifact_app.command("delete")
def artifact_delete(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete an artifact and all its data."""
    if not force:
        data = _api("GET", f"/api/artifacts/{artifact_id}")
        if not typer.confirm(f"Delete artifact '{data['name']}' ({artifact_id})? This is irreversible."):
            raise typer.Abort()

    _api("DELETE", f"/api/artifacts/{artifact_id}")
    console.print(f"[red]Deleted:[/red] {artifact_id}")


# ============================================================================
# state
# ============================================================================

@artifact_app.command("state")
def artifact_state(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
    target: str = typer.Argument(..., help="Target state (building, testing, approved, etc.)"),
):
    """Transition an artifact to a new state."""
    data = _api("POST", f"/api/artifacts/{artifact_id}/state",
                json={"target_state": target, "actor": "cli"})
    console.print(f"[green]{artifact_id}:[/green] → {data['state']}")


# ============================================================================
# start / stop (backend process)
# ============================================================================

@artifact_app.command("start")
def artifact_start(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
):
    """Start the artifact's backend process."""
    data = _api("POST", f"/api/artifacts/{artifact_id}/backend/start")
    console.print(f"[green]Backend started:[/green] pid={data.get('pid', '?')}")


@artifact_app.command("stop")
def artifact_stop(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
):
    """Stop the artifact's backend process."""
    _api("POST", f"/api/artifacts/{artifact_id}/backend/stop")
    console.print(f"[yellow]Backend stopped:[/yellow] {artifact_id}")


# ============================================================================
# call (backend action)
# ============================================================================

@artifact_app.command("call")
def artifact_call(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
    action: str = typer.Argument(..., help="Action name (get_config, test, generate, etc.)"),
    params_json: str = typer.Argument("{}", help="JSON params string"),
    json_output: bool = typer.Option(False, "--json", help="Output raw JSON"),
):
    """Call a backend action on an artifact (e.g. test, generate, get_config)."""
    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as e:
        console.print(f"[red]Invalid JSON params:[/red] {e}")
        raise typer.Exit(1)

    # Pipeline actions need longer timeout
    slow = action in {"test", "generate", "run_runbook"}
    data = _api("POST", f"/api/artifacts/{artifact_id}/backend/call",
                json={"action": action, "params": params},
                timeout=120.0 if slow else 30.0)

    if json_output:
        rprint(json.dumps(data, indent=2))
        return

    result = data.get("result", data)

    # Pretty-print known action results
    if action == "get_config":
        stages = result.get("stages", [])
        console.print(f"[cyan]Pipeline:[/cyan] {len(stages)} stages")
        for s in stages:
            console.print(f"  {s.get('id', '?')}: {s.get('type', '?')} → {s.get('connector', '?')}/{s.get('model', '?')}")

    elif action == "test":
        text = result.get("text", "")
        score = result.get("score")
        feedback = result.get("feedback", "")
        console.print(f"[cyan]Run:[/cyan] {result.get('run_id', '?')}")
        console.print(f"[cyan]Text:[/cyan] {len(text)} chars")
        if text:
            console.print(Panel(text[:800] + ("..." if len(text) > 800 else ""),
                               title="Generated Content", border_style="dim"))
        if score is not None:
            color = "green" if score >= 7 else "yellow" if score >= 5 else "red"
            console.print(f"[cyan]Score:[/cyan] [{color}]{score:.1f}[/{color}]/10")
        if feedback:
            console.print(f"[cyan]Feedback:[/cyan] {feedback[:200]}")
        console.print(f"[cyan]Cost:[/cyan] ${result.get('cost_usd', 0):.6f}")

    else:
        rprint(json.dumps(result, indent=2))


# ============================================================================
# templates
# ============================================================================

@artifact_app.command("templates")
def artifact_templates():
    """List available artifact templates."""
    data = _api("GET", "/api/artifacts/templates")
    if not data:
        console.print("[dim]No templates found.[/dim]")
        return

    table = Table(title="Artifact Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for t in data:
        table.add_row(t.get("name", "?"), t.get("description", ""))
    console.print(table)


# ============================================================================
# files (list remote)
# ============================================================================

@artifact_app.command("files")
def artifact_files(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """List all files in an artifact with sizes and content hashes."""
    data = _api("GET", f"/api/artifacts/{artifact_id}/files")

    if json_output:
        rprint(json.dumps(data, indent=2))
        return

    files = data.get("files", [])
    if not files:
        console.print("[dim]No files.[/dim]")
        return

    table = Table(title=f"Artifact {artifact_id} — {len(files)} files")
    table.add_column("Path", style="cyan")
    table.add_column("Size", justify="right")
    table.add_column("Hash", style="dim", width=16)

    for f in files:
        size = f["size"]
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f}M"
        elif size > 1024:
            size_str = f"{size / 1024:.1f}K"
        else:
            size_str = f"{size}B"
        table.add_row(f["path"], size_str, f["sha256"])
    console.print(table)


# ============================================================================
# sync (push local folder → artifact)
# ============================================================================

# Dirs/patterns never included in sync zips
_SYNC_EXCLUDE = {".vlt", ".git", "__pycache__", ".deps", "node_modules",
                 ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def _make_sync_zip(folder: Path) -> bytes:
    """Zip a local folder for sync upload, skipping excluded dirs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(folder)
            if any(p in _SYNC_EXCLUDE for p in rel.parts):
                continue
            zf.write(file_path, str(rel))
    return buf.getvalue()


@artifact_app.command("sync")
def artifact_sync(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
    folder: str = typer.Argument(".", help="Local folder to sync from"),
    delete: bool = typer.Option(False, "--delete", help="Remove files on artifact not in local folder"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show what would change without writing"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """Push a local folder to an artifact (like rsync/git push).

    Compares local files against the artifact's disk and writes only changes.
    New files are added, modified files are overwritten, unchanged files are
    skipped.  Use --delete to remove files on the artifact that don't exist
    locally.
    """
    import httpx

    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        console.print(f"[red]Not a directory:[/red] {folder}")
        raise typer.Exit(1)

    zip_bytes = _make_sync_zip(folder_path)
    file_count = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        file_count = sum(1 for n in zf.namelist() if not n.endswith("/"))

    if not dry_run:
        console.print(f"Syncing {file_count} files from [cyan]{folder_path}[/cyan] → artifact [cyan]{artifact_id}[/cyan]")

    params = {}
    if delete:
        params["delete"] = "true"
    if dry_run:
        params["dry_run"] = "true"

    try:
        resp = httpx.post(
            f"{_daemon_url()}/api/artifacts/{artifact_id}/sync",
            params=params,
            files={"file": ("sync.zip", zip_bytes, "application/zip")},
            timeout=30.0,
        )
        if resp.status_code >= 400:
            detail = resp.text[:300]
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            console.print(f"[red]Error ({resp.status_code}):[/red] {detail}")
            raise typer.Exit(1)
        data = resp.json()
    except httpx.RequestError as e:
        console.print(f"[red]Connection error:[/red] {e}")
        raise typer.Exit(1)

    if json_output:
        rprint(json.dumps(data, indent=2))
        return

    added = data.get("added", [])
    modified = data.get("modified", [])
    unchanged = data.get("unchanged", [])
    deleted = data.get("deleted", [])
    prefix = "[dim](dry run)[/dim] " if dry_run else ""

    if added:
        for f in added:
            console.print(f"  {prefix}[green]+ {f}[/green]")
    if modified:
        for f in modified:
            console.print(f"  {prefix}[yellow]~ {f}[/yellow]")
    if deleted:
        for f in deleted:
            console.print(f"  {prefix}[red]- {f}[/red]")

    changes = len(added) + len(modified) + len(deleted)
    if changes == 0:
        console.print("[dim]Already up to date.[/dim]")
    else:
        console.print(
            f"\n{prefix}[green]+{len(added)}[/green]  "
            f"[yellow]~{len(modified)}[/yellow]  "
            f"[red]-{len(deleted)}[/red]  "
            f"[dim]={len(unchanged)} unchanged[/dim]"
        )


# ============================================================================
# pull (download artifact → local folder)
# ============================================================================

@artifact_app.command("pull")
def artifact_pull(
    artifact_id: str = typer.Argument(..., help="Artifact ID"),
    folder: str = typer.Argument(".", help="Local folder to download into"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files without asking"),
):
    """Download an artifact's files to a local folder (like git clone/pull).

    Exports the artifact as a zip and extracts it to the target folder.
    Existing files are overwritten; extra local files are NOT deleted.
    """
    import httpx

    folder_path = Path(folder).resolve()
    folder_path.mkdir(parents=True, exist_ok=True)

    # Check if target has files already
    existing = list(folder_path.iterdir())
    if existing and not force:
        if not typer.confirm(f"{folder_path} has {len(existing)} items. Overwrite?"):
            raise typer.Abort()

    try:
        resp = httpx.get(
            f"{_daemon_url()}/api/artifacts/{artifact_id}/export",
            timeout=30.0,
        )
        if resp.status_code >= 400:
            console.print(f"[red]Error ({resp.status_code}):[/red] {resp.text[:300]}")
            raise typer.Exit(1)
    except httpx.RequestError as e:
        console.print(f"[red]Connection error:[/red] {e}")
        raise typer.Exit(1)

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    extracted = 0
    for member in zf.namelist():
        if member.endswith("/"):
            continue
        target = folder_path / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(zf.read(member))
        extracted += 1

    console.print(f"[green]Pulled {extracted} files[/green] → {folder_path}")
