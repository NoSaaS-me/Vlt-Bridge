import typer
from pathlib import Path
from typing import List
from rich import print
from rich.table import Table
from rich.markdown import Markdown
from rich.panel import Panel
import json
import os
import time
import logging
from vlt.core.migrations import init_db
from vlt.core.service import SqliteVaultService
from vlt.core.librarian import Librarian
from vlt.lib.llm import OpenRouterLLMProvider
from vlt.config import Settings

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
app.add_typer(thread_app, name="thread")
app.add_typer(config_app, name="config")
app.add_typer(sync_app, name="sync")

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

from vlt.core.identity import create_vlt_toml, load_project_identity

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
    """
    from vlt.core.sync import ThreadSyncClient
    import asyncio

    client = ThreadSyncClient()
    result = asyncio.run(client.retry_queue())

    print(f"[green]Success: {result['success']}[/green]")
    print(f"[red]Failed: {result['failed']}[/red]")
    print(f"[yellow]Skipped (max retries): {result['skipped']}[/yellow]")


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

@app.command()
def init(
    project: str = typer.Option(None, "--project", "-p", help="Initialize a vlt.toml for this directory with the given project name.")
):
    """
    Initialize the Vault DB or a Project Context.
    
    - Default: Initializes the local DB (~/.vlt/vault.db).
    - With --project: Creates a 'vlt.toml' file in the current directory, anchoring it to a project.
    """
    if project:
        # Create vlt.toml
        project_id = project.lower().replace(" ", "-")
        create_vlt_toml(Path("."), name=project, id=project_id)
        print(f"[bold green]Initialized project '{project}' (id: {project_id}) in vlt.toml[/bold green]")
        
        # Ensure project exists in DB too
        try:
            service.create_project(name=project, description="Initialized via vlt init")
        except Exception:
            pass
        return

    print("[bold green]Initializing Vault database...[/bold green]")
    init_db()
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
    from vlt.core.sync import sync_thread_entry
    from vlt.config import settings
    import asyncio

    # Only attempt sync if server is configured
    if settings.is_server_configured:
        try:
            thread_info = service.get_thread_state(thread_slug, limit=1)
            if thread_info:
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
                    print("[dim]Synced to server[/dim]")
                else:
                    print("[dim yellow]Queued for sync (will retry)[/dim yellow]")
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

@coderag_app.command("init")
def coderag_init(
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    path: Path = typer.Option(None, "--path", help="Directory to index (defaults to current directory)"),
    force: bool = typer.Option(False, "--force", help="Force full re-index (ignore incremental)"),
):
    """
    Initialize all indexes for a project.

    This command performs a full codebase index:
    - Parses files using tree-sitter
    - Generates context-enriched semantic chunks
    - Creates vector embeddings (qwen/qwen3-embedding-8b)
    - Builds BM25 keyword index
    - Constructs import/call graph
    - Generates repository map
    - Runs ctags for symbol index

    By default, uses incremental indexing (only indexes changed files).
    Use --force to re-index everything.
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
            console.print("Usage: vlt coderag init --project <project>")
            console.print("Or run: vlt init --project <name> to create vlt.toml")
            raise typer.Exit(code=1)

    # Resolve path
    if not path:
        path = Path(".")

    console.print(f"[bold blue]Initializing CodeRAG index for project '{project}'[/bold blue]")
    console.print(f"Path: {path.resolve()}")
    console.print(f"Mode: {'Full re-index' if force else 'Incremental'}")
    console.print()

    # Create indexer
    indexer = CodeRAGIndexer(path, project)

    # Run indexing with progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Parsing files...", total=None)

        try:
            # Run index
            stats = indexer.index_full(force=force)

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
                    console.print(f"  • {error}")
                if len(stats.errors) > 10:
                    console.print(f"  ... and {len(stats.errors) - 10} more errors")

        except Exception as e:
            progress.update(task, description="[red]Indexing failed![/red]")
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(code=1)


@coderag_app.command("status")
def coderag_status(
    project: str = typer.Option(None, "--project", "-p", help="Project ID (auto-detected from vlt.toml if not specified)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
):
    """
    Display index health and statistics.

    Shows:
    - Files count
    - Chunks count
    - Symbols count
    - Graph nodes/edges count
    - Last indexed time
    - Repository map statistics
    - Delta queue count (pending changes)
    """
    from vlt.core.identity import load_project_identity
    from vlt.core.coderag.indexer import CodeRAGIndexer
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

    # Get status
    indexer = CodeRAGIndexer(Path("."), project)
    status = indexer.get_index_status()

    if json_output:
        console.print(json_lib.dumps(status, indent=2))
        return

    # Display as table
    console.print(f"[bold blue]CodeRAG Index Status[/bold blue]")
    console.print(f"Project: {status['project_id']}")
    console.print()

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
        if queued_files > 0 and not json_output:
            console.print()
            console.print("[bold]Queued Files:[/bold]")
            for entry in delta_queue.get('queued_entries', [])[:5]:  # Show first 5
                file_path = entry['file_path']
                change_type = entry['change_type']
                lines = entry['lines_changed']
                age_min = entry['age_seconds'] // 60
                console.print(f"  • {file_path} ({change_type}, +{lines} lines, {age_min}m ago)")

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
        console.print(json_lib.dumps(json_results, indent=2))
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
):
    """
    Ask Oracle a question about the codebase.

    Oracle is a multi-source intelligent context retrieval system that:
    - Searches code index (vector + BM25 + graph)
    - Searches documentation vault (markdown notes)
    - Searches development threads (historical context)
    - Reranks results for relevance
    - Synthesizes a comprehensive answer with citations

    Examples:
        vlt oracle "How does authentication work?"
        vlt oracle "Where is UserService defined?" --source code
        vlt oracle "What calls the login function?" --explain
        vlt oracle "Why did we choose SQLite?" --source threads

    The response includes:
    - A synthesized answer from an LLM
    - Source citations [file.py:42], [note.md], [thread:id#node]
    - Repository structure context
    - Cost and timing information
    """
    import asyncio
    from vlt.core.identity import load_project_identity
    from vlt.core.oracle import OracleOrchestrator
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
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

    # Check if API key is configured
    # Oracle currently needs OpenRouter for synthesis, but in the future
    # this will be handled server-side like the Librarian
    settings = Settings()
    if not settings.openrouter_api_key and not settings.sync_token:
        console.print("[red]Error: No API credentials configured.[/red]")
        console.print()
        console.print("Option 1 (Recommended): Configure server sync token:")
        console.print("  vlt config set-key <your-sync-token>")
        console.print()
        console.print("Option 2 (Legacy): Set OpenRouter API key directly:")
        console.print("  export VLT_OPENROUTER_API_KEY=<your-api-key>")
        raise typer.Exit(code=1)

    # Warn if using legacy mode
    if settings.openrouter_api_key and not settings.sync_token:
        console.print("[yellow]Note: Using legacy OpenRouter mode. Consider migrating to server sync.[/yellow]")
        console.print("[dim]Run: vlt config set-key <sync-token> --server <url>[/dim]")
        console.print()

    # Display query
    console.print()
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {question}",
        title="Oracle Query",
        border_style="blue"
    ))
    console.print()

    # Show status while processing
    with console.status("[bold blue]Searching knowledge sources...[/bold blue]") as status:
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
                    "path": source.source_path,
                    "type": source.source_type.value,
                    "method": source.retrieval_method.value,
                    "score": source.score
                }
                for source in response.sources
            ],
            "query_type": response.query_type,
            "model": response.model,
            "tokens_used": response.tokens_used,
            "cost_cents": response.cost_cents,
            "duration_ms": response.duration_ms,
        }

        if response.traces:
            output_data["traces"] = response.traces

        console.print(json_lib.dumps(output_data, indent=2))
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
        for i, source in enumerate(response.sources[:5], 1):  # Show top 5
            score_color = "green" if source.score >= 0.8 else "yellow"
            console.print(
                f"  {i}. [{score_color}]{source.source_path}[/{score_color}] "
                f"({source.source_type.value} via {source.retrieval_method.value}, "
                f"score: {source.score:.2f})"
            )

    # Show metadata
    console.print()
    console.print(
        f"[dim]Query type: {response.query_type} | "
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


if __name__ == "__main__":
    app()
