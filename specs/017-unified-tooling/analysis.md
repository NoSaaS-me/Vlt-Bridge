# Tool Surface Fragmentation Analysis

## Executive Summary

The Vlt-Bridge codebase suffers from severe tool architecture fragmentation across **6 distinct patterns** for tool definition and **4 different execution paths**. This analysis identifies the root causes, documents each pattern's quirks, and proposes a unified Tool abstraction that can serve all clients (MCP, REST, Agent, Plugins) while enabling proper agent-tool composition.

---

## 1. Current State Audit

### 1.1 Tool Definition Locations

Tools are defined in **six different places** with **incompatible schemas**:

| Location | Pattern | Schema Format | Count |
|----------|---------|---------------|-------|
| `backend/prompts/tools.json` | Static JSON | OpenRouter function calling | 20 tools |
| `backend/src/mcp/server.py` | FastMCP decorators | Pydantic Field + docstring | 12 tools |
| `backend/src/services/tool_executor.py` | Handler registry | Inline Python methods | 20 handlers |
| `backend/src/api/routes/*.py` | FastAPI routes | Pydantic request models | 18 routes |
| `backend/src/services/research/behaviors.py` | Behavior classes | Custom async methods | 7 behaviors |
| `backend/src/services/plugins/engine.py` | Rule actions | ActionType enum | 6 action types |

### 1.2 Tool Definition #1: `prompts/tools.json`

**Location:** `/backend/prompts/tools.json`

This is the **canonical source** for Oracle Agent tools, using OpenRouter's function calling schema:

```json
{
  "type": "function",
  "function": {
    "name": "search_code",
    "description": "Search the codebase...",
    "parameters": {
      "type": "object",
      "properties": {
        "query": { "type": "string", "description": "..." },
        "limit": { "type": "integer", "minimum": 1, "maximum": 20, "default": 5 }
      },
      "required": ["query"]
    }
  },
  "agent_scope": ["oracle", "librarian"],  // Custom extension
  "category": "code"                        // Custom extension
}
```

**Issues:**
- Custom extensions (`agent_scope`, `category`) not part of OpenRouter spec
- No validation at load time
- Duplicates information from handler implementations
- Agent definitions (`oracle`, `librarian`) mixed with tool definitions

### 1.3 Tool Definition #2: MCP Server FastMCP Decorators

**Location:** `/backend/src/mcp/server.py`

MCP tools use FastMCP's decorator pattern with Pydantic Field descriptions:

```python
@mcp.tool(
    name="search_notes",
    description="Full-text search with snippets...",
    meta={
        "openai/outputTemplate": "ui://widget/note.html",
        "openai/resultCanProduceWidget": True,
        "openai/toolInvocation/invoking": "Searching...",
        "openai/toolInvocation/invoked": "Search complete."
    }
)
def search_notes(
    query: str = Field(..., description="Non-empty search query"),
    limit: int = Field(50, ge=1, le=100, description="Result cap"),
    tags: Optional[List[str]] = Field(default=None, description="Filter by tags"),
) -> ToolResult:
    # Direct service call
    results = indexer_service.search_notes(user_id, query, tags=tags, limit=limit)
    return ToolResult(
        content=[TextContent(type="text", text=f"Found {len(results)} notes")],
        structured_content={"results": structured_results},
        meta={...}
    )
```

**Issues:**
- Schema defined inline via Pydantic Field, not shared with `tools.json`
- ChatGPT-specific meta keys (`openai/outputTemplate`) leak into non-OpenAI clients
- Return type (`ToolResult`) has MCP-specific structure
- Validation is separate from tool_executor validation

### 1.4 Tool Definition #3: ToolExecutor Handler Registry

**Location:** `/backend/src/services/tool_executor.py`

The giant dispatcher pattern:

```python
class ToolExecutor:
    def __init__(self, ...):
        self._tools: Dict[str, Any] = {
            "search_code": self._search_code,
            "find_definition": self._find_definition,
            # ... 20 more entries
        }

    async def execute(self, name: str, arguments: Dict[str, Any], user_id: str, timeout: Optional[float] = None) -> str:
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})
        handler = self._tools[name]
        result = await asyncio.wait_for(handler(user_id, **arguments), timeout=actual_timeout)
        return json.dumps(result, default=str)
```

**Issues:**
- Tool registration is implicit (handler name must match JSON definition)
- No schema validation before execution
- Response is always JSON string (loses typing)
- Error handling is bespoke per-handler
- Timeouts defined both here and in `tools.json` (duplication)

### 1.5 Tool Definition #4: REST API Routes

**Location:** `/backend/src/api/routes/notes.py`, `oracle.py`, etc.

REST endpoints that overlap with agent tools:

```python
@router.get("/api/notes/{path:path}", response_model=Note)
async def get_note(
    path: str,
    project_id: str = Query(DEFAULT_PROJECT_ID),
    auth: AuthContext = Depends(require_auth_context),
):
    vault_service = VaultService()
    note_data = vault_service.read_note(user_id, note_path, project_id)
    return Note(...)
```

**Issues:**
- Duplicates `vault_read` tool logic
- Different parameter names (`path` vs `note_path`)
- Different auth context injection pattern
- Different response model (`Note` vs `Dict`)
- No way to share schema/validation with agent tools

### 1.6 Tool Definition #5: Research Behavior Classes

**Location:** `/backend/src/services/research/behaviors.py`

Internal "tool-like" behaviors with yet another pattern:

```python
class ResearcherBehavior(ResearchBehavior):
    async def run_single(self, state: ResearchState, researcher: ResearcherState) -> ResearcherState:
        queries = await self._generate_search_queries(state, researcher)
        if self.search_provider == "tavily":
            search_responses = await self.tavily.search_parallel(queries=queries, ...)
        elif self.search_provider == "openrouter":
            search_responses = await self.openrouter_search.search_parallel(...)
        # ...
```

**Issues:**
- Not exposed as tools to agents
- Hardcoded provider switching (Tavily vs OpenRouter)
- State mutation pattern differs from stateless tools
- No schema, no validation, no discoverability

### 1.7 Tool Definition #6: Plugin Rule Actions

**Location:** `/backend/src/services/plugins/engine.py`

Actions triggered by rule evaluation:

```python
class ActionType(Enum):
    NOTIFY_SELF = "notify_self"
    LOG = "log"
    SET_STATE = "set_state"
    EMIT_EVENT = "emit_event"
    MODIFY_PROMPT = "modify_prompt"
    INJECT_CONTEXT = "inject_context"
```

**Issues:**
- Not discoverable by agents
- Different dispatch mechanism (ActionDispatcher vs ToolExecutor)
- Actions can modify agent state (side effects)
- No schema for action parameters

---

## 2. Tool Execution Fragmentation

### 2.1 Execution Path #1: OracleAgent -> ToolExecutor

```
OracleAgent.query()
  -> LLM returns tool_calls
  -> _parse_xml_tool_calls() or native function calling
  -> ToolExecutor.execute(name, arguments, user_id)
  -> handler method (e.g., _search_code)
  -> Return JSON string
  -> Append to conversation as tool result
```

**Characteristics:**
- Async with configurable timeout
- JSON string response
- ANS event emission on failures
- No streaming

### 2.2 Execution Path #2: MCP Server -> Service

```
MCP client sends tool request
  -> FastMCP routes to @mcp.tool function
  -> _current_user_id() extracts auth
  -> Direct service call (vault_service.read_note())
  -> Return ToolResult with TextContent + structured_content
```

**Characteristics:**
- Sync (wrapped by FastMCP)
- ToolResult response with meta
- No event emission
- No timeout handling (relies on client)

### 2.3 Execution Path #3: REST API -> Service

```
HTTP request to /api/notes/{path}
  -> FastAPI routing
  -> Depends() for auth
  -> Direct service call
  -> Pydantic model response
```

**Characteristics:**
- Async
- Pydantic response model
- HTTPException for errors
- Standard HTTP timeout

### 2.4 Execution Path #4: Research Orchestrator -> Behaviors

```
ResearchOrchestrator.run_research()
  -> GenerateBriefBehavior.run()
  -> ParallelResearchersBehavior.run()
  -> Each behavior calls services directly
  -> State mutation
```

**Characteristics:**
- Behavior tree pattern
- Mutable state
- No external tool interface
- Progress tracking via state

---

## 3. The MCP iFrame Problem

### 3.1 Location of ChatGPT-Specific Code

**File:** `/backend/src/mcp/server.py`

The widget resource with ChatGPT iFrame handling:

```python
@mcp.resource("ui://widget/note.html", mime_type="text/html+skybridge")
def widget_resource() -> str:
    """Return the widget HTML bundle."""
    widget_path = PROJECT_ROOT / "frontend" / "dist" / "widget.html"
    html_content = widget_path.read_text(encoding="utf-8")

    # Inject API_BASE_URL for widget
    html_content = html_content.replace(
        '<head>',
        f'<head><script>window.API_BASE_URL = "{base_url}";</script>'
    )

    # Fix asset paths for iframe context
    html_content = html_content.replace('src="/assets/', f'src="{base_url}/assets/')
    html_content = html_content.replace('href="/assets/', f'href="{base_url}/assets/')
    return html_content
```

### 3.2 OpenAI-Specific Meta in Tool Responses

```python
@mcp.tool(name="read_note", ...)
def read_note(path: str = Field(...)) -> dict:
    return ToolResult(
        content=[TextContent(...)],
        structured_content={"note": structured_note},
        meta={
            "openai/outputTemplate": "ui://widget/note.html",
            "openai/resultCanProduceWidget": True,
            "openai/toolInvocation/invoking": f"Opening {note['title']}...",
            "openai/toolInvocation/invoked": f"Loaded {note['title']}"
        }
    )
```

### 3.3 Why This Is Problematic

1. **Non-OpenAI MCP clients** (Claude Desktop, VS Code, Raycast) see meaningless `openai/*` keys
2. **Widget resource MIME type** (`text/html+skybridge`) is ChatGPT-specific
3. **Asset path injection** assumes iframe context
4. **No abstraction layer** to swap rendering strategies per client
5. **Mixed concerns**: tool logic + rendering + client-specific workarounds

### 3.4 Making It Client-Aware

A proper solution would:

1. Detect client type from MCP transport or headers
2. Load client-specific rendering configuration
3. Transform responses at the edge (not in tool code)

```python
# Conceptual fix
class ClientAwareRenderer:
    def render_tool_result(self, result: ToolResult, client: ClientType) -> Any:
        if client == ClientType.CHATGPT:
            return self._wrap_for_chatgpt(result)
        elif client == ClientType.CLAUDE:
            return self._wrap_for_claude(result)
        else:
            return result.structured_content  # Plain JSON
```

---

## 4. The Crossroads: Tools and Agents

### 4.1 Current Agent-Tool Relationship

```
                    +----------------+
                    |  OracleAgent   |
                    |  (owns tools)  |
                    +-------+--------+
                            |
              +-------------+-------------+
              |                           |
    +---------v---------+     +-----------v-----------+
    |   ToolExecutor    |     |   LibrarianAgent      |
    | (20 hardcoded)    |     | (10 subset tools)     |
    +-------------------+     +-----------------------+
```

**Problems:**
- Agents have fixed tool sets defined in `tools.json`
- Adding a tool requires updating JSON, executor, and agent config
- Sub-agents (Librarian) get tools via `agent_scope` filter
- No runtime tool composition

### 4.2 What Agents Need From Tools

1. **Discovery**: Which tools exist and what do they do?
2. **Schema**: What parameters do they accept?
3. **Invocation**: How to call them (sync/async, timeout)?
4. **Results**: Structured responses with typing
5. **Scoping**: Which tools am I allowed to use?
6. **Composition**: Can I add/remove tools dynamically?

### 4.3 What Tools Need From Agents

1. **Context**: Who is calling (user_id, project_id)?
2. **Auth**: What permissions does the caller have?
3. **Lifecycle**: When was I started, can I be cancelled?
4. **Observability**: Log my execution, emit events

### 4.4 Proposed Agent-Tool Architecture

```
                    +------------------------+
                    |     ToolRegistry       |
                    | (discovers all tools)  |
                    +-----------+------------+
                                |
         +----------------------+----------------------+
         |                      |                      |
+--------v--------+    +--------v--------+    +--------v--------+
|   BaseTool      |    |   BaseTool      |    |   BaseTool      |
| (vault_read)    |    | (search_code)   |    | (web_search)    |
+-----------------+    +-----------------+    +-----------------+
         ^                      ^                      ^
         |                      |                      |
+--------+--------+    +--------+--------+    +--------+--------+
|  ToolCapability |    |  ToolCapability |    |  ToolCapability |
| (scope: oracle) |    | (scope: both)   |    | (scope: oracle) |
+-----------------+    +-----------------+    +-----------------+
         |                      |                      |
         +----------------------+----------------------+
                                |
                    +-----------v------------+
                    |     AgentToolset       |
                    | (filtered by scope)    |
                    +-----------+------------+
                                |
                    +-----------v------------+
                    |     OracleAgent        |
                    | (uses toolset)         |
                    +------------------------+
```

### 4.5 Should Agents Define Their Own Tools?

**Yes, but with constraints:**

```python
class OracleAgent:
    def define_tools(self) -> List[Tool]:
        # Agent can add synthetic/dynamic tools
        return [
            self._create_delegation_tool(),  # delegate_librarian
            self._create_memory_tool(),      # thread_push
        ]

    def get_toolset(self, registry: ToolRegistry) -> AgentToolset:
        # Combine registered tools + agent-specific tools
        base_tools = registry.get_by_scope("oracle")
        agent_tools = self.define_tools()
        return AgentToolset(base_tools + agent_tools)
```

### 4.6 How MCP Fits In

MCP becomes **one exposure surface** for tools, not the source of truth:

```
+----------------+     +----------------+     +----------------+
|   MCP Client   |     |   REST Client  |     |   Agent        |
+-------+--------+     +-------+--------+     +-------+--------+
        |                      |                      |
        v                      v                      v
+-------+--------+     +-------+--------+     +-------+--------+
|  MCPAdapter    |     |  RESTAdapter   |     |  AgentAdapter  |
| (ToolResult)   |     | (Pydantic)     |     | (JSON string)  |
+-------+--------+     +-------+--------+     +-------+--------+
        |                      |                      |
        +----------------------+----------------------+
                               |
                    +----------v-----------+
                    |    Unified Tool      |
                    | execute(ctx, params) |
                    +----------------------+
```

---

## 5. Unified Tool Architecture

### 5.1 The BaseTool Abstract Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar
from pydantic import BaseModel

TParams = TypeVar("TParams", bound=BaseModel)
TResult = TypeVar("TResult", bound=BaseModel)


@dataclass
class ToolContext:
    """Context passed to every tool invocation."""
    user_id: str
    project_id: str
    request_id: str
    timeout_seconds: float
    permissions: set[str]
    client_type: str  # "mcp", "rest", "agent", "plugin"
    metadata: Dict[str, Any]


@dataclass
class ToolCapability:
    """Metadata about a tool's capabilities."""
    name: str
    description: str
    category: str
    agent_scopes: list[str]
    is_async: bool = True
    supports_streaming: bool = False
    timeout_default: float = 30.0
    requires_auth: bool = True
    side_effects: bool = False  # Does this tool modify state?


class ToolError(Exception):
    """Structured tool error with category for agent hints."""
    def __init__(
        self,
        message: str,
        category: str,  # "validation", "not_found", "timeout", "auth", "config"
        recoverable: bool = True,
        suggestion: Optional[str] = None,
    ):
        self.message = message
        self.category = category
        self.recoverable = recoverable
        self.suggestion = suggestion


class BaseTool(ABC, Generic[TParams, TResult]):
    """Base class for all tools in the system."""

    @property
    @abstractmethod
    def capability(self) -> ToolCapability:
        """Return this tool's capability metadata."""
        pass

    @property
    @abstractmethod
    def params_model(self) -> type[TParams]:
        """Return the Pydantic model for input parameters."""
        pass

    @property
    @abstractmethod
    def result_model(self) -> type[TResult]:
        """Return the Pydantic model for output."""
        pass

    @abstractmethod
    async def execute(self, ctx: ToolContext, params: TParams) -> TResult:
        """Execute the tool with validated parameters."""
        pass

    def to_openrouter_schema(self) -> Dict[str, Any]:
        """Generate OpenRouter function calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.capability.name,
                "description": self.capability.description,
                "parameters": self.params_model.model_json_schema(),
            }
        }

    def to_mcp_schema(self) -> Dict[str, Any]:
        """Generate MCP tool schema."""
        return {
            "name": self.capability.name,
            "description": self.capability.description,
            "inputSchema": self.params_model.model_json_schema(),
        }
```

### 5.2 Concrete Tool Example

```python
from pydantic import BaseModel, Field


class VaultReadParams(BaseModel):
    """Parameters for reading a vault note."""
    path: str = Field(..., description="Path to the note (e.g., 'research/auth.md')")


class VaultReadResult(BaseModel):
    """Result of reading a vault note."""
    path: str
    title: str
    content: str
    metadata: Dict[str, Any]


class VaultReadTool(BaseTool[VaultReadParams, VaultReadResult]):
    """Read a markdown note from the documentation vault."""

    def __init__(self, vault_service: VaultService):
        self.vault_service = vault_service

    @property
    def capability(self) -> ToolCapability:
        return ToolCapability(
            name="vault_read",
            description="Read a markdown note from the documentation vault.",
            category="vault",
            agent_scopes=["oracle", "librarian"],
            timeout_default=10.0,
        )

    @property
    def params_model(self) -> type[VaultReadParams]:
        return VaultReadParams

    @property
    def result_model(self) -> type[VaultReadResult]:
        return VaultReadResult

    async def execute(self, ctx: ToolContext, params: VaultReadParams) -> VaultReadResult:
        note = self.vault_service.read_note(ctx.user_id, params.path, project_id=ctx.project_id)
        return VaultReadResult(
            path=params.path,
            title=note.get("title", ""),
            content=note.get("body", ""),
            metadata=note.get("metadata", {}),
        )
```

### 5.3 ToolRegistry

```python
class ToolRegistry:
    """Central registry for all tools in the system."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        name = tool.capability.name
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_by_scope(self, scope: str) -> List[BaseTool]:
        """Get all tools available to an agent scope."""
        return [
            t for t in self._tools.values()
            if scope in t.capability.agent_scopes
        ]

    def get_by_category(self, category: str) -> List[BaseTool]:
        """Get all tools in a category."""
        return [
            t for t in self._tools.values()
            if t.capability.category == category
        ]

    def all_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def to_openrouter_schemas(self, scope: Optional[str] = None) -> List[Dict[str, Any]]:
        """Generate OpenRouter schemas for tools, optionally filtered by scope."""
        tools = self.get_by_scope(scope) if scope else self.all_tools()
        return [t.to_openrouter_schema() for t in tools]
```

### 5.4 Adapters for Different Clients

```python
class ToolAdapter(ABC):
    """Base class for adapters that transform tool I/O for different clients."""

    @abstractmethod
    def build_context(self, raw_request: Any) -> ToolContext:
        """Build ToolContext from client-specific request."""
        pass

    @abstractmethod
    def format_result(self, result: BaseModel, tool: BaseTool) -> Any:
        """Format tool result for client."""
        pass

    @abstractmethod
    def format_error(self, error: ToolError, tool: BaseTool) -> Any:
        """Format tool error for client."""
        pass


class MCPToolAdapter(ToolAdapter):
    """Adapter for MCP clients (Claude Desktop, ChatGPT, etc.)."""

    def __init__(self, client_type: str = "generic"):
        self.client_type = client_type

    def build_context(self, raw_request: Any) -> ToolContext:
        # Extract user_id from MCP transport
        user_id = self._extract_user_from_mcp(raw_request)
        return ToolContext(
            user_id=user_id,
            project_id="default",
            request_id=str(uuid.uuid4()),
            timeout_seconds=30.0,
            permissions=set(),
            client_type=f"mcp:{self.client_type}",
            metadata={},
        )

    def format_result(self, result: BaseModel, tool: BaseTool) -> ToolResult:
        content = [TextContent(type="text", text=f"Executed {tool.capability.name}")]
        structured = result.model_dump()

        meta = {}
        if self.client_type == "chatgpt":
            # Add ChatGPT-specific widget meta
            meta["openai/resultCanProduceWidget"] = True

        return ToolResult(content=content, structured_content=structured, meta=meta)


class AgentToolAdapter(ToolAdapter):
    """Adapter for agent tool execution (OracleAgent, LibrarianAgent)."""

    def format_result(self, result: BaseModel, tool: BaseTool) -> str:
        # Agents expect JSON string
        return json.dumps(result.model_dump(), default=str)

    def format_error(self, error: ToolError, tool: BaseTool) -> str:
        return json.dumps({
            "error": error.message,
            "category": error.category,
            "tool": tool.capability.name,
            "suggestion": error.suggestion,
        })


class RESTToolAdapter(ToolAdapter):
    """Adapter for REST API exposure."""

    def format_result(self, result: BaseModel, tool: BaseTool) -> BaseModel:
        # REST returns Pydantic model directly
        return result

    def format_error(self, error: ToolError, tool: BaseTool) -> HTTPException:
        status_map = {
            "validation": 400,
            "not_found": 404,
            "auth": 403,
            "timeout": 504,
            "config": 500,
        }
        return HTTPException(
            status_code=status_map.get(error.category, 500),
            detail={"error": error.message, "suggestion": error.suggestion},
        )
```

### 5.5 Unified Tool Executor

```python
class UnifiedToolExecutor:
    """Executes tools with adapter-specific I/O transformation."""

    def __init__(self, registry: ToolRegistry, event_bus: EventBus):
        self.registry = registry
        self.event_bus = event_bus

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        adapter: ToolAdapter,
        raw_request: Any,
    ) -> Any:
        """Execute a tool with proper context and response formatting."""

        tool = self.registry.get(tool_name)
        if tool is None:
            raise ToolError(f"Unknown tool: {tool_name}", category="not_found")

        # Build context
        ctx = adapter.build_context(raw_request)

        # Validate params
        try:
            validated_params = tool.params_model.model_validate(params)
        except ValidationError as e:
            raise ToolError(str(e), category="validation", suggestion="Check parameter types")

        # Emit pre-execution event
        self.event_bus.emit(Event(
            type=EventType.TOOL_CALL_PENDING,
            source="tool_executor",
            severity=Severity.INFO,
            payload={"tool_name": tool_name, "params": params},
        ))

        # Execute with timeout
        try:
            result = await asyncio.wait_for(
                tool.execute(ctx, validated_params),
                timeout=tool.capability.timeout_default,
            )

            # Emit success event
            self.event_bus.emit(Event(
                type=EventType.TOOL_CALL_SUCCESS,
                source="tool_executor",
                severity=Severity.INFO,
                payload={"tool_name": tool_name},
            ))

            return adapter.format_result(result, tool)

        except asyncio.TimeoutError:
            error = ToolError(
                f"Tool timed out after {tool.capability.timeout_default}s",
                category="timeout",
                suggestion="Try with smaller scope or simpler query",
            )
            self.event_bus.emit(Event(
                type=EventType.TOOL_CALL_TIMEOUT,
                source="tool_executor",
                severity=Severity.WARNING,
                payload={"tool_name": tool_name, "timeout": tool.capability.timeout_default},
            ))
            return adapter.format_error(error, tool)

        except ToolError as e:
            self.event_bus.emit(Event(
                type=EventType.TOOL_CALL_FAILURE,
                source="tool_executor",
                severity=Severity.WARNING,
                payload={"tool_name": tool_name, "error": e.message},
            ))
            return adapter.format_error(e, tool)
```

---

## 6. Migration Path

### Phase 1: Create Infrastructure (Week 1)

1. Create `backend/src/tools/` package with:
   - `base.py` - BaseTool, ToolContext, ToolCapability, ToolError
   - `registry.py` - ToolRegistry
   - `adapters/` - MCPToolAdapter, AgentToolAdapter, RESTToolAdapter
   - `executor.py` - UnifiedToolExecutor

2. Create `backend/src/tools/definitions/` with migrated tools:
   - Start with `vault_read.py`, `vault_write.py` (simple, well-defined)
   - Each tool is a single file with Params, Result, and Tool class

### Phase 2: Migrate Vault Tools (Week 2)

1. Create unified tool implementations:
   ```
   tools/definitions/
   ├── vault_read.py
   ├── vault_write.py
   ├── vault_search.py
   ├── vault_list.py
   └── vault_move.py
   ```

2. Update ToolExecutor to delegate to new tools:
   ```python
   # In tool_executor.py
   async def _vault_read(self, user_id: str, path: str, **kwargs) -> Dict[str, Any]:
       # Delegate to unified tool
       tool = self.registry.get("vault_read")
       result = await tool.execute(ctx, VaultReadParams(path=path))
       return result.model_dump()
   ```

3. Update MCP server to use registry:
   ```python
   # In mcp/server.py
   @mcp.tool(name="vault_read", ...)
   async def vault_read(path: str) -> ToolResult:
       adapter = MCPToolAdapter(client_type=detect_client())
       return await executor.execute("vault_read", {"path": path}, adapter, request)
   ```

### Phase 3: Migrate Code Tools (Week 3)

1. Migrate `search_code`, `find_definition`, `find_references`, `get_repo_map`
2. Handle special cases (CodeRAG initialization check)
3. Update OracleBridge to use unified tools where applicable

### Phase 4: Migrate Thread & Web Tools (Week 4)

1. Migrate thread tools (`thread_push`, `thread_read`, etc.)
2. Migrate web tools (`web_search`, `web_fetch`)
3. Migrate GitHub tools

### Phase 5: Migrate Meta Tools (Week 5)

1. Migrate `delegate_librarian` (complex, spawns sub-agent)
2. Migrate `notify_self` (ANS integration)
3. Migrate `deep_research` (orchestrator pattern)

### Phase 6: Delete Legacy Code (Week 6)

1. Remove `prompts/tools.json` (schemas generated from code)
2. Remove handler methods from ToolExecutor (keep as thin dispatcher)
3. Update tests to use unified tools

### Phase 7: Client-Aware MCP (Week 7)

1. Detect MCP client type from transport/headers
2. Create client-specific rendering configurations
3. Remove hardcoded `openai/*` meta keys
4. Add proper widget handling per client

---

## 7. Benefits of Unified Architecture

| Before | After |
|--------|-------|
| 6 tool definition locations | 1 canonical location (`tools/definitions/`) |
| 4 execution paths | 1 executor with adapters |
| No validation at call time | Pydantic validation on every call |
| JSON string responses | Typed Pydantic models |
| Hardcoded agent scopes | Capability-based scoping |
| ChatGPT-specific MCP | Client-aware adapters |
| Manual schema sync | Schema generated from code |
| No tool discoverability | Registry with filtering |
| Bespoke error handling | Structured ToolError |
| No observability | Event emission on all calls |

---

## 8. Open Questions

1. **Streaming tools**: How do we handle tools that can stream results (e.g., `deep_research`)?
   - Proposal: Add `stream_execute()` method to BaseTool, adapter transforms to SSE/WebSocket

2. **State-mutating tools**: How do we handle tools like `notify_self` that affect agent state?
   - Proposal: Mark with `side_effects=True`, allow adapters to handle differently

3. **Composable toolsets**: Can agents dynamically add/remove tools?
   - Proposal: AgentToolset wraps registry filter, allows runtime additions

4. **Plugin tools**: How do plugins expose new tools?
   - Proposal: Plugins register tools via ToolRegistry on load

5. **Backward compatibility**: How long do we support legacy `tools.json` format?
   - Proposal: Generate `tools.json` from registry for transition period

---

## 9. Conclusion

The Vlt-Bridge tool surface is severely fragmented across 6 definition locations and 4 execution paths. The proposed unified architecture uses:

- **BaseTool** as the single abstraction for all tools
- **ToolRegistry** for discovery and scoping
- **Adapters** for client-specific I/O transformation
- **UnifiedToolExecutor** for consistent execution with events

This enables proper agent-tool composition, eliminates schema duplication, and provides a clean path to client-aware MCP rendering without ChatGPT-specific hacks.

**Confidence: 9/10** - This analysis is based on direct reading of all relevant source files and provides concrete code examples. The one uncertainty is around streaming tools, which would need experimentation to validate the proposed approach.
