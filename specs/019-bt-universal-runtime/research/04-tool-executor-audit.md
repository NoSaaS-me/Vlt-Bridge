# Tool Executor Deep Audit for Leaf Node Integration

**Date:** 2026-01-06
**Spec:** 019-bt-universal-runtime
**Source:** `/mnt/Samsung2tb/Projects/00Tooling/Vlt-Bridge/backend/src/services/tool_executor.py` (2,539 lines)

---

## 1. Tool Registry Architecture

### Registration Pattern

The `ToolExecutor` uses a dictionary-based registry pattern initialized in `__init__`:

```python
self._tools: Dict[str, Any] = {
    # Code tools
    "search_code": self._search_code,
    "find_definition": self._find_definition,
    "find_references": self._find_references,
    "get_repo_map": self._get_repo_map,
    # Vault tools
    "vault_read": self._vault_read,
    "vault_write": self._vault_write,
    "vault_search": self._vault_search,
    "vault_list": self._vault_list,
    "vault_move": self._vault_move,
    "vault_create_index": self._vault_create_index,
    # Thread tools
    "thread_push": self._thread_push,
    "thread_read": self._thread_read,
    "thread_seek": self._thread_seek,
    "thread_list": self._thread_list,
    # Web tools
    "web_search": self._web_search,
    "web_fetch": self._web_fetch,
    # Meta tools
    "delegate_librarian": self._delegate_librarian,
    "notify_self": self._notify_self,
    # GitHub tools
    "github_read": self._github_read,
    "github_search": self._github_search,
    # Research tools
    "deep_research": self._deep_research,
}
```

### Schema Loading

Tool schemas are loaded from JSON files:
- Primary: `backend/prompts/tools.json`
- Fallback: `specs/009-oracle-agent/contracts/tools.json`

Schemas include:
- OpenRouter function calling format
- `agent_scope` field for filtering (oracle, librarian)
- `category` field for grouping

---

## 2. Service Dependencies

### Constructor Signature

```python
def __init__(
    self,
    vault_service: Optional[VaultService] = None,
    indexer_service: Optional[IndexerService] = None,
    thread_service: Optional[ThreadService] = None,
    oracle_bridge: Optional[OracleBridge] = None,
    db_service: Optional[DatabaseService] = None,
    user_settings_service: Optional[UserSettingsService] = None,
    librarian_service: Optional[LibrarianService] = None,
    github_service: Optional[GitHubService] = None,
    default_timeout: Optional[float] = None,
) -> None:
```

### Service Matrix

| Service | Instance Attr | Used By Tools |
|---------|---------------|---------------|
| `DatabaseService` | `self._db` | Thread tools, indexer |
| `VaultService` | `self.vault` | Vault tools |
| `IndexerService` | `self.indexer` | vault_write (indexing), vault_search |
| `ThreadService` | `self.threads` | All thread_* tools |
| `OracleBridge` | `self.oracle_bridge` | Code tools (search_code, find_*, get_repo_map) |
| `UserSettingsService` | `self.user_settings` | Model/API key lookups for LLM calls |
| `LibrarianService` | `self.librarian` | delegate_librarian (legacy) |
| `GitHubService` | `self.github` | github_read, github_search |

### For BT Integration

When creating leaf nodes, each tool's service dependencies must be available on the blackboard or injected via the BT context:

```python
# Example blackboard setup for vault tools
blackboard = {
    "vault_service": VaultService(),
    "indexer_service": IndexerService(db),
    "user_id": "user-123",
    "project_id": "default",
}
```

---

## 3. Execution Pattern

### Single Tool Execution

```python
async def execute(
    self,
    name: str,
    arguments: Dict[str, Any],
    user_id: str,
    timeout: Optional[float] = None,
) -> str:
```

**Flow:**
1. Validate tool exists in registry
2. Get handler function from `self._tools[name]`
3. Resolve timeout via `get_timeout()`
4. Wrap handler in `asyncio.wait_for()` for timeout protection
5. Call handler: `await handler(user_id, **arguments)`
6. Serialize result to JSON string
7. Catch and categorize exceptions

**Return Format:**
- Always returns JSON string
- Success: `{"key": "value", ...}`
- Error: `{"error": "...", "category": "...", "tool": "...", "suggestion": "..."}`

### Timeout Resolution

```python
def get_timeout(self, tool_name: str, override: Optional[float] = None, user_id: Optional[str] = None) -> float:
```

**Resolution Order:**
1. Per-call `override` parameter
2. User settings (for `delegate_librarian` only)
3. Tool-specific timeout from `TOOL_TIMEOUTS` dict
4. Instance-level `_default_timeout`
5. Class-level `DEFAULT_TIMEOUT` (30.0s)

---

## 4. Batch Execution

```python
async def execute_batch(
    self,
    tool_calls: List[Dict[str, Any]],
    user_id: str,
    timeout: Optional[float] = None,
    include_call_ids: bool = False,
) -> List[Any]:
```

**Flow:**
1. Create async tasks for each tool call
2. Execute all in parallel via `asyncio.gather(*tasks, return_exceptions=True)`
3. Process results, converting exceptions to error JSON
4. Optionally include call IDs for correlation

**Input Format:**
```python
[
    {"name": "search_code", "arguments": {"query": "auth"}, "id": "call_1"},
    {"name": "vault_list", "arguments": {}, "id": "call_2"},
]
```

---

## 5. ANS Event Emission

### Events Emitted

| Event Type | When | Severity | Payload Keys |
|------------|------|----------|--------------|
| `tool.call.timeout` | Tool exceeds timeout | WARNING | tool_name, error_type, error_message, timeout_seconds |
| `source.stale` | File changed since last read | WARNING | path, project_id, previous_hash, current_hash, message |
| `agent.self.notify` | notify_self tool (non-reminder) | Varies | message, priority, category, deliver_at, inject_at, persist_cross_session, user_id |
| `agent.self.remind` | notify_self tool (reminder category) | Varies | (same as above) |

### Timeout Event Example

```python
get_event_bus().emit(Event(
    type=EventType.TOOL_CALL_TIMEOUT,
    source="tool_executor",
    severity=Severity.WARNING,
    payload={
        "tool_name": name,
        "error_type": "timeout",
        "error_message": f"Timed out after {actual_timeout}s",
        "timeout_seconds": actual_timeout,
    }
))
```

### Missing Events

The current implementation does NOT emit:
- `tool.call.success` - Not emitted on successful completion
- `tool.call.failure` - Not emitted on exception (only logged)
- `tool.call.pending` - Not emitted before execution

**BT Integration Opportunity:** Add these events for full observability.

---

## 6. Staleness Detection

### File Read Tracking

```python
# Cache: (user_id, project_id, path) -> (content_hash, mtime)
self._file_read_times: Dict[tuple, tuple] = {}
```

### Detection Flow (in `_vault_read`)

1. Read file content
2. Hash content: `hashlib.md5(content.encode()).hexdigest()[:8]`
3. Get file mtime from filesystem
4. Check cache for previous read
5. If hash or mtime changed:
   - Update cache
   - Emit `SOURCE_STALE` event
   - Return staleness info

---

## 7. Complete Tool Inventory

### Tool Table

| Tool Name | Handler | Timeout (s) | Async | Category | Side Effects | Agent Scope |
|-----------|---------|-------------|-------|----------|--------------|-------------|
| `search_code` | `_search_code` | 30 | Yes | code | Read-only | oracle, librarian |
| `find_definition` | `_find_definition` | 30 | Yes | code | Read-only | oracle |
| `find_references` | `_find_references` | 30 | Yes | code | Read-only | oracle |
| `get_repo_map` | `_get_repo_map` | 45 | Yes | code | Read-only | oracle |
| `vault_read` | `_vault_read` | 10 | Yes | vault | Updates read cache | oracle, librarian |
| `vault_write` | `_vault_write` | 10 | Yes | vault | Writes file, updates index | oracle, librarian |
| `vault_search` | `_vault_search` | 15 | Yes | vault | Read-only | oracle, librarian |
| `vault_list` | `_vault_list` | 10 | Yes | vault | Read-only | oracle, librarian |
| `vault_move` | `_vault_move` | 10 | Yes | vault | NOT IMPLEMENTED | librarian |
| `vault_create_index` | `_vault_create_index` | 20 | Yes | vault | NOT IMPLEMENTED | librarian |
| `thread_push` | `_thread_push` | 10 | Yes | thread | Creates thread/entry in DB | oracle |
| `thread_read` | `_thread_read` | 10 | Yes | thread | Read-only | oracle |
| `thread_seek` | `_thread_seek` | 15 | Yes | thread | Read-only | oracle |
| `thread_list` | `_thread_list` | 10 | Yes | thread | Read-only | oracle |
| `web_search` | `_web_search` | 60 | Yes | web | External API call (DDG) | oracle, librarian |
| `web_fetch` | `_web_fetch` | 60 | Yes | web | External HTTP request | oracle, librarian |
| `delegate_librarian` | `_delegate_librarian` | 1200 | Yes | meta | Spawns subagent, may write vault | oracle |
| `notify_self` | `_notify_self` | 5 | Yes | meta | Emits ANS event, may persist | oracle |
| `github_read` | `_github_read` | 30 | Yes | github | External API call | oracle, librarian |
| `github_search` | `_github_search` | 45 | Yes | github | External API call | oracle |
| `deep_research` | `_deep_research` | 1800 | Yes | research | Spawns researchers, writes vault | oracle |

---

## 8. Leaf Node Specifications

### Blackboard Contract Per Tool

Each tool as a leaf node needs specific blackboard keys:

#### Code Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `search_code` | `query`, `limit?`, `language?`, `user_id` | `code_search_results` | `results` array non-empty |
| `find_definition` | `symbol`, `scope?`, `kind?`, `user_id` | `definition_result` | Has `path` and `line` |
| `find_references` | `symbol`, `limit?`, `user_id` | `references_result` | `references` array exists |
| `get_repo_map` | `scope?`, `max_tokens?`, `user_id` | `repo_map` | `map_text` non-empty |

#### Vault Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `vault_read` | `path`, `project_id?`, `user_id` | `note_content`, `note_metadata` | `content` exists |
| `vault_write` | `path`, `body`, `title?`, `project_id?`, `user_id` | `write_status` | `status == "ok"` |
| `vault_search` | `query`, `limit?`, `project_id?`, `user_id` | `vault_search_results` | `results` array exists |
| `vault_list` | `folder?`, `project_id?`, `user_id` | `vault_notes_list` | `notes` array exists |

#### Thread Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `thread_push` | `thread_id`, `content`, `entry_type?`, `user_id` | `push_result` | `status == "ok"` |
| `thread_read` | `thread_id`, `limit?`, `user_id` | `thread_entries` | No `error` key |
| `thread_seek` | `query`, `limit?`, `project_id?`, `user_id` | `thread_search_results` | `results` array exists |
| `thread_list` | `status?`, `project_id?`, `user_id` | `threads_list` | `threads` array exists |

#### Web Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `web_search` | `query`, `limit?`, `user_id` | `web_search_results` | `results` array exists |
| `web_fetch` | `url`, `max_tokens?`, `user_id` | `fetched_content` | `content` non-empty |

#### Meta Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `delegate_librarian` | `task`, `task_type?`, `files?`, `folder?`, `content_items?`, `user_id` | `librarian_result` | `success == True` |
| `notify_self` | `message`, `priority?`, `category?`, `deliver_at?`, `user_id` | `notification_status` | `status == "ok"` |

#### GitHub Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `github_read` | `repo`, `path`, `branch?`, `user_id` | `github_file_content` | `success == True` |
| `github_search` | `query`, `repo?`, `language?`, `limit?`, `user_id` | `github_search_results` | `results` array exists |

#### Research Tools

| Tool | Reads | Writes | Success Condition |
|------|-------|--------|-------------------|
| `deep_research` | `query`, `depth?`, `save_to_vault?`, `output_folder?`, `user_id` | `research_result` | `success == True` |

---

## 9. Failure Conditions

### Exception Categories

The executor categorizes exceptions via `_categorize_error()`:

| Category | Exception Types | Example |
|----------|-----------------|---------|
| `configuration_error` | NameError, AttributeError, ImportError | Missing module |
| `timeout_error` | TimeoutError, asyncio.TimeoutError | Exceeded limit |
| `network_error` | ConnectionError | API unreachable |
| `resource_error` | FileNotFoundError, OSError | File not found |
| `user_input_error` | ValueError | Invalid arguments |
| `api_error` | HTTPError, HTTPStatusError | Bad API response |
| `runtime_error` | (default) | Unexpected errors |

### Explicit Exception Handling

```python
except asyncio.TimeoutError:
    # Emit ANS event, return timeout error JSON
except FileNotFoundError as e:
    # Return file_error JSON
except ValueError as e:
    # Return user_input_error JSON
except (NameError, AttributeError) as e:
    # Return configuration_error JSON
except PermissionError as e:
    # Return permission_error JSON
except TimeoutError as e:
    # Return timeout_error JSON (non-asyncio)
except Exception as e:
    # Catch-all, categorize and return
```

---

## 10. Leaf Node BT Wrapper Pattern

### Proposed Interface

```python
class ToolLeafNode(LeafNode):
    """BT leaf node that wraps a ToolExecutor tool."""

    def __init__(
        self,
        name: str,
        tool_name: str,
        argument_mapping: Dict[str, str],  # blackboard_key -> tool_arg
        result_key: str,                   # where to write result
        timeout_override: Optional[float] = None,
    ):
        super().__init__(name)
        self.tool_name = tool_name
        self.argument_mapping = argument_mapping
        self.result_key = result_key
        self.timeout_override = timeout_override

    async def tick(self, blackboard: Blackboard) -> NodeStatus:
        # Get tool executor from blackboard
        executor: ToolExecutor = blackboard.get("tool_executor")
        user_id = blackboard.get("user_id")

        # Build arguments from blackboard
        arguments = {}
        for bb_key, tool_arg in self.argument_mapping.items():
            if bb_key in blackboard:
                arguments[tool_arg] = blackboard.get(bb_key)

        try:
            # Execute tool
            result_json = await executor.execute(
                self.tool_name,
                arguments,
                user_id,
                timeout=self.timeout_override,
            )
            result = json.loads(result_json)

            # Write result to blackboard
            blackboard.set(self.result_key, result)

            # Determine success
            if "error" in result:
                return NodeStatus.FAILURE
            return NodeStatus.SUCCESS

        except Exception as e:
            blackboard.set(f"{self.result_key}_error", str(e))
            return NodeStatus.FAILURE
```

---

## 11. E2E Test Scenarios

### Single Tool Execution

```python
async def test_single_tool_execution():
    """Test: vault_read returns note content."""
    executor = ToolExecutor()
    result = await executor.execute(
        "vault_read",
        {"path": "test/note.md"},
        user_id="test-user"
    )
    data = json.loads(result)
    assert "content" in data
    assert "error" not in data
```

### Batch Tool Execution

```python
async def test_batch_execution():
    """Test: Multiple tools execute in parallel."""
    executor = ToolExecutor()
    results = await executor.execute_batch(
        [
            {"name": "vault_list", "arguments": {}, "id": "list"},
            {"name": "thread_list", "arguments": {}, "id": "threads"},
        ],
        user_id="test-user",
        include_call_ids=True,
    )
    assert len(results) == 2
    assert results[0][0] == "list"
    assert results[1][0] == "threads"
```

### Tool Timeout

```python
async def test_tool_timeout():
    """Test: Timeout triggers ANS event."""
    executor = ToolExecutor(default_timeout=0.001)  # Very short
    result = await executor.execute(
        "web_search",
        {"query": "test"},
        user_id="test-user"
    )
    data = json.loads(result)
    assert data["timed_out"] == True
    assert "timeout" in data["error"].lower()
```

### Tool Failure

```python
async def test_tool_failure():
    """Test: Invalid path returns error with category."""
    executor = ToolExecutor()
    result = await executor.execute(
        "vault_read",
        {"path": "nonexistent/path.md"},
        user_id="test-user"
    )
    data = json.loads(result)
    assert "error" in data
    assert data.get("category") in ["resource_error", "file_error"]
```

### ANS Event Emission

```python
async def test_ans_event_on_timeout():
    """Test: Timeout emits tool.call.timeout event."""
    from backend.src.services.ans.bus import get_event_bus

    bus = get_event_bus()
    events_received = []

    def handler(event):
        events_received.append(event)

    bus.subscribe("tool.call.timeout", handler)

    executor = ToolExecutor(default_timeout=0.001)
    await executor.execute("web_search", {"query": "test"}, "test-user")

    assert len(events_received) == 1
    assert events_received[0].payload["tool_name"] == "web_search"
```

### Staleness Detection

```python
async def test_staleness_detection():
    """Test: Re-reading modified file emits source.stale event."""
    from backend.src.services.ans.bus import get_event_bus

    bus = get_event_bus()
    stale_events = []
    bus.subscribe("source.stale", lambda e: stale_events.append(e))

    executor = ToolExecutor()

    # First read
    await executor.execute("vault_read", {"path": "test.md"}, "user")

    # Modify file externally...

    # Second read
    await executor.execute("vault_read", {"path": "test.md"}, "user")

    assert len(stale_events) == 1
    assert stale_events[0].payload["path"] == "test.md"
```

---

## 12. BT Integration Recommendations

### 1. Add Missing ANS Events

```python
# In execute(), emit success/failure events:
get_event_bus().emit(Event(
    type=EventType.TOOL_CALL_SUCCESS,
    source="tool_executor",
    severity=Severity.DEBUG,
    payload={"tool_name": name, "duration_ms": duration}
))
```

### 2. Create Tool Node Factory

```python
def create_tool_leaf(tool_name: str, **config) -> ToolLeafNode:
    """Factory for creating BT leaf nodes from tool definitions."""
    schema = get_tool_schema(tool_name)
    return ToolLeafNode(
        name=f"Tool:{tool_name}",
        tool_name=tool_name,
        argument_mapping=config.get("args", {}),
        result_key=config.get("result_key", f"{tool_name}_result"),
    )
```

### 3. Blackboard Service Injection

```python
# At BT tree initialization:
blackboard.set("tool_executor", get_tool_executor())
blackboard.set("vault_service", VaultService())
blackboard.set("thread_service", ThreadService(db))
# etc.
```

### 4. Result Standardization

Consider standardizing tool results for easier BT condition checking:

```python
@dataclass
class ToolResult:
    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    category: Optional[str]
    duration_ms: float
```

---

## 13. Summary

The `ToolExecutor` is well-architected for BT integration:

**Strengths:**
- Clean registry pattern with uniform async interface
- Robust timeout handling with per-tool configurability
- Built-in batch execution for parallel tools
- ANS integration for timeout events
- Comprehensive exception categorization

**Gaps to Address:**
- Missing `tool.call.success` and `tool.call.failure` ANS events
- No built-in result standardization
- Two tools not implemented (`vault_move`, `vault_create_index`)
- Service dependencies require explicit injection for BT context

**Recommended Next Steps:**
1. Define `ToolLeafNode` base class
2. Create factory for tool schema to leaf node conversion
3. Add missing ANS events
4. Build blackboard adapter for service injection
5. Write integration tests for BT leaf nodes

---

## Appendix: Timeout Configuration Reference

```python
TOOL_TIMEOUTS: Dict[str, float] = {
    "web_search": 60.0,
    "web_fetch": 60.0,
    "search_code": 30.0,
    "find_definition": 30.0,
    "find_references": 30.0,
    "get_repo_map": 45.0,
    "vault_read": 10.0,
    "vault_write": 10.0,
    "vault_search": 15.0,
    "vault_list": 10.0,
    "vault_move": 10.0,
    "vault_create_index": 20.0,
    "thread_push": 10.0,
    "thread_read": 10.0,
    "thread_seek": 15.0,
    "thread_list": 10.0,
    "delegate_librarian": 1200.0,  # 20 minutes
    "notify_self": 5.0,
    "github_read": 30.0,
    "github_search": 45.0,
    "deep_research": 1800.0,  # 30 minutes
}
DEFAULT_TIMEOUT: float = 30.0
```
