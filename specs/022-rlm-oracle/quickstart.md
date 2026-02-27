# Quickstart: RLM Oracle Development

**Branch**: `022-rlm-oracle` | **Date**: 2026-02-22

---

## What This Feature Does

Replaces the Behavior Tree Oracle with a REPL-centric inference harness. The LLM is given a Python environment where the entire project lives as variables (`project`, `sub_oracle`, `Final`). It writes code to explore and synthesize answers programmatically rather than relying on retrieval to pre-select context.

**Before (BT Oracle)**: Query classifier → prompt composer → BT XML signals → multi-turn loop
**After (RLM Oracle)**: REPL environment → LLM writes Python → `Final` variable terminates loop

---

## Prerequisites

```bash
# Install new dependency (RestrictedPython)
cd backend
uv pip install RestrictedPython

# Verify tree-sitter still works (used by ProjectContext)
uv run python -c "from vlt.core.coderag.parser import parse_file; print('OK')"
```

---

## Running the Backend

```bash
# Start the backend server (no change from current)
cd backend
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Testing the Oracle

### Via HTTP (non-streaming)
```bash
# Get a token first (local mode)
TOKEN=$(curl -s -X POST http://localhost:8000/api/tokens \
  -H "Content-Type: application/json" \
  -d '{"user_id": "local-dev"}' | jq -r .access_token)

# Query the oracle
curl -X POST http://localhost:8000/api/oracle \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the connection lifecycle flow from a vlt-mcp tool call to SQLite?",
    "project_id": "vlt-bridge"
  }'
```

### Via SSE (streaming with progress)
```bash
curl -N -X POST http://localhost:8000/api/oracle/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What files handle the oracle streaming logic?",
    "project_id": "vlt-bridge"
  }'
# Expect: progress events as REPL runs, then content events with the answer
```

### Via MCP tool (vlt-mcp)
```bash
# The vlt_oracle_query MCP tool is unchanged (FR-020)
# Call from any Claude Code session with vlt-mcp registered
```

---

## Running Tests

```bash
# Unit tests for new RLM services
cd backend
uv run pytest tests/unit/test_rlm_oracle.py -v
uv run pytest tests/unit/test_project_context.py -v
uv run pytest tests/unit/test_repl_executor.py -v

# All backend tests (verify nothing broken)
uv run pytest tests/ -v

# Verify BT oracle tests are removed (not just passing)
# Should show "no tests ran" or test file not found:
uv run pytest tests/unit/test_bt_oracle.py 2>&1 | grep "no tests"
```

---

## Key Files After Implementation

```
backend/src/
├── services/
│   ├── rlm_oracle.py          # RLMOracleWrapper (replaces bt/wrappers/oracle_wrapper.py)
│   ├── project_context.py     # ProjectContext + TextHandle + FileManifest
│   ├── repl_executor.py       # REPLExecutor + REPLNamespace (RestrictedPython)
│   ├── openrouter_client.py   # Moved from bt/services/openrouter_client.py
│   └── oracle_bridge.py       # UNCHANGED (conversation history)
├── api/routes/oracle.py       # UPDATED: swap OracleBTWrapper → RLMOracleWrapper
└── models/oracle.py           # UNCHANGED (OracleStreamChunk, OracleRequest, etc.)

# DELETED:
# backend/src/bt/               (entire directory)
# backend/src/models/signals.py
# backend/src/services/signal_parser.py
# backend/src/services/query_classifier.py
# backend/src/services/prompt_composer.py
```

---

## Verifying Success Criteria During Development

**SC-002** — Root context < 4000 tokens:
```python
# Add to rlm_oracle.py logging:
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
token_count = len(enc.encode(system_prompt))
assert token_count < 4000, f"System prompt too large: {token_count} tokens"
```

**SC-003** — Focused queries < 20 seconds:
```bash
time curl -X POST http://localhost:8000/api/oracle \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "What does vlt_code_lookup return?", "project_id": "vlt-bridge"}'
```

**SC-007** — SSE progress events before final answer:
```bash
# Watch the SSE stream; you should see type="progress" events before type="content"
curl -N ... | grep -E '"type":"(progress|content)"'
```

**SC-008** — BT completely removed:
```bash
# Should return nothing:
grep -r "from.*bt\." backend/src/ --include="*.py" | grep -v test
grep -r "import.*bt\." backend/src/ --include="*.py" | grep -v test
```

---

## Troubleshooting

**RestrictedPython blocks a needed stdlib function**:
Check `ALLOWED_MODULES` in `repl_executor.py`. Add the module to the allowlist if it's safe (matches FR-015 criteria: no filesystem, no process execution).

**Sub-oracle recursion depth error**:
`RecursionDepthExceeded` means LLM tried to call `sub_oracle()` from within a depth-2 session. The prompt guardrail should prevent this — check if the system prompt's recursion constraint section is loading correctly.

**REPL timeout (30s)**:
The LLM wrote an infinite loop or a very slow operation. The iteration result will include `error: "Execution timeout after 30s"`. The LLM sees this in the next iteration and can adjust.

**ProjectContext has no files (empty manifest)**:
The vlt project record has no `project_path` set. Run `vlt coderag init --project <id>` to set the path and build an index. Oracle will still work for threads/notes.
