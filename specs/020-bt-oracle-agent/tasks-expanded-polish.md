# Expanded Tasks: Phase 8 - Polish & Cross-Cutting Concerns

**Feature**: 020-bt-oracle-agent
**Phase**: 8 (Polish)
**Tasks**: T050-T056

---

## T050: Update shadow_mode.py to Compare Signal Emission

**Goal**: Extend `ShadowModeRunner._compare_chunks()` to detect and track signal emission differences between legacy and BT implementations.

### Function Signature

```python
# In ShadowModeRunner class (backend/src/bt/wrappers/shadow_mode.py)

def _extract_signals_from_chunks(
    self,
    chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Extract signals from accumulated content in chunk stream."""

def _compare_signals(
    self,
    bt_signals: List[Dict[str, Any]],
    legacy_signals: List[Dict[str, Any]],
) -> List[ChunkDiscrepancy]:
    """Compare signal emissions between implementations."""
```

### Core Algorithm

1. Extract final content from both chunk lists (from `done` chunk)
2. Parse signals from each content using `signal_parser.parse_signal()`
3. Compare:
   - Signal presence (BT emits but legacy doesn't, or vice versa)
   - Signal type mismatch
   - Confidence score difference > 0.2
   - Key field differences (reason, sources_found, etc.)
4. Return list of `ChunkDiscrepancy` with `severity="warning"` for mismatches

### Key Code Snippet

```python
def _extract_signals_from_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract signals from chunk stream content."""
    from ...services.signal_parser import parse_signal

    signals = []
    for chunk in chunks:
        content = chunk.get("content", "") or chunk.get("accumulated_content", "")
        if content:
            signal = parse_signal(content)
            if signal:
                signals.append(signal.model_dump() if hasattr(signal, 'model_dump') else signal)
    return signals

def _compare_signals(
    self,
    bt_signals: List[Dict[str, Any]],
    legacy_signals: List[Dict[str, Any]],
) -> List[ChunkDiscrepancy]:
    """Compare signal emissions between BT and legacy."""
    discrepancies = []

    # Compare signal counts
    if len(bt_signals) != len(legacy_signals):
        discrepancies.append(ChunkDiscrepancy(
            field="signal_count",
            bt_value=len(bt_signals),
            legacy_value=len(legacy_signals),
            severity="warning",
        ))

    # Compare signal types in order
    bt_types = [s.get("type") for s in bt_signals]
    legacy_types = [s.get("type") for s in legacy_signals]

    if bt_types != legacy_types:
        discrepancies.append(ChunkDiscrepancy(
            field="signal_types",
            bt_value=bt_types,
            legacy_value=legacy_types,
            severity="error" if set(bt_types) != set(legacy_types) else "warning",
        ))

    # Compare individual signals
    for i, (bt_sig, leg_sig) in enumerate(zip(bt_signals, legacy_signals)):
        if bt_sig.get("type") != leg_sig.get("type"):
            discrepancies.append(ChunkDiscrepancy(
                field=f"signal[{i}].type",
                bt_value=bt_sig.get("type"),
                legacy_value=leg_sig.get("type"),
                index=i,
                severity="error",
            ))

        # Compare confidence if both have same type
        bt_conf = bt_sig.get("confidence", 0)
        leg_conf = leg_sig.get("confidence", 0)
        if abs(bt_conf - leg_conf) > 0.2:
            discrepancies.append(ChunkDiscrepancy(
                field=f"signal[{i}].confidence",
                bt_value=bt_conf,
                legacy_value=leg_conf,
                index=i,
                severity="info",
            ))

    return discrepancies
```

### Verification Checklist

- [ ] Import `parse_signal` from `services.signal_parser`
- [ ] Add `_extract_signals_from_chunks()` method
- [ ] Add `_compare_signals()` method
- [ ] Call `_compare_signals()` from `_generate_report()`
- [ ] Unit test: Both emit same signal type -> no discrepancy
- [ ] Unit test: BT emits signal, legacy doesn't -> discrepancy logged
- [ ] Unit test: Different signal types -> severity=error
- [ ] Integration test: Run shadow mode with signal-producing query

---

## T051: Add Signal Discrepancy Tracking to Comparison Report

**Goal**: Extend `ComparisonReport` dataclass and `to_dict()` to include signal-specific metrics.

### Function Signature

```python
# In ComparisonReport dataclass (backend/src/bt/wrappers/shadow_mode.py)

@dataclass
class ComparisonReport:
    # ... existing fields ...

    # New fields for signal comparison
    bt_signals: List[Dict[str, Any]] = field(default_factory=list)
    legacy_signals: List[Dict[str, Any]] = field(default_factory=list)
    signal_discrepancies: List[ChunkDiscrepancy] = field(default_factory=list)
    signal_match: bool = True
```

### Core Algorithm

1. Add new fields to `ComparisonReport` dataclass
2. Update `to_dict()` to serialize signal data
3. Update `_generate_report()` to populate signal fields
4. Add signal section to JSON log output
5. Update `_log_discrepancies()` to highlight signal mismatches

### Key Code Snippet

```python
@dataclass
class ComparisonReport:
    """Full comparison report between BT and legacy execution."""

    timestamp: datetime
    user_id: str
    query_preview: str

    bt_chunks: List[Dict[str, Any]]
    legacy_chunks: List[Dict[str, Any]]

    discrepancies: List[ChunkDiscrepancy] = field(default_factory=list)
    match_rate: float = 0.0

    bt_duration_ms: float = 0.0
    legacy_duration_ms: float = 0.0

    bt_error: Optional[str] = None
    legacy_error: Optional[str] = None

    # Signal comparison fields
    bt_signals: List[Dict[str, Any]] = field(default_factory=list)
    legacy_signals: List[Dict[str, Any]] = field(default_factory=list)
    signal_discrepancies: List[ChunkDiscrepancy] = field(default_factory=list)
    signal_match: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            # ... existing fields ...
            "bt_signals": self.bt_signals,
            "legacy_signals": self.legacy_signals,
            "signal_discrepancy_count": len(self.signal_discrepancies),
            "signal_discrepancies": [
                {
                    "field": d.field,
                    "bt_value": str(d.bt_value)[:100],
                    "legacy_value": str(d.legacy_value)[:100],
                    "index": d.index,
                    "severity": d.severity,
                }
                for d in self.signal_discrepancies
            ],
            "signal_match": self.signal_match,
        }
```

### Verification Checklist

- [ ] Add `bt_signals`, `legacy_signals` fields to dataclass
- [ ] Add `signal_discrepancies` field to dataclass
- [ ] Add `signal_match` boolean field
- [ ] Update `to_dict()` to include signal fields
- [ ] Update `_generate_report()` to set `signal_match = len(signal_discrepancies) == 0`
- [ ] Verify JSON log files include signal section
- [ ] Test serialization of signal data (handles None, empty lists)

---

## T052: Update config.py with ORACLE_USE_BT Documentation

**Goal**: Add ORACLE_USE_BT and related BT Oracle configuration to `AppConfig` with proper documentation.

### Function Signature

```python
# In AppConfig class (backend/src/services/config.py)

class AppConfig(BaseModel):
    # ... existing fields ...

    oracle_use_bt: str = Field(
        default="false",
        description="Oracle mode: 'false' (legacy), 'true' (BT-only), 'shadow' (parallel comparison)"
    )
    oracle_max_turns: int = Field(
        default=30,
        description="Maximum turns per Oracle query before forced completion"
    )
    oracle_prompt_budget: int = Field(
        default=8000,
        description="Maximum tokens for composed Oracle system prompt"
    )
```

### Core Algorithm

1. Add three new fields to `AppConfig` dataclass
2. Add validators for `oracle_use_bt` (must be "false", "true", or "shadow")
3. Add environment variable reading in `get_config()`
4. Add comments documenting the migration path

### Key Code Snippet

```python
# Add to AppConfig class
oracle_use_bt: str = Field(
    default="false",
    description=(
        "Oracle execution mode. Options:\n"
        "  'false' - Use legacy OracleAgent (default)\n"
        "  'true'  - Use BT-controlled Oracle exclusively\n"
        "  'shadow' - Run both in parallel, compare outputs, yield legacy"
    )
)
oracle_max_turns: int = Field(
    default=30,
    description="Maximum agent loop iterations before forced completion (safety limit)"
)
oracle_prompt_budget: int = Field(
    default=8000,
    description="Token budget for composed system prompt (excludes tool schemas)"
)

@field_validator("oracle_use_bt", mode="before")
@classmethod
def _validate_oracle_mode(cls, value: str) -> str:
    allowed = {"false", "true", "shadow"}
    v = value.lower().strip()
    if v not in allowed:
        raise ValueError(f"ORACLE_USE_BT must be one of {allowed}, got: {value!r}")
    return v

# Add to get_config() function
oracle_use_bt = _read_env("ORACLE_USE_BT", "false")
oracle_max_turns = int(_read_env("ORACLE_MAX_TURNS", "30"))
oracle_prompt_budget = int(_read_env("ORACLE_PROMPT_BUDGET", "8000"))
```

### Verification Checklist

- [ ] Add `oracle_use_bt` field with default "false"
- [ ] Add `oracle_max_turns` field with default 30
- [ ] Add `oracle_prompt_budget` field with default 8000
- [ ] Add `@field_validator` for `oracle_use_bt` (validates allowed values)
- [ ] Update `get_config()` to read from environment
- [ ] Pass new fields to `AppConfig()` constructor
- [ ] Test: invalid ORACLE_USE_BT raises ValueError
- [ ] Test: defaults work when env vars not set
- [ ] Update `.env.example` with new variables

---

## T053: Update CLAUDE.md with 020-bt-oracle-agent Technology Additions

**Goal**: Document the BT Oracle feature in the project CLAUDE.md for future AI agents.

### Core Steps

1. Add entry to "Recent Changes" section
2. Add entry to "Active Technologies" section
3. Add BT Oracle configuration to Environment Configuration section
4. Briefly document shadow mode workflow

### Key Content to Add

```markdown
## Recent Changes
- 020-bt-oracle-agent: Added Python 3.11+ (backend), TypeScript 5.x (frontend - no changes) + FastAPI, Pydantic, lupa (Lua), existing BT runtime (019)
...

## Active Technologies
- Python 3.11+ (backend), TypeScript 5.x (frontend - no changes) + FastAPI, Pydantic, lupa (Lua), existing BT runtime (019) (020-bt-oracle-agent)
- SQLite (existing index.db for state persistence) (020-bt-oracle-agent)

## Environment Configuration
...
**BT Oracle variables** (for controlled rollout):
- ORACLE_USE_BT: Execution mode (false/true/shadow)
- ORACLE_MAX_TURNS: Max iterations (default 30)
- ORACLE_PROMPT_BUDGET: Token limit for system prompt (default 8000)

## BT Oracle Shadow Mode

Enable shadow mode to compare legacy and BT implementations:

```bash
export ORACLE_USE_BT=shadow
# Run queries, check data/shadow_logs/*.json for discrepancy reports
```

Switch to BT-only when shadow mode shows >95% match rate:
```bash
export ORACLE_USE_BT=true
```
```

### Verification Checklist

- [ ] Add entry to "Recent Changes" section
- [ ] Add entry to "Active Technologies" section
- [ ] Add BT Oracle env vars to Environment Configuration
- [ ] Add brief Shadow Mode section with usage instructions
- [ ] Verify markdown formatting is correct
- [ ] Run `grep -c "020-bt-oracle-agent" CLAUDE.md` returns >= 2

---

## T054: Full Test Suite Command

**Goal**: Run all tests related to BT Oracle and verify they pass.

### Command

```bash
cd backend && uv run pytest \
    tests/unit/bt/ \
    tests/unit/test_signal_parser.py \
    tests/unit/test_query_classifier.py \
    tests/unit/test_prompt_composer.py \
    tests/integration/test_oracle_bt_integration.py \
    -v --tb=short
```

### Core Algorithm

1. Activate backend environment
2. Run pytest with all BT-related test modules
3. Verify all tests pass (exit code 0)
4. Document any failures with file:line references
5. If failures, fix before proceeding

### Expected Test Coverage

| Module | Tests | Description |
|--------|-------|-------------|
| `tests/unit/bt/` | ~20-30 | BT conditions, actions, tree execution |
| `test_signal_parser.py` | ~10 | Signal XML parsing, edge cases |
| `test_query_classifier.py` | ~8 | Query type classification |
| `test_prompt_composer.py` | ~8 | Segment composition logic |
| `test_oracle_bt_integration.py` | ~5 | End-to-end BT Oracle flow |

### Verification Checklist

- [ ] All unit tests pass (exit code 0)
- [ ] All integration tests pass
- [ ] No skipped tests without documented reason
- [ ] Coverage report shows >80% for new code (if coverage enabled)
- [ ] No deprecation warnings from BT code
- [ ] Test output captured and reviewed

---

## T055: Run quickstart.md Validation

**Goal**: Follow quickstart.md step-by-step and verify shadow mode works.

### Core Steps

1. **Set environment**:
   ```bash
   export ORACLE_USE_BT=shadow
   ```

2. **Copy prompt templates** (if not done):
   ```bash
   cp -r specs/020-bt-oracle-agent/prompts/oracle/* backend/src/prompts/oracle/
   ```

3. **Run unit tests**:
   ```bash
   cd backend
   uv run pytest tests/unit/test_signal_parser.py -v
   uv run pytest tests/unit/test_query_classifier.py -v
   ```

4. **Start dev server**:
   ```bash
   cd backend && uv run uvicorn src.api.main:app --reload --port 8000
   ```

5. **Send test query**:
   ```bash
   curl -X POST http://localhost:8000/api/oracle/query \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer local-dev-token" \
     -d '{"query": "What is the weather in Paris?", "project_id": "test"}'
   ```

6. **Check shadow logs**:
   ```bash
   ls -la data/shadow_logs/
   cat data/shadow_logs/shadow_*.json | jq '.discrepancies'
   ```

### Verification Checklist

- [ ] Environment variable set correctly
- [ ] Prompt templates exist in `backend/src/prompts/oracle/`
- [ ] Signal parser tests pass
- [ ] Query classifier tests pass
- [ ] Server starts without errors
- [ ] Test query returns valid response
- [ ] Shadow log file created in `data/shadow_logs/`
- [ ] JSON log contains `bt_signals`, `legacy_signals` fields
- [ ] `match_rate` field present in log

---

## T056: Manual E2E Test with Shadow Mode (10 Diverse Queries)

**Goal**: Execute 10 diverse queries in shadow mode and analyze comparison logs.

### Test Query Set

| # | Query | Expected Type | Expected Signal |
|---|-------|---------------|-----------------|
| 1 | "What is the weather in Paris?" | research | context_sufficient |
| 2 | "How does the auth middleware work?" | code | need_turn / context_sufficient |
| 3 | "Thanks, that helps!" | conversational | (none) |
| 4 | "What did we decide about caching?" | documentation | context_sufficient |
| 5 | "Create a note about testing" | action | need_turn |
| 6 | "Compare React vs Vue for our frontend" | research | need_turn |
| 7 | "Where is the login function defined?" | code | context_sufficient |
| 8 | "What's the latest on TypeScript 5.5?" | research | context_sufficient |
| 9 | "Explain how signals work in this project" | code/documentation | context_sufficient |
| 10 | "Fix the bug in user settings" | action | need_turn / stuck |

### Core Algorithm

1. Start server with `ORACLE_USE_BT=shadow`
2. Execute each query via curl or UI
3. Wait for completion, note response quality
4. After all queries, analyze shadow logs:
   ```bash
   for f in data/shadow_logs/shadow_*.json; do
     echo "=== $f ==="
     jq '{query: .query_preview, match_rate: .match_rate, signal_match: .signal_match, discrepancy_count: .discrepancy_count}' "$f"
   done
   ```
5. Calculate aggregate metrics:
   - Average match rate
   - Signal match rate
   - Common discrepancy patterns

### Analysis Template

```markdown
## Shadow Mode E2E Test Results

**Date**: YYYY-MM-DD
**Queries**: 10
**Environment**: ORACLE_USE_BT=shadow

### Summary Metrics

| Metric | Value |
|--------|-------|
| Average Match Rate | X.XX% |
| Signal Match Rate | X/10 |
| Queries with Discrepancies | X/10 |

### Per-Query Results

| Query # | Match Rate | Signal Match | Discrepancies |
|---------|------------|--------------|---------------|
| 1 | 98% | Yes | 0 |
| 2 | 95% | No | 1 (signal_type) |
| ... | ... | ... | ... |

### Common Discrepancy Patterns

1. **Pattern A**: Description
   - Occurrences: X
   - Severity: warning/error
   - Root cause: ...

### Recommendation

[ ] Ready for ORACLE_USE_BT=true
[ ] Need fixes before switching (list issues)
```

### Verification Checklist

- [ ] All 10 queries executed
- [ ] Shadow log created for each query
- [ ] Logs contain both BT and legacy chunks
- [ ] Signal comparison data present
- [ ] Analysis template completed
- [ ] Match rate >= 90% for non-signal discrepancies
- [ ] Signal match rate >= 70%
- [ ] No error-severity discrepancies in signal types
- [ ] Results documented in `Ai-notes/` or spec directory

---

## Summary

| Task | Type | Est. Effort | Dependencies |
|------|------|-------------|--------------|
| T050 | Code | 1-2 hours | signal_parser.py exists |
| T051 | Code | 30 min | T050 |
| T052 | Config | 30 min | None |
| T053 | Docs | 20 min | None |
| T054 | Test | 10 min | All code tasks |
| T055 | Validation | 30 min | Server running |
| T056 | E2E Test | 1 hour | T050, T051, server |

**Parallel Opportunities**:
- T052 and T053 can run in parallel (different files)
- T054 must wait for code tasks
- T055 and T056 are sequential (need server)
