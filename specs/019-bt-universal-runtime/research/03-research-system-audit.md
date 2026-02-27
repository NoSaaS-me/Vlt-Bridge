# Research System Audit for BT Subtree Conversion

## Executive Summary

The Deep Research system (~1,985 lines) implements a custom "behavior-like" pattern with 7 sequential behaviors orchestrated by `ResearchOrchestrator`. While structurally similar to behavior trees (sequential execution, state passing), it lacks true BT composability:

- **No composite nodes**: Pure linear sequence, no selectors/fallbacks
- **No tick semantics**: Each behavior runs to completion (blocking)
- **No parallel policy control**: `ParallelResearchersBehavior` uses raw asyncio, not BT parallel semantics
- **No interruption/memory**: State is ephemeral within a single research execution

Converting to a BT subtree will provide proper composability, stuck detection, and integration with the Oracle agent's event-driven tick loop.

---

## 1. Current Behavior Pattern Analysis

### Base Class: `ResearchBehavior`

```python
# backend/src/services/research/behaviors.py:48-68
class ResearchBehavior(ABC):
    @abstractmethod
    async def run(
        self,
        state: ResearchState,
    ) -> ResearchState:
        """Execute the behavior and update state."""
        pass

    def get_progress_message(self) -> str:
        return self.__class__.__name__
```

**Key Observations:**
- Abstract base with single `run()` method returning updated `ResearchState`
- No `tick()` semantics - behaviors run to completion
- No `SUCCESS`/`FAILURE`/`RUNNING` return status
- Progress messages are static strings, not dynamic

### Comparison: ResearchBehavior vs BT Node

| Aspect | ResearchBehavior | BT Leaf Node |
|--------|------------------|--------------|
| Execution | `run()` - async, runs to completion | `tick()` - returns status, may return RUNNING |
| State | Mutable `ResearchState` passed through | Blackboard with scoped access |
| Progress | Static string message | Can emit progress via blackboard/callbacks |
| Failure | Exception or `status=FAILED` on state | Returns `FAILURE` status |
| Composition | Sequential only (orchestrator hardcoded) | Composable via composites |
| Cancellation | Not supported | Interruptible via tick context |

---

## 2. Orchestrator Workflow Analysis

### Exact Behavior Sequence

From `backend/src/services/research/orchestrator.py:176-273`:

```
1. GenerateBriefBehavior      -> brief + status=PLANNING
2. PlanSubtopicsBehavior      -> researchers[] + status=RESEARCHING
3. ParallelResearchersBehavior -> sources[] + total_searches
4. CompressFindingsBehavior   -> compressed_findings + status=COMPRESSING
5. GenerateReportBehavior     -> report + status=GENERATING
6. PersistToVaultBehavior     -> vault_folder (optional)
7. -> status=COMPLETED
```

### Progress Streaming Implementation

The `run_research_streaming()` method (lines 275-477) yields `ResearchProgress` objects at fixed points:

```python
# Progress percentages (hardcoded):
# 0%   - initializing
# 5%   - brief generation started
# 10%  - brief complete
# 15%  - planning complete
# 20%  - parallel research started
# 50%  - parallel research complete
# 60%  - compression started
# 70%  - compression complete
# 75%  - report generation started
# 90%  - report complete
# 95%  - vault persistence started
# 100% - completed
```

**Problem**: Progress is manually interpolated, not derived from actual node completion. A true BT would derive progress from node depth/completion.

---

## 3. Parallel Execution Analysis

### ParallelResearchersBehavior (lines 433-531)

```python
async def run(self, state: ResearchState) -> ResearchState:
    tasks = [
        self.researcher.run_single(state, researcher)
        for researcher in state.researchers
        if not researcher.completed
    ]

    semaphore = asyncio.Semaphore(self.max_concurrent)

    async def run_with_limit(task):
        async with semaphore:
            return await task

    results = await asyncio.gather(
        *[run_with_limit(t) for t in tasks],
        return_exceptions=True,  # Continue on failure
    )
```

**Current Semantics:**
- Uses `asyncio.gather` with `return_exceptions=True`
- Failed researchers don't abort siblings
- No configurable policy (hardcoded REQUIRE_NONE effectively)
- Semaphore limits concurrency but doesn't respect BT parallel policies

### BT Parallel Node Mapping

| Current Behavior | BT Equivalent |
|------------------|---------------|
| `return_exceptions=True` | `:policy :continue-on-failure` |
| Semaphore limit | `:max-concurrent N` |
| No cancellation | Needs `:on-child-fail :cancel-siblings` option |
| No success threshold | Needs `:policy :require-one` option |

---

## 4. State Machine Analysis

### ResearchStatus Transitions

From `backend/src/models/research.py:11-19`:

```
PLANNING -> RESEARCHING -> COMPRESSING -> GENERATING -> COMPLETED
    |            |              |              |
    v            v              v              v
  FAILED      FAILED         FAILED         FAILED
```

**Status Values:**
```python
class ResearchStatus(str, Enum):
    PLANNING = "planning"       # Brief generation
    RESEARCHING = "researching" # Parallel research
    COMPRESSING = "compressing" # Finding synthesis
    GENERATING = "generating"   # Report generation
    COMPLETED = "completed"     # Success
    FAILED = "failed"           # Any error
```

### State Persistence

- `ResearchState` is a dataclass, not persisted during execution
- Only final result persisted to vault via `PersistToVaultBehavior`
- No checkpointing or resume capability
- Token/search counts accumulated but not persisted incrementally

---

## 5. LLM Service Comparison

### ResearchLLMService vs Oracle's LLM Usage

| Aspect | ResearchLLMService | Oracle Agent |
|--------|-------------------|--------------|
| Provider | OpenRouter or Gemini | OpenRouter + Anthropic direct |
| Streaming | Not supported | Full SSE streaming |
| Model selection | User settings (subagent model) | Per-call model selection |
| Error handling | Raise exception | Event-based (ANS) |
| Token tracking | Debug logging only | Budget tracking + ANS |
| Retry logic | None | Built-in retry |

### ResearchLLMService Methods

```python
# backend/src/services/research/llm_service.py
async def generate(prompt, model, provider, max_tokens, temperature, system_prompt) -> str
async def generate_json(prompt, ...) -> Dict[str, Any]  # Parses JSON from response
```

**Key Difference**: Research LLM service is a thin wrapper over HTTP calls. Oracle uses the full streaming infrastructure with chunk callbacks.

---

## 6. BT Subtree Design

### LISP Tree Structure

```lisp
(subtree "deep-research"
  :description "Multi-source research with parallel investigators"
  :blackboard-schema {
    :input {:query string :depth enum :config ResearchConfig}
    :artifacts {:brief ResearchBrief :researchers [] :sources [] :findings [] :report ResearchReport}
    :progress {:phase string :pct float :message string}
    :output {:vault_path string :research_id string}
  }

  (sequence
    ;; Phase 1: Generate research brief
    (action set-phase
      :fn "research.set_phase"
      :args {:phase "brief" :pct 5 :message "Generating research brief..."})

    (llm-call generate-brief
      :fn "research.generate_brief"
      :model [:input :config :planning_model]
      :prompt-template "research/brief.md"
      :output-key [:artifacts :brief]
      :budget [:input :config :brief_max_tokens]
      :timeout 60)

    (condition brief-valid?
      :fn "research.validate_brief"
      :on-failure (action create-fallback-brief :fn "research.fallback_brief"))

    ;; Phase 2: Plan subtopics
    (action plan-subtopics
      :fn "research.plan_subtopics"
      :output-key [:artifacts :researchers])

    ;; Phase 3: Parallel research
    (action set-phase
      :fn "research.set_phase"
      :args {:phase "researching" :pct 20 :message "Starting parallel research..."})

    (parallel research-parallel
      :policy :require-all
      :on-child-fail :continue
      :max-concurrent [:input :config :max_concurrent_researchers]
      :memory true

      (for-each [:artifacts :researchers]
        (subtree-ref "single-researcher"
          :bind {:subtopic [:current :subtopic]
                 :max_tool_calls [:current :max_tool_calls]})))

    ;; Phase 4: Compress findings
    (action set-phase
      :fn "research.set_phase"
      :args {:phase "compressing" :pct 60 :message "Synthesizing findings..."})

    (llm-call compress-findings
      :fn "research.compress_findings"
      :model [:input :config :compression_model]
      :prompt-template "research/compress.md"
      :input-keys [[:artifacts :brief] [:artifacts :sources]]
      :output-key [:artifacts :findings]
      :budget [:input :config :compression_max_tokens]
      :timeout 120)

    ;; Phase 5: Generate report
    (action set-phase
      :fn "research.set_phase"
      :args {:phase "generating" :pct 75 :message "Generating report..."})

    (llm-call generate-report
      :fn "research.generate_report"
      :model [:input :config :report_model]
      :prompt-template "research/report.md"
      :input-keys [[:artifacts :brief] [:artifacts :findings] [:artifacts :sources]]
      :output-key [:artifacts :report]
      :budget [:input :config :report_max_tokens]
      :timeout 180)

    ;; Phase 6: Persist to vault (conditional)
    (selector persist-selector
      (sequence
        (condition should-persist?
          :fn "research.should_persist")
        (action persist-to-vault
          :fn "research.persist_to_vault"
          :output-key [:output :vault_path]))
      (action skip-persist :fn "research.noop"))

    ;; Final phase
    (action set-phase
      :fn "research.set_phase"
      :args {:phase "completed" :pct 100 :message "Research completed"})))


;; Single researcher subtree
(subtree "single-researcher"
  :description "Research a single subtopic"
  :blackboard-schema {
    :subtopic string
    :max_tool_calls int
    :sources []
    :tool_calls_made int
  }

  (sequence
    ;; Generate search queries
    (action generate-queries
      :fn "research.generate_search_queries"
      :output-key [:queries])

    ;; Execute searches (with retry)
    (retry-decorator :max-attempts 3 :backoff-ms 1000
      (selector search-selector
        ;; Try Tavily first
        (sequence
          (condition tavily-available? :fn "research.has_tavily")
          (action search-tavily
            :fn "research.search_tavily"
            :input-keys [[:queries]]
            :output-key [:search_results]))

        ;; Fallback to OpenRouter Perplexity
        (sequence
          (condition openrouter-available? :fn "research.has_openrouter")
          (action search-openrouter
            :fn "research.search_openrouter"
            :input-keys [[:queries]]
            :output-key [:search_results]))

        ;; No search provider - use LLM knowledge
        (action search-llm-only
          :fn "research.search_llm_fallback"
          :output-key [:search_results])))

    ;; Convert results to sources
    (action convert-to-sources
      :fn "research.convert_search_results"
      :output-key [:sources])

    ;; Extract findings from sources
    (condition has-sources?
      :predicate (> (count [:sources]) 0))

    (llm-call extract-findings
      :fn "research.extract_findings"
      :model "research-model"
      :input-keys [[:subtopic] [:sources]]
      :output-key [:extracted]
      :budget 2000)))
```

---

## 7. Blackboard Keys Specification

### Input Keys (Set Before Subtree Invocation)

| Key | Type | Source | Description |
|-----|------|--------|-------------|
| `input.query` | string | User request | The research question |
| `input.depth` | ResearchDepth | User request | quick/standard/thorough |
| `input.config` | ResearchConfig | Computed | Config for depth level |
| `input.save_to_vault` | bool | User request | Whether to persist |
| `input.user_id` | string | Session | User identifier |
| `input.vault_path` | string | Session | Path to user vault |

### Artifact Keys (Generated During Execution)

| Key | Type | Producer | Consumer |
|-----|------|----------|----------|
| `artifacts.brief` | ResearchBrief | generate-brief | plan-subtopics, compress-findings, generate-report |
| `artifacts.researchers` | ResearcherState[] | plan-subtopics | research-parallel |
| `artifacts.sources` | ResearchSource[] | single-researcher (aggregated) | compress-findings, generate-report |
| `artifacts.findings` | ResearchFinding[] | compress-findings | generate-report |
| `artifacts.report` | ResearchReport | generate-report | persist-to-vault |

### Progress Keys (Updated Throughout)

| Key | Type | Description |
|-----|------|-------------|
| `progress.phase` | string | Current phase name |
| `progress.pct` | float | Progress percentage (0-100) |
| `progress.message` | string | Human-readable status |
| `progress.sources_found` | int | Running source count |
| `progress.current_subtopic` | string | Active researcher topic |

### Output Keys (Available After Completion)

| Key | Type | Description |
|-----|------|-------------|
| `output.research_id` | string | Unique research identifier |
| `output.vault_path` | string | Path to saved research folder |
| `output.status` | ResearchStatus | Final status |
| `output.error` | string | Error message if failed |

---

## 8. E2E Test Scenarios

### Test 1: Basic Research Flow (Quick Depth)

**Setup:**
```python
blackboard.set("input.query", "What is quantum computing?")
blackboard.set("input.depth", ResearchDepth.QUICK)
blackboard.set("input.config", ResearchConfig.for_depth(ResearchDepth.QUICK))
blackboard.set("input.save_to_vault", False)
```

**Expectations:**
- Tree completes with SUCCESS
- `artifacts.brief` contains refined question
- `artifacts.researchers` has 1 entry (QUICK = 1 researcher)
- `artifacts.sources` has 3-5 entries
- `artifacts.report` exists with title and summary
- Total execution < 60 seconds

**Verification:**
```python
assert tree.status == SUCCESS
assert blackboard.get("artifacts.brief").subtopics
assert len(blackboard.get("artifacts.researchers")) == 1
assert blackboard.get("artifacts.report").executive_summary
```

### Test 2: Parallel Researcher Execution

**Setup:**
```python
blackboard.set("input.depth", ResearchDepth.STANDARD)  # 3 researchers
```

**Expectations:**
- All 3 researchers run concurrently
- Semaphore respects max_concurrent limit
- Results aggregated in `artifacts.sources`
- Failed researcher doesn't abort siblings

**Verification:**
```python
assert len(blackboard.get("artifacts.researchers")) == 3
for r in blackboard.get("artifacts.researchers"):
    assert r.completed  # All should complete, even if failed
```

### Test 3: One Researcher Fails, Others Continue

**Setup:**
```python
# Mock search service to fail for subtopic "subtopic-2"
mock_search.fail_on = "subtopic-2"
```

**Expectations:**
- Tree completes with SUCCESS (policy: continue-on-failure)
- 2 of 3 researchers contribute sources
- Failed researcher logged with error
- Report generated with partial data

**Verification:**
```python
assert tree.status == SUCCESS
successful = [r for r in blackboard.get("artifacts.researchers") if not r.error]
assert len(successful) >= 2
assert blackboard.get("artifacts.report")  # Report still generated
```

### Test 4: Token Budget Exceeded Mid-Research

**Setup:**
```python
blackboard.set("input.config.report_max_tokens", 100)  # Artificially low
```

**Expectations:**
- Report generation LLM node returns FAILURE
- Error captured in blackboard
- ANS event emitted for budget exceeded
- Tree returns FAILURE with budget reason

**Verification:**
```python
assert tree.status == FAILURE
assert "budget" in blackboard.get("output.error").lower()
# ANS event check
assert any(e.type == "budget.token.exceeded" for e in event_bus.events)
```

### Test 5: Progress Streaming to Frontend

**Setup:**
```python
progress_updates = []
def on_progress(progress):
    progress_updates.append(progress)

tree.set_progress_callback(on_progress)
```

**Expectations:**
- Progress updates received at each phase transition
- Percentages increase monotonically
- Messages are human-readable
- Final progress is 100%

**Verification:**
```python
assert len(progress_updates) >= 7  # At least one per phase
percentages = [p.pct for p in progress_updates]
assert percentages == sorted(percentages)  # Monotonic
assert progress_updates[-1].pct == 100
assert progress_updates[-1].phase == "completed"
```

### Test 6: Vault Persistence

**Setup:**
```python
blackboard.set("input.save_to_vault", True)
blackboard.set("input.vault_path", "/tmp/test-vault")
```

**Expectations:**
- Folder created: `/tmp/test-vault/research/{research_id}/`
- Files created: `index.md`, `brief.md`, `report.md`, `sources.md`, `methodology.md`
- Source notes in `notes/` folder
- Wikilinks resolve correctly

**Verification:**
```python
vault_path = blackboard.get("output.vault_path")
assert (Path(vault_path) / "index.md").exists()
assert (Path(vault_path) / "report.md").exists()
assert len(list((Path(vault_path) / "notes").glob("*.md"))) > 0
```

---

## 9. Migration Considerations

### State Mapping

| ResearchState Field | Blackboard Key |
|---------------------|----------------|
| `research_id` | `output.research_id` |
| `user_id` | `input.user_id` |
| `request` | `input.*` (decomposed) |
| `status` | Derived from tree status |
| `brief` | `artifacts.brief` |
| `researchers` | `artifacts.researchers` |
| `all_sources` | `artifacts.sources` |
| `compressed_findings` | `artifacts.findings` |
| `report` | `artifacts.report` |
| `started_at` | `_tree_started_at` (metadata) |
| `completed_at` | `_tree_completed_at` (metadata) |
| `total_tokens` | Accumulated by LLM nodes |
| `total_searches` | Count of search actions |
| `vault_folder` | `output.vault_path` |

### Service Dependencies

| Current Service | BT Service Injection |
|-----------------|---------------------|
| `ResearchLLMService` | Shared LLM service via `ctx.services.llm` |
| `TavilySearchService` | `ctx.services.tavily` |
| `OpenRouterSearchService` | `ctx.services.openrouter_search` |
| `PromptLoader` | `ctx.services.prompts` |
| `ResearchVaultPersister` | Action leaf, receives vault_path from blackboard |

### Breaking Changes

1. **Progress API**: `ResearchProgress` model changes to derive from tree state
2. **Streaming**: Will use SSE integration via tree callbacks, not generator
3. **Error handling**: Failures return FAILURE status, not exceptions
4. **Cancellation**: New capability - can cancel mid-execution

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM call integration complexity | High | Medium | Reuse Oracle's LLM node implementation |
| Parallel semantics change | Medium | Medium | Test all current behaviors in new parallel node |
| Progress calculation regression | Medium | Low | Add comprehensive progress streaming tests |
| Vault persistence path changes | Low | Low | Keep `ResearchVaultPersister` as leaf action |
| Token counting accuracy | Medium | Medium | Use same token counting as Oracle |

---

## Appendix A: File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `behaviors.py` | 976 | 7 behavior classes |
| `orchestrator.py` | 509 | Sequential execution + streaming |
| `llm_service.py` | 296 | OpenRouter/Gemini wrapper |
| `vault_persister.py` | 479 | Jinja2 template rendering |
| `__init__.py` | 41 | Package exports |
| **Total** | **2,301** | - |

| Model | Lines | Purpose |
|-------|-------|---------|
| `research.py` | 235 | All research domain models |

---

## Appendix B: Prompt Templates

Research uses Jinja2 templates from `backend/prompts/research/`:

| Template | Used By | Purpose |
|----------|---------|---------|
| `brief.md` | GenerateBriefBehavior | Transform query to structured brief |
| `compress.md` | CompressFindingsBehavior | Synthesize findings |
| `report.md` | GenerateReportBehavior | Generate final report |
| `vault/index_template.md` | VaultPersister | Hub file |
| `vault/brief_template.md` | VaultPersister | Brief document |
| `vault/sources_template.md` | VaultPersister | Bibliography |
| `vault/methodology_template.md` | VaultPersister | Methodology doc |
| `vault/source_note_template.md` | VaultPersister | Individual source notes |

These templates can be reused directly by BT action nodes via `ctx.services.prompts.load()`.
