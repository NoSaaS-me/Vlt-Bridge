# Deep Research Implementation Audit Report

**Date**: 2026-01-04
**Auditor**: Claude Opus 4.5
**Scope**: Comparison of Vlt-Bridge Deep Research vs LangChain open_deep_research

---

## Executive Summary

**Overall Completion: ~35-40%**

Our implementation has the basic pipeline structure but is fundamentally missing the core innovation of the LangChain reference: **agentic researchers that iteratively call tools**. What we built is a sophisticated multi-step batch processing pipeline, not a true research agent system.

### Critical Gaps

1. **Researchers do NOT iterate with tools** - They make a single batch of search calls, not agentic loops
2. **No supervisor agent** - The orchestrator is procedural, not an LLM-driven supervisor
3. **No think tool in execution** - Think.md prompt exists but is never used during research
4. **No tool calling capability for researchers** - Researchers cannot dynamically decide to search more
5. **No clarification phase** - No ability to ask user clarifying questions before research
6. **No message-based state management** - Uses simple dataclasses instead of message history

### What Works

1. Basic pipeline structure (brief -> research -> compress -> report -> persist)
2. Tavily and OpenRouter search services
3. Parallel execution of researchers
4. Vault persistence with templates
5. Progress streaming
6. Quality metrics in reports

---

## Detailed Comparison Table

| Feature/Component | LangChain Reference | Our Implementation | Gap Analysis | Priority |
|-------------------|--------------------|--------------------|--------------|----------|
| **Three-Tier Agent Architecture** | Main Graph -> Supervisor Subgraph -> Researcher Subgraph with LangGraph | Single orchestrator with behavior classes | **MISSING**: No hierarchical agent structure, no LangGraph-style state machines | Critical |
| **Supervisor Agent** | LLM-based supervisor that calls `ConductResearch`, `ResearchComplete`, `think_tool` to dynamically decide research strategy | Procedural `PlanSubtopicsBehavior` that just creates researcher states from brief | **MISSING**: No LLM-driven supervisor, no dynamic research planning during execution | Critical |
| **Researcher Agent Loop** | ReAct loop: researcher() -> researcher_tools() -> repeat until exit condition | Single `run_single()` call that does batch search + extract, no iteration | **MISSING**: Researchers don't iterate, can't call search repeatedly based on findings | Critical |
| **Tool Calling from Researchers** | Researchers have `search_web_tavily` + `think_tool`, call them iteratively up to `max_react_tool_calls` | Researchers generate queries upfront, execute batch search, no tool calling | **MISSING**: No actual tool execution loop, queries are pre-generated | Critical |
| **Think Tool Integration** | Used for strategic reflection between searches, recorded in messages | `think.md` prompt exists but is never rendered or used | **MISSING**: Think tool is dead code | High |
| **Message-Based State** | `AgentState`, `SupervisorState`, `ResearcherState` all track `messages: list[MessageLikeRepresentation]` | Dataclasses with simple fields, no message history | **PARTIAL**: Have state classes but without message threading | High |
| **Clarify With User** | Optional first node that asks clarifying questions before research | Not implemented | **MISSING**: No user clarification phase | Medium |
| **Research Brief Generation** | `write_research_brief` node with structured output | `GenerateBriefBehavior` with JSON generation | **MATCHES**: Both generate structured brief | - |
| **Parallel Research Execution** | Uses LangGraph's `Send()` to spawn parallel researcher subgraphs | `ParallelResearchersBehavior` with `asyncio.gather()` | **MATCHES**: Both support parallel execution | - |
| **Compression Phase** | `compress_research()` with token limit handling, progressive truncation | `CompressFindingsBehavior` with regex parsing | **PARTIAL**: Basic compression but no token limit handling | Medium |
| **Token Limit Management** | `is_token_limit_exceeded()`, progressive 10% truncation, message pruning | Not implemented | **MISSING**: No token overflow handling | High |
| **Final Report Generation** | Multi-model support, retry logic, token limit handling | `GenerateReportBehavior` with single LLM call | **PARTIAL**: Basic report generation, no retry/limit handling | Medium |
| **Structured Output Schemas** | `ConductResearch`, `ResearchComplete`, `ClarifyWithUser`, `ResearchQuestion`, `Summary` Pydantic models | Basic JSON generation without structured output enforcement | **PARTIAL**: Have models but don't use LLM structured output | Medium |
| **Search Tool Implementation** | Multi-query parallel execution, deduplication, content summarization | `TavilySearchService` with parallel search and dedup | **MATCHES**: Good implementation | - |
| **OpenAI/Anthropic Native Search** | `web_search_preview` for OpenAI, `web_search_20250305` for Anthropic | Not implemented | **MISSING**: No native web search support | Low |
| **MCP Integration** | Full MCP support for extending tools | Oracle already has MCP but not exposed to researchers | **PARTIAL**: Have MCP but not connected to research | Low |
| **Prompt Quality** | Extensive prompts with date context, tool budgets, citation rules, language detection | Basic prompts with some of these elements | **PARTIAL**: Prompts exist but less detailed | Medium |
| **Vault Persistence** | Not included in reference | Full vault structure with templates | **BETTER**: Our addition | - |
| **Progress Streaming** | LangGraph provides streaming natively | Custom `ResearchProgress` with SSE-style updates | **MATCHES**: Different approach, same result | - |
| **Configuration** | Extensive `Configuration` class with model selection, limits, MCP config | Basic `ResearchConfig` with depth presets | **PARTIAL**: Have config but less flexible | Low |
| **Exit Conditions** | Multiple: `ResearchComplete` tool, no tool calls, max iterations | Fixed: researcher runs once and completes | **MISSING**: No dynamic exit conditions | High |
| **Error Recovery** | Retry logic with `max_structured_output_retries` | Basic try/except with fallback states | **PARTIAL**: Has fallbacks but no retry loops | Medium |

---

## Architecture Gap Analysis

### LangChain Architecture (What We Should Have)

```
[User Query]
    |
    v
[Clarify With User] -- LLM decides if clarification needed
    |
    v
[Write Research Brief] -- LLM generates structured brief
    |
    v
[Research Supervisor] <---------------------------+
    |                                              |
    | LLM decides: call tools                      |
    |   - ConductResearch(topic) -> spawn researcher
    |   - think_tool(reflection) -> continue loop
    |   - ResearchComplete() -> exit
    |                                              |
    +-- [Researcher 1] --+                        |
    |   [Researcher 2]   |- parallel              |
    |   [Researcher N] --+                        |
    |        |                                     |
    |        | Each researcher loops:              |
    |        | - search_web_tavily(queries)        |
    |        | - think_tool(assess)                |
    |        | - repeat until max_tool_calls       |
    |        |                                     |
    +--------<------- results roll up ------------+
    |
    v
[Compress Research] -- LLM synthesizes
    |
    v
[Final Report] -- LLM generates
```

### Our Architecture (What We Built)

```
[User Query]
    |
    v
[GenerateBriefBehavior] -- LLM generates brief (OK)
    |
    v
[PlanSubtopicsBehavior] -- Just slice subtopics into researchers (NO LLM)
    |
    v
[ParallelResearchersBehavior]
    |
    +-- [Researcher 1] --+
    |   [Researcher 2]   |- parallel (OK)
    |   [Researcher N] --+
    |        |
    |        | Each researcher:
    |        | 1. _generate_search_queries() -- STATIC, not LLM loop
    |        | 2. tavily.search_parallel() -- SINGLE batch call
    |        | 3. _extract_findings() -- SINGLE LLM call
    |        | 4. DONE (no iteration!)
    |        |
    v
[CompressFindingsBehavior] -- LLM synthesizes (OK)
    |
    v
[GenerateReportBehavior] -- LLM generates (OK)
    |
    v
[PersistToVaultBehavior] -- Save to vault (BONUS)
```

---

## Code Changes Required for Parity

### 1. [CRITICAL] Implement Agentic Researcher Loop

**Effort**: Large (3-5 days)

**Current State**: `ResearcherBehavior.run_single()` makes one batch search call.

**Required**: Transform into a ReAct loop that:
1. Generates tool calls via LLM
2. Executes tools (search, think)
3. Appends results to message history
4. Loops until exit condition

**Files to Modify**:
- `backend/src/services/research/behaviors.py`: Rewrite `ResearcherBehavior`
- `backend/src/models/research.py`: Add `researcher_messages` field
- `backend/prompts/research/researcher.md`: Add tool calling format

**Example Pattern**:
```python
async def run_single(self, state, researcher):
    while researcher.tool_calls < researcher.max_tool_calls:
        # 1. Call LLM with message history + tools
        response = await self.llm.generate_with_tools(
            messages=researcher.messages,
            tools=[search_tool, think_tool, complete_tool],
        )

        # 2. Check for exit
        if response.calls_tool("ResearchComplete") or not response.tool_calls:
            break

        # 3. Execute tools
        for tool_call in response.tool_calls:
            result = await self.execute_tool(tool_call)
            researcher.messages.append(result)

        researcher.tool_calls += 1
```

### 2. [CRITICAL] Implement Supervisor Agent

**Effort**: Large (2-3 days)

**Current State**: `PlanSubtopicsBehavior` is procedural code.

**Required**: Create an LLM-based supervisor that:
1. Receives research brief
2. Calls `ConductResearch(topic)` to spawn researchers
3. Uses `think_tool` for reflection
4. Calls `ResearchComplete()` when done
5. Can iterate (spawn more researchers if needed)

**Files to Create/Modify**:
- `backend/src/services/research/supervisor.py` (NEW)
- `backend/prompts/research/supervisor.md` (NEW)
- `backend/src/models/research.py`: Add `supervisor_messages`, `research_iterations`

### 3. [CRITICAL] Add Tool Calling to LLM Service

**Effort**: Medium (1-2 days)

**Current State**: `ResearchLLMService` only has `generate()` and `generate_json()`.

**Required**: Add `generate_with_tools()` method that:
1. Accepts tool schemas
2. Parses tool call responses
3. Returns structured tool calls

**Files to Modify**:
- `backend/src/services/research/llm_service.py`: Add tool calling methods

### 4. [HIGH] Implement Think Tool Execution

**Effort**: Small (0.5 days)

**Current State**: `think.md` prompt exists but is never used.

**Required**:
1. Create `ThinkTool` that records reflection to messages
2. Wire it into researcher tool set
3. Actually render and use `think.md` prompt

**Files to Modify**:
- `backend/src/services/research/behaviors.py`: Add think tool handling

### 5. [HIGH] Add Token Limit Management

**Effort**: Medium (1 day)

**Current State**: No token counting or overflow handling.

**Required**:
1. Add token counting utility
2. Implement `is_token_limit_exceeded()` check
3. Add progressive truncation (10% reduction per retry)
4. Add message pruning for context overflow

**Files to Create/Modify**:
- `backend/src/services/research/utils.py` (NEW)
- `backend/src/services/research/behaviors.py`: Add to compression/report behaviors

### 6. [MEDIUM] Add User Clarification Phase

**Effort**: Medium (1 day)

**Current State**: Not implemented.

**Required**:
1. Create `ClarifyWithUserBehavior`
2. Add interactive loop for clarification
3. Integrate with frontend ChatPanel

**Files to Create/Modify**:
- `backend/src/services/research/behaviors.py`: Add clarification behavior
- `backend/prompts/research/clarify.md` (NEW)

### 7. [MEDIUM] Improve Structured Output Handling

**Effort**: Small (0.5 days)

**Current State**: Uses `generate_json()` with regex JSON extraction.

**Required**:
1. Use proper JSON mode for OpenRouter
2. Add retry logic for malformed JSON
3. Add structured output schemas for tool calls

**Files to Modify**:
- `backend/src/services/research/llm_service.py`

### 8. [MEDIUM] Enhance Prompts

**Effort**: Small (0.5 days)

**Current State**: Prompts exist but are less detailed than reference.

**Required**:
1. Add explicit tool call budgets to prompts
2. Add more detailed citation requirements
3. Add example outputs

**Files to Modify**:
- All files in `backend/prompts/research/`

---

## Effort Summary

| Change | Effort | Priority | Status |
|--------|--------|----------|--------|
| Agentic Researcher Loop | Large (3-5 days) | Critical | Not Started |
| Supervisor Agent | Large (2-3 days) | Critical | Not Started |
| Tool Calling in LLM Service | Medium (1-2 days) | Critical | Not Started |
| Think Tool Execution | Small (0.5 days) | High | Not Started |
| Token Limit Management | Medium (1 day) | High | Not Started |
| User Clarification Phase | Medium (1 day) | Medium | Not Started |
| Improved Structured Output | Small (0.5 days) | Medium | Not Started |
| Enhanced Prompts | Small (0.5 days) | Medium | Not Started |

**Total Effort for Parity**: ~10-14 days of focused development

---

## Recommendations

### Option A: Full Parity (Recommended)

Implement all critical and high priority changes to achieve true agentic research. This transforms our batch pipeline into an actual agent system.

**Pros**:
- Matches reference quality
- Researchers can adapt to findings
- Supervisor can adjust strategy
- Better research outcomes

**Cons**:
- 2+ weeks of work
- More complex system
- Higher token costs

### Option B: Partial Enhancement

Keep batch approach but add:
1. Multi-round search (researcher calls search 2-3 times)
2. Basic think tool integration
3. Token limit handling

**Pros**:
- ~3-4 days of work
- Incremental improvement
- Lower complexity

**Cons**:
- Still not truly agentic
- Can't adapt to unexpected findings

### Option C: LangGraph Integration

Instead of reimplementing, use LangGraph directly or port their code.

**Pros**:
- Fastest path to feature parity
- Battle-tested implementation

**Cons**:
- New dependency
- Different state management
- Integration complexity with Oracle

---

## Conclusion

Our Deep Research implementation is a well-structured batch processing pipeline, but it fundamentally misses the agentic nature of the LangChain reference. The key innovation of open_deep_research is that **researchers iterate with tools**, making decisions about what to search next based on what they've found. Our researchers run once and complete.

To achieve parity, we need to:
1. Transform researchers into ReAct agents with tool loops
2. Add an LLM-based supervisor that directs research strategy
3. Actually use the think tool for strategic reflection
4. Add proper exit conditions and iteration limits

The current implementation is ~35-40% complete in terms of the intended functionality.

---

**Confidence: 9/10**

I have high confidence in this assessment because I read both the complete reference documentation (research.md) and every file of our implementation. The gap is clear and architectural - our code runs sequentially through behavior classes while the reference uses nested agent loops with tool calling.
