# Open Deep Research - Implementation Analysis

## Overview

**Repository**: https://github.com/langchain-ai/open_deep_research
**Version Analyzed**: 0.0.16
**Last Updated**: August 2025 (includes GPT-5 support)
**License**: MIT

Open Deep Research is a configurable, fully open-source deep research agent that automates multi-step research and produces comprehensive reports. It ranked #6 on the Deep Research Bench Leaderboard with a RACE score of 0.4344.

## Architecture Summary

### High-Level Flow

```
User Query
    |
    v
[Clarify with User] -- optional, can ask clarifying questions
    |
    v
[Write Research Brief] -- transforms query into detailed research brief
    |
    v
[Research Supervisor] -- delegates research to sub-agents
    |         |
    v         v
[Researcher 1] [Researcher 2] ... [Researcher N]  -- parallel execution
    |         |                         |
    v         v                         v
[Compress Research] -- summarizes findings
    |
    v
[Final Report Generation] -- synthesizes all findings into report
    |
    v
Final Report (Markdown)
```

### Three-Tier Agent Architecture

1. **Main Graph** (`deep_researcher`):
   - Entry point: `clarify_with_user` -> `write_research_brief` -> `research_supervisor` -> `final_report_generation`
   - Manages overall research workflow

2. **Supervisor Subgraph** (`supervisor_subgraph`):
   - Decides research strategy using tools: `ConductResearch`, `ResearchComplete`, `think_tool`
   - Delegates work to multiple researcher sub-agents
   - Controls parallelism (up to `max_concurrent_research_units`)

3. **Researcher Subgraph** (`researcher_subgraph`):
   - Individual research agents for specific topics
   - Uses search tools + `think_tool` for strategic reflection
   - Compresses findings before returning to supervisor

## Key Files and Components

### Core Implementation (`src/open_deep_research/`)

| File | Purpose |
|------|---------|
| `deep_researcher.py` | Main LangGraph workflow (~720 lines) |
| `configuration.py` | Pydantic configuration schema |
| `state.py` | State definitions and structured outputs |
| `prompts.py` | All system prompts and templates |
| `utils.py` | Search tools, MCP loading, token management |

### Entry Point

The graph is exported via `langgraph.json`:
```json
{
  "graphs": {
    "Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher"
  }
}
```

### State Definitions

**AgentState** (main state):
```python
class AgentState(MessagesState):
    supervisor_messages: list[MessageLikeRepresentation]
    research_brief: Optional[str]
    raw_notes: list[str]  # Accumulated from researchers
    notes: list[str]      # Processed notes
    final_report: str
```

**SupervisorState**:
```python
class SupervisorState(TypedDict):
    supervisor_messages: list[MessageLikeRepresentation]
    research_brief: str
    notes: list[str]
    research_iterations: int
    raw_notes: list[str]
```

**ResearcherState**:
```python
class ResearcherState(TypedDict):
    researcher_messages: list[MessageLikeRepresentation]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: list[str]
```

### Structured Outputs (Tools)

| Schema | Purpose |
|--------|---------|
| `ConductResearch` | Supervisor delegates research topic (paragraph-length) |
| `ResearchComplete` | Signals supervisor to finish |
| `ClarifyWithUser` | Ask clarifying questions before research |
| `ResearchQuestion` | Structured research brief |
| `Summary` | Webpage summarization output |

## Configuration Options

### Model Configuration

| Field | Default | Purpose |
|-------|---------|---------|
| `summarization_model` | `openai:gpt-4.1-mini` | Summarizes Tavily search results |
| `research_model` | `openai:gpt-4.1` | Powers supervisor + researchers |
| `compression_model` | `openai:gpt-4.1` | Compresses research findings |
| `final_report_model` | `openai:gpt-4.1` | Writes final report |

### Research Limits

| Field | Default | Purpose |
|-------|---------|---------|
| `max_concurrent_research_units` | 5 | Parallel researchers (1-20) |
| `max_researcher_iterations` | 6 | Supervisor reflection cycles |
| `max_react_tool_calls` | 10 | Tool calls per researcher |
| `max_structured_output_retries` | 3 | LLM retry attempts |
| `max_content_length` | 50000 | Chars before summarization |

### Search Configuration

| Field | Default | Options |
|-------|---------|---------|
| `search_api` | `tavily` | `tavily`, `openai`, `anthropic`, `none` |

### MCP Configuration

```python
class MCPConfig(BaseModel):
    url: Optional[str]      # MCP server URL
    tools: Optional[List[str]]  # Tools to expose
    auth_required: Optional[bool]  # OAuth requirement
```

## Search Tool Implementation

### Tavily Search Tool

The primary search implementation (`tavily_search`):

1. **Multi-query execution**: Accepts list of queries, runs in parallel
2. **Deduplication**: Removes duplicate URLs across queries
3. **Content summarization**: Uses `summarization_model` to compress webpage content
4. **Structured output**: Returns formatted source list with summaries

```python
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    config: RunnableConfig = None
) -> str
```

### Native Web Search

- **OpenAI**: `{"type": "web_search_preview"}`
- **Anthropic**: `{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}`

### Think Tool (Strategic Reflection)

```python
@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making."""
    return f"Reflection recorded: {reflection}"
```

Used for:
- Analyzing search results
- Planning next research steps
- Assessing research completeness
- Strategic decision-making

## Agent Loop Details

### Supervisor Loop

1. **supervisor()**: Generates tool calls using `ConductResearch`, `ResearchComplete`, or `think_tool`
2. **supervisor_tools()**: Executes tools
   - `think_tool`: Records reflection, continues loop
   - `ConductResearch`: Spawns researcher subgraphs in parallel
   - `ResearchComplete`: Exits loop, proceeds to report generation

Exit conditions:
- `ResearchComplete` tool called
- No tool calls made
- Exceeded `max_researcher_iterations`

### Researcher Loop

1. **researcher()**: Uses search tools + `think_tool`
2. **researcher_tools()**: Executes tools, checks exit conditions

Exit conditions:
- `ResearchComplete` tool called
- Exceeded `max_react_tool_calls`
- No tool calls made

### Compression Phase

After research completes, `compress_research()`:
1. Takes all researcher messages
2. Removes duplicates and irrelevant information
3. Preserves all sources with citations
4. Handles token limit exceeded with progressive truncation

### Final Report Generation

`final_report_generation()`:
1. Combines research brief + user messages + findings
2. Generates structured markdown report
3. Includes proper citations and source links
4. Handles token limits with retry logic

## Prompts Analysis

### Key Prompt Patterns

1. **Date Context**: All prompts include `{date}` for temporal awareness
2. **Hard Limits**: Explicit constraints on tool calls and iterations
3. **Structured Output Guidance**: JSON format requirements
4. **Citation Rules**: Sequential numbering, URL format requirements
5. **Language Detection**: Reports match user's language

### Prompt Templates

| Prompt | Purpose | Key Features |
|--------|---------|--------------|
| `clarify_with_user_instructions` | Clarification questions | JSON structured output |
| `transform_messages_into_research_topic_prompt` | Create research brief | First-person perspective |
| `lead_researcher_prompt` | Supervisor instructions | Task delegation budgets |
| `research_system_prompt` | Researcher instructions | Tool call budgets |
| `compress_research_system_prompt` | Compression instructions | Citation preservation |
| `final_report_generation_prompt` | Report writing | Multiple structure templates |
| `summarize_webpage_prompt` | Webpage summarization | Key excerpt extraction |

## Dependencies

### Core Dependencies

```toml
dependencies = [
    "langgraph>=0.5.4",
    "langchain-community>=0.3.9",
    "langchain-openai>=0.3.28",
    "langchain-anthropic>=0.3.15",
    "langchain-mcp-adapters>=0.1.6",
    "langchain-deepseek>=0.1.2",
    "langchain-tavily",
    "langchain-groq>=0.2.4",
    "openai>=1.99.2",
    "tavily-python>=0.5.0",
    "mcp>=1.9.4",
]
```

### Search API Dependencies

- `tavily-python` - Primary search API
- `duckduckgo-search>=3.0.0` - Alternative free search
- `exa-py>=1.8.8` - Exa search API
- `arxiv>=2.1.3` - Academic paper search
- `pymupdf>=1.25.3` - PDF processing

### External Services

| Service | Purpose | Required |
|---------|---------|----------|
| **Tavily** | Web search | Yes (default) |
| **OpenAI** | LLM provider | Yes (default models) |
| **Anthropic** | Alternative LLM | Optional |
| **Google/Gemini** | Alternative LLM | Optional |
| **LangSmith** | Tracing/evaluation | Optional |
| **MCP Servers** | Extended tools | Optional |
| **Supabase** | OAP auth | Optional |

### Environment Variables

```bash
OPENAI_API_KEY=         # Required for default models
ANTHROPIC_API_KEY=      # For Claude models
GOOGLE_API_KEY=         # For Gemini models
TAVILY_API_KEY=         # For Tavily search
LANGSMITH_API_KEY=      # For tracing
LANGSMITH_PROJECT=
LANGSMITH_TRACING=
```

## Performance Considerations

### Token Usage

From README evaluation results:
- **GPT-5 run**: 204,640,896 tokens total
- **Default (GPT-4.1)**: 58,015,332 tokens, $45.98
- **Claude Sonnet 4**: 138,917,050 tokens, $187.09

### Estimated Costs

| Configuration | Cost per Research Task |
|--------------|------------------------|
| Default (GPT-4.1 all) | ~$0.50-1.00 |
| Full benchmark (100 tasks) | $20-$100+ |

### Typical Research Duration

Based on architecture:
- Simple queries: 2-5 minutes
- Complex research: 5-15 minutes
- Full benchmark tasks: 10-30+ minutes

### Parallelism Impact

- `max_concurrent_research_units`: Higher values = faster but more API rate limit risk
- Default 5 parallel researchers balances speed/reliability

### Token Limit Handling

1. **Detection**: `is_token_limit_exceeded()` checks for provider-specific errors
2. **Truncation**: Progressive reduction (10% per retry) for findings
3. **Message pruning**: `remove_up_to_last_ai_message()` for context overflow

### Model Token Limits (from utils.py)

```python
MODEL_TOKEN_LIMITS = {
    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1": 1047576,
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "google:gemini-1.5-pro": 2097152,
}
```

## Legacy Implementations

Two alternative approaches available in `src/legacy/`:

### 1. Plan-and-Execute Workflow (`graph.py`)

- **Human-in-the-loop**: Review report plan before execution
- **Sequential sections**: One section at a time with reflection
- **Quality focus**: Iterative refinement with grading

### 2. Multi-Agent Architecture (`multi_agent.py`)

- **Supervisor-Researcher pattern**: Similar to current implementation
- **Parallel research**: Multiple researchers work simultaneously
- **Less performant**: Replaced by current implementation

## Vlt-Bridge Integration Considerations

### Fit as Oracle Plugin/Mode

Open Deep Research could integrate as a "deep research mode" for the Oracle agent:

**Strengths**:
1. Already uses similar supervisor-researcher pattern
2. MCP support built-in
3. Configurable models match Oracle's multi-model approach
4. Streaming-compatible output

**Challenges**:
1. LangGraph dependency (Oracle uses custom agent loop)
2. Different state management
3. No direct code/vault integration

### Leveraging Existing Vlt-Bridge Tools

| ODR Need | Vlt-Bridge Equivalent |
|----------|----------------------|
| Web search | `web_search` tool in Oracle |
| Structured output | Already have via XML/JSON parsing |
| State management | Context tree (oracle_context_service) |
| MCP tools | MCP server already available |

### New Capabilities Needed

1. **Research Brief Generation**: Transform user queries into structured research plans
2. **Parallel Research Delegation**: Fork multiple research sub-tasks
3. **Research Compression**: Synthesize findings from multiple sources
4. **Report Generation**: Structured markdown output with citations

### Proposed Integration Architecture

```
Oracle Agent
    |
    +-- Deep Research Mode (new)
    |       |
    |       +-- Research Planning (generates research brief)
    |       |
    |       +-- Parallel Research (uses existing tools)
    |       |       |
    |       |       +-- web_search
    |       |       +-- vault tools (documentation context)
    |       |       +-- coderag tools (code understanding)
    |       |       +-- github_read (repo analysis)
    |       |
    |       +-- Synthesis (compress + cite)
    |       |
    |       +-- Report Generation (markdown output)
    |
    +-- Save to Vault (optional)
```

### Key Design Decisions

1. **Keep or replace LangGraph?**
   - Option A: Embed LangGraph as dependency
   - Option B: Port patterns to Oracle's agent loop (recommended)

2. **Research persistence**
   - Store intermediate research in vault
   - Use vlt threads for research history
   - Enable research resume/continuation

3. **Tool integration**
   - Reuse existing Oracle tools
   - Add research-specific prompts
   - Configure research limits in Oracle settings

4. **Output format**
   - Markdown report (can save to vault)
   - Streaming chunks during research
   - Progress indicators for long research

### Migration Path

1. **Phase 1**: Add research prompts and brief generation
2. **Phase 2**: Implement parallel research execution
3. **Phase 3**: Add compression and report generation
4. **Phase 4**: Vault integration for persistence

## Evaluation and Quality

### Deep Research Bench Metrics

| Metric | Description |
|--------|-------------|
| RACE Score | LLM-as-judge evaluation (Gemini) |
| Overall Quality | Well-researched, accurate, professional |
| Relevance | Topic and section relevance |
| Structure | Logical flow and organization |
| Correctness | Factual accuracy |
| Groundedness | Source-backed claims |
| Completeness | Comprehensive coverage |

### Quality Criteria (from evaluation)

1. Topic Relevance (Overall)
2. Section Relevance (Critical)
3. Structure and Flow
4. Introduction Quality
5. Conclusion Quality
6. Structural Elements (tables, lists)
7. Section Headers (proper Markdown)
8. Citations
9. Overall Quality

## Conclusion

Open Deep Research provides a well-architected pattern for multi-step research automation. Key takeaways for Vlt-Bridge integration:

1. **Supervisor-Researcher pattern** is proven and effective
2. **Parallel research** significantly speeds up comprehensive research
3. **Compression step** is essential for managing context limits
4. **Structured prompts** with explicit limits improve consistency
5. **Citation preservation** is critical for research credibility

The implementation can be adapted to Oracle's existing architecture by:
- Porting the research planning and delegation patterns
- Leveraging existing tools (web_search, vault, coderag)
- Adding research-specific prompts and configuration
- Implementing compression and report generation phases
