# Deep Research Implementation Mapping: Ours vs LangChain's open_deep_research

This document provides an **exact, code-level comparison** between our Deep Research implementation and LangChain's `open_deep_research`. This is based on actual source code reading, not assumptions.

---

## Executive Summary

| Aspect | LangChain open_deep_research | Our Implementation | Gap Severity |
|--------|------------------------------|-------------------|--------------|
| **Architecture** | LangGraph state machine with subgraphs | Behavior tree pattern with orchestrator | Different but comparable |
| **Clarification Phase** | Full user clarification flow | **Missing entirely** | **CRITICAL** |
| **Supervisor Loop** | Iterative with tool calls, think_tool | One-shot subtopic planning | **HIGH** |
| **Researcher Loop** | ReAct loop with max_react_tool_calls iterations | **Single search iteration** | **CRITICAL** |
| **Token Management** | Progressive truncation with model-specific limits | **No token tracking** | **HIGH** |
| **Compression** | Full message history compression with retry | Basic regex parsing | **MEDIUM** |
| **Report Generation** | Multi-retry with 10% truncation fallback | Single attempt | **MEDIUM** |
| **MCP Integration** | Full MCP tool support with auth | None | **LOW** (feature gap) |

---

## 1. Function-by-Function Mapping

### Phase 1: User Input Processing

| Their Function | Their File | Our Equivalent | Our File | Match Quality | Gap Description |
|----------------|------------|----------------|----------|---------------|-----------------|
| `clarify_with_user()` | deep_researcher.py:52-97 | **NONE** | - | **None** | We skip clarification entirely. They ask follow-up questions if query is ambiguous. |
| `write_research_brief()` | deep_researcher.py:100-153 | `GenerateBriefBehavior.run()` | behaviors.py:93-144 | **Partial** | We generate a brief but don't use structured output or retry logic |

**Their clarify_with_user() actual code:**
```python
# They check if clarification is enabled
if not configurable.allow_clarification:
    return Command(goto="write_research_brief")

# They use structured output to decide
clarification_model = (
    configurable_model
    .with_structured_output(ClarifyWithUser)
    .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    .with_config(model_config)
)

response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

if response.need_clarification:
    return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
```

**Our equivalent: DOES NOT EXIST**

---

### Phase 2: Research Planning (Supervisor)

| Their Function | Their File | Our Equivalent | Our File | Match Quality | Gap Description |
|----------------|------------|----------------|----------|---------------|-----------------|
| `supervisor()` | deep_researcher.py:156-199 | `PlanSubtopicsBehavior.run()` | behaviors.py:165-193 | **None** | They have iterative supervisor loop; we just slice subtopics |
| `supervisor_tools()` | deep_researcher.py:202-319 | **NONE** | - | **None** | We don't have a supervisor tool execution loop |
| `think_tool` | utils.py:162-191 | **NONE** | - | **None** | Strategic reflection tool missing entirely |

**Their supervisor() loop pattern:**
```python
# They call LLM with tools: [ConductResearch, ResearchComplete, think_tool]
lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

research_model = (
    configurable_model
    .bind_tools(lead_researcher_tools)
    .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    .with_config(research_model_config)
)

response = await research_model.ainvoke(supervisor_messages)

return Command(
    goto="supervisor_tools",
    update={
        "supervisor_messages": [response],
        "research_iterations": state.get("research_iterations", 0) + 1
    }
)
```

**Our PlanSubtopicsBehavior (the "equivalent"):**
```python
# We just take subtopics from brief and create researcher states
subtopics = state.brief.subtopics[:config.max_concurrent_researchers]
state.researchers = [
    ResearcherState(
        subtopic=subtopic,
        max_tool_calls=config.max_tool_calls_per_researcher,
    )
    for subtopic in subtopics
]
```

**Gap Analysis:**
- They have an **iterative** supervisor that can:
  1. Use `think_tool` to plan strategy
  2. Call `ConductResearch` to delegate (potentially multiple rounds)
  3. Call `ResearchComplete` when satisfied
- We have **one-shot** planning that just slices subtopics

---

### Phase 3: Individual Research (Researcher Loop)

| Their Function | Their File | Our Equivalent | Our File | Match Quality | Gap Description |
|----------------|------------|----------------|----------|---------------|-----------------|
| `researcher()` | deep_researcher.py:401-460 | `ResearcherBehavior.run_single()` | behaviors.py:243-327 | **Partial** | They iterate up to `max_react_tool_calls`; we do ONE search |
| `researcher_tools()` | deep_researcher.py:470-543 | **NONE** | - | **None** | We don't have a researcher tool loop |
| `tavily_search` tool | utils.py:32-118 | `TavilySearchService.search()` | tavily_service.py:51-87 | **Partial** | They summarize raw_content; we just return snippets |
| `summarize_webpage()` | utils.py:152-178 | **NONE** | - | **None** | We don't summarize long webpage content |

**Their researcher() ReAct loop:**
```python
# They have a LOOP that iterates up to max_react_tool_calls
async def researcher_tools(state, config):
    # Check iteration limit
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls

    if exceeded_iterations or research_complete_called:
        return Command(goto="compress_research", ...)

    # Otherwise continue looping
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})
```

**Our ResearcherBehavior.run_single() (single pass):**
```python
async def run_single(self, state, researcher):
    # Generate queries
    queries = await self._generate_search_queries(state, researcher)

    # Execute searches - ONE TIME
    search_responses = await self.tavily.search_parallel(
        queries=queries,
        max_results_per_query=3,
        deduplicate=True,
    )

    # Done - no loop
    researcher.completed = True
```

**Gap Analysis:**
- Their researcher can iterate **up to 10 tool calls** (default `max_react_tool_calls=10`)
- Our researcher makes **exactly 1 search call** per subtopic
- Their researcher can use `think_tool` between searches to assess progress
- We don't have any reflection or iteration capability

---

### Phase 4: Research Compression

| Their Function | Their File | Our Equivalent | Our File | Match Quality | Gap Description |
|----------------|------------|----------------|----------|---------------|-----------------|
| `compress_research()` | deep_researcher.py:546-609 | `CompressFindingsBehavior.run()` | behaviors.py:556-688 | **Partial** | They compress full message history; we use regex parsing |

**Their compress_research():**
```python
# They compress the ENTIRE message history
researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))

while synthesis_attempts < max_attempts:
    try:
        messages = [SystemMessage(content=compression_prompt)] + researcher_messages
        response = await synthesizer_model.ainvoke(messages)
        return {"compressed_research": str(response.content), ...}
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.research_model):
            # THEY HANDLE TOKEN OVERFLOW
            researcher_messages = remove_up_to_last_ai_message(researcher_messages)
            continue
```

**Our CompressFindingsBehavior:**
```python
# We use regex to parse a specific format
def _parse_compressed_findings(self, response: str) -> List[ResearchFinding]:
    finding_pattern = re.compile(
        r"-\s*(.+?)\s*\[sources?:\s*([\d,\s]+)\]",
        re.IGNORECASE
    )

    for match in finding_pattern.finditer(response):
        # ... parse matches
```

**Gap Analysis:**
- They compress full tool call history with retry for token overflow
- We rely on regex parsing a specific format - **brittle**
- They preserve raw notes; we don't track raw notes at all

---

### Phase 5: Final Report Generation

| Their Function | Their File | Our Equivalent | Our File | Match Quality | Gap Description |
|----------------|------------|----------------|----------|---------------|-----------------|
| `final_report_generation()` | deep_researcher.py:642-721 | `GenerateReportBehavior.run()` | behaviors.py:715-916 | **Partial** | They retry with truncation; we make one attempt |

**Their final_report_generation() with retry:**
```python
while current_retry <= max_retries:
    try:
        final_report = await configurable_model.ainvoke(...)
        return {"final_report": final_report.content, ...}
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.final_report_model):
            current_retry += 1
            if current_retry == 1:
                model_token_limit = get_model_token_limit(configurable.final_report_model)
                findings_token_limit = model_token_limit * 4
            else:
                # REDUCE BY 10% EACH RETRY
                findings_token_limit = int(findings_token_limit * 0.9)
            findings = findings[:findings_token_limit]
            continue
```

**Our GenerateReportBehavior.run():**
```python
try:
    response = await self.llm.generate(
        prompt=prompt,
        max_tokens=15000,
        temperature=0.5,
    )
    state.report = self._parse_report(response, state)
except Exception as e:
    # Single failure path - no retry
    state.status = ResearchStatus.FAILED
```

---

## 2. Prompt-by-Prompt Comparison

### Clarification Prompt

| Aspect | Their Prompt | Our Prompt |
|--------|-------------|------------|
| **Variable** | `clarify_with_user_instructions` | **NONE** |
| **Date Context** | `Today's date is {date}` | N/A |
| **JSON Output** | Yes, structured with `need_clarification`, `question`, `verification` | N/A |
| **Message History** | Full `{messages}` context | N/A |

**Their clarification prompt excerpt:**
```
Assess whether you need to ask a clarifying question, or if the user has already
provided enough information for you to start research.
IMPORTANT: If you can see in the messages history that you have already asked a
clarifying question, you almost always do not need to ask another one.
```

**Our clarification prompt: DOES NOT EXIST**

---

### Research Brief/Topic Prompt

| Aspect | Their Prompt | Our Prompt |
|--------|-------------|------------|
| **Variable** | `transform_messages_into_research_topic_prompt` | `research/brief.md` |
| **Date Context** | `Today's date is {date}` | `{{ current_date }}` |
| **Output** | Single `research_brief` string | JSON with `refined_question`, `scope`, `subtopics`, `constraints`, `language` |
| **Guidance** | Maximizing specificity, avoiding assumptions | Basic structured breakdown |

**Their prompt guidance:**
```
1. Maximize Specificity and Detail
2. Fill in Unstated But Necessary Dimensions as Open-Ended
3. Avoid Unwarranted Assumptions
4. Use the First Person
5. Sources - prioritize primary sources, official websites, original papers
```

**Our brief.md prompt:**
```markdown
1. **Refined Question**: Clarify the core research question. Be specific and measurable.
2. **Scope**: What is IN scope and OUT of scope for this research.
3. **Subtopics**: Break down into 3-5 specific subtopics...
```

**Gap:** Their prompt focuses on **single research question** with detailed guidance on source prioritization. Ours structures into subtopics but less guidance on source quality.

---

### Lead Researcher (Supervisor) Prompt

| Aspect | Their Prompt | Our Prompt |
|--------|-------------|------------|
| **Variable** | `lead_researcher_prompt` | **NONE** (we don't have supervisor) |
| **Tool Budget** | `max_researcher_iterations` limit | N/A |
| **Concurrent Limit** | `max_concurrent_research_units` | Via config |
| **Think Tool Usage** | Explicitly required before/after `ConductResearch` | N/A |

**Their supervisor prompt excerpt:**
```
<Hard Limits>
**Task Delegation Budgets** (Prevent excessive delegation):
- **Bias towards single agent** - Use single agent for simplicity unless clear opportunity for parallelization
- **Stop when you can answer confidently** - Don't keep delegating research for perfection
- **Limit tool calls** - Always stop after {max_researcher_iterations} tool calls

**Maximum {max_concurrent_research_units} parallel agents per iteration**
</Hard Limits>

<Show Your Thinking>
Before you call ConductResearch tool call, use think_tool to plan your approach
After each ConductResearch tool call, use think_tool to analyze the results
</Show Your Thinking>
```

**Our equivalent: DOES NOT EXIST** - We skip to direct subtopic research

---

### Researcher Prompt

| Aspect | Their Prompt | Our Prompt |
|--------|-------------|------------|
| **Variable** | `research_system_prompt` | `research/researcher.md` |
| **Tool Budget** | 2-5 search calls max, hard stop at 5 | `{{ max_tool_calls }}` variable, but not enforced |
| **Think Tool** | Required after each search | Mentioned but not implemented |
| **Stop Conditions** | Explicit: "3+ relevant sources", "last 2 searches similar" | "aim for 3-5 quality sources" |

**Their researcher prompt excerpt:**
```
<Hard Limits>
**Tool Call Budgets** (Prevent excessive searching):
- **Simple queries**: Use 2-3 search tool calls maximum
- **Complex queries**: Use up to 5 search tool calls maximum
- **Always stop**: After 5 search tool calls if you cannot find the right sources

**Stop Immediately When**:
- You can answer the user's question comprehensively
- You have 3+ relevant examples/sources for the question
- Your last 2 searches returned similar information
</Hard Limits>

<Show Your Thinking>
After each search tool call, use think_tool to analyze the results
</Show Your Thinking>
```

**Our researcher.md:**
```markdown
## Instructions
1. **PLAN FIRST**: Use the think tool to plan 2-3 search strategies before searching.
2. **SEARCH STRATEGICALLY**:
   - Maximum {{ max_tool_calls }} tool calls
3. **ASSESS COMPLETENESS**: Use think tool to assess...
```

**Gap:** We mention think tool but don't implement it. Their prompt enforces hard limits; ours is aspirational.

---

### Compression Prompt

| Aspect | Their Prompt | Our Prompt |
|--------|-------------|------------|
| **Variable** | `compress_research_system_prompt` | `research/compress.md` |
| **Preservation** | "repeat key information verbatim" | "preserve all relevant information" |
| **Citation Rules** | Numbered sequentially, explicit format | Basic source reference |
| **Length** | "can be as long as necessary" | "aim for 30-50% compression" |

**Their compression prompt excerpt:**
```
<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...)
</Citation Rules>

Critical Reminder: It is extremely important that any information that is even
remotely relevant to the user's research topic is preserved verbatim
```

**Our compress.md:**
```markdown
- **BE CONCISE**: Aim for 30-50% compression while preserving meaning
- **PRESERVE SOURCE IDS**: Keep all citation references
```

**Gap:** Their compression preserves everything verbatim; ours aims for 30-50% reduction which can lose information.

---

### Final Report Prompt

| Aspect | Their Prompt | Our Prompt |
|--------|-------------|------------|
| **Variable** | `final_report_generation_prompt` | `research/report.md` |
| **Language Matching** | **CRITICAL** - match user's language | `{{ language }}` variable |
| **Structure Examples** | Multiple examples: comparison, list, summary, single section | Basic template |
| **Citation Format** | `[1] Source Title: URL` with sequential numbering | `[N]` inline citations |

**Their report prompt language enforcement:**
```
CRITICAL: Make sure the answer is written in the same language as the human messages!
For example, if the user's messages are in English, then MAKE SURE you write your
response in English. If the user's messages are in Chinese, then MAKE SURE you
write your entire response in Chinese.
```

**Their report structure flexibility:**
```
You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things:
1/ list of things or table of things

REMEMBER: Section is a VERY fluid and loose concept.
```

---

## 3. State Field Mapping

### Their AgentState vs Our ResearchState

| Their Field | Type | Our Equivalent | Our Type | Match |
|-------------|------|----------------|----------|-------|
| `messages` | `list[Message]` | **NONE** | - | None |
| `supervisor_messages` | `Annotated[list, override_reducer]` | **NONE** | - | None |
| `research_brief` | `Optional[str]` | `brief` | `Optional[ResearchBrief]` | Partial (ours is structured) |
| `raw_notes` | `Annotated[list[str], override_reducer]` | **NONE** | - | None |
| `notes` | `Annotated[list[str], override_reducer]` | `compressed_findings` | `List[ResearchFinding]` | Partial |
| `final_report` | `str` | `report` | `Optional[ResearchReport]` | Partial (ours is structured) |

### Their SupervisorState vs Our ???

| Their Field | Type | Our Equivalent | Notes |
|-------------|------|----------------|-------|
| `supervisor_messages` | `Annotated[list, override_reducer]` | **NONE** | We don't have a supervisor |
| `research_brief` | `str` | N/A | |
| `notes` | `Annotated[list[str], override_reducer]` | N/A | |
| `research_iterations` | `int` | N/A | **We don't track iterations** |
| `raw_notes` | `Annotated[list[str], override_reducer]` | N/A | |

### Their ResearcherState vs Our ResearcherState

| Their Field | Type | Our Equivalent | Our Type | Match |
|-------------|------|----------------|----------|-------|
| `researcher_messages` | `Annotated[list, operator.add]` | `messages` | `List[Any]` | Partial (not used) |
| `tool_call_iterations` | `int` | `tool_calls` | `int` | **Yes but not enforced** |
| `research_topic` | `str` | `subtopic` | `str` | Yes |
| `compressed_research` | `str` | **NONE** | - | None |
| `raw_notes` | `Annotated[list[str], override_reducer]` | **NONE** | - | None |

---

## 4. Where We Cut Corners (Specific Code Evidence)

### 4.1 No Clarification Phase

**Their code:**
```python
# deep_researcher.py lines 52-97
async def clarify_with_user(state: AgentState, config: RunnableConfig):
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])

    if response.need_clarification:
        return Command(goto=END, update={"messages": [AIMessage(content=response.question)]})
```

**Our code:** Does not exist. We go directly from query to brief generation.

---

### 4.2 Researcher Runs Exactly Once (Not ReAct Loop)

**Their code (researcher loop):**
```python
# deep_researcher.py lines 470-543
async def researcher_tools(state: ResearcherState, config: RunnableConfig):
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls

    if exceeded_iterations or research_complete_called:
        return Command(goto="compress_research", ...)

    # Execute tools
    observations = await asyncio.gather(*tool_execution_tasks)

    # LOOP BACK to researcher
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})
```

**Our code (single pass):**
```python
# behaviors.py lines 264-327
async def run_single(self, state: ResearchState, researcher: ResearcherState):
    queries = await self._generate_search_queries(state, researcher)

    # Execute searches - NO LOOP
    search_responses = await self.tavily.search_parallel(queries=queries, ...)

    # Mark done immediately
    researcher.completed = True
    return researcher
```

---

### 4.3 No Token Overflow Handling

**Their code:**
```python
# deep_researcher.py lines 662-697
while current_retry <= max_retries:
    try:
        final_report = await configurable_model.ainvoke(...)
    except Exception as e:
        if is_token_limit_exceeded(e, configurable.final_report_model):
            current_retry += 1
            # Reduce by 10% each retry
            findings_token_limit = int(findings_token_limit * 0.9)
            findings = findings[:findings_token_limit]
            continue
```

**Their utils.py token detection:**
```python
# utils.py lines 300-400
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    # Provider-specific checks for OpenAI, Anthropic, Gemini
    return (
        _check_openai_token_limit(exception, error_str) or
        _check_anthropic_token_limit(exception, error_str) or
        _check_gemini_token_limit(exception, error_str)
    )
```

**Our code:** No token tracking or overflow handling. Single attempt, crash on failure.

---

### 4.4 No think_tool for Strategic Reflection

**Their code:**
```python
# utils.py lines 162-191
@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    """
    return f"Reflection recorded: {reflection}"
```

**Our code:** We have `think.md` prompt but no tool implementation or usage.

---

### 4.5 Compression Uses Regex Instead of LLM Understanding

**Their code:**
```python
# deep_researcher.py lines 546-609
async def compress_research(state: ResearcherState, config: RunnableConfig):
    # They use full LLM compression
    messages = [SystemMessage(content=compression_prompt)] + researcher_messages
    response = await synthesizer_model.ainvoke(messages)
    return {"compressed_research": str(response.content), ...}
```

**Our code:**
```python
# behaviors.py lines 642-688
def _parse_compressed_findings(self, response: str) -> List[ResearchFinding]:
    # We rely on regex to parse expected format
    finding_pattern = re.compile(
        r"-\s*(.+?)\s*\[sources?:\s*([\d,\s]+)\]",
        re.IGNORECASE
    )
    # If pattern matching failed, create basic findings
    if not findings:
        # Fallback is very crude
```

---

### 4.6 No Webpage Summarization

**Their code:**
```python
# utils.py lines 152-178
async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    prompt_content = summarize_webpage_prompt.format(
        webpage_content=webpage_content,
        date=get_today_str()
    )
    summary = await asyncio.wait_for(
        model.ainvoke([HumanMessage(content=prompt_content)]),
        timeout=60.0
    )
    return f"<summary>\n{summary.summary}\n</summary>\n\n<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
```

**Our code:** We use raw Tavily snippets without summarization:
```python
# behaviors.py lines 274-284
source = ResearchSource(
    content_summary=result.content[:500] if result.content else "",  # Just truncate
    raw_content=result.raw_content or result.content,
)
```

---

### 4.7 No Structured Output with Retry

**Their pattern (used everywhere):**
```python
model = (
    configurable_model
    .with_structured_output(SomeModel)
    .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    .with_config(model_config)
)
```

**Our pattern:**
```python
# llm_service.py - No structured output, no retry
async def generate_json(self, prompt: str, ...):
    response = await self.generate(prompt=prompt, ...)
    # Just try to parse JSON, no retry
    return json.loads(content)
```

---

## 5. Configuration Comparison

| Their Config Field | Default | Our Equivalent | Our Default | Notes |
|-------------------|---------|----------------|-------------|-------|
| `max_structured_output_retries` | 3 | **NONE** | - | We don't retry |
| `allow_clarification` | True | **NONE** | - | We don't clarify |
| `max_concurrent_research_units` | 5 | `max_concurrent_researchers` | 5 | Match |
| `max_researcher_iterations` | 6 | **NONE** | - | Supervisor iterations |
| `max_react_tool_calls` | 10 | `max_tool_calls_per_researcher` | 10 | **But we don't enforce it** |
| `search_api` | Tavily | `search_provider` | "none" | Match concept |
| `summarization_model` | gpt-4.1-mini | **NONE** | - | No summarization |
| `max_content_length` | 50000 | **NONE** | - | No content length check |
| `research_model` | gpt-4.1 | Via user settings | - | Different mechanism |
| `compression_model` | gpt-4.1 | Via user settings | - | Different mechanism |
| `final_report_model` | gpt-4.1 | Via user settings | - | Different mechanism |

---

## 6. Graph/Workflow Comparison

### Their LangGraph Structure

```
START
  |
  v
clarify_with_user ----[needs_clarification]--> END (ask question)
  |
  [no clarification needed]
  v
write_research_brief
  |
  v
research_supervisor (subgraph)
  |  |
  |  +---> supervisor
  |         |
  |         v
  |      supervisor_tools
  |         |
  |         +--[ConductResearch]--> researcher_subgraph (parallel)
  |         |                          |
  |         |                          +---> researcher
  |         |                          |        |
  |         |                          |        v
  |         |                          |     researcher_tools
  |         |                          |        |
  |         |                          |        +--[loop up to max_react_tool_calls]
  |         |                          |        |
  |         |                          |        v
  |         |                          |     compress_research
  |         |                          |        |
  |         |                          +--------+
  |         |
  |         +--[ResearchComplete]--> END supervisor
  |
  v
final_report_generation
  |
  v
END
```

### Our Behavior Tree Structure

```
START
  |
  v
GenerateBriefBehavior (one-shot)
  |
  v
PlanSubtopicsBehavior (one-shot, just slices subtopics)
  |
  v
ParallelResearchersBehavior
  |
  +---> ResearcherBehavior.run_single() (ONE search per subtopic)
  |        |
  |        +---> _generate_search_queries()
  |        |
  |        +---> tavily.search_parallel() - ONE TIME
  |        |
  |        +---> _extract_findings() - ONE TIME
  |        |
  +--------+
  |
  v
CompressFindingsBehavior (one-shot, regex parsing)
  |
  v
GenerateReportBehavior (one-shot, no retry)
  |
  v
PersistToVaultBehavior (optional)
  |
  v
END
```

---

## 7. Critical Gaps Summary (Prioritized)

### P0 - Critical (Must Fix for Parity)

1. **Missing Clarification Phase** - Users with ambiguous queries get poor results
2. **Single Search Iteration** - Researchers don't iterate, miss follow-up queries
3. **No Token Overflow Handling** - Will crash on large reports

### P1 - High (Significantly Impacts Quality)

4. **No Supervisor Loop** - Can't dynamically adjust research strategy
5. **No think_tool** - No strategic reflection during research
6. **No Webpage Summarization** - Raw snippets instead of summarized content

### P2 - Medium (Quality of Life)

7. **Brittle Regex Compression** - Relies on specific format instead of LLM understanding
8. **No Structured Output Retry** - JSON parsing can fail silently
9. **Single Report Attempt** - No retry on generation failure

### P3 - Low (Nice to Have)

10. **No MCP Integration** - Feature gap, not quality issue
11. **No Native Search Support** - Anthropic/OpenAI web search APIs

---

## 8. Recommendations

### Immediate Actions

1. **Add researcher iteration loop** - Allow 5-10 search iterations per researcher
2. **Implement think_tool** - Strategic reflection between searches
3. **Add token overflow handling** - Progressive truncation like theirs

### Short-term Actions

4. **Add clarification phase** - Ask follow-up questions for ambiguous queries
5. **Implement webpage summarization** - Summarize long content before extraction
6. **Add structured output retry** - Use retry logic for JSON generation

### Medium-term Actions

7. **Implement supervisor loop** - Allow dynamic research strategy adjustment
8. **Replace regex compression** - Use LLM-based compression with full context
9. **Add model-specific token limits** - Track token usage across providers

---

*Document generated by code analysis on 2026-01-04*
