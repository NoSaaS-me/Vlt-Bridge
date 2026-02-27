# Deep Research Implementation Plan

**Feature**: 016-deep-research
**Created**: 2026-01-04
**Status**: Implementation

## Research Findings Summary

### Quality Frameworks (from Deep Research Bench)

**RACE Framework** - Report Quality:
- **R**elevance: Topic and section relevance
- **A**nalytical depth: Not surface-level
- **C**omprehensiveness: Complete coverage
- **E**xpression: Readability, structure, flow

**FACT Framework** - Citation Quality:
- **F**idelity: Citation accuracy (source matches claim)
- **A**ttribution: Every claim has a source
- **C**overage: Uses diverse sources
- **T**raceability: URLs are valid and accessible

### DEFT Failure Taxonomy (Top Failures to Avoid)

| Failure Mode | Frequency | Mitigation |
|--------------|-----------|------------|
| Strategic Content Fabrication | 19% | Strict citation-first approach |
| Insufficient Info Acquisition | 16% | Multiple search queries, parallel researchers |
| Lack of Analytical Depth | 11% | Synthesis step with think tool |
| Verification Mechanism Failure | 9% | Cross-reference sources |

**Key Insight**: Agents struggle with **evidence integration**, not task comprehension. Focus on connecting sources to claims.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Oracle Agent                              │
│  (detects research request, enters research mode)           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               Research Orchestrator (Behavior Tree)          │
│  Sequence: Brief → Research → Compress → Report → Save      │
└─────────────────────────────┬───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  Researcher 1 │     │  Researcher 2 │     │  Researcher N │
│  (subtopic A) │     │  (subtopic B) │     │  (subtopic N) │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Source Accumulator                         │
│  Deduplicates, scores, organizes by relevance               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Compression Step                           │
│  Synthesizes findings, preserves citations                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Report Generator                           │
│  RACE-quality markdown with inline citations                 │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Vault Persister                            │
│  Saves to research/{id}/ folder structure                    │
└─────────────────────────────────────────────────────────────┘
```

## Vault Deliverable Structure

```
vault/
  research/
    {research-id}/                    # e.g., "2026-01-deep-learning-trends"
      index.md                        # Research project hub
      brief.md                        # Original question + scope
      report.md                       # Final synthesized report
      notes/
        source-001-{slug}.md          # Per-source notes
        source-002-{slug}.md
        ...
      methodology.md                  # Search queries, decisions
      sources.md                      # Bibliography with scores
```

### index.md Template
```markdown
---
title: "Research: {TOPIC}"
type: research-project
status: in-progress | completed
created: {DATE}
completed: {DATE}
tags: [research, deep-research, {topic-tags}]
condensed: |
  {2-3 sentence summary for AI context loading}
---

# {TOPIC}

**Brief**: [[brief]]
**Status**: {STATUS_EMOJI} {STATUS}

## Summary
{Executive summary}

## Report
→ [[report]]

## Sources
→ [[sources]] ({N} sources evaluated, {M} cited)

## Research Notes
{List of source note wikilinks}

## Methodology
→ [[methodology]]
```

## Implementation Phases

### Phase 1: Tavily API Integration (Backend)

**Files to create/modify**:
- `backend/src/services/tavily_service.py` (NEW)
- `backend/src/models/search.py` (extend)
- `backend/src/models/settings.py` (add search provider setting)
- `backend/pyproject.toml` (add tavily-python)

**TavilySearchService**:
```python
class TavilySearchService:
    async def search(
        self,
        queries: List[str],
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
    ) -> List[SearchResult]:
        """Execute parallel search queries with deduplication."""

    async def summarize_page(self, url: str, prompt: str) -> str:
        """Fetch and summarize a webpage."""
```

**User Setting**: `search_provider: "tavily" | "openrouter" | "none"`

### Phase 2: Research Mode Behavior Tree

**Files to create/modify**:
- `backend/src/services/research/orchestrator.py` (NEW)
- `backend/src/services/research/behaviors.py` (NEW)
- `backend/src/services/research/state.py` (NEW)

**Behavior Tree Structure**:
```python
research_tree = Sequence([
    # 1. Generate research brief from user query
    GenerateBriefBehavior(),

    # 2. Plan research subtopics
    PlanSubtopicsBehavior(),

    # 3. Execute parallel research
    ParallelResearchBehavior(max_concurrent=5),

    # 4. Accumulate and deduplicate sources
    AccumulateSourcesBehavior(),

    # 5. Compress findings
    CompressFindingsBehavior(),

    # 6. Generate report with citations
    GenerateReportBehavior(),

    # 7. Save to vault (optional)
    PersistToVaultBehavior(),
])
```

### Phase 3: Research Prompts

**Files to create**:
- `backend/prompts/research/brief.md`
- `backend/prompts/research/subtopics.md`
- `backend/prompts/research/researcher.md`
- `backend/prompts/research/compress.md`
- `backend/prompts/research/report.md`

**Key Prompt Patterns** (from open_deep_research):
- Include current date for temporal awareness
- Explicit tool call budgets
- Citation format requirements
- Language detection for reports

### Phase 4: Oracle Tool Integration

**New Oracle Tools**:
- `deep_research`: Trigger research mode with query
- `search_web_tavily`: Direct Tavily search (uses user's API key or OpenRouter)

**Tool Schema**:
```python
@dataclass
class DeepResearchTool:
    query: str  # Research topic/question
    depth: Literal["quick", "standard", "thorough"] = "standard"
    save_to_vault: bool = True
    output_folder: Optional[str] = None
```

### Phase 5: Frontend Integration

**Files to modify**:
- `frontend/src/pages/Settings.tsx` (add search provider setting)
- `frontend/src/types/oracle.ts` (research mode types)
- `frontend/src/components/ChatPanel.tsx` (research progress display)

**Research Progress Display**:
- Show research phases as they complete
- Display sources being gathered
- Stream report generation

## Search Provider Settings

### Option 1: Tavily (Recommended)
- User provides TAVILY_API_KEY
- Direct API access, best quality
- Cost: ~$0.01-0.05 per search

### Option 2: OpenRouter Fallback
- Uses user's existing OpenRouter API key
- Routes through Perplexity or similar via OpenRouter
- More expensive but uses existing key

### Option 3: None (Disabled)
- Research mode unavailable
- Falls back to basic web_search tool

## Quality Assurance

### Citation-First Approach
Every claim in the report MUST:
1. Be extracted from a specific source
2. Include inline citation `[1]`
3. Have corresponding entry in sources.md
4. Have URL that was verified accessible

### Anti-Fabrication Measures
1. **Source-locked generation**: Report generator only uses accumulated sources
2. **Think tool for planning**: Strategic reflection before searches
3. **Verification step**: Cross-reference claims across sources
4. **Transparency**: methodology.md logs all decisions

### Quality Metrics (Self-Evaluation)
Generate a quality assessment in report frontmatter:
```yaml
quality:
  comprehensiveness: 0.85  # 0-1
  analytical_depth: 0.72
  source_diversity: 0.90
  citation_density: 0.95  # claims with citations / total claims
```

## Token/Cost Management

### Budget Limits (Configurable)
| Phase | Default Limit |
|-------|---------------|
| Brief generation | 2K tokens |
| Per-researcher | 10K tokens |
| Compression | 5K tokens |
| Report generation | 15K tokens |
| **Total** | ~50K tokens |

### Compression Strategy
1. After each researcher completes, compress findings
2. Preserve all source URLs and key quotes
3. Remove redundant information
4. Keep within context window

## Dependencies

### Python (backend)
```toml
dependencies = [
    "tavily-python>=0.5.0",
    # Existing: httpx, pydantic, etc.
]
```

### No LangGraph
Use existing behavior tree implementation in:
- `backend/src/services/plugins/behavior_tree.py`

## Success Criteria

1. **Functional**: Can execute deep research queries end-to-end
2. **Quality**: Reports pass manual RACE framework review
3. **Citation**: >90% of claims have valid citations
4. **Performance**: Complete standard research in <10 minutes
5. **Persistence**: Research saved to vault correctly
6. **User Control**: Can configure search provider and limits

## Open Questions

1. Should we support research resume/continuation?
2. Do we need human-in-the-loop for expensive research?
3. How to handle Tavily API key management securely?
4. Should research mode be a separate Oracle context tree branch?
