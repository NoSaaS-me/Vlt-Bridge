--[[
Single Researcher Subtree - Research a single subtopic.

This subtree handles research for one subtopic:
1. Generate search queries from subtopic
2. Execute search (Tavily with OpenRouter fallback)
3. Convert results to sources
4. Extract key findings from sources via LLM

Invoked from research.lua via parallel for_each over researchers array.

Part of the BT Universal Runtime (spec 019).
Tasks covered: Phase 6.1 Research Migration

Blackboard Schema (inherited from parent + local):
- Input: subtopic, max_tool_calls, researcher_index
- Local: queries, search_results, sources, extracted_findings
- Output: Updated researcher state in parent's researchers array

@author BT Migration
@version 1.0.0
]]

return BT.tree("single-researcher", {
    description = "Research a single subtopic with search and extraction",
    version = "1.0.0",

    -- Contract specification
    contract = BT.contract({
        requires = {
            "subtopic",        -- string: The subtopic to research
        },
        optional = {
            "max_tool_calls",  -- number: Maximum search calls (default: 10)
            "researcher_index", -- number: Index in researchers array
            "brief",           -- table: Research brief for context
        },
        produces = {
            "researcher_sources",  -- table[]: Sources found by this researcher
            "researcher_error",    -- string: Error if failed
            "researcher_completed", -- boolean: Completion status
        },
        on_violation = "log_and_continue",  -- Don't fail whole research for one researcher
    }),

    root = BT.sequence({
        id = "researcher-main",
        description = "Single researcher workflow",

        -- =====================================================================
        -- Step 1: Initialize Researcher State
        -- =====================================================================
        BT.action("init-researcher", {
            fn = "research.init_researcher",
            description = "Initialize researcher-local state",
        }),

        BT.action("emit-researcher-start", {
            fn = "research.emit_researcher_event",
            args = {
                event_type = "start",
            },
        }),

        -- =====================================================================
        -- Step 2: Generate Search Queries
        -- =====================================================================
        BT.action("generate-queries", {
            fn = "research.generate_search_queries",
            description = "Generate targeted search queries for subtopic",
        }),

        -- =====================================================================
        -- Step 3: Execute Search (with fallback)
        -- =====================================================================
        -- Retry wrapper for transient failures
        BT.retry({
            id = "search-retry",
            max_attempts = 3,
            backoff_ms = 1000,

            BT.selector({
                id = "search-selector",
                description = "Try search providers with fallback",

                -- Option 1: Tavily Search
                BT.sequence({
                    id = "tavily-search",

                    BT.condition("tavily-available", {
                        fn = "research.has_tavily",
                    }),

                    BT.action("search-tavily", {
                        fn = "research.search_tavily",
                        description = "Search using Tavily API",
                        timeout = 30,
                    }),
                }),

                -- Option 2: OpenRouter Perplexity Search
                BT.sequence({
                    id = "openrouter-search",

                    BT.condition("openrouter-available", {
                        fn = "research.has_openrouter_search",
                    }),

                    BT.action("search-openrouter", {
                        fn = "research.search_openrouter",
                        description = "Search using OpenRouter Perplexity",
                        timeout = 30,
                    }),
                }),

                -- Option 3: LLM Knowledge Fallback (no external search)
                BT.action("search-llm-fallback", {
                    fn = "research.search_llm_fallback",
                    description = "Use LLM knowledge as fallback",
                    timeout = 60,
                }),
            }),
        }),

        -- =====================================================================
        -- Step 4: Convert Search Results to Sources
        -- =====================================================================
        BT.action("convert-to-sources", {
            fn = "research.convert_search_results",
            description = "Convert raw search results to ResearchSource objects",
        }),

        -- =====================================================================
        -- Step 5: Extract Findings (conditional on having sources)
        -- =====================================================================
        BT.selector({
            id = "extract-selector",
            description = "Extract findings if sources available",

            BT.sequence({
                id = "extract-with-sources",

                BT.condition("researcher-has-sources", {
                    fn = "research.researcher_has_sources",
                }),

                BT.llm_call("extract-findings", {
                    fn = "research.extract_findings",
                    description = "Extract key quotes and findings from sources",
                    model_key = "research_model",
                    max_tokens = 2000,
                    temperature = 0.2,
                    timeout = 60,
                }),
            }),

            -- No sources - skip extraction
            BT.action("skip-extraction", {
                fn = "research.noop",
            }),
        }),

        -- =====================================================================
        -- Step 6: Mark Researcher Complete
        -- =====================================================================
        BT.action("mark-researcher-complete", {
            fn = "research.mark_researcher_complete",
            description = "Update researcher state to completed",
        }),

        BT.action("emit-researcher-complete", {
            fn = "research.emit_researcher_event",
            args = {
                event_type = "complete",
            },
        }),

        -- =====================================================================
        -- Step 7: Update Progress
        -- =====================================================================
        BT.action("update-research-progress", {
            fn = "research.update_researcher_progress",
            description = "Update parent research progress based on researcher completion",
        }),
    }),

    -- Error handler for researcher failures
    on_failure = BT.sequence({
        id = "researcher-failure-handler",

        BT.action("capture-researcher-error", {
            fn = "research.capture_researcher_error",
            description = "Capture error details for this researcher",
        }),

        BT.action("emit-researcher-error", {
            fn = "research.emit_researcher_event",
            args = {
                event_type = "error",
            },
        }),

        -- Mark as completed even on failure (prevents retry loops)
        BT.action("mark-researcher-failed", {
            fn = "research.mark_researcher_failed",
        }),
    }),
})
