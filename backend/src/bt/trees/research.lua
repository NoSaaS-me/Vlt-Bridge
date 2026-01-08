--[[
Research Tree - Deep Research BT subtree for multi-source research.

This tree orchestrates the complete research workflow:
1. Generate research brief from query
2. Plan subtopics for parallel research
3. Execute parallel researchers (via single-researcher subtree)
4. Compress findings into synthesized insights
5. Generate final research report
6. Persist to vault (optional)

Part of the BT Universal Runtime (spec 019).
Tasks covered: Phase 6.1 Research Migration

Blackboard Schema:
- Input: query, depth, config, save_to_vault, user_id, vault_path
- Artifacts: brief, researchers, sources, findings, report
- Progress: phase, pct, message, sources_found
- Output: research_id, vault_path, status, error

@author BT Migration
@version 1.0.0
]]

return BT.tree("research", {
    description = "Multi-source deep research with parallel investigators",
    version = "1.0.0",

    -- Contract specification
    contract = BT.contract({
        requires = {
            "query",           -- string: The research question
            "user_id",         -- string: User identifier
        },
        optional = {
            "depth",           -- string: quick/standard/thorough (default: standard)
            "save_to_vault",   -- boolean: Whether to persist (default: true)
            "vault_path",      -- string: Path to user vault
            "project_id",      -- string: Project context
            "max_sources",     -- number: Maximum sources to gather
        },
        produces = {
            "research_id",     -- string: Unique research identifier
            "report",          -- table: The research report
            "status",          -- string: Final status
        },
        on_violation = "log_and_fail",
    }),

    -- Root is a sequence of research phases
    root = BT.sequence({
        id = "research-main",
        description = "Main research workflow sequence",

        -- =====================================================================
        -- Phase 1: Initialization
        -- =====================================================================
        BT.action("init-research", {
            fn = "research.init_research",
            description = "Initialize research state and generate research_id",
        }),

        BT.action("emit-start-event", {
            fn = "research.emit_start_event",
            description = "Emit research.start event to ANS",
        }),

        -- =====================================================================
        -- Phase 2: Generate Research Brief
        -- =====================================================================
        BT.action("set-phase-brief", {
            fn = "research.set_phase",
            args = {
                phase = "brief",
                pct = 5,
                message = "Generating research brief...",
            },
        }),

        BT.llm_call("generate-brief", {
            fn = "research.generate_brief",
            description = "Transform query into structured research brief",
            model_key = "planning_model",
            prompt_template = "research/brief.md",
            output_key = "brief",
            max_tokens = 2000,
            temperature = 0.3,
            timeout = 60,
        }),

        -- Validate brief and create fallback if needed
        BT.selector({
            id = "brief-validation",
            description = "Ensure valid brief exists",

            -- Try to validate existing brief
            BT.condition("brief-valid", {
                fn = "research.validate_brief",
            }),

            -- Create fallback brief on validation failure
            BT.action("create-fallback-brief", {
                fn = "research.create_fallback_brief",
                description = "Create minimal brief from query",
            }),
        }),

        BT.action("set-phase-brief-complete", {
            fn = "research.set_phase",
            args = {
                phase = "brief",
                pct = 10,
                message = "Research brief generated",
            },
        }),

        -- =====================================================================
        -- Phase 3: Plan Subtopics
        -- =====================================================================
        BT.action("plan-subtopics", {
            fn = "research.plan_subtopics",
            description = "Create researcher assignments from brief subtopics",
        }),

        BT.action("set-phase-plan-complete", {
            fn = "research.set_phase",
            args = {
                phase = "planning",
                pct = 15,
                message = "Research plan created",
            },
        }),

        -- =====================================================================
        -- Phase 4: Parallel Research
        -- =====================================================================
        BT.action("set-phase-researching", {
            fn = "research.set_phase",
            args = {
                phase = "researching",
                pct = 20,
                message = "Starting parallel research...",
            },
        }),

        -- Parallel execution of researchers
        BT.parallel({
            id = "research-parallel",
            description = "Run researchers in parallel",
            policy = "require_all",
            on_child_fail = "continue",  -- Continue even if one researcher fails
            memory = true,

            -- ForEach over researchers array
            BT.for_each({
                id = "researchers-loop",
                collection_key = "researchers",
                item_key = "current_researcher",
                index_key = "researcher_index",

                -- Each researcher runs the single-researcher subtree
                BT.subtree_ref("single-researcher", {
                    bind = {
                        subtopic = {"current_researcher", "subtopic"},
                        max_tool_calls = {"current_researcher", "max_tool_calls"},
                        researcher_index = "researcher_index",
                    },
                }),
            }),
        }),

        -- Aggregate sources from all researchers
        BT.action("aggregate-sources", {
            fn = "research.aggregate_sources",
            description = "Collect all sources from researchers into unified list",
        }),

        BT.action("set-phase-research-complete", {
            fn = "research.set_phase",
            args = {
                phase = "researching",
                pct = 50,
            },
            -- Message will be dynamic based on sources found
        }),

        -- =====================================================================
        -- Phase 5: Compress Findings
        -- =====================================================================
        BT.action("set-phase-compressing", {
            fn = "research.set_phase",
            args = {
                phase = "compressing",
                pct = 60,
                message = "Synthesizing findings...",
            },
        }),

        -- Only compress if we have sources
        BT.selector({
            id = "compress-selector",
            description = "Compress findings if sources available",

            BT.sequence({
                id = "compress-with-sources",

                BT.condition("has-sources", {
                    fn = "research.has_sources",
                }),

                BT.llm_call("compress-findings", {
                    fn = "research.compress_findings",
                    description = "Synthesize findings from all sources",
                    model_key = "compression_model",
                    prompt_template = "research/compress.md",
                    output_key = "findings",
                    max_tokens = 5000,
                    temperature = 0.3,
                    timeout = 120,
                }),
            }),

            -- No sources - create empty findings
            BT.action("empty-findings", {
                fn = "research.create_empty_findings",
            }),
        }),

        BT.action("set-phase-compress-complete", {
            fn = "research.set_phase",
            args = {
                phase = "compressing",
                pct = 70,
                message = "Findings synthesized",
            },
        }),

        -- =====================================================================
        -- Phase 6: Generate Report
        -- =====================================================================
        BT.action("set-phase-generating", {
            fn = "research.set_phase",
            args = {
                phase = "generating",
                pct = 75,
                message = "Generating report...",
            },
        }),

        BT.llm_call("generate-report", {
            fn = "research.generate_report",
            description = "Create final research report with citations",
            model_key = "report_model",
            prompt_template = "research/report.md",
            output_key = "report",
            max_tokens = 15000,
            temperature = 0.5,
            timeout = 180,
        }),

        BT.action("set-phase-report-complete", {
            fn = "research.set_phase",
            args = {
                phase = "generating",
                pct = 90,
                message = "Report generated",
            },
        }),

        -- =====================================================================
        -- Phase 7: Vault Persistence (Conditional)
        -- =====================================================================
        BT.selector({
            id = "persist-selector",
            description = "Persist to vault if enabled",

            -- Persistence path
            BT.sequence({
                id = "persist-sequence",

                BT.condition("should-persist", {
                    fn = "research.should_persist",
                }),

                BT.action("set-phase-persisting", {
                    fn = "research.set_phase",
                    args = {
                        phase = "saving",
                        pct = 95,
                        message = "Saving to vault...",
                    },
                }),

                BT.action("persist-to-vault", {
                    fn = "research.persist_to_vault",
                    description = "Save research artifacts to user vault",
                }),
            }),

            -- Skip persistence
            BT.action("skip-persist", {
                fn = "research.noop",
            }),
        }),

        -- =====================================================================
        -- Phase 8: Finalization
        -- =====================================================================
        BT.action("set-phase-completed", {
            fn = "research.set_phase",
            args = {
                phase = "completed",
                pct = 100,
                message = "Research completed",
            },
        }),

        BT.action("emit-complete-event", {
            fn = "research.emit_complete_event",
            description = "Emit research.complete event to ANS",
        }),

        BT.action("finalize-research", {
            fn = "research.finalize",
            description = "Set output status and cleanup",
        }),
    }),
})
