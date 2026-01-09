--[[
Oracle Agent Behavior Tree

Main agent tree for Oracle chat functionality. Maps the oracle_agent.py
query() method to behavior tree semantics.

Migration from: backend/src/services/oracle_agent.py (~2,765 lines)
Target: ~300-500 lines Lua DSL

Phases:
1. Initialization - Reset state, emit events
2. Context Loading - Load tree/legacy context
3. Message Building - Build system prompt, history, tools
4. Agent Loop - Repeater with budget check, LLM calls, tool execution
5. Finalization - Save exchange, emit done

Part of the BT Universal Runtime (spec 019).
Tasks covered: 5.1.1-5.1.5 from tasks.md
--]]

return BT.tree("oracle-agent", {
    description = "Main Oracle agent for AI chat with tool calling",

    -- Blackboard schema for type validation
    blackboard = {
        -- Input parameters
        query = "OracleQuery",
        user_id = "str",
        project_id = "str",
        context_id = "str",
        model = "str",
        max_tokens = "int",

        -- State tracking
        turn = "int",
        cancelled = "bool",
        messages = "list",
        tools = "list",
        tool_calls = "list",
        tool_results = "list",

        -- Budget tracking
        tokens_used = "int",
        max_context_tokens = "int",
        context_tokens = "int",
        iteration_warning_emitted = "bool",
        token_warning_emitted = "bool",

        -- Context state
        tree_root_id = "str",
        current_node_id = "str",
        context = "OracleContext",
        message_history = "list",

        -- Output state
        accumulated_content = "str",
        reasoning_content = "str",
        partial_response = "str",
        collected_sources = "list",
        system_messages = "list",
        pending_notifications = "list",

        -- Loop detection
        recent_tool_patterns = "list",
        loop_detected = "bool",
        loop_warning = "str",

        -- Query Classification (US1 - Intelligent Context Selection)
        query_classification = "dict",  -- {query_type, needs_code, needs_vault, needs_web, confidence}
        needs_code = "bool",
        needs_vault = "bool",
        needs_web = "bool",

        -- Signal state tracking (T023 - US2: Agent Self-Reflection via Signals)
        -- Per data-model.md AgentSignalState entity
        last_signal = "dict",              -- Most recent parsed signal {type, confidence, fields, raw_xml}
        signals_emitted = "list",          -- All signals this session
        consecutive_same_reason = "int",   -- Loop detection counter (3+ = stuck)
        turns_without_signal = "int",      -- Fallback trigger counter (3+ = BERT fallback)

        -- Budget enforcement (T030 - US3: Budget and Loop Enforcement)
        -- Per tasks-expanded-us3.md
        force_complete = "bool",           -- Force agent to complete on next check
        loop_warning_emitted = "bool",     -- Loop warning ANS event emitted
        iteration_exceeded_emitted = "bool", -- Budget exceeded ANS event emitted
    },

    -- Note: Budget config (max_turns, etc.) comes from OracleConfig in Python
    -- These values are accessed via src.services.config.get_oracle_config()

    root = BT.sequence({
        --[[
            Phase 1: Initialization
            Reset state, emit query start event
        --]]
        BT.action("reset-state", {
            fn = "src.bt.actions.oracle.reset_state",
            description = "Reset cancellation, loop detection, budget tracking"
        }),

        BT.action("emit-query-start", {
            fn = "src.bt.actions.oracle.emit_query_start",
            description = "Emit QUERY_START event for plugin system"
        }),

        --[[
            Phase 1.5: Context Assessment (US1 - Intelligent Context Selection)
            Classify query to determine what context sources are needed.
            This informs tool selection and prompt composition.
        --]]
        BT.action("analyze-query", {
            fn = "src.bt.actions.query_analysis.analyze_query",
            description = "Classify query type and determine context needs"
        }),

        --[[
            Phase 2: Context Loading
            Load tree context or create new tree, load legacy context
            Note: Context needs are now in bb.needs_code, bb.needs_vault, bb.needs_web
        --]]
        BT.selector({
            -- Try to load existing context if context_id provided
            BT.sequence({
                BT.condition("has-context-id", {
                    expression = "bb.context_id ~= nil and bb.context_id ~= ''"
                }),
                BT.action("load-tree-node", {
                    fn = "src.bt.actions.oracle.load_tree_node",
                    description = "Load existing tree node from context_id"
                })
            }),
            -- Otherwise get or create tree for user/project
            BT.action("get-or-create-tree", {
                fn = "src.bt.actions.oracle.get_or_create_tree",
                description = "Get active tree or create new one"
            })
        }),

        -- Load legacy context for backwards compatibility
        BT.always_succeed(
            BT.action("load-legacy-context", {
                fn = "src.bt.actions.oracle.load_legacy_context",
                description = "Load OracleContextService context (fallback)"
            })
        ),

        -- Load cross-session notifications
        BT.always_succeed(
            BT.action("load-cross-session-notifications", {
                fn = "src.bt.actions.oracle.load_cross_session_notifications",
                description = "Load ANS notifications persisted across sessions"
            })
        ),

        --[[
            Phase 3: Message Building
            Build system prompt, add history, inject notifications, get tools
        --]]
        BT.action("build-system-prompt", {
            fn = "src.bt.actions.oracle.build_system_prompt",
            description = "Load vault files/threads, render system.md template"
        }),

        BT.action("add-tree-history", {
            fn = "src.bt.actions.oracle.add_tree_history",
            description = "Walk root to current node, add exchanges to messages"
        }),

        BT.always_succeed(
            BT.action("inject-notifications", {
                fn = "src.bt.actions.oracle.inject_notifications",
                description = "Inject cross-session notifications as system messages"
            })
        ),

        BT.action("add-user-question", {
            fn = "src.bt.actions.oracle.add_user_question",
            description = "Add current user question to messages"
        }),

        BT.action("get-tool-schemas", {
            fn = "src.bt.actions.oracle.get_tool_schemas",
            description = "Get tool definitions from tool_executor"
        }),

        BT.action("init-context-tracking", {
            fn = "src.bt.actions.oracle.init_context_tracking",
            description = "Set max_context_tokens from model, estimate initial tokens"
        }),

        BT.action("yield-initial-context-update", {
            fn = "src.bt.actions.oracle.yield_context_update",
            description = "Yield initial context_update chunk to frontend"
        }),

        --[[
            Phase 4: Agent Loop
            Repeater with budget check, agent turns until done or max turns

            Per oracle agent: MAX_TURNS = 30, uses tool calling loop

            Budget Guards (T030 - US3: Budget and Loop Enforcement):
            1. Check if forced to complete (from previous stuck/budget detection)
            2. Check if over budget (turn >= max_turns)
            3. Check if stuck in loop (consecutive_same_reason >= 3)
            4. Normal agent turn processing
        --]]
        BT.selector({
            -- Main agent loop - repeats until SUCCESS (done) or max turns
            BT.retry(30,
                BT.selector({
                    --[[
                        Budget Guard 1: Check if forced to complete
                        If force_complete flag is set, emit done and exit loop.
                        This flag is set by force_completion() action.
                    --]]
                    BT.sequence({
                        BT.condition("is-force-complete", {
                            expression = "bb.force_complete == true"
                        }),
                        BT.action("emit-done-forced", {
                            fn = "src.bt.actions.oracle.emit_done",
                            description = "Yield done chunk when force_complete is set"
                        })
                        -- Returns SUCCESS to break the loop
                    }),

                    -- Check cancellation
                    BT.sequence({
                        BT.condition("is-cancelled", {
                            expression = "bb.cancelled == true"
                        }),
                        BT.action("emit-cancelled", {
                            fn = "src.bt.actions.oracle.emit_cancelled",
                            description = "Yield cancelled error chunk"
                        }),
                        BT.always_fail(
                            BT.action("noop", { fn = "src.bt.actions.oracle.noop" })
                        )
                    }),

                    --[[
                        Budget Guard 2: Check if over budget (US3-AC2)
                        Uses budget.is_over_budget condition from config.
                        If over budget, force completion and exit.
                    --]]
                    BT.sequence({
                        BT.condition("is-over-budget", {
                            fn = "src.bt.conditions.budget.is_over_budget"
                        }),
                        BT.action("force-completion-budget", {
                            fn = "src.bt.actions.budget_actions.force_completion",
                            description = "Force completion when budget exceeded"
                        }),
                        BT.action("emit-done-budget", {
                            fn = "src.bt.actions.oracle.emit_done",
                            description = "Yield done chunk after budget-forced completion"
                        })
                        -- Returns SUCCESS to break the loop
                    }),

                    --[[
                        Budget Guard 3: Check for stuck loop (US3-AC3)
                        Uses loop_detection.is_stuck_loop condition.
                        If stuck in loop (3+ same reason signals), force completion.
                    --]]
                    BT.sequence({
                        BT.condition("is-stuck-loop", {
                            fn = "src.bt.conditions.loop_detection.is_stuck_loop"
                        }),
                        BT.action("emit-loop-warning", {
                            fn = "src.bt.actions.budget_actions.emit_loop_warning",
                            description = "Emit agent.loop.detected ANS event"
                        }),
                        BT.action("force-completion-loop", {
                            fn = "src.bt.actions.budget_actions.force_completion",
                            description = "Force completion when stuck in loop"
                        }),
                        BT.action("emit-done-loop", {
                            fn = "src.bt.actions.oracle.emit_done",
                            description = "Yield done chunk after loop-forced completion"
                        })
                        -- Returns SUCCESS to break the loop
                    }),

                    -- Legacy: Check iteration budget exceeded (fallback)
                    BT.sequence({
                        BT.condition("iteration-exceeded", {
                            expression = "bb.turn >= 30"
                        }),
                        BT.action("handle-max-turns-exceeded", {
                            fn = "src.bt.actions.oracle.handle_max_turns_exceeded",
                            description = "Emit iteration_exceeded event, save partial, yield done"
                        })
                        -- This returns SUCCESS to break the loop
                    }),

                    -- Normal agent turn
                    BT.sequence({
                        -- Budget warnings (non-blocking, T030)
                        -- Uses budget_actions.emit_budget_warning which checks threshold
                        BT.always_succeed(
                            BT.action("check-iteration-budget", {
                                fn = "src.bt.actions.budget_actions.check_budget_and_warn",
                                description = "Emit warning at 70% of max turns (via OracleConfig)"
                            })
                        ),

                        -- Drain turn_start notifications
                        BT.always_succeed(
                            BT.action("drain-turn-start-notifications", {
                                fn = "src.bt.actions.oracle.drain_turn_start_notifications",
                                description = "Drain and yield turn_start notifications"
                            })
                        ),

                        -- Increment turn counter
                        BT.action("increment-turn", {
                            fn = "src.bt.actions.oracle.increment_turn"
                        }),

                        -- Execute agent turn (LLM call + tool handling)
                        BT.subtree_ref("agent-turn", { lazy = true }),

                        -- Check if we got a final response (no tool calls)
                        -- Use Python condition function to reliably check tool_calls list
                        BT.selector({
                            -- Branch 1: No tool calls - save and exit loop
                            BT.sequence({
                                BT.condition("no-tool-calls", {
                                    fn = "src.bt.conditions.oracle.no_tool_calls"
                                }),
                                BT.action("save-exchange", {
                                    fn = "src.bt.actions.oracle.save_exchange",
                                    description = "Save Q&A to tree and legacy context"
                                }),
                                BT.action("yield-sources", {
                                    fn = "src.bt.actions.oracle.yield_sources",
                                    description = "Yield collected sources to frontend"
                                }),
                                BT.action("emit-done", {
                                    fn = "src.bt.actions.oracle.emit_done",
                                    description = "Yield done chunk to complete response"
                                })
                                -- Returns SUCCESS to break loop
                            }),

                            -- Branch 2: Has tool calls - clear and fail to continue loop
                            BT.sequence({
                                BT.condition("has-tool-calls", {
                                    fn = "src.bt.conditions.oracle.has_tool_calls"
                                }),
                                -- Clear tool calls after processing
                                BT.action("clear-tool-calls", {
                                    fn = "src.bt.actions.oracle.clear_tool_calls"
                                }),
                                -- Return FAILURE to continue loop via retry
                                BT.always_fail(
                                    BT.action("continue-loop", { fn = "src.bt.actions.oracle.noop" })
                                )
                            })
                        })
                    })
                })
            ),

            -- Fallback: max turns exceeded without explicit handling
            BT.sequence({
                BT.action("emit-iteration-exceeded", {
                    fn = "src.bt.actions.oracle.emit_iteration_exceeded",
                    description = "Emit BUDGET_ITERATION_EXCEEDED event"
                }),
                BT.always_succeed(
                    BT.action("drain-immediate-notifications", {
                        fn = "src.bt.actions.oracle.drain_immediate_notifications"
                    })
                ),
                BT.action("save-partial-exchange", {
                    fn = "src.bt.actions.oracle.save_partial_exchange",
                    description = "Save partial exchange for recovery"
                }),
                BT.action("emit-done-with-warning", {
                    fn = "src.bt.actions.oracle.emit_done_with_warning",
                    description = "Yield done chunk with max_turns warning"
                })
            })
        }),

        --[[
            Phase 5: Finalization
            This runs after main loop completes (success or failure)
            Note: actual cleanup is in finally block below
        --]]
        BT.always_succeed(
            BT.action("finalize-response", {
                fn = "src.bt.actions.oracle.finalize_response",
                description = "Final cleanup and metrics"
            })
        ),

        -- Finalization actions (previously in finally block)
        BT.always_succeed(
            BT.action("save-partial-if-needed", {
                fn = "src.bt.actions.oracle.save_partial_if_needed",
                description = "Save partial exchange if connection dropped"
            })
        ),
        BT.always_succeed(
            BT.action("emit-session-end", {
                fn = "src.bt.actions.oracle.emit_session_end",
                description = "Emit SESSION_END event for plugin system"
            })
        )
    })
})

-- Subtrees (agent-turn, execute-tools) are defined in separate files
-- and loaded via the TreeRegistry. They are referenced via BT.subtree_ref()
-- with lazy=true for runtime resolution.
