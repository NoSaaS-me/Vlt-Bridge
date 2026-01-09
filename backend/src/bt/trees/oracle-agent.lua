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

    -- Tree configuration
    max_turns = 30,
    iteration_warning_threshold = 0.70,
    token_warning_threshold = 0.80,
    context_warning_threshold = 0.70,

    root = BT.sequence({
        --[[
            Phase 1: Initialization
            Reset state, emit query start event
        --]]
        BT.action("reset-state", {
            fn = "backend.src.bt.actions.oracle.reset_state",
            description = "Reset cancellation, loop detection, budget tracking"
        }),

        BT.action("emit-query-start", {
            fn = "backend.src.bt.actions.oracle.emit_query_start",
            description = "Emit QUERY_START event for plugin system"
        }),

        --[[
            Phase 1.5: Context Assessment (US1 - Intelligent Context Selection)
            Classify query to determine what context sources are needed.
            This informs tool selection and prompt composition.
        --]]
        BT.action("analyze-query", {
            fn = "backend.src.bt.actions.query_analysis.analyze_query",
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
                    fn = "backend.src.bt.actions.oracle.load_tree_node",
                    description = "Load existing tree node from context_id"
                })
            }),
            -- Otherwise get or create tree for user/project
            BT.action("get-or-create-tree", {
                fn = "backend.src.bt.actions.oracle.get_or_create_tree",
                description = "Get active tree or create new one"
            })
        }),

        -- Load legacy context for backwards compatibility
        BT.always_succeed(
            BT.action("load-legacy-context", {
                fn = "backend.src.bt.actions.oracle.load_legacy_context",
                description = "Load OracleContextService context (fallback)"
            })
        ),

        -- Load cross-session notifications
        BT.always_succeed(
            BT.action("load-cross-session-notifications", {
                fn = "backend.src.bt.actions.oracle.load_cross_session_notifications",
                description = "Load ANS notifications persisted across sessions"
            })
        ),

        --[[
            Phase 3: Message Building
            Build system prompt, add history, inject notifications, get tools
        --]]
        BT.action("build-system-prompt", {
            fn = "backend.src.bt.actions.oracle.build_system_prompt",
            description = "Load vault files/threads, render system.md template"
        }),

        BT.action("add-tree-history", {
            fn = "backend.src.bt.actions.oracle.add_tree_history",
            description = "Walk root to current node, add exchanges to messages"
        }),

        BT.always_succeed(
            BT.action("inject-notifications", {
                fn = "backend.src.bt.actions.oracle.inject_notifications",
                description = "Inject cross-session notifications as system messages"
            })
        ),

        BT.action("add-user-question", {
            fn = "backend.src.bt.actions.oracle.add_user_question",
            description = "Add current user question to messages"
        }),

        BT.action("get-tool-schemas", {
            fn = "backend.src.bt.actions.oracle.get_tool_schemas",
            description = "Get tool definitions from tool_executor"
        }),

        BT.action("init-context-tracking", {
            fn = "backend.src.bt.actions.oracle.init_context_tracking",
            description = "Set max_context_tokens from model, estimate initial tokens"
        }),

        BT.action("yield-initial-context-update", {
            fn = "backend.src.bt.actions.oracle.yield_context_update",
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
                            fn = "backend.src.bt.actions.oracle.emit_done",
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
                            fn = "backend.src.bt.actions.oracle.emit_cancelled",
                            description = "Yield cancelled error chunk"
                        }),
                        BT.always_fail(
                            BT.action("noop", { fn = "backend.src.bt.actions.oracle.noop" })
                        )
                    }),

                    --[[
                        Budget Guard 2: Check if over budget (US3-AC2)
                        Uses budget.is_over_budget condition from config.
                        If over budget, force completion and exit.
                    --]]
                    BT.sequence({
                        BT.condition("is-over-budget", {
                            fn = "backend.src.bt.conditions.budget.is_over_budget"
                        }),
                        BT.action("force-completion-budget", {
                            fn = "backend.src.bt.actions.budget_actions.force_completion",
                            description = "Force completion when budget exceeded"
                        }),
                        BT.action("emit-done-budget", {
                            fn = "backend.src.bt.actions.oracle.emit_done",
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
                            fn = "backend.src.bt.conditions.loop_detection.is_stuck_loop"
                        }),
                        BT.action("emit-loop-warning", {
                            fn = "backend.src.bt.actions.budget_actions.emit_loop_warning",
                            description = "Emit agent.loop.detected ANS event"
                        }),
                        BT.action("force-completion-loop", {
                            fn = "backend.src.bt.actions.budget_actions.force_completion",
                            description = "Force completion when stuck in loop"
                        }),
                        BT.action("emit-done-loop", {
                            fn = "backend.src.bt.actions.oracle.emit_done",
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
                            fn = "backend.src.bt.actions.oracle.handle_max_turns_exceeded",
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
                                fn = "backend.src.bt.actions.budget_actions.check_budget_and_warn",
                                description = "Emit warning at 70% of max turns (via OracleConfig)"
                            })
                        ),

                        -- Drain turn_start notifications
                        BT.always_succeed(
                            BT.action("drain-turn-start-notifications", {
                                fn = "backend.src.bt.actions.oracle.drain_turn_start_notifications",
                                description = "Drain and yield turn_start notifications"
                            })
                        ),

                        -- Increment turn counter
                        BT.action("increment-turn", {
                            fn = "backend.src.bt.actions.oracle.increment_turn"
                        }),

                        -- Execute agent turn (LLM call + tool handling)
                        BT.subtree_ref("agent-turn", { lazy = true }),

                        -- Check if we got a final response (no tool calls)
                        BT.selector({
                            -- If we have tool calls, continue loop (return FAILURE)
                            BT.sequence({
                                BT.condition("has-tool-calls", {
                                    expression = "bb.tool_calls ~= nil and #bb.tool_calls > 0"
                                }),
                                -- Clear tool calls after processing
                                BT.action("clear-tool-calls", {
                                    fn = "backend.src.bt.actions.oracle.clear_tool_calls"
                                }),
                                -- Return FAILURE to continue loop
                                BT.always_fail(
                                    BT.action("continue-loop", { fn = "backend.src.bt.actions.oracle.noop" })
                                )
                            }),

                            -- Final response - save exchange and succeed
                            BT.sequence({
                                BT.action("save-exchange", {
                                    fn = "backend.src.bt.actions.oracle.save_exchange",
                                    description = "Save Q&A to tree and legacy context"
                                }),
                                BT.action("yield-sources", {
                                    fn = "backend.src.bt.actions.oracle.yield_sources",
                                    description = "Yield collected sources to frontend"
                                }),
                                BT.action("emit-done", {
                                    fn = "backend.src.bt.actions.oracle.emit_done",
                                    description = "Yield done chunk to complete response"
                                })
                                -- Returns SUCCESS to break loop
                            })
                        })
                    })
                })
            ),

            -- Fallback: max turns exceeded without explicit handling
            BT.sequence({
                BT.action("emit-iteration-exceeded", {
                    fn = "backend.src.bt.actions.oracle.emit_iteration_exceeded",
                    description = "Emit BUDGET_ITERATION_EXCEEDED event"
                }),
                BT.always_succeed(
                    BT.action("drain-immediate-notifications", {
                        fn = "backend.src.bt.actions.oracle.drain_immediate_notifications"
                    })
                ),
                BT.action("save-partial-exchange", {
                    fn = "backend.src.bt.actions.oracle.save_partial_exchange",
                    description = "Save partial exchange for recovery"
                }),
                BT.action("emit-done-with-warning", {
                    fn = "backend.src.bt.actions.oracle.emit_done_with_warning",
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
                fn = "backend.src.bt.actions.oracle.finalize_response",
                description = "Final cleanup and metrics"
            })
        )
    }),

    -- Finally block - runs on both success and failure
    finally = BT.sequence({
        BT.always_succeed(
            BT.action("save-partial-if-needed", {
                fn = "backend.src.bt.actions.oracle.save_partial_if_needed",
                description = "Save partial exchange if connection dropped"
            })
        ),
        BT.always_succeed(
            BT.action("emit-session-end", {
                fn = "backend.src.bt.actions.oracle.emit_session_end",
                description = "Emit SESSION_END event for plugin system"
            })
        )
    })
})


--[[
    Agent Turn Subtree

    Single turn of the agent loop: LLM call, response handling, tool execution.
    Separated as subtree for clarity and potential reuse.
--]]

BT.subtree("agent-turn", {
    description = "Single agent turn: LLM call, response processing, tool execution",

    blackboard = {
        -- Inherits from parent tree
    },

    root = BT.sequence({
        -- Build request body
        BT.action("build-llm-request", {
            fn = "backend.src.bt.actions.oracle.build_llm_request",
            description = "Build request with model, messages, tools, max_tokens"
        }),

        -- Make LLM call with streaming
        BT.llm_call({
            name = "agent-llm-call",
            model_key = "model",
            messages_key = "messages",
            tools_key = "tools",
            max_tokens_key = "max_tokens",
            stream_to = "partial_response",
            response_key = "llm_response",
            reasoning_key = "reasoning_content",
            tool_calls_key = "tool_calls",
            interruptible = true,
            timeout = 120000,  -- 2 minutes
            on_chunk = "backend.src.bt.actions.oracle.on_llm_chunk"
        }),

        -- Update budget tracking
        BT.always_succeed(
            BT.action("update-token-budget", {
                fn = "backend.src.bt.actions.oracle.update_token_budget",
                description = "Update tokens_used, check budget warnings"
            })
        ),

        --[[
            Signal Processing Phase (T024 - US2: Agent Self-Reflection via Signals)
            Parse, log, and strip XML signals from LLM response.
            This enables the agent to communicate its internal state.
        --]]
        BT.always_succeed(
            BT.sequence({
                -- 1. Parse XML signal from accumulated content
                BT.action("parse-signal", {
                    fn = "backend.src.bt.actions.signal_actions.parse_response_signal",
                    description = "Parse XML signal from LLM response"
                }),
                -- 2. Update signal state (consecutive reason tracking)
                BT.action("update-signal-state", {
                    fn = "backend.src.bt.actions.signal_actions.update_signal_state",
                    description = "Track consecutive reasons for loop detection"
                }),
                -- 3. Log signal to ANS event bus (best-effort)
                BT.always_succeed(
                    BT.action("log-signal", {
                        fn = "backend.src.bt.actions.signal_actions.log_signal",
                        description = "Emit signal event to ANS for audit"
                    })
                ),
                -- 4. Strip signal XML from user-visible content
                BT.action("strip-signal", {
                    fn = "backend.src.bt.actions.signal_actions.strip_signal_from_response",
                    description = "Remove signal XML from accumulated_content"
                })
            })
        ),

        --[[
            Signal-Based Routing (T024)
            Check parsed signal and route accordingly:
            - stuck: Let response through (agent acknowledged limitation)
            - context_sufficient: Ready for final answer
            - need_turn: Continue loop if budget allows
        --]]
        BT.selector({
            -- If stuck signal, acknowledge and let response through
            BT.sequence({
                BT.condition("is-stuck-signal", {
                    fn = "backend.src.bt.conditions.signals.signal_type_is",
                    args = { expected_type = "stuck" }
                }),
                -- Let the response through - agent already acknowledged limitation
                BT.action("noop-stuck", {
                    fn = "backend.src.bt.actions.oracle.noop",
                    description = "Stuck signal - let response complete"
                })
            }),

            -- If context_sufficient, ready for final answer
            BT.sequence({
                BT.condition("context-sufficient-signal", {
                    fn = "backend.src.bt.conditions.signals.signal_type_is",
                    args = { expected_type = "context_sufficient" }
                }),
                -- Continue to tool handling / response completion
                BT.action("noop-context-sufficient", {
                    fn = "backend.src.bt.actions.oracle.noop",
                    description = "Context sufficient - proceed to response"
                })
            }),

            -- Check for loop (3+ consecutive same reason)
            BT.sequence({
                BT.condition("consecutive-same-reason", {
                    fn = "backend.src.bt.conditions.signals.consecutive_same_reason_gte",
                    args = { count = 3 }
                }),
                -- Agent is stuck in a loop - treat as stuck
                BT.action("noop-loop-detected", {
                    fn = "backend.src.bt.actions.oracle.noop",
                    description = "Loop detected via signals - proceed to completion"
                })
            }),

            -- Default: proceed with normal tool handling
            BT.action("noop-continue", {
                fn = "backend.src.bt.actions.oracle.noop",
                description = "No signal routing - continue normal flow"
            })
        }),

        -- Handle tool calls if present
        BT.selector({
            -- Native tool calls from LLM response
            BT.sequence({
                BT.condition("has-native-tool-calls", {
                    expression = "bb.tool_calls ~= nil and #bb.tool_calls > 0"
                }),
                BT.subtree_ref("execute-tools", { lazy = true })
            }),

            -- Check for XML tool calls in content (some models use this)
            BT.sequence({
                BT.action("extract-xml-tool-calls", {
                    fn = "backend.src.bt.actions.oracle.extract_xml_tool_calls",
                    description = "Parse XML tool syntax from content/reasoning"
                }),
                BT.condition("has-xml-tool-calls", {
                    expression = "bb.tool_calls ~= nil and #bb.tool_calls > 0"
                }),
                BT.subtree_ref("execute-tools", { lazy = true })
            }),

            -- No tool calls - just accumulate content
            BT.action("accumulate-content", {
                fn = "backend.src.bt.actions.oracle.accumulate_content",
                description = "Add LLM response to accumulated_content"
            })
        }),

        --[[
            Fallback Check Phase (US5 - BERT Fallback for Edge Cases)
            After content accumulation/tool handling, check if fallback is needed.
            Triggers when: no signal 3+ turns OR confidence < 0.3 OR stuck signal.
            Wrapped in always_succeed so fallback doesn't break main loop.
        --]]
        BT.always_succeed(
            BT.selector({
                -- Check if fallback should trigger and apply if needed
                BT.sequence({
                    BT.condition("needs-fallback", {
                        fn = "backend.src.bt.conditions.fallback.needs_fallback",
                        description = "Check if fallback should activate (no signal 3+ turns, low confidence, or stuck)"
                    }),
                    BT.action("trigger-fallback", {
                        fn = "backend.src.bt.actions.fallback_actions.trigger_fallback",
                        description = "Run heuristic classification"
                    }),
                    BT.action("apply-fallback", {
                        fn = "backend.src.bt.actions.fallback_actions.apply_heuristic_classification",
                        description = "Apply fallback action (inject hint, force response, or escalate)"
                    })
                }),
                -- No fallback needed - noop
                BT.action("noop-no-fallback", {
                    fn = "backend.src.bt.actions.oracle.noop",
                    description = "No fallback trigger - continue normally"
                })
            })
        )
    })
})


--[[
    Tool Execution Subtree

    Executes tool calls with parallel support, loop detection, and ANS events.
--]]

BT.subtree("execute-tools", {
    description = "Execute tool calls with loop detection and parallel execution",

    root = BT.sequence({
        -- Loop detection
        BT.always_succeed(
            BT.sequence({
                BT.action("detect-loop", {
                    fn = "backend.src.bt.actions.oracle.detect_loop",
                    description = "Check for repeated tool call patterns"
                }),
                BT.selector({
                    BT.sequence({
                        BT.condition("loop-detected", {
                            expression = "bb.loop_detected == true"
                        }),
                        BT.action("emit-loop-event", {
                            fn = "backend.src.bt.actions.oracle.emit_loop_event",
                            description = "Emit AGENT_LOOP_DETECTED event, inject warning"
                        }),
                        BT.action("yield-loop-warning", {
                            fn = "backend.src.bt.actions.oracle.yield_loop_warning"
                        })
                    }),
                    BT.action("noop", { fn = "backend.src.bt.actions.oracle.noop" })
                })
            })
        ),

        -- Parse tool calls
        BT.action("parse-tool-calls", {
            fn = "backend.src.bt.actions.oracle.parse_tool_calls",
            description = "Extract call_id, name, arguments from each call"
        }),

        -- Yield pending status for each tool
        BT.action("yield-tool-pending", {
            fn = "backend.src.bt.actions.oracle.yield_tool_pending",
            description = "Yield tool_call chunks with status=pending"
        }),

        -- Inject project context for scoped tools
        BT.always_succeed(
            BT.action("inject-project-context", {
                fn = "backend.src.bt.actions.oracle.inject_project_context",
                description = "Add project_id to PROJECT_SCOPED_TOOLS"
            })
        ),

        -- Execute tools in parallel
        BT.parallel({
            policy = "require_all",
            merge_strategy = "collect",
            continue_on_failure = true
        }, {
            BT.for_each("tool_calls", {
                item_key = "current_tool",
                continue_on_failure = true,
                children = {
                    BT.action("execute-single-tool", {
                        fn = "backend.src.bt.actions.oracle.execute_single_tool",
                        description = "Execute tool and collect result"
                    })
                }
            })
        }),

        -- Process results and yield chunks
        BT.action("process-tool-results", {
            fn = "backend.src.bt.actions.oracle.process_tool_results",
            description = "Yield result chunks, emit events, extract sources"
        }),

        -- Update context tokens
        BT.always_succeed(
            BT.action("update-context-tokens", {
                fn = "backend.src.bt.actions.oracle.update_context_tokens",
                description = "Update context_tokens estimate after tool results"
            })
        ),

        -- Add tool results to messages
        BT.action("add-tool-results-to-messages", {
            fn = "backend.src.bt.actions.oracle.add_tool_results_to_messages",
            description = "Append tool results to conversation messages"
        }),

        -- Drain after_tool notifications
        BT.always_succeed(
            BT.action("drain-after-tool-notifications", {
                fn = "backend.src.bt.actions.oracle.drain_after_tool_notifications",
                description = "Drain and yield after_tool notifications"
            })
        )
    })
})
