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
            Phase 2: Context Loading
            Load tree context or create new tree, load legacy context
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
        --]]
        BT.selector({
            -- Main agent loop - repeats until SUCCESS (done) or max turns
            BT.retry(30,
                BT.selector({
                    -- Check cancellation first
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

                    -- Check iteration budget exceeded
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
                        -- Budget warnings (non-blocking)
                        BT.always_succeed(
                            BT.action("check-iteration-budget", {
                                fn = "backend.src.bt.actions.oracle.check_iteration_budget",
                                description = "Emit warning at 70% of max turns"
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
        })
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
