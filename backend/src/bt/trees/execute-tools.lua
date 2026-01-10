--[[
    Tool Execution Subtree

    Executes tool calls with parallel support, loop detection, and ANS events.

    Part of the BT Universal Runtime (spec 019).
--]]

return BT.tree("execute-tools", {
    description = "Execute tool calls with loop detection and parallel execution",

    root = BT.sequence({
        -- Parse tool calls (moved before loop detection)
        BT.action("parse-tool-calls", {
            fn = "src.bt.actions.oracle.parse_tool_calls",
            description = "Extract call_id, name, arguments from each call"
        }),

        -- Yield pending status for each tool
        BT.action("yield-tool-pending", {
            fn = "src.bt.actions.oracle.yield_tool_pending",
            description = "Yield tool_call chunks with status=pending"
        }),

        -- Inject project context for scoped tools
        BT.always_succeed(
            BT.action("inject-project-context", {
                fn = "src.bt.actions.oracle.inject_project_context",
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
                        fn = "src.bt.actions.oracle.execute_single_tool",
                        description = "Execute tool and collect result"
                    })
                }
            })
        }),

        -- Process results and yield chunks
        BT.action("process-tool-results", {
            fn = "src.bt.actions.oracle.process_tool_results",
            description = "Yield result chunks, emit events, extract sources"
        }),

        -- Loop detection (AFTER tools execute and return results)
        -- This ensures we only detect loops based on executed tool patterns,
        -- not just requested tools. The agent gets a chance to see results
        -- before we decide it's stuck in a loop.
        BT.always_succeed(
            BT.sequence({
                BT.action("detect-loop", {
                    fn = "src.bt.actions.oracle.detect_loop",
                    description = "Check for repeated tool call patterns (after execution)"
                }),
                BT.selector({
                    BT.sequence({
                        BT.condition("loop-detected", {
                            expression = "bb.loop_detected == true"
                        }),
                        BT.action("emit-loop-event", {
                            fn = "src.bt.actions.oracle.emit_loop_event",
                            description = "Emit AGENT_LOOP_DETECTED event, inject warning"
                        }),
                        BT.action("yield-loop-warning", {
                            fn = "src.bt.actions.oracle.yield_loop_warning"
                        })
                    }),
                    BT.action("noop", { fn = "src.bt.actions.oracle.noop" })
                })
            })
        ),

        -- Update context tokens
        BT.always_succeed(
            BT.action("update-context-tokens", {
                fn = "src.bt.actions.oracle.update_context_tokens",
                description = "Update context_tokens estimate after tool results"
            })
        ),

        -- Add tool results to messages
        BT.action("add-tool-results-to-messages", {
            fn = "src.bt.actions.oracle.add_tool_results_to_messages",
            description = "Append tool results to conversation messages"
        }),

        -- Drain after_tool notifications
        BT.always_succeed(
            BT.action("drain-after-tool-notifications", {
                fn = "src.bt.actions.oracle.drain_after_tool_notifications",
                description = "Drain and yield after_tool notifications"
            })
        )
    })
})
