--[[
    Agent Turn Subtree

    Single turn of the agent loop: LLM call, response handling, tool execution.
    Separated as subtree for clarity and potential reuse.

    Part of the BT Universal Runtime (spec 019).
--]]

return BT.tree("agent-turn", {
    description = "Single agent turn: LLM call, response processing, tool execution",

    blackboard = {
        -- Inherits from parent tree
    },

    root = BT.sequence({
        -- Build request body
        BT.action("build-llm-request", {
            fn = "src.bt.actions.oracle.build_llm_request",
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
            on_chunk = "src.bt.actions.oracle.on_llm_chunk"
        }),

        -- Update budget tracking
        BT.always_succeed(
            BT.action("update-token-budget", {
                fn = "src.bt.actions.oracle.update_token_budget",
                description = "Update tokens_used, check budget warnings"
            })
        ),

        -- Handle tool calls if present OR accumulate content
        -- This must happen BEFORE signal processing so signals are parsed
        -- from the actual accumulated content
        BT.selector({
            -- Native tool calls from LLM response
            BT.sequence({
                BT.condition("has-native-tool-calls", {
                    fn = "src.bt.conditions.oracle.has_tool_calls"
                }),
                BT.subtree_ref("execute-tools", { lazy = true, id = "exec-tools-native" })
            }),

            -- Check for XML tool calls in content (some models use this)
            BT.sequence({
                BT.action("extract-xml-tool-calls", {
                    fn = "src.bt.actions.oracle.extract_xml_tool_calls",
                    description = "Parse XML tool syntax from content/reasoning"
                }),
                BT.condition("has-xml-tool-calls", {
                    fn = "src.bt.conditions.oracle.has_tool_calls"
                }),
                BT.subtree_ref("execute-tools", { lazy = true, id = "exec-tools-xml" })
            }),

            -- No tool calls - accumulate content first
            BT.action("accumulate-content", {
                fn = "src.bt.actions.oracle.accumulate_content",
                description = "Add LLM response to accumulated_content"
            })
        }),

        --[[
            Signal Processing Phase (T024 - US2: Agent Self-Reflection via Signals)
            Parse, log, and strip XML signals from LLM response.
            This enables the agent to communicate its internal state.
            NOTE: Must happen AFTER content accumulation so we parse from
            the actual accumulated_content.
        --]]
        BT.always_succeed(
            BT.sequence({
                -- 1. Parse XML signal from accumulated content
                BT.action("parse-signal", {
                    fn = "src.bt.actions.signal_actions.parse_response_signal",
                    description = "Parse XML signal from LLM response"
                }),
                -- 2. Update signal state (consecutive reason tracking)
                BT.action("update-signal-state", {
                    fn = "src.bt.actions.signal_actions.update_signal_state",
                    description = "Track consecutive reasons for loop detection"
                }),
                -- 3. Log signal to ANS event bus (best-effort)
                BT.always_succeed(
                    BT.action("log-signal", {
                        fn = "src.bt.actions.signal_actions.log_signal",
                        description = "Emit signal event to ANS for audit"
                    })
                ),
                -- 4. Strip signal XML from user-visible content
                BT.action("strip-signal", {
                    fn = "src.bt.actions.signal_actions.strip_signal_from_response",
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
                    fn = "src.bt.conditions.signals.signal_type_is",
                    args = { expected_type = "stuck" }
                }),
                -- Let the response through - agent already acknowledged limitation
                BT.action("noop-stuck", {
                    fn = "src.bt.actions.oracle.noop",
                    description = "Stuck signal - let response complete"
                })
            }),

            -- If context_sufficient, ready for final answer
            BT.sequence({
                BT.condition("context-sufficient-signal", {
                    fn = "src.bt.conditions.signals.signal_type_is",
                    args = { expected_type = "context_sufficient" }
                }),
                -- Continue to tool handling / response completion
                BT.action("noop-context-sufficient", {
                    fn = "src.bt.actions.oracle.noop",
                    description = "Context sufficient - proceed to response"
                })
            }),

            -- Check for loop (3+ consecutive same reason)
            BT.sequence({
                BT.condition("consecutive-same-reason", {
                    fn = "src.bt.conditions.signals.consecutive_same_reason_gte",
                    args = { count = 3 }
                }),
                -- Agent is stuck in a loop - treat as stuck
                BT.action("noop-loop-detected", {
                    fn = "src.bt.actions.oracle.noop",
                    description = "Loop detected via signals - proceed to completion"
                })
            }),

            -- Default: proceed with normal tool handling
            BT.action("noop-continue", {
                fn = "src.bt.actions.oracle.noop",
                description = "No signal routing - continue normal flow"
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
                        fn = "src.bt.conditions.fallback.needs_fallback",
                        description = "Check if fallback should activate (no signal 3+ turns, low confidence, or stuck)"
                    }),
                    BT.action("trigger-fallback", {
                        fn = "src.bt.actions.fallback_actions.trigger_fallback",
                        description = "Run heuristic classification"
                    }),
                    BT.action("apply-fallback", {
                        fn = "src.bt.actions.fallback_actions.apply_heuristic_classification",
                        description = "Apply fallback action (inject hint, force response, or escalate)"
                    })
                }),
                -- No fallback needed - noop
                BT.action("noop-no-fallback", {
                    fn = "src.bt.actions.oracle.noop",
                    description = "No fallback trigger - continue normally"
                })
            })
        )
    })
})
