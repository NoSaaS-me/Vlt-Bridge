-- repeated_failure_detector.lua
-- Detects repeated failures of the same tool and suggests course correction
--
-- This script demonstrates complex temporal pattern matching that
-- exceeds the capabilities of simpleeval expressions.

-- Configuration thresholds
local FAILURE_THRESHOLD = 3
local RECENT_TOOLS_WINDOW = 5

-- Check if a specific tool has failed multiple times
local function get_failure_count(tool_name)
    local failures = context.history.failures
    if failures and failures[tool_name] then
        return failures[tool_name]
    end
    return 0
end

-- Analyze recent tool calls to find patterns
local function analyze_tool_patterns()
    local tools = context.history.tools
    if not tools then return {} end

    local patterns = {
        consecutive_failures = 0,
        same_tool_attempts = {},
        last_tool_name = nil
    }

    -- Count from end of list (most recent)
    local count = 0
    for i = #tools, 1, -1 do
        if count >= RECENT_TOOLS_WINDOW then break end

        local tool = tools[i]
        if tool then
            -- Track consecutive failures
            if not tool.success then
                if patterns.last_tool_name == nil or
                   patterns.last_tool_name == tool.name then
                    patterns.consecutive_failures = patterns.consecutive_failures + 1
                end
            else
                -- Reset consecutive counter on success
                patterns.consecutive_failures = 0
            end

            -- Track same-tool attempts
            patterns.same_tool_attempts[tool.name] =
                (patterns.same_tool_attempts[tool.name] or 0) + 1
            patterns.last_tool_name = tool.name
        end

        count = count + 1
    end

    return patterns
end

-- Check if we're in a failure loop
local function is_in_failure_loop()
    local total_failures = context.history.total_failures
    if total_failures < FAILURE_THRESHOLD then
        return false, nil
    end

    local patterns = analyze_tool_patterns()

    -- Check for consecutive failures
    if patterns.consecutive_failures >= FAILURE_THRESHOLD then
        return true, patterns.last_tool_name
    end

    -- Check for repeated failures of same tool
    for tool_name, count in pairs(patterns.same_tool_attempts) do
        if count >= FAILURE_THRESHOLD then
            local failures = get_failure_count(tool_name)
            if failures >= 2 then  -- At least 2 actual failures
                return true, tool_name
            end
        end
    end

    return false, nil
end

-- Build appropriate message based on context
local function build_message(tool_name)
    local base_msg = "Repeated failures detected"
    if tool_name then
        base_msg = base_msg .. " with " .. tool_name
    end

    -- Add specific suggestions based on tool
    local suggestions = {}

    if tool_name == "vault_search" then
        table.insert(suggestions, "Try broader search terms")
        table.insert(suggestions, "Check if the vault contains the expected content")
    elseif tool_name == "web_search" then
        table.insert(suggestions, "Rephrase the search query")
        table.insert(suggestions, "Check network connectivity")
    elseif tool_name == "file_read" then
        table.insert(suggestions, "Verify the file path exists")
        table.insert(suggestions, "Check file permissions")
    else
        table.insert(suggestions, "Consider a different approach")
        table.insert(suggestions, "Review error messages for clues")
    end

    -- Format suggestions
    if #suggestions > 0 then
        base_msg = base_msg .. ". Suggestions: "
        base_msg = base_msg .. table.concat(suggestions, "; ")
    end

    return base_msg
end

-- Main execution
local in_loop, failing_tool = is_in_failure_loop()

if in_loop then
    return {
        type = "notify_self",
        message = build_message(failing_tool),
        category = "warning",
        priority = "high"
    }
end

-- No action needed
return nil
