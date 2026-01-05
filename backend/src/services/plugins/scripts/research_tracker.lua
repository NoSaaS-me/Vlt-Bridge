-- Track search tool usage and fire when threshold reached
local SEARCH_TOOLS = {
    vault_search = true,
    web_search = true,
    github_search = true,
    thread_seek = true
}

-- Get current count from state
local search_count = context.state.get("search_count") or 0

-- Check if last completed tool was a search
local tools = context.history.tools
if #tools > 0 then
    local last_tool = tools[#tools]
    if SEARCH_TOOLS[last_tool.name] then
        search_count = search_count + 1
        context.state.set("search_count", search_count)
    end
end

-- Fire every 5 searches
if search_count > 0 and search_count % 5 == 0 then
    -- Make count available to action template
    state.search_count = search_count
    return true
end

return false
