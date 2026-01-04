# Large Result Hint Rule

**Rule ID**: `large-result-hint`
**File**: `backend/src/services/plugins/rules/large_result.toml`
**Core**: No (can be disabled)

## Purpose

Suggests summarization when tool results exceed 2000 characters, helping agents manage large outputs efficiently.

## Configuration

```toml
[rule]
id = "large-result-hint"
name = "Large Result Hint"
description = "Suggest summarization when tool results exceed 6 items"
version = "1.0.0"
trigger = "on_tool_complete"
priority = 50
enabled = true
core = false

[condition]
# Check if the result contains indicators of large result sets
# This checks for common patterns in result text
expression = "context.result is not None and context.result.success and len(context.result.result or '') > 2000"

[action]
type = "notify_self"
message = "Large result from {{ context.result.tool_name }}. Consider summarizing key findings rather than processing all items."
category = "info"
priority = "normal"
deliver_at = "after_tool"
```

## Behavior

### When It Fires

The rule evaluates after each successful tool completion (`on_tool_complete`). It fires when:
- `context.result` exists
- `context.result.success` is true
- Result text length exceeds 2000 characters

### Notification

When triggered, the agent receives an informational message:

```
Large result from vault_search. Consider summarizing key findings rather than processing all items.
```

The message is injected immediately `after_tool`, helping the agent decide how to process the result.

## Context Used

| Field | Type | Description |
|-------|------|-------------|
| `context.result` | ToolResult | Tool execution result |
| `context.result.success` | bool | Whether tool succeeded |
| `context.result.result` | str | Tool output text |
| `context.result.tool_name` | str | Name of the tool |

## Why Not Core?

This rule is marked as `core = false` because:
1. **Optional Optimization**: Not critical for agent safety
2. **User Preference**: Some users may want full results
3. **Context-Dependent**: Large results aren't always problematic

Users can disable this rule if they prefer to see full results without summarization hints.

## Customization

### Disable the Rule

```bash
curl -X POST http://localhost:8000/api/rules/large-result-hint/toggle \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

### Custom Threshold

Create a rule with a different threshold:

```toml
[rule]
id = "very-large-result"
name = "Very Large Result Warning"
trigger = "on_tool_complete"
priority = 60

[condition]
expression = "context.result is not None and context.result.success and len(context.result.result or '') > 5000"

[action]
type = "notify_self"
message = "Very large result ({{ len(context.result.result or '') }} chars). Strong recommendation to summarize."
category = "warning"
priority = "high"
deliver_at = "after_tool"
```

### Tool-Specific Rules

```toml
[rule]
id = "large-search-result"
name = "Large Search Result"
trigger = "on_tool_complete"
priority = 55

[condition]
expression = "context.result is not None and 'search' in context.result.tool_name and len(context.result.result or '') > 1500"

[action]
type = "notify_self"
message = "Search returned extensive results. Consider filtering or focusing on top matches."
category = "info"
deliver_at = "after_tool"
```

### Count-Based Detection

For structured results with item counts:

```toml
[rule]
id = "many-items-result"
name = "Many Items Result"
trigger = "on_tool_complete"
priority = 52

[condition]
# Look for patterns like "Found 25 results" or lists
expression = "context.result is not None and context.result.success and ('Found' in (context.result.result or '') or (context.result.result or '').count('\\n') > 20)"

[action]
type = "notify_self"
message = "Result contains many items. Consider processing in batches or extracting key items."
category = "info"
```

## Use Cases

### When This Rule Helps

1. **Search Results**: Long lists of search matches
2. **File Listings**: Large directory contents
3. **API Responses**: Verbose JSON/XML data
4. **Code Search**: Many grep/find results

### Example Scenario

```
User: "Find all Python files with 'auth' in them"
Tool: grep_search returns 3500 characters of matches
Rule fires: "Large result from grep_search. Consider summarizing key findings."
Agent: Instead of processing every match, summarizes the main patterns found
```

## Related Rules

- [token-budget-warning](./token-budget.md) - Budget management (large results consume tokens)
- [repeated-failure-warning](./repeated-failure.md) - May trigger if processing large results fails

## See Also

- [Hook Points](../hooks/lifecycle.md#on_tool_complete)
- [Context API](../context-api/reference.md#toolresult)
- [Action Types](../rules/actions.md#action-type-notify_self)
