# Data Model: ChatGPT Integration

## Auth Entities

### ServiceTokenStrategy
Strategy for validating static service tokens.

| Field | Type | Description |
|-------|------|-------------|
| `token` | `str` | The static token to match against. |
| `user_id` | `str` | The user ID to impersonate (e.g. "demo-user"). |

## Configuration

### AppConfig Updates
New fields added to `AppConfig`.

| Field | Type | Description |
|-------|------|-------------|
| `chatgpt_service_token` | `Optional[str]` | Static token for Apps SDK auth. |
| `chatgpt_cors_origin` | `str` | Allowed CORS origin (default: `https://chatgpt.com`). |

## Tool Responses

### WidgetMeta
Structure of the `_meta` field in `CallToolResult`.

```json
{
  "openai": {
    "outputTemplate": "https://your-space.hf.space/widget",
    "toolInvocation": {
      "invoking": "Searching notes...",
      "invoked": "Found 3 notes."
    }
  }
}
```
