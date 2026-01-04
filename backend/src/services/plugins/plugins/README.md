# Oracle Plugin System - Plugins Directory

This directory contains plugins for the Oracle agent. Each plugin is a subdirectory
with a `manifest.toml` file that defines the plugin's metadata, dependencies, and settings.

## Plugin Structure

```
plugins/
  my-plugin/
    manifest.toml       # Plugin manifest (required)
    rules/              # TOML rule definitions
      rule-one.toml
      rule-two.toml
    scripts/            # Lua scripts for complex logic
      helper.lua
```

## Creating a Plugin

### 1. Create a Plugin Directory

```bash
mkdir -p plugins/my-plugin/rules
```

### 2. Create the Manifest

Create `plugins/my-plugin/manifest.toml`:

```toml
[plugin]
id = "my-plugin"
name = "My Plugin"
version = "1.0.0"
description = "A collection of custom rules"

[capabilities]
requires = ["vault_search"]  # Optional: capabilities your plugin needs

[settings.threshold]
type = "float"
default = 0.8
min = 0.0
max = 1.0
description = "Threshold for triggering warnings"

[settings.enabled_feature]
type = "boolean"
default = true
description = "Enable the special feature"
```

### 3. Add Rules

Create rules in `plugins/my-plugin/rules/`:

```toml
# plugins/my-plugin/rules/my-rule.toml
[rule]
id = "my-rule"
name = "My Custom Rule"
description = "Does something useful"
trigger = "on_turn_start"
priority = 100

[condition]
expression = "context.turn.token_usage > 0.8"

[action]
type = "notify_self"
message = "Token budget is getting low!"
priority = "high"
```

## Manifest Reference

### [plugin] Section (Required)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | Yes | Unique identifier (kebab-case) |
| name | string | No | Display name (defaults to id) |
| version | string | No | Semantic version (default: "1.0.0") |
| description | string | No | What the plugin does |

### [capabilities] Section (Optional)

| Field | Type | Description |
|-------|------|-------------|
| requires | list[string] | Capabilities the plugin needs |

Available capabilities:
- `vault_search` - Search vault notes
- `vault_read` - Read vault notes
- `vault_write` - Write vault notes
- `web_search` - Web search
- `code_search` - CodeRAG search
- `thread_search` - Thread/memory search

### [settings.*] Sections (Optional)

Define user-configurable settings:

```toml
[settings.my_setting]
name = "My Setting"           # Display name
type = "integer"              # integer, float, string, boolean
default = 10                  # Default value
description = "What it does"  # Help text
min = 1                       # Minimum (numeric types)
max = 100                     # Maximum (numeric types)
options = ["a", "b", "c"]     # Valid values (string type)
```

## Rule Format

Rules within plugins use the same TOML format as standalone rules.
See `../rules/README.md` for the full rule reference.

Key difference: When a rule is loaded from a plugin, its `plugin_id`
is automatically set to the parent plugin's ID.

## Best Practices

1. **Use descriptive IDs**: Plugin and rule IDs should be clear and unique
2. **Document settings**: Provide helpful descriptions for all settings
3. **Declare dependencies**: List all required capabilities upfront
4. **Version properly**: Follow semantic versioning for compatibility
5. **Group related rules**: Put related rules in the same plugin
6. **Test thoroughly**: Use the `/api/rules/{id}/test` endpoint
