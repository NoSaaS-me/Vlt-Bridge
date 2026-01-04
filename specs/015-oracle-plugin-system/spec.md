# Feature Specification: Oracle Plugin System

**Feature Branch**: `015-oracle-plugin-system`
**Created**: 2026-01-04
**Status**: Draft
**Input**: Rule engine and plugin architecture built on ANS that enables reactive and proactive agent behaviors with TOML rules, Lua scripting, and extensible hook points

## Overview

The Oracle Plugin System extends the Agent Notification System (ANS) with a rule engine that enables both reactive and proactive agent behaviors. Where ANS provides event-driven notifications, the Plugin System adds conditional logic, custom actions, and an extensible plugin architecture.

The system follows a tiered complexity model: 80% of use cases are handled by simple TOML rule definitions, while complex logic can escape to a scripting language. This allows operators to configure behavior without coding, while power users can implement sophisticated patterns like multi-step research workflows or agent coordination.

The architecture draws inspiration from game bot frameworks (Honorbuddy, Onyx) that separate core abstractions (pathfinding, inventory) from user-implementable logic (quest completion). Here, the Oracle core handles tool execution, context management, and event routing, while plugins implement domain-specific behaviors.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Simple Threshold Rules (Priority: P1)

Operators can create rules that trigger notifications based on simple conditions without writing code. For example, warning when token usage exceeds 80% or suggesting summarization when search results exceed a threshold.

**Why this priority**: This is the most common use case - simple reactive behaviors that don't require programming knowledge. It validates the core rule engine and TOML-based configuration.

**Independent Test**: Can be tested by creating a threshold rule, triggering the condition, and verifying the resulting notification appears in agent context.

**Acceptance Scenarios**:

1. **Given** a rule configured with threshold condition `context.token_usage > 0.8`, **When** the agent's token usage exceeds 80%, **Then** the agent receives the configured notification message.
2. **Given** a rule with trigger `on_tool_complete` and condition `result.count > 6`, **When** a search returns 7+ results, **Then** the suggestion notification is delivered.
3. **Given** multiple rules match the same event, **When** the event occurs, **Then** all matching rules fire in priority order.

---

### User Story 2 - Hook Point Integration (Priority: P1)

Rules can attach to specific points in the agent lifecycle: turn start, turn end, tool call, tool completion, session boundaries. This allows precise timing of interventions.

**Why this priority**: Hook points are the foundation of the rule system. Without them, rules cannot observe or react to agent behavior.

**Independent Test**: Can be tested by creating rules attached to different hook points, running an agent session, and verifying rules fire at the expected moments.

**Acceptance Scenarios**:

1. **Given** a rule attached to `on_turn_start`, **When** a new agent turn begins, **Then** the rule's condition is evaluated and action executed if true.
2. **Given** a rule attached to `on_tool_complete` for tool "vault_search", **When** vault_search completes, **Then** the rule fires with access to the tool result.
3. **Given** a rule attached to `on_session_end`, **When** the session closes, **Then** the rule can persist state for the next session.

---

### User Story 3 - Context API Access (Priority: P1)

Rules and plugins can read agent state through a defined context API: current turn number, token usage, recent tool history, user settings, and plugin-scoped persistent state.

**Why this priority**: Rules need access to context to make decisions. The API boundary defines what rules can observe and how they interact with the system.

**Independent Test**: Can be tested by creating a rule that references various context fields, triggering it, and verifying the values are accessible and accurate.

**Acceptance Scenarios**:

1. **Given** a rule condition referencing `context.turn.token_usage`, **When** the rule evaluates, **Then** it receives the current token consumption ratio (0.0-1.0).
2. **Given** a rule referencing `context.history.tools`, **When** evaluated, **Then** it can access the list of recent tool calls with names and results.
3. **Given** a plugin using `context.state.get(key)`, **When** called, **Then** it retrieves plugin-scoped persistent state that survives across turns.

---

### User Story 4 - Script Escape Hatch (Priority: P2)

For complex logic that exceeds TOML expression capabilities, rules can delegate to script files. Scripts have access to the same context API and can execute multiple actions.

**Why this priority**: TOML expressions cover 80% of cases, but power users need escape hatches for complex temporal patterns, aggregations, or multi-step logic.

**Independent Test**: Can be tested by creating a rule that delegates to a script, triggering it, and verifying the script executes with full context access.

**Acceptance Scenarios**:

1. **Given** a rule with script reference instead of inline condition, **When** the trigger fires, **Then** the script is executed with context object available.
2. **Given** a script that calls multiple actions (notify, write state), **When** executed, **Then** all actions complete successfully.
3. **Given** a script that runs too long (>5 seconds), **When** timeout is reached, **Then** execution is terminated and an error is logged.

---

### User Story 5 - Rule Management UI (Priority: P2)

Users can view, enable/disable, and configure rules through the Settings interface. Built-in rules ship with the system; custom rules can be added via configuration.

**Why this priority**: Users need visibility and control over active rules. However, the rules must work before the UI to manage them matters.

**Independent Test**: Can be tested by opening Settings, navigating to Rules section, toggling a rule off, and verifying it no longer fires.

**Acceptance Scenarios**:

1. **Given** a user opens Settings and navigates to Rules, **When** the page loads, **Then** they see a list of all rules with enabled/disabled status.
2. **Given** a rule with configurable parameters (e.g., threshold value), **When** the user edits the parameter, **Then** the rule uses the new value on next trigger.
3. **Given** a core system rule (cannot be disabled), **When** the user views it, **Then** the toggle is disabled with explanation.

---

### User Story 6 - Plugin Manifest and Discovery (Priority: P3)

Plugins are packaged with a manifest file declaring their rules, capabilities, and dependencies. The system discovers and loads plugins from a designated directory.

**Why this priority**: Plugin packaging is important for distribution and organization but not critical for core rule functionality.

**Independent Test**: Can be tested by adding a plugin directory with manifest, restarting the service, and verifying the plugin's rules appear and function.

**Acceptance Scenarios**:

1. **Given** a plugin directory with valid manifest, **When** the system starts, **Then** the plugin is loaded and its rules registered.
2. **Given** a plugin declares dependency on a capability not present, **When** loading is attempted, **Then** the plugin is skipped with a warning.
3. **Given** a plugin has configurable settings, **When** the user opens Settings, **Then** plugin-specific configuration options appear.

---

## Functional Requirements *(mandatory)*

### FR-1: Rule Definition Schema

The system shall support rule definitions in a declarative configuration format with:
- **Trigger specification**: Which hook point activates the rule
- **Condition expression**: When the rule should fire (evaluated against context)
- **Action definition**: What happens when the rule fires
- **Priority**: Ordering when multiple rules match
- **Enabled flag**: Whether the rule is active

### FR-2: Hook Points

The system shall provide the following hook points for rule attachment:
- `on_query_start`: When a new user query is received
- `on_turn_start`: Before agent processes each turn
- `on_turn_end`: After agent completes each turn
- `on_tool_call`: Before a tool is executed
- `on_tool_complete`: After a tool returns
- `on_tool_failure`: When a tool fails or times out
- `on_session_end`: When the session closes

### FR-3: Context API

Rules shall have read access to:
- `context.turn`: Current turn number, token usage (ratio), iteration count
- `context.history`: Recent messages, recent tool calls, tool failure counts
- `context.user`: User ID, user settings
- `context.project`: Project ID, project settings
- `context.state`: Plugin-scoped persistent key-value storage

### FR-4: Expression Language

Simple conditions shall be expressible as inline expressions supporting:
- Comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
- Boolean operators: `and`, `or`, `not`
- Field access: `context.turn.token_usage`, `result.count`
- List operations: `any()`, `all()`, `len()`
- Basic arithmetic: `+`, `-`, `*`, `/`

### FR-5: Action Types

Rules shall support the following action types:
- `notify_self`: Inject a notification into agent context
- `log`: Write to system log for debugging
- `set_state`: Store plugin-scoped persistent state
- `emit_event`: Emit an ANS event for other subscribers

### FR-6: Script Execution

Rules may specify a script file instead of inline condition/action. Scripts shall:
- Have access to the full context API
- Be subject to execution timeout (configurable, default 5 seconds)
- Be sandboxed from system resources (no filesystem access, no network)

### FR-7: Rule Ordering and Conflicts

- Rules shall execute in priority order (higher priority first)
- Multiple rules may fire for the same event
- Rules cannot cancel or modify other rules' execution

### FR-8: Built-in Rules

The system shall ship with these built-in rules:
- Token budget warning (80% threshold)
- Iteration budget warning (70% threshold)
- Large result summarization hint (>6 results)
- Repeated failure warning (3+ failures of same tool)

---

## Success Criteria *(mandatory)*

### SC-1: Rule Authoring Efficiency
Operators can create a new threshold-based rule in under 5 minutes using only the configuration format, without writing code.

### SC-2: Rule Execution Reliability
Rules fire within 100ms of their trigger event 99% of the time under normal load.

### SC-3: System Stability
Rule execution failures (syntax errors, timeouts, exceptions) do not crash the agent or block message processing.

### SC-4: User Control
Users can enable/disable non-core rules without restarting the system. Changes take effect within 1 second.

### SC-5: Extensibility
New rules can be added by placing configuration files in the designated directory; no code changes required.

### SC-6: Context Accuracy
Context API values (token usage, tool history, state) are accurate to within 1 second of real-time at evaluation.

---

## Scope Boundaries *(mandatory)*

### In Scope (MVP)
- TOML-based rule definition schema
- Expression language for conditions (restricted safe evaluation)
- Scripting escape hatch with one supported language
- Core hook points (turn start/end, tool complete/failure, session end)
- Context API (read-only for MVP)
- Built-in rules (4-6 rules covering common patterns)
- Settings UI for rule listing and enable/disable
- Test button for manual rule triggering

### Out of Scope (Future Enhancements)
- Visual rule builder UI (drag-and-drop)
- Semantic conditions using embedding similarity (BERT)
- Alternative rule languages (LISP-style S-expressions)
- Complex plugin types (Skills, Behaviors, Swarms)
- Plugin marketplace or import system
- Agent-created rules (AI authoring rules at runtime)
- Rule chaining (one rule triggering another)
- Context API write operations (modifying agent state)

---

## Key Entities *(if applicable)*

### Rule
- **id**: Unique identifier (string, kebab-case)
- **name**: Human-readable name
- **description**: What the rule does
- **trigger**: Hook point specification
- **condition**: Expression or script reference
- **action**: Action type and parameters
- **priority**: Numeric priority (higher = earlier)
- **enabled**: Boolean, whether rule is active
- **core**: Boolean, whether rule can be disabled

### Plugin
- **id**: Unique identifier
- **name**: Display name
- **version**: Semantic version
- **description**: What the plugin provides
- **rules**: List of rule IDs this plugin provides
- **requires**: List of capabilities/dependencies
- **settings**: Configurable parameters

### RuleContext
- **turn**: Turn state (number, token_usage, iteration_count)
- **history**: Historical data (messages, tools, failures)
- **user**: User information (id, settings)
- **project**: Project information (id, settings)
- **state**: Plugin-scoped persistent storage
- **event**: The triggering event (if applicable)
- **result**: Tool result (for tool completion triggers)

---

## Dependencies

- **ANS Event Bus**: Rules subscribe to ANS events for tool failures, budget warnings
- **Oracle Agent Loop**: Hook points require integration with agent turn processing
- **User Settings Service**: Rule enable/disable state stored in user settings
- **Persistence Layer**: Plugin state requires database storage

---

## Assumptions

1. **Scripting language**: MVP will use Lua as the scripting escape hatch (well-established embedding story, good sandboxing)
2. **Rule storage**: Rules are stored as configuration files, not in database (simplifies version control, editing)
3. **Evaluation frequency**: Conditions are evaluated at hook points, not continuously polled
4. **Single-tenant**: Rules are per-user/per-project, not shared globally
5. **Safe expressions**: Expression evaluation uses a restricted evaluator preventing arbitrary code execution
6. **Performance budget**: Rule evaluation adds <50ms overhead to affected operations

---

## Stretch Goals (Documented for Future)

### Semantic Conditions (BERT)
Enable conditions that match based on semantic similarity rather than exact values:
```
semantic = {concept = "authentication", field = "tool.args.query", threshold = 0.7}
```
Requires embedding model integration and concept vector precomputation.

### LISP Rule Language
Alternative expression syntax using S-expressions for complex pattern matching:
```lisp
(and (> (ctx-token-usage) 0.8)
     (any (ctx-recent-tools 5) (lambda (t) (eq (tool-name t) "vault_search"))))
```
Requires LISP interpreter integration.

### Skill Plugins
Higher-level plugins that define callable capabilities with structured inputs/outputs:
- Deep Researcher: Multi-step web research with source synthesis
- Agent Swarm: Coordinate multiple sub-agents for parallel work

### Visual Rule Builder
Drag-and-drop UI for constructing rules without editing configuration files.
