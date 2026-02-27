# Data Model: BT-Controlled Oracle Agent

**Date**: 2026-01-08
**Feature**: 020-bt-oracle-agent

## Overview

This document defines the data entities for the BT-controlled Oracle agent. The primary new entities are related to signal parsing and query classification.

---

## Entity: Signal

Structured agent self-reflection extracted from LLM response.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| type | SignalType (enum) | Yes | Signal category |
| confidence | float | Yes | 0.0-1.0 confidence score |
| fields | dict[str, Any] | Yes | Type-specific fields |
| raw_xml | str | Yes | Original XML for debugging |
| timestamp | datetime | Yes | When signal was parsed |

### Signal Types (Enum)

```
need_turn           - Agent needs more iterations
context_sufficient  - Agent has enough information
stuck               - Agent cannot proceed
need_capability     - Agent needs unavailable tool
partial_answer      - Answering with caveats
delegation_recommended - Task should be delegated
```

### Type-Specific Fields

**need_turn:**
- reason (str, required): Why another turn is needed
- expected_turns (int, optional): Estimated turns needed

**context_sufficient:**
- sources_found (int, required): Number of relevant sources
- source_types (list[str], optional): Types of sources found

**stuck:**
- attempted (list[str], required): Tools/approaches tried
- blocker (str, required): What prevents progress
- suggestions (list[str], optional): What might help

**need_capability:**
- capability (str, required): What capability is needed
- reason (str, required): Why it's needed
- workaround (str, optional): Partial solution

**partial_answer:**
- missing (str, required): What information is missing
- caveat (str, optional): Important caveat for user

**delegation_recommended:**
- reason (str, required): Why delegation helps
- scope (str, required): What to delegate
- estimated_tokens (int, optional): Token budget needed
- subagent_type (str, optional): Recommended subagent

### Validation Rules

1. confidence must be between 0.0 and 1.0
2. type must be a valid SignalType enum value
3. required fields for signal type must be present
4. raw_xml must be non-empty string

### State Transitions

Signals don't have state - they are immutable once parsed.

---

## Entity: QueryClassification

Result of analyzing user query to determine context needs.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| query_type | QueryType (enum) | Yes | Primary classification |
| needs_code | bool | Yes | Should search code |
| needs_vault | bool | Yes | Should search vault |
| needs_web | bool | Yes | Should search web |
| confidence | float | Yes | Classification confidence |
| keywords_matched | list[str] | No | Which keywords triggered |

### Query Types (Enum)

```
code           - Questions about implementation
documentation  - Questions about decisions/architecture
research       - External information needs
conversational - Follow-ups, acknowledgments
action         - Write operations
```

### Validation Rules

1. query_type must be valid QueryType enum value
2. confidence must be between 0.0 and 1.0
3. At least one of needs_code/needs_vault/needs_web should be true unless conversational

### Derivation Rules

| Query Type | needs_code | needs_vault | needs_web |
|------------|------------|-------------|-----------|
| code | true | false | false |
| documentation | false | true | false |
| research | false | false | true |
| conversational | false | false | false |
| action | false | true | false |

---

## Entity: PromptSegment

Reusable prompt component for dynamic composition.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | str | Yes | Unique segment identifier |
| content | str | Yes | Prompt text content |
| token_estimate | int | Yes | Estimated token count |
| conditions | list[str] | No | When to include |
| priority | int | Yes | Load order (lower = first) |

### Segment Registry (Static)

| ID | File | Priority | Conditions |
|----|------|----------|------------|
| base | oracle/base.md | 0 | always |
| signals | oracle/signals.md | 1 | always |
| tools | oracle/tools-reference.md | 2 | always |
| code | oracle/code-analysis.md | 10 | query_type=code |
| docs | oracle/documentation.md | 10 | query_type=documentation |
| research | oracle/research.md | 10 | query_type=research |
| conversation | oracle/conversation.md | 10 | query_type=conversational |

### Validation Rules

1. id must be unique across registry
2. content must be non-empty
3. priority must be non-negative integer
4. File must exist at specified path

---

## Entity: AgentSignalState

Tracking of signals across agent loop iterations.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| signals_emitted | list[Signal] | Yes | All signals this session |
| last_signal | Signal | No | Most recent signal |
| consecutive_same_reason | int | Yes | Count of same need_turn reason |
| turns_without_signal | int | Yes | Turns with no signal emitted |

### Validation Rules

1. consecutive_same_reason triggers stuck detection at >= 3
2. turns_without_signal triggers BERT fallback at >= 3

### State Transitions

```
Initial State:
  signals_emitted = []
  last_signal = None
  consecutive_same_reason = 0
  turns_without_signal = 0

On Signal Received:
  signals_emitted.append(signal)
  if signal.type == "need_turn":
    if last_signal?.reason == signal.reason:
      consecutive_same_reason += 1
    else:
      consecutive_same_reason = 1
  last_signal = signal
  turns_without_signal = 0

On Turn Without Signal:
  turns_without_signal += 1
  consecutive_same_reason = 0
```

---

## Relationships

```
┌─────────────────┐
│ OracleSession   │
│ (existing)      │
└────────┬────────┘
         │ 1:N
         ▼
┌─────────────────┐     ┌─────────────────┐
│ AgentSignalState│────►│ Signal          │
│                 │ 1:N │                 │
└─────────────────┘     └─────────────────┘
         │
         │ uses
         ▼
┌─────────────────┐     ┌─────────────────┐
│QueryClassification│──►│ PromptSegment   │
│                 │ 1:N │ (composed)      │
└─────────────────┘     └─────────────────┘
```

---

## Storage

### Signal Persistence

Signals are logged to:
1. **Blackboard** (runtime) - for BT condition evaluation
2. **ANS Events** (events table) - for audit/debugging
3. **Exchange metadata** (context_nodes) - for session continuity

### Query Classification

Not persisted - computed fresh for each query.

### Prompt Segments

Stored as static markdown files in `backend/src/prompts/oracle/`.
Loaded at startup, cached in memory.

---

## Migration Notes

No database migrations required. All new entities are:
- Runtime-only (blackboard)
- Static files (prompts)
- Existing table columns (events, context_nodes JSON fields)
