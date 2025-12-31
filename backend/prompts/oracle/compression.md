# Oracle Context Compression Prompt

The conversation has reached {{ tokens_used }} tokens out of a {{ token_budget }} token budget ({{ usage_percent }}%). You need to compress older exchanges while preserving essential context for continuity.

## Compression Principles

### What to Preserve (High Priority)

1. **Key Decisions**
   - Architectural choices made during the conversation
   - User preferences or constraints expressed
   - Conclusions reached after analysis

2. **Important Discoveries**
   - Critical code paths identified
   - Bugs or issues found
   - Relationships between components

3. **Recent Exchanges** (last 3-5 turns)
   - Keep verbatim for immediate context
   - Preserve exact citations and code snippets

4. **Active References**
   - Files currently being discussed
   - Symbols the user has asked about
   - Threads that have been read or written to

### What to Summarize (Medium Priority)

1. **Tool Call Results**
   - Replace raw results with key findings
   - Keep citations, remove verbose output

2. **Exploratory Searches**
   - Note what was searched and what was found
   - Remove detailed listings if not directly relevant

3. **Background Context**
   - Summarize earlier explanations
   - Preserve the gist, not the details

### What to Remove (Low Priority)

1. **Redundant Information**
   - Repeated explanations
   - Superseded findings (if corrected later)

2. **Failed Searches**
   - Unless the failure itself is informative

3. **Verbose Tool Output**
   - Full file contents if only a section was relevant
   - Long code listings if already explained

## Compression Format

Generate a compressed summary in this structure:

```markdown
## Session Summary

**Started**: {{ session_start }}
**Turns compressed**: {{ turns_compressed }}

### Key Decisions
- [List each decision with brief rationale]

### Discoveries
- [List significant findings with citations]

### Current Focus
- [What the user is currently working on]
- [Active files/symbols being discussed]

### Context for Continuation
- [Any information needed to answer follow-up questions]
```

## Current Conversation State

**Recent exchanges to preserve verbatim**:
{{ recent_exchanges_json }}

**Key decisions recorded**:
{{ key_decisions_json }}

**Files mentioned**: {{ mentioned_files }}
**Symbols mentioned**: {{ mentioned_symbols }}

## Compression Task

Compress the older portions of this conversation while:

1. Keeping the last {{ preserve_recent }} exchanges intact
2. Preserving all key decisions listed above
3. Maintaining enough context for coherent continuation
4. Reducing total token count by at least {{ reduction_target }}%

Output the compressed summary that will replace older exchanges. The summary should be self-contained - someone reading only the summary and recent exchanges should understand the conversation's context.

---

## Important Notes

- **Never lose decisions**: If a decision was made, it must appear in the summary
- **Preserve citations**: Even in summary form, keep source references
- **Note what was removed**: If significant context is being compressed, briefly note what topics were covered
- **Err on the side of keeping**: When uncertain, preserve rather than discard
- **Recency Bias**: Prioritize recent exchanges and decisions over older ones when making compression decisions, if the path of the conversation indicates it is pivoting towards something else with high importance in the context window, that information should be preserved.
- **The Polished Gem**: Your context window is your memory. Treasure it, and polish it.
