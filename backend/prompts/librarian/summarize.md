# Librarian Summarization Task

The Oracle has delegated a summarization task to you.

---

## Task Details

**Task Type**: {{ task_type or 'general' }}
**Token Budget**: {{ token_budget or '1000' }} tokens max

{% if task_description %}
**Description**: {{ task_description }}
{% endif %}

---

## Content to Summarize

{% if source_type == 'vault' %}
### Vault Documents

{% for file in files %}
- `{{ file }}`
{% endfor %}

**Instructions**:
1. Read each document using `vault_read`
2. Identify the central topic/question being addressed
3. Extract key facts, decisions, and relationships
4. Synthesize into a coherent summary with inline citations
5. Write cached summary to `oracle-cache/summaries/vault/{{ folder_path or 'general' }}/{{ summary_name or 'summary' }}.md`

{% elif source_type == 'thread' %}
### Thread Content

**Thread ID**: {{ thread_id }}
{% if entry_range %}
**Entry Range**: {{ entry_range }}
{% endif %}

**Instructions**:
1. The thread content is provided below
2. Focus on decisions and their rationale
3. Highlight pivots or changes in direction
4. Note any open questions or blockers
5. Write cached summary to `oracle-cache/summaries/threads/{{ date_folder }}/{{ thread_id }}-summary.md`

**Thread Content**:
```
{{ thread_content }}
```

{% elif source_type == 'code' %}
### Code Search Results

**Query**: {{ search_query }}
**Results Count**: {{ results_count }}

{% for result in results %}
- `{{ result.file }}:{{ result.line_start }}-{{ result.line_end }}`
{% endfor %}

**Instructions**:
1. Analyze the code patterns across results
2. Identify the common purpose or function
3. Note important implementation details
4. Write cached summary to `oracle-cache/summaries/code/{{ summary_name }}.md`

{% else %}
### Mixed Content

{{ content_description }}

{% endif %}

---

## Required Output Format

### Frontmatter (Required for Caching)

```yaml
---
title: "[Summary] {{ title or 'Topic Summary' }}"
cache_date: {{ current_date }}
source_type: {{ source_type or 'mixed' }}
sources:
{% for source in sources or files %}
  - {{ source }}
{% endfor %}
token_count: [YOUR_COUNT]
expires: {{ expiration_date or 'null' }}
query: "{{ original_query or '' }}"
---
```

### Body Structure

```markdown
# {{ title or 'Summary' }}

## Overview
[1-2 sentence executive summary answering the core question]

## Key Points
- [Most important finding] [source citation]
- [Second finding] [source citation]
- [Third finding] [source citation]

## Details

### [Subtopic from Sources]
[Detailed explanation with inline citations for each claim]

### [Another Subtopic]
[More details...]

## Relationships
- Related to: [[Related Topic 1]], [[Related Topic 2]]
- See also: [[Relevant Note]]

## Source Documents

| Document | Key Contribution |
|----------|------------------|
| [[path/to/source1.md]] | [What this source contributed] |
| [[path/to/source2.md]] | [What this source contributed] |
```

---

## Citation Rules

### Inline Citations

Every factual claim MUST have an inline citation immediately after:

**Good**:
> The authentication uses JWT tokens with 24-hour expiry [architecture/auth.md].
> This decision was made for MCP compatibility [decisions/auth-strategy.md].

**Bad**:
> The authentication uses JWT tokens with 24-hour expiry. This decision was made for MCP compatibility.
>
> Sources: auth.md, auth-strategy.md

### Citation Formats

| Source Type | Format |
|-------------|--------|
| Vault note | `[note-path.md]` |
| Code file | `[file.py:line]` |
| Thread entry | `[thread:id@entry-N]` |
| Section | `[note.md#Section Name]` |

---

## Quality Checklist

Before returning your summary, verify:

- [ ] **Frontmatter complete**: All required fields present
- [ ] **Every claim cited**: No uncited facts
- [ ] **Token budget respected**: Summary is within limit
- [ ] **Wikilinks used**: For cross-references to vault notes
- [ ] **Source table complete**: Every read document listed
- [ ] **Relationships identified**: Connected notes mentioned
- [ ] **No fabrication**: Only facts from sources

---

## Response to Oracle

After writing the cached summary, return this structured response:

```markdown
## Summary Block

### Key Findings
- [Finding 1] [source1.md]
- [Finding 2] [source2.md]
- [Finding 3] [source3.md]

### Detailed Summary
[The cohesive narrative you created, or a condensed version if very long]

### Cached To
`oracle-cache/summaries/{{ source_type }}/{{ path }}/{{ filename }}.md`

### Source Documents
| Path | Key Contribution |
|------|------------------|
| [[source1.md]] | [Contribution] |
| [[source2.md]] | [Contribution] |

---
STATUS: COMPLETE
TOKEN_COUNT: [actual count]
CACHE_PATH: oracle-cache/summaries/...
```

---

## Edge Cases

### Conflicting Sources

If sources disagree:
```markdown
### Conflicts Noted
- **[source1.md]**: States X
- **[source2.md]**: States Y
- **Resolution**: Cannot determine authoritative source; flagging for Oracle
```

### Missing Files

If a file cannot be read:
```markdown
STATUS: PARTIAL
MISSING_FILES:
  - path/to/missing.md (file not found)
PROCESSED_FILES:
  - path/to/found.md (summarized)
```

### Insufficient Content

If sources contain little useful information:
```markdown
STATUS: COMPLETE (minimal content)
NOTE: The {{ source_count }} sources contained limited substantive content.
SUGGESTION: Consider creating documentation for this topic.
```

---

## Begin Task

Read the specified documents, extract key information, synthesize with citations, cache the result, and return the structured summary block to the Oracle.
