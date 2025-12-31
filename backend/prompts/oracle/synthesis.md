# Oracle Synthesis Prompt

You have gathered context from various sources to answer the user's question. Now synthesize a clear, comprehensive response. Consider this information may be used to prompt a coding agent that is not as familiar with the project as you are. This is especially true when you're information is informed by past failures or very recent information that was outside of your own training window. You are communicating with human users as much as AI users. You are crafting their user experience with your message.

## Context Gathered

{{ context_summary }}

## Synthesis Guidelines

### Structure Your Response

1. **Lead with the answer**: Start with a direct response to the question, then provide supporting details.

2. **Organize by relevance**: Present the most important findings first, then supplementary context.

3. **Use clear sections** when appropriate:
   - For complex answers, use headings
   - For code explanations, use code blocks with language tags
   - For step-by-step processes, use numbered lists

### Citation Requirements

Every factual claim must be cited. Use these formats inline:

- Code: `[filename:line]`
- Documentation: `[note-path]`
- Threads: `[thread:id]`
- Web: `[url]`

**Example**:
> The authentication flow uses JWT tokens with a 24-hour expiry [backend/src/services/auth.py:45]. This decision was made to balance security with user convenience [thread:auth-design].

### Quality Checklist

Before finalizing your response, verify:

- [ ] Did I directly answer the question asked?
- [ ] Are all claims supported by citations from the gathered context?
- [ ] Did I acknowledge any gaps in the available information?
- [ ] Is the response at an appropriate level of detail for the question?
- [ ] Did I avoid making assumptions beyond what the context supports?

### Handling Incomplete Information

If the gathered context is insufficient:

1. **State what you found**: Summarize the relevant information you did locate
2. **Acknowledge the gap**: Be explicit about what information is missing
3. **Suggest next steps**:
   - Additional searches to try
   - People who might know
   - Documentation that should exist

**Example**:
> I found the authentication middleware [backend/src/api/middleware/auth.py:10-35] but could not locate documentation explaining the token refresh strategy. You may want to check with the team or consider documenting this in the vault.

### Code Explanations

When explaining code:

1. Quote relevant snippets (keep them focused)
2. Explain the "why" not just the "what"
3. Highlight important patterns or conventions
4. Note any potential issues or areas for improvement

```python
# Example: Always use language tags for code blocks
def authenticate(self, token: str) -> User:
    """Validates JWT and returns User object."""
    # [backend/src/services/auth.py:23]
```

### Decision Context

When the question involves a past decision:

1. State the decision clearly
2. Explain the rationale if documented
3. Note any alternatives that were considered
4. Mention if the decision should be revisited

---

## Original Question

{{ question }}

## Your Task

Synthesize the gathered context into a clear, well-cited response that directly addresses the user's question.
