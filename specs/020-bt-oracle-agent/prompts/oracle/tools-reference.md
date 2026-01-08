# Available Tools

## Code Tools

### search_code
Search the codebase for relevant code snippets.

**When to use**: Questions about implementation, "where is X defined", "how does X work"

**Parameters:**
- `query`: Natural language search query
- `limit`: Max results (default: 10)

**Tips:**
- Use specific terms from the domain
- Include function/class names if known
- Results include file paths and line numbers

### find_definition
Find where a symbol (function, class, variable) is defined.

**When to use**: "Where is X defined?", "Show me the X class"

**Parameters:**
- `symbol`: Name of the symbol to find

### find_references
Find all usages of a symbol.

**When to use**: "What calls X?", "Where is X used?"

**Parameters:**
- `symbol`: Name of the symbol to find references for

---

## Documentation Tools

### vault_search
Search the documentation vault for relevant notes.

**When to use**: Questions about decisions, architecture, requirements, meeting notes

**Parameters:**
- `query`: Search query
- `limit`: Max results (default: 5)

### vault_read
Read a specific document from the vault.

**When to use**: When you know the exact document path

**Parameters:**
- `path`: Path to the document

### thread_seek
Search past conversation threads.

**When to use**: "What did we discuss about X?", context from previous sessions

**Parameters:**
- `query`: Search query

---

## Web Tools

### web_search
Search the web for external information.

**When to use**: Current events, best practices, external documentation, "what's the latest on X"

**Parameters:**
- `query`: Search query
- `limit`: Max results (default: 5)

**Tips:**
- Be specific - include version numbers, dates if relevant
- Use for information NOT in the codebase

---

## Action Tools

### vault_write
Create or update a document in the vault.

**When to use**: User explicitly asks to save/create a note

**Parameters:**
- `path`: Where to save
- `content`: What to save
- `mode`: "create" or "update"

### thread_push
Save insight to the current thread.

**When to use**: Recording important decisions or context

**Parameters:**
- `content`: What to record

---

## Tool Selection Matrix

| User Intent | First Choice | Second Choice | Avoid |
|-------------|--------------|---------------|-------|
| Find code | search_code | find_definition | web_search |
| Understand decision | vault_search | thread_seek | search_code |
| External info | web_search | - | search_code |
| Previous discussion | thread_seek | vault_search | web_search |
| Save something | vault_write | thread_push | - |
