# Oracle Agent Tools Reference

Comprehensive reference for all tools available to Oracle agents. Tools are organized by category with detailed usage examples and best practices.

---

## Code Tools

Tools for understanding and navigating the codebase.

### search_code

**Purpose**: Search the codebase using hybrid retrieval (vector + BM25) for relevant code chunks.

**When to Use**:
- Finding implementations of a concept
- Locating code related to a feature
- Understanding how something is done in the codebase

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | Natural language search query |
| `limit` | integer | No | Max results (1-20, default 5) |
| `language` | string | No | Filter by language (e.g., "python", "typescript") |

**Examples**:
```json
// Find authentication code
{"query": "JWT token validation middleware", "limit": 5}

// Find Python-specific code
{"query": "database connection pool", "language": "python", "limit": 10}

// Broad feature search
{"query": "how are notes indexed for search"}
```

**Best Practices**:
- Use natural language, not code syntax
- Start with broader queries, then narrow down
- Combine with `find_references` for complete picture

---

### find_definition

**Purpose**: Locate where a symbol (function, class, variable) is defined.

**When to Use**:
- Finding the source of a class or function
- Understanding where a constant is declared
- Locating type definitions

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Symbol name (e.g., "OracleAgent", "search_code") |
| `scope` | string | No | Limit to directory or pattern |

**Examples**:
```json
// Find a class definition
{"symbol": "DatabaseService"}

// Find function in specific area
{"symbol": "validate_note_path", "scope": "backend/src/services"}

// Find a React component
{"symbol": "NoteViewer", "scope": "frontend/src"}
```

**Best Practices**:
- Use exact symbol names (case-sensitive)
- Add scope for large codebases to speed up search
- Follow up with `find_references` to see usage

---

### find_references

**Purpose**: Find all usages of a symbol throughout the codebase.

**When to Use**:
- Understanding how widely a function is used
- Finding callers of a method
- Impact analysis before refactoring

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `symbol` | string | Yes | Symbol name to find references for |
| `limit` | integer | No | Max references (1-50, default 20) |

**Examples**:
```json
// Find all uses of a service
{"symbol": "VaultService", "limit": 30}

// Find function calls
{"symbol": "index_note"}
```

**Best Practices**:
- Increase limit for commonly-used symbols
- Combine with `find_definition` for complete understanding
- Note: may include false positives for common names

---

### get_repo_map

**Purpose**: Get a structural overview of the repository showing key files, classes, and functions.

**When to Use**:
- Starting to explore a new codebase
- Understanding project structure
- Finding entry points

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `scope` | string | No | Directory to focus on |
| `max_tokens` | integer | No | Token budget (500-8000, default 2000) |

**Examples**:
```json
// Get full repo overview
{}

// Focus on a subsystem
{"scope": "backend/src/services", "max_tokens": 3000}

// Quick glance at a folder
{"scope": "frontend/src/components", "max_tokens": 500}
```

**Best Practices**:
- Start with broad view, then scope down
- Use larger token budget for complex areas
- Good first step before detailed searches

---

## Vault Tools

Tools for reading and writing documentation.

### vault_read

**Purpose**: Read a markdown note from the documentation vault.

**When to Use**:
- Reading specific documentation
- Checking note contents before linking
- Verifying cached summaries against originals

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | Yes | Path to note (e.g., "architecture/auth.md") |

**Examples**:
```json
// Read a specific note
{"path": "architecture/auth.md"}

// Read an index file
{"path": "decisions/index.md"}

// Read cached summary
{"path": "oracle-cache/summaries/threads/2025-01/auth-design-summary.md"}
```

**Best Practices**:
- Paths are relative to vault root
- Include `.md` extension
- Check note exists before reading (use vault_list first if unsure)

---

### vault_write

**Purpose**: Create or update a markdown note in the vault.

**When to Use**:
- Saving research findings
- Creating cached summaries
- Documenting decisions

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `path` | string | Yes | Path for the note |
| `title` | string | No | Note title (for frontmatter) |
| `body` | string | Yes | Markdown content |

**Examples**:
```json
// Create a research note
{
  "path": "research/auth-investigation.md",
  "title": "Authentication Investigation",
  "body": "## Findings\\n\\nThe auth flow uses JWT tokens..."
}

// Create cached summary
{
  "path": "oracle-cache/summaries/vault/architecture/index-summary.md",
  "title": "[Summary] Architecture Overview",
  "body": "---\\ncache_date: 2025-01-15\\nsources:\\n  - architecture/overview.md\\n---\\n\\n## Summary..."
}
```

**Best Practices**:
- Use descriptive paths that reflect content
- Include proper frontmatter for cached summaries
- Add wikilinks to related notes
- Cite sources when summarizing

---

### vault_search

**Purpose**: Search the vault using full-text search (SQLite FTS5).

**When to Use**:
- Finding notes on a topic
- Discovering related documentation
- Checking for existing content before creating new

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `limit` | integer | No | Max results (1-20, default 5) |

**Examples**:
```json
// Topic search
{"query": "authentication JWT", "limit": 10}

// Find all notes mentioning a component
{"query": "VaultService", "limit": 20}

// Search with phrase
{"query": "\"error handling\""}
```

**Best Practices**:
- Uses BM25 ranking (title weighted 3x)
- Supports standard FTS5 operators
- Combine multiple terms for precision

---

### vault_list

**Purpose**: List notes in a vault folder.

**When to Use**:
- Exploring folder structure
- Finding all notes in a category
- Checking what exists before creating

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `folder` | string | No | Folder path (empty for root) |

**Examples**:
```json
// List root folder
{}

// List specific folder
{"folder": "architecture"}

// List cached summaries
{"folder": "oracle-cache/summaries/threads"}
```

**Best Practices**:
- Returns files and subfolders
- Use to verify paths before read/write
- Helpful for understanding organization

---

### vault_move

**Purpose**: Move or rename a note. Automatically updates wikilinks in other notes.

**When to Use**:
- Reorganizing vault structure
- Renaming notes
- Moving notes to appropriate folders

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `old_path` | string | Yes | Current note path |
| `new_path` | string | Yes | New note path |

**Examples**:
```json
// Move to different folder
{
  "old_path": "misc/auth-notes.md",
  "new_path": "architecture/auth-notes.md"
}

// Rename in place
{
  "old_path": "architecture/auth.md",
  "new_path": "architecture/authentication.md"
}
```

**Best Practices**:
- **Librarian only** - Oracle should delegate moves
- Wikilinks are updated automatically
- Verify with vault_list after moving

**Agent Scope**: Librarian only

---

### vault_create_index

**Purpose**: Create an index.md file for a folder with links to all notes.

**When to Use**:
- Folders with 3+ notes need navigation
- After reorganizing a folder
- Creating topic hubs

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `folder` | string | Yes | Folder to create index for |
| `title` | string | No | Title for index page |
| `include_summaries` | boolean | No | Include first paragraph of each note (default true) |

**Examples**:
```json
// Create basic index
{"folder": "architecture", "title": "Architecture Documentation"}

// Create minimal index
{"folder": "meeting-notes", "title": "Meeting Notes", "include_summaries": false}
```

**Best Practices**:
- Index files make folders navigable
- Include summaries for discoverability
- Update after adding new notes

**Agent Scope**: Librarian only

---

## Thread Tools

Tools for persistent memory across sessions.

### thread_push

**Purpose**: Record a thought, decision, or finding to long-term memory.

**When to Use**:
- Capturing important decisions
- Recording discoveries that should persist
- Logging insights for future sessions

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `thread_id` | string | Yes | Thread identifier (e.g., "auth-design") |
| `content` | string | Yes | The insight to record |
| `entry_type` | string | No | Type: "thought", "decision", "research", "insight" |

**Examples**:
```json
// Record a decision
{
  "thread_id": "api-design",
  "content": "Decided to use optimistic concurrency with version counters rather than locking. Simpler for MCP use case.",
  "entry_type": "decision"
}

// Record a discovery
{
  "thread_id": "auth-investigation",
  "content": "Found that HF OAuth tokens expire after 90 days, not 30 as assumed.",
  "entry_type": "research"
}
```

**Best Practices**:
- Focus on "why" not "what"
- Record pivots and reasoning
- Keep entries self-contained

---

### thread_read

**Purpose**: Read a thread to get context and summary of past work.

**When to Use**:
- Starting work on a topic with history
- Checking what was previously decided
- Getting up to speed on past sessions

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `thread_id` | string | Yes | Thread identifier |
| `limit` | integer | No | Recent entries to include (1-50, default 10) |

**Examples**:
```json
// Read recent thread activity
{"thread_id": "auth-design"}

// Read more history
{"thread_id": "api-design", "limit": 30}
```

**Best Practices**:
- Start with default limit, increase if needed
- Thread includes auto-generated summary
- Delegate to Librarian if thread is very long

---

### thread_seek

**Purpose**: Search across all threads using semantic similarity.

**When to Use**:
- Finding past context on a topic
- Discovering related previous work
- Cross-thread research

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | What to search for |
| `limit` | integer | No | Max results (1-20, default 5) |

**Examples**:
```json
// Find past auth discussions
{"query": "authentication implementation decisions"}

// Find related work
{"query": "performance optimization attempts", "limit": 10}
```

**Best Practices**:
- Uses semantic search, so natural language works well
- Results include thread ID and entry ID for follow-up
- Combine with thread_read for full context

---

### thread_list

**Purpose**: List all threads for the current project.

**When to Use**:
- Discovering available threads
- Finding thread IDs
- Understanding project history

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `status` | string | No | Filter: "active", "archived", "all" (default "active") |

**Examples**:
```json
// List active threads
{}

// List all threads
{"status": "all"}
```

**Best Practices**:
- Check available threads before creating new
- Use meaningful thread IDs when creating

---

## Web Tools

Tools for external information gathering.

### web_search

**Purpose**: Search the web for documentation or current information.

**When to Use**:
- Library documentation not in codebase
- Current information (versions, standards)
- General concepts or patterns

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `query` | string | Yes | Search query |
| `limit` | integer | No | Number of results (1-10, default 5) |

**Examples**:
```json
// Find documentation
{"query": "FastAPI middleware authentication", "limit": 5}

// Find current info
{"query": "Python 3.11 new features"}
```

**Best Practices**:
- Use after exhausting internal sources
- Cite web sources with URL
- Prefer official documentation

---

### web_fetch

**Purpose**: Fetch and extract content from a URL.

**When to Use**:
- Reading specific documentation pages
- Extracting content from search results
- Saving external info to vault

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `url` | string | Yes | URL to fetch |
| `max_tokens` | integer | No | Max tokens to extract (100-4000, default 1000) |

**Examples**:
```json
// Fetch documentation
{"url": "https://fastapi.tiangolo.com/tutorial/middleware/"}

// Fetch with more content
{"url": "https://example.com/long-article", "max_tokens": 3000}
```

**Best Practices**:
- Increases token budget for longer pages
- Content is cleaned and extracted
- Always cite the URL in responses

---

## Meta Tools

Tools for agent coordination.

### delegate_librarian

**Purpose**: Delegate vault organization or summarization tasks to the Librarian subagent.

**When to Use**:
- **Large search results**: More than 6 relevant results with similar scores
- **Folder overviews**: User asks about content spanning multiple files
- **Thread summarization**: Long thread history needs condensing
- **Token pressure**: Response would exceed ~4000 tokens
- **Organization tasks**: Moving, renaming, creating indexes

**Parameters**:
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `task` | string | Yes | Description of what Librarian should do |
| `files` | array | No | Specific files to operate on |
| `folder` | string | No | Target folder for organization |

**Examples**:
```json
// Summarize search results
{
  "task": "Summarize these 12 authentication-related notes into a coherent overview, saving to oracle-cache",
  "files": [
    "architecture/auth.md",
    "architecture/jwt.md",
    "decisions/2024-auth-strategy.md",
    "..."
  ]
}

// Organize a folder
{
  "task": "Create an index for the architecture folder with summaries of each document",
  "folder": "architecture"
}

// Summarize thread
{
  "task": "Read and summarize the auth-design thread, caching the summary for future queries",
  "files": ["thread:auth-design"]
}
```

**Best Practices**:
- Provide clear, specific task descriptions
- Include all relevant file paths
- Librarian returns structured summary
- Oracle incorporates Librarian output into response

---

## Common Workflow Patterns

### Pattern 1: Answering "How does X work?"

```
1. search_code("X implementation") -> Find code
2. vault_search("X") -> Find documentation
3. thread_seek("X decisions") -> Find historical context
4. [If many results] delegate_librarian(summarize)
5. Synthesize answer with citations
```

### Pattern 2: Research and Document

```
1. search_code + vault_search -> Gather info
2. web_search (if needed) -> External context
3. vault_write -> Save findings
4. thread_push -> Record key decision
```

### Pattern 3: Vault Reorganization

```
1. Oracle: delegate_librarian(task="reorganize X folder")
2. Librarian: vault_list -> Understand structure
3. Librarian: vault_read (key files) -> Analyze content
4. Librarian: vault_move + vault_create_index
5. Librarian: Return summary to Oracle
```

### Pattern 4: Context Restoration (New Session)

```
1. thread_list -> Find relevant threads
2. thread_read (key threads) -> Load context
3. vault_search (recent topics) -> Find notes
4. [If long threads] delegate_librarian(summarize threads)
```

---

## Tool Selection Quick Reference

| User Intent | Primary Tools | Fallback |
|-------------|---------------|----------|
| "How does X work?" | search_code, find_definition | vault_search |
| "Why is X this way?" | thread_seek, vault_search | find_references |
| "Show me X" | vault_read, find_definition | search_code |
| "Where is X used?" | find_references | search_code |
| "Organize the X folder" | delegate_librarian | - |
| "Summarize X" | delegate_librarian | vault_read |
| "What's the current Y?" | web_search | vault_search |
| "Document X" | vault_write | - |
| "Remember X" | thread_push | vault_write |
