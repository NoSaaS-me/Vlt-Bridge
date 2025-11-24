# MCP Tools Test Report

## Overview
Comprehensive testing of all MCP obsidian-docs tools exposed by the server.

## Test Results

### ✅ `list_notes` - **PASSING**
**Functionality**: Lists all notes in the vault
**Test Results**:
- Successfully listed 10 notes from the vault
- Returns array with: `path`, `title`, `last_modified` timestamp
- Works with optional `folder` parameter (tested with `null` and empty string)
- **Status**: ✅ Working correctly

**Sample Output**:
```json
[
  {"path":"API Documentation.md","title":"API Documentation 123","last_modified":"2025-11-17T01:17:52.721411+00:00"},
  {"path":"test-from-cursor-over-mcp.md","title":"test from cursor over mcp","last_modified":"2025-11-17T02:16:14.156806+00:00"}
]
```

---

### ✅ `read_note` - **PASSING**
**Functionality**: Reads a note with metadata and body
**Test Results**:
- Successfully read existing notes
- Returns: `path`, `title`, `metadata` (created, updated, title), and `body`
- Properly handles deleted notes (returns "Note not found" error)
- **Status**: ✅ Working correctly

**Sample Output**:
```json
{
  "path":"test-from-cursor-over-mcp.md",
  "title":"test from cursor over mcp",
  "metadata":{
    "created":"2025-11-17T02:13:39+00:00",
    "title":"test from cursor over mcp",
    "updated":"2025-11-17T02:16:14+00:00"
  },
  "body":"# test from cursor over mcp\n\nThis is a test note created from Cursor using the MCP server.\n\n**Updated**: This note has been updated to test the write_note functionality.\n\nNow referencing [[API Documentation]] to test backlinks."
}
```

**Error Handling**:
- ✅ Returns proper error for non-existent notes: "Note not found: test-delete-me.md"

---

### ✅ `write_note` - **PASSING**
**Functionality**: Creates or updates notes with optional metadata
**Test Results**:
- Successfully created new notes
- Successfully updated existing notes (preserves creation timestamp, updates modified timestamp)
- Supports optional `title` parameter
- Supports optional `metadata` parameter (not fully tested with tags)
- Automatically updates frontmatter timestamps
- **Status**: ✅ Working correctly

**Test Cases**:
1. ✅ Created new note: `test-from-cursor-over-mcp.md`
2. ✅ Updated existing note: Modified `test-from-cursor-over-mcp.md` and confirmed timestamp updated
3. ✅ Created note for deletion testing: `test-delete-me.md`

**Sample Output**:
```json
{"status":"ok","path":"test-from-cursor-over-mcp.md"}
```

---

### ✅ `delete_note` - **PASSING**
**Functionality**: Deletes a note from the vault
**Test Results**:
- Successfully deleted test note `test-delete-me.md`
- Returns status confirmation: `{"status":"ok"}`
- Deleted note removed from vault (verified by attempting to read it)
- **Status**: ✅ Working correctly

**Test Case**:
1. ✅ Created `test-delete-me.md`
2. ✅ Verified it exists by reading it
3. ✅ Deleted it successfully
4. ✅ Confirmed deletion by attempting to read (returned "Note not found" error)

---

### ✅ `search_notes` - **MOSTLY PASSING**
**Functionality**: Full-text search with snippets and recency-aware scoring
**Test Results**:
- Successfully searches notes with proper queries
- Returns results with highlighted snippets using `<mark>` tags
- Supports `limit` parameter (tested with 5 and 10)
- Returns: `path`, `title`, `snippet` (with markdown highlights)
- **Status**: ⚠️ Working but has SQL syntax issues with special characters

**Test Cases**:
1. ✅ Search "test cursor" - Found matching note with highlights
2. ✅ Search "API" - Found multiple notes with proper highlighting
3. ❌ Search with special characters (apostrophe) - **Error**: `fts5: syntax error near "'"`

**Sample Output**:
```json
[
  {
    "path":"test-from-cursor-over-mcp.md",
    "title":"test from cursor over mcp",
    "snippet":"# <mark>test</mark> from <mark>cursor</mark> over mcp\n\nThis is a <mark>test</mark> note created from <mark>Cursor</mark> using the MCP server."
  }
]
```

**Known Issue**:
- ❌ SQL syntax error when search query contains special characters like apostrophes
- **Recommendation**: Implement input sanitization/escaping for search queries

---

### ✅ `get_backlinks` - **PASSING**
**Functionality**: Lists notes that reference the target note via wikilinks
**Test Results**:
- Successfully retrieved backlinks for existing notes
- Returns array of notes that reference the target
- Includes: `path` and `title` for each backlink
- **Status**: ✅ Working correctly

**Test Cases**:
1. ✅ Retrieved backlinks for "API Documentation.md" - Found 6 notes referencing it
2. ✅ Confirmed backlinks include the test note we created that references it

**Sample Output**:
```json
[
  {"path":"test-from-cursor-over-mcp.md","title":"test from cursor over mcp"},
  {"path":"Architecture Overview.md","title":"Architecture Overview"},
  {"path":"FAQ.md","title":"FAQ"}
]
```

---

### ✅ `get_tags` - **PASSING**
**Functionality**: Lists all tags and their associated note counts
**Test Results**:
- Successfully retrieved all tags from the vault
- Returns array with: `tag` name and `count` of notes using that tag
- **Status**: ✅ Working correctly

**Sample Output**:
```json
[
  {"tag":"guide","count":2},
  {"tag":"agents","count":1},
  {"tag":"ai","count":1},
  {"tag":"architecture","count":1},
  {"tag":"mcp","count":1}
]
```

---

## Summary

### Tools Tested: 7
### Passing: 6
### Passing with Issues: 1

| Tool | Status | Notes |
|------|--------|-------|
| `list_notes` | ✅ PASS | Works perfectly |
| `read_note` | ✅ PASS | Works perfectly, proper error handling |
| `write_note` | ✅ PASS | Works perfectly, updates timestamps correctly |
| `delete_note` | ✅ PASS | Works perfectly, proper cleanup |
| `search_notes` | ⚠️ PASS (with issues) | SQL syntax error with special characters |
| `get_backlinks` | ✅ PASS | Works perfectly |
| `get_tags` | ✅ PASS | Works perfectly |

## Issues Found

### 1. Search Query SQL Syntax Error
**Severity**: Medium
**Description**: The `search_notes` tool fails when search queries contain special characters like apostrophes
**Error Message**: `fts5: syntax error near "'"`
**Recommendation**: Implement proper SQL escaping/sanitization for FTS5 queries in the backend

## Recommendations

1. **Fix SQL injection vulnerability** in search functionality
2. **Add input validation** for all tool parameters
3. **Consider adding**:
   - Error codes for better error handling
   - Rate limiting information
   - Validation for path parameters (no '..' or '\\' as documented)

## Test Notes Created/Modified

- ✅ `test-from-cursor-over-mcp.md` - Created and updated
- ✅ `test-delete-me.md` - Created and deleted (for deletion testing)

## Conclusion

Overall, the MCP server is working well with 6 out of 7 tools functioning perfectly. The only issue is with the search functionality when handling special characters, which should be addressed for production use.

