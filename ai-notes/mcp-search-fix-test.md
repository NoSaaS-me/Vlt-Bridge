# MCP Search Input Fix Test Report

## Test Date
2025-11-17

## Issue Being Tested
SQL syntax errors in FTS5 search queries when special characters (apostrophes, ampersands) are present.

## Test Results

### ❌ Still Failing

The search functionality is **still experiencing SQL syntax errors** with special characters:

1. **Query: `test's`**
   - **Error**: `fts5: syntax error near "'"`
   - **Status**: ❌ FAILING

2. **Query: `don't`**
   - **Error**: `fts5: syntax error near "'"`
   - **Status**: ❌ FAILING

3. **Query: `user's guide`**
   - **Error**: `fts5: syntax error near "'"`
   - **Status**: ❌ FAILING

4. **Query: `API & documentation`**
   - **Error**: `fts5: syntax error near "&"`
   - **Status**: ❌ FAILING

### ✅ Working Correctly

1. **Query: `getting started`**
   - **Status**: ✅ WORKING
   - **Results**: Found 5 matching notes with proper highlighting

## Code Analysis

### Sanitization Function Exists

The `_prepare_match_query` function in `backend/src/services/indexer.py` (lines 39-75) is implemented and should:
- Split queries on whitespace
- Wrap each token in double quotes
- Escape embedded double quotes
- Preserve trailing `*` for prefix searches

### Function Output Test

Tested the sanitization function directly:
```python
_prepare_match_query("test's")     # Returns: '"test\'s"'
_prepare_match_query("don't")      # Returns: '"don\'t"'
_prepare_match_query("API & documentation")  # Returns: '"API" "&" "documentation"'
```

The function is producing the expected output format.

## Possible Causes

1. **MCP Server Not Restarted**: The running MCP server process may not have picked up the code changes. The server needs to be restarted for changes to take effect.

2. **FTS5 Tokenizer Behavior**: The `unicode61` tokenizer with `porter` stemming may be treating apostrophes as word separators even inside double-quoted phrases. FTS5 might require additional escaping.

3. **SQL Parameter Binding**: While the query is being sanitized, FTS5 might be interpreting the apostrophe before the parameter binding occurs.

## Recommendations

1. **Restart MCP Server**: Ensure the MCP server process has been restarted to load the updated code.

2. **Test Direct SQL**: Test the sanitized queries directly against the SQLite database to verify if the issue is with FTS5 or the sanitization logic.

3. **Alternative Escaping**: Consider removing or replacing apostrophes in search queries, or using a different FTS5 query syntax.

4. **Add Logging**: Add logging to see the exact query string being passed to FTS5 MATCH.

## Other Tools Status

All other MCP tools are working correctly:
- ✅ `list_notes` - Working
- ✅ `read_note` - Working
- ✅ `write_note` - Working
- ✅ `delete_note` - Working
- ✅ `get_backlinks` - Working
- ✅ `get_tags` - Working
- ⚠️ `search_notes` - Failing with special characters

## Next Steps

1. Verify MCP server has been restarted
2. Test with direct database queries to isolate the issue
3. Consider additional escaping for apostrophes in FTS5 queries
4. Check FTS5 documentation for proper handling of special characters in quoted phrases

