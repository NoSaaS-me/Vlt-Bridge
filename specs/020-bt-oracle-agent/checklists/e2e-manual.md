# Manual E2E Test Checklist: BT Oracle

**Feature**: 020-bt-oracle-agent
**Phase**: 8 - Polish (T056)
**Created**: 2026-01-08

## Prerequisites

- [ ] Backend server running: `cd backend && uv run uvicorn src.api.main:app --reload`
- [ ] Frontend running: `cd frontend && npm run dev`
- [ ] Environment set: `export ORACLE_USE_BT=shadow`
- [ ] Shadow logs directory exists: `mkdir -p data/shadow_logs`

## Test Queries

Execute each query and verify the expected behavior.

### Query 1: Research - External Information
**Input**: "What's the weather in Paris?"
**Expected Type**: `research`
**Expected Signal**: `context_sufficient` or no signal (web search needed)
**Expected Behavior**:
- [ ] Should NOT search code or vault
- [ ] Should use web search (if available) or acknowledge limitation
- [ ] Should NOT emit `need_turn` for code/vault searches

### Query 2: Code - Implementation Understanding
**Input**: "How does the auth middleware work?"
**Expected Type**: `code`
**Expected Signal**: `context_sufficient` or `need_turn`
**Expected Behavior**:
- [ ] Should search code first
- [ ] Should locate `auth_middleware.py` or equivalent
- [ ] Should explain authentication flow
- [ ] Response cites specific file locations

### Query 3: Documentation - Project Memory
**Input**: "What did we decide about caching?"
**Expected Type**: `documentation`
**Expected Signal**: `context_sufficient` or `partial_answer`
**Expected Behavior**:
- [ ] Should search vault/threads first
- [ ] Should NOT search web first
- [ ] Acknowledges if no decision found

### Query 4: Conversational - Simple Acknowledgment
**Input**: "Thanks, that helps!"
**Expected Type**: `conversational`
**Expected Signal**: None (no signal expected)
**Expected Behavior**:
- [ ] NO tool calls made
- [ ] Quick conversational response
- [ ] Does not search anything

### Query 5: Action - Write Operation
**Input**: "Create a note about today's meeting"
**Expected Type**: `action`
**Expected Signal**: `need_turn` (awaiting content) or `context_sufficient`
**Expected Behavior**:
- [ ] Asks for meeting details OR
- [ ] Creates note if context available
- [ ] Uses vault write tool

### Query 6: Research - Comparison Analysis
**Input**: "Compare React vs Vue for our frontend"
**Expected Type**: `research`
**Expected Signal**: `need_turn` (gathering info) or `context_sufficient`
**Expected Behavior**:
- [ ] May search web for latest comparisons
- [ ] Provides balanced comparison
- [ ] Considers project-specific factors if context available

### Query 7: Code - Symbol Location
**Input**: "Where is the login function?"
**Expected Type**: `code`
**Expected Signal**: `context_sufficient`
**Expected Behavior**:
- [ ] Uses code search
- [ ] Returns specific file path and line
- [ ] Short, direct answer

### Query 8: Documentation - Architecture Understanding
**Input**: "Explain the BT runtime architecture"
**Expected Type**: `documentation` or `code`
**Expected Signal**: `context_sufficient`
**Expected Behavior**:
- [ ] Searches vault/docs or code
- [ ] Explains BT concepts (nodes, composites, blackboard)
- [ ] References relevant files/docs

### Query 9: Research - External Best Practices
**Input**: "Find the latest Python best practices"
**Expected Type**: `research`
**Expected Signal**: `context_sufficient` or `need_turn`
**Expected Behavior**:
- [ ] May use web search
- [ ] Provides current (2025-2026) practices
- [ ] Acknowledges if web unavailable

### Query 10: Conversational - Minimal Input
**Input**: "OK"
**Expected Type**: `conversational`
**Expected Signal**: None
**Expected Behavior**:
- [ ] NO tool calls
- [ ] Brief acknowledgment or question
- [ ] Fast response (<1s)

---

## Shadow Mode Verification

After running all queries, analyze shadow logs:

```bash
# List all shadow logs
ls -la data/shadow_logs/

# Check overall match rates
for f in data/shadow_logs/shadow_*.json; do
  echo "=== $f ==="
  jq '{
    query: .query_preview,
    match_rate: .match_rate,
    signal_match: .signal_match,
    bt_signals: [.bt_signals[].type],
    legacy_signals: [.legacy_signals[].type],
    discrepancy_count: .discrepancy_count,
    signal_discrepancy_count: .signal_discrepancy_count
  }' "$f"
done

# Calculate aggregate metrics
cat data/shadow_logs/shadow_*.json | jq -s '
{
  total_queries: length,
  avg_match_rate: ([.[].match_rate] | add / length),
  signal_matches: ([.[].signal_match] | map(select(. == true)) | length),
  queries_with_discrepancies: ([.[].discrepancy_count | select(. > 0)] | length)
}'
```

### Metrics Targets

| Metric | Target | Critical |
|--------|--------|----------|
| Average Match Rate | >= 90% | >= 80% |
| Signal Match Count | >= 7/10 | >= 5/10 |
| Error-severity Discrepancies | 0 | <= 2 |
| Queries with Discrepancies | <= 3 | <= 5 |

---

## Result Summary Template

```markdown
## E2E Test Results

**Date**: YYYY-MM-DD
**Tester**: [Name]
**Environment**: ORACLE_USE_BT=shadow

### Summary

| Metric | Value | Pass/Fail |
|--------|-------|-----------|
| Queries Tested | 10 | - |
| Average Match Rate | X.XX% | |
| Signal Matches | X/10 | |
| Error Discrepancies | X | |

### Per-Query Results

| # | Query | Type OK | Signal OK | Tool Calls OK | Notes |
|---|-------|---------|-----------|---------------|-------|
| 1 | Weather in Paris | | | | |
| 2 | Auth middleware | | | | |
| 3 | Caching decision | | | | |
| 4 | Thanks | | | | |
| 5 | Create note | | | | |
| 6 | React vs Vue | | | | |
| 7 | Login function | | | | |
| 8 | BT architecture | | | | |
| 9 | Python practices | | | | |
| 10 | OK | | | | |

### Issues Found

1. [Issue description, query #, severity]
2. ...

### Recommendation

- [ ] Ready for ORACLE_USE_BT=true
- [ ] Needs fixes (list below)
```

---

## Common Issues and Fixes

### Signal Not Emitted

**Symptoms**: Legacy emits signal, BT does not (or vice versa)
**Cause**: Different prompt templates or LLM behavior
**Fix**: Check `backend/src/prompts/oracle/signals.md` is loaded

### Wrong Query Classification

**Symptoms**: Code query treated as research, or vice versa
**Cause**: Keyword list gaps
**Fix**: Update `backend/src/services/query_classifier.py` keyword lists

### Tool Call Differences

**Symptoms**: BT makes different tool calls than legacy
**Cause**: Different prompt composition
**Fix**: Review `backend/src/services/prompt_composer.py` segment loading

### Timeout or Slow Response

**Symptoms**: BT takes significantly longer than legacy
**Cause**: BT overhead or additional tool calls
**Fix**: Check budget settings, review `ORACLE_MAX_TURNS`

---

## Post-Test Actions

1. [ ] Document results in this checklist
2. [ ] Create issues for any bugs found
3. [ ] Update quickstart.md if issues discovered
4. [ ] If all pass: Set `ORACLE_USE_BT=true` and re-test one query
5. [ ] Archive shadow logs to `data/shadow_logs/archive/`
