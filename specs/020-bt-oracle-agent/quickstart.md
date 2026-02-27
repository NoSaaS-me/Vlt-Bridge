# Quick Start: BT-Controlled Oracle Agent

**Feature**: 020-bt-oracle-agent
**Estimated Setup Time**: 15 minutes

## Prerequisites

- Python 3.11+
- Existing Vlt-Bridge development environment
- BT Universal Runtime (019) completed

## Getting Started

### 1. Enable Shadow Mode

Start with shadow mode to run both implementations in parallel:

```bash
export ORACLE_USE_BT=shadow
```

This runs:
- Legacy OracleAgent (output shown to user)
- New BT-based Oracle (runs silently in background)
- Comparison logs saved to `data/shadow_logs/`

### 2. Copy Prompt Templates

Copy the prompt templates from spec to backend:

```bash
cp -r specs/020-bt-oracle-agent/prompts/oracle/* backend/src/prompts/oracle/
```

### 3. Run Tests

Verify signal parsing and query classification:

```bash
cd backend
uv run pytest tests/unit/test_signal_parser.py -v
uv run pytest tests/unit/test_query_classifier.py -v
```

### 4. Test Signal Emission

Start the dev server and send a test query:

```bash
# Terminal 1: Start backend
cd backend && uv run uvicorn src.api.main:app --reload

# Terminal 2: Test query
curl -X POST http://localhost:8000/api/oracle/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather in Paris?", "project_id": "test"}'
```

Check shadow logs:
```bash
cat data/shadow_logs/shadow_*.json | jq '.discrepancies'
```

### 5. Switch to BT Mode

Once shadow mode shows high match rate (>95%):

```bash
export ORACLE_USE_BT=true
```

## Key Files

| File | Purpose |
|------|---------|
| `backend/src/models/signals.py` | Signal dataclasses |
| `backend/src/services/signal_parser.py` | XML signal extraction |
| `backend/src/services/query_classifier.py` | Query type classification |
| `backend/src/services/prompt_composer.py` | Dynamic prompt assembly |
| `backend/src/bt/conditions/signals.py` | BT signal conditions |
| `backend/src/prompts/oracle/*.md` | Prompt templates |

## Development Workflow

### Adding a New Signal Type

1. Add to `SignalType` enum in `models/signals.py`
2. Define fields in `data-model.md`
3. Add parsing logic to `signal_parser.py`
4. Add BT condition in `conditions/signals.py`
5. Update `prompts/oracle/signals.md` with examples
6. Add tests

### Modifying Query Classification

1. Edit keyword lists in `query_classifier.py`
2. Update derivation rules if needed
3. Add test cases for edge cases
4. Monitor accuracy in production logs

### Testing Signal Parsing

```python
from backend.src.services.signal_parser import parse_signal

response = """
Here's what I found...

<signal type="context_sufficient">
  <sources_found>3</sources_found>
  <confidence>0.85</confidence>
</signal>
"""

signal = parse_signal(response)
assert signal.type == "context_sufficient"
assert signal.confidence == 0.85
assert signal.fields["sources_found"] == 3
```

### Testing Prompt Composition

```python
from backend.src.services.prompt_composer import compose_prompt
from backend.src.services.query_classifier import classify_query

classification = classify_query("How does the auth middleware work?")
assert classification.query_type == "code"

prompt = compose_prompt(classification, project_context={...})
assert "code-analysis" in prompt  # Included code segment
assert "signals" in prompt  # Always included
```

## Troubleshooting

### Signal Not Parsed

- Check response ends with signal (not inline)
- Verify XML is well-formed
- Check for multiple signals (only first is parsed)
- Look for parsing errors in logs

### Wrong Query Classification

- Check keyword lists in classifier
- Consider adding domain-specific keywords
- Review misclassified examples for patterns

### Shadow Mode Discrepancies

- High mismatch rate: Check prompt differences
- Timing issues: BT may be slower/faster
- Missing signals: Check prompt includes signals.md

## Configuration

| Env Variable | Values | Default | Description |
|--------------|--------|---------|-------------|
| ORACLE_USE_BT | false/true/shadow | false | Oracle mode |
| ORACLE_MAX_TURNS | integer | 30 | Max turns per query |
| ORACLE_PROMPT_BUDGET | integer | 8000 | Max prompt tokens |

## Monitoring

### Key Metrics

- **Signal parse rate**: % of responses with valid signals
- **Query classification accuracy**: Manual sampling
- **Shadow match rate**: BT vs legacy comparison
- **Turn consumption**: Avg turns per query

### Logging

Signal events are logged at INFO level:
```
INFO oracle.signals: Parsed signal type=context_sufficient confidence=0.85
INFO oracle.classification: Query classified as code (matched: ['how does', 'middleware'])
```

## Next Steps

1. Review shadow mode logs for discrepancies
2. Tune query classification keywords
3. Collect training data for future BERT classifier
4. Monitor signal emission patterns
