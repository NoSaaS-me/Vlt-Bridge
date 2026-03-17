# ContentForge — Implementation Plan

## Tech Stack

- **Backend**: Python 3.11+ (daemon artifact harness, connector implementations)
- **Frontend**: TypeScript / React 19 (artifact iframe content, Settings UI extensions)
- **Connectors**: vlt-connectors package (declarative connectors with Fernet encryption)
- **Database**: SQLite (connector configs, cost tracking, prompt versions in context_nodes)
- **AI Providers**: OpenRouter, z.ai, HuggingFace Inference API, Google Gemini API, ElevenLabs API
- **Existing Infrastructure**: ANS EventBus, injection queues, connector_configs table, proxy_profiles, cronban scheduler

## Libraries

- `httpx` — HTTP client for AI provider APIs (already in deps)
- `httpx[socks]` / `socksio` — SOCKS proxy support for connector routing
- `cryptography` / `Fernet` — API key encryption (already in deps via vlt-connectors)
- `croniter` — Cron schedule parsing (already in deps)

## Project Structure

### Modified Files

```
packages/vlt-cli/src/vlt/daemon/
├── artifact_harness.py          # Extend: bidirectional _type message handling
├── artifact_service.py          # Extend: stdout reader dispatches _type messages
├── artifact_routes.py           # Extend: runbook endpoints, cost display
├── server.py                    # Extend: cronban runbook trigger type
└── cronban_routes.py            # Extend: artifact runbook scheduling

packages/vlt-cli/src/vlt/mcp/
└── artifact_tools.py            # Extend: runbook MCP tools

backend/src/services/
└── database.py                  # Extend: prompt_versions table, AI provider configs

backend/src/api/routes/
└── connectors.py                # Extend: AI provider CRUD endpoints

frontend/src/pages/
└── Settings.tsx                  # Extend: AI Providers section

frontend/src/components/artifacts/
├── ArtifactsCompositorView.tsx   # Extend: template picker
└── NewArtifactDialog.tsx         # Extend: template selection
```

### New Files

```
packages/vlt-connectors/src/vlt_connectors/connectors/
├── openrouter.py                # OpenRouter multimodal connector
├── zai.py                       # z.ai text connector
├── custom_openai.py             # Custom OpenAI-compatible connector
├── huggingface_inference.py     # HuggingFace Inference API connector
├── gemini.py                    # Google Gemini connector
├── elevenlabs.py                # ElevenLabs TTS connector
└── ai_base.py                   # Base class for AI model connectors

packages/vlt-cli/src/vlt/daemon/
├── harness_dispatcher.py        # _type message dispatch logic
├── cost_tracker.py              # Per-connector cost tracking & enforcement
└── artifact_templates/          # Template manifests + scaffold files
    ├── text_factory/
    ├── image_factory/
    └── blank/

frontend/src/components/settings/
└── AIProvidersSection.tsx        # AI Provider configuration UI

artifacts/content-factory/       # The Content Factory artifact itself
├── manifest.json
├── frontend/
│   ├── index.html               # Pipeline config + test + preview UI
│   ├── style.css
│   └── app.js                   # Pipeline editor, test runner, queue
├── backend/
│   ├── main.py                  # handle() dispatcher
│   ├── pipeline.py              # Stage execution engine
│   ├── queue.py                 # Content queue management
│   └── runbook.py               # Runbook save/load/execute
└── prompts/
    └── example.md               # Sample prompt template
```

## Architecture

### Bidirectional Harness Protocol

```
Artifact Backend (main.py)
    │ stdout: {"_type": "connector_call", "_id": "out_1", ...}
    ▼
Harness stdout reader (artifact_service.py)
    ├─→ _type=connector_call → harness_dispatcher → connector_service → response on stdin
    ├─→ _type=storage → artifact storage service
    ├─→ _type=event → artifact event bus
    └─→ _type=notification → ANS EventBus → injection_queues → Claude session
```

### Content Factory Data Flow

```
User → [Configure Pipeline] → [Test Button] → Backend handle("test")
    → Stage 1: connector_call(openrouter, generate_text) → text
    → Stage 2: connector_call(huggingface, text_to_image) → image
    → Stage 3: connector_call(gemini, analyze_image) → score
    → Result → Frontend Preview
    → User: Approve/Reject/Retry
    → Approve → storage(content/{id}) → event(content.approved)
```

### Runbook Automation Flow

```
Cronban Timer → daemon /api/artifacts/{id}/runbook/{name}/run
    → artifact_service.call_backend(artifact_id, "run_runbook", {name, params})
    → Backend executes pipeline with saved config
    → Auto-approve if score > threshold
    → event(content.approved) → Distribution Hub picks up
```

## Constitution Check

- [x] Uses existing connector_configs table with Fernet encryption
- [x] Extends existing ANS pub/sub (no new event system)
- [x] Follows artifact sandbox state machine (draft → building → testing → approved → deployed)
- [x] Cost enforcement at harness level (before connector proxy)
- [x] Prompt versioning reuses Oracle context_nodes pattern
- [x] Cronban integration extends existing scheduler
- [x] All AI provider credentials stored server-side, never in iframe
