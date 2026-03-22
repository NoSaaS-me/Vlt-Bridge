# Tasks: ContentForge

**Input**: Design documents from `/specs/026-contentforge/`
**Prerequisites**: plan.md, spec.md, system-spec (Ai-notes/03-16-2026/ContentForge-Spec/)

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to

---

## Phase 1: Setup

**Purpose**: Spec registration and directory scaffolding

- [ ] T001 Create 026-contentforge spec directory with spec.md and plan.md in specs/026-contentforge/
- [ ] T002 Create AI model connector base class in packages/vlt-connectors/src/vlt_connectors/connectors/ai_base.py
- [ ] T003 [P] Create harness dispatcher module skeleton in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py
- [ ] T004 [P] Create cost tracker module skeleton in packages/vlt-cli/src/vlt/daemon/cost_tracker.py
- [ ] T005 [P] Create artifact templates directory structure at packages/vlt-cli/src/vlt/daemon/artifact_templates/

---

## Phase 2: Foundational — Bidirectional Harness (Blocking)

**Purpose**: Core infrastructure that MUST be complete before any connector or Content Factory work

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [ ] T006 [US1] Add message ID field to harness request/response protocol in packages/vlt-cli/src/vlt/daemon/artifact_harness.py — extend stdin/stdout JSON to include `id` field for request correlation
- [ ] T007 [US1] Implement `_type` message detection in harness stdout reader in packages/vlt-cli/src/vlt/daemon/artifact_service.py — modify `_backend_stdout_reader()` to distinguish response messages from `_type`-prefixed backend-initiated messages
- [ ] T008 [US1] Implement harness dispatcher for connector_call routing in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — receive `_type: connector_call` messages, validate against manifest, proxy to connector service, send response back on stdin
- [ ] T009 [US1] Implement harness dispatcher for storage routing in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — handle `_type: storage` messages (get/set/list/delete) routed to artifact storage service
- [ ] T010 [US1] Implement harness dispatcher for event emission in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — handle `_type: event` messages routed to artifact event bus
- [ ] T011 [US1] Implement harness dispatcher for notification routing in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — handle `_type: notification` messages routed to ANS EventBus with severity mapping
- [ ] T012 [US1] Implement cost limit enforcement in packages/vlt-cli/src/vlt/daemon/cost_tracker.py — check cumulative daily cost before proxying connector calls, return error if limit exceeded
- [ ] T013 [US1] Implement cost tracking persistence in packages/vlt-cli/src/vlt/daemon/cost_tracker.py — log each connector call cost to artifact storage at `.vlt/costs/{date}.json`, daily reset at midnight UTC
- [ ] T014 [US1] Wire harness dispatcher into artifact_service backend process management in packages/vlt-cli/src/vlt/daemon/artifact_service.py — connect dispatcher to start_backend/stop_backend lifecycle, pass manifest for validation
- [ ] T015 [US1] Add manifest connector validation in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — reject connector calls for connectors not declared in the artifact's manifest
- [ ] T016 [US1] Unit tests for bidirectional harness protocol in packages/vlt-cli/src/vlt/tests/unit/test_harness_dispatcher.py — test message routing, manifest validation, cost enforcement, error handling

**Checkpoint**: Bidirectional harness ready — artifact backends can now call connectors, storage, events, and notifications through stdout

---

## Phase 3: AI Model Connectors (US2) [P1]

**Goal**: Implement all AI model provider connectors so artifacts can generate content

**Independent Test**: Configure an API key, call `list_models`, call `generate_text` from a test artifact

### Implementation

- [ ] T017 [P] [US2] Implement OpenRouter connector in packages/vlt-connectors/src/vlt_connectors/connectors/openrouter.py — actions: chat_completion, generate_text, generate_image, generate_video, generate_audio, list_models. Wrap existing openrouter_client.py patterns
- [ ] T018 [P] [US2] Implement z.ai connector in packages/vlt-connectors/src/vlt_connectors/connectors/zai.py — actions: chat_completion, generate_text
- [ ] T019 [P] [US2] Implement Custom/OpenAI-compatible connector in packages/vlt-connectors/src/vlt_connectors/connectors/custom_openai.py — configurable endpoint URL, optional API key, multi-instance support
- [ ] T020 [P] [US2] Implement HuggingFace Inference connector in packages/vlt-connectors/src/vlt_connectors/connectors/huggingface_inference.py — actions: text_to_image, image_to_image, text_generation with provider selection (fal, replicate, together)
- [ ] T021 [P] [US2] Implement Google Gemini connector in packages/vlt-connectors/src/vlt_connectors/connectors/gemini.py — actions: generate_text, generate_image (Imagen 3), generate_video (Veo 2), analyze_image
- [ ] T022 [P] [US2] Implement ElevenLabs connector in packages/vlt-connectors/src/vlt_connectors/connectors/elevenlabs.py — actions: text_to_speech, list_voices. Wrap existing ElevenLabs integration
- [ ] T023a [P] [US2] Implement z.ai Vision connector in packages/vlt-connectors/src/vlt_connectors/connectors/zai_vision.py — wraps z.ai Vision MCP Server (GLM-4.6V). Actions: image_analysis, video_analysis, ui_diff_check, extract_text, diagnose_error, understand_diagram, analyze_visualization, ui_to_artifact. Spawns `npx -y @z_ai/mcp-server` subprocess with MCP STDIO protocol
- [ ] T023b [US2] Implement vision QC fallback in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — when a vision_review stage's configured model doesn't support vision, fall back to zai_vision connector; if zai_vision not configured, skip QC with warning
- [ ] T024 [US2] Register all AI connectors in packages/vlt-connectors/src/vlt_connectors/connectors/__init__.py — add to CONNECTOR_REGISTRY with proper imports (openrouter, zai, custom_openai, huggingface_inference, gemini, elevenlabs, zai_vision)
- [ ] T024 [US2] Add connector service integration for AI connectors in backend/src/services/connector_service.py — ensure AI connectors are discoverable and callable through the existing connector proxy
- [ ] T025 [US2] Add AI provider config CRUD endpoints in backend/src/api/routes/connectors.py — GET/PUT/DELETE for AI provider configs with connection health check endpoint
- [ ] T026 [US2] Unit tests for AI connectors in packages/vlt-connectors/tests/test_ai_connectors.py — test action schemas, config validation, mock API calls

**Checkpoint**: All AI connectors registered and callable. `connector_call("openrouter", "generate_text", {...})` works through the harness

---

## Phase 4: AI Provider Settings UI (US3) [P1]

**Goal**: Users can configure AI provider API keys and verify connections in the Settings page

**Independent Test**: Open Settings, add an OpenRouter API key, click "Test Connection", see green checkmark

### Implementation

- [ ] T027 [P] [US3] Create AIProvidersSection component in frontend/src/components/settings/AIProvidersSection.tsx — provider cards with masked API key input, default model selector, connection test button, health status dot
- [ ] T028 [P] [US3] Create CustomEndpointDialog component in frontend/src/components/settings/CustomEndpointDialog.tsx — dialog for adding custom OpenAI-compatible endpoints (URL, optional API key, model name)
- [ ] T029 [US3] Integrate AIProvidersSection into Settings page in frontend/src/pages/Settings.tsx — add as new tab alongside existing Oracle/Vision tabs
- [ ] T030 [US3] Add AI provider API service in frontend/src/services/provider-api.ts — REST client for AI provider CRUD, connection test, model listing
- [ ] T031 [US3] Add backend health check endpoint per provider in backend/src/api/routes/connectors.py — POST /api/connectors/{name}/health-check that validates API key and returns model count

**Checkpoint**: Users can add/edit/test AI provider connections from the Settings page

---

## Phase 5: Content Factory — Pipeline Config (US4) [P2]

**Goal**: Users can configure a multi-stage content generation pipeline within the Content Factory artifact

**Independent Test**: Create a Content Factory artifact from template, add text_generation + text_review stages, save config

### Implementation

- [ ] T032 [US4] Create Content Factory artifact template in packages/vlt-cli/src/vlt/daemon/artifact_templates/text_factory/ — manifest.json with pipeline config schema, scaffold frontend (index.html, style.css, app.js), scaffold backend (main.py, pipeline.py), sample prompt file
- [ ] T033 [US4] Implement pipeline configuration data model in artifacts/content-factory/backend/pipeline.py — PipelineConfig, StageConfig dataclasses, stage type validation, input_from/prompt_source resolution
- [ ] T034 [US4] Implement pipeline stage executor in artifacts/content-factory/backend/pipeline.py — execute_stage() dispatches to appropriate connector based on stage type, chains stage outputs
- [ ] T035 [US4] Implement Content Factory backend handler in artifacts/content-factory/backend/main.py — handle() dispatcher for get_config, update_config, test, generate, approve, reject, get_queue, get_history
- [ ] T036 [US4] Implement Content Factory frontend pipeline editor in artifacts/content-factory/frontend/app.js — stage list with add/remove/reorder, per-stage model selector, prompt file editor, parameter sliders
- [ ] T037 [US4] Implement template loading in daemon artifact creation in packages/vlt-cli/src/vlt/daemon/artifact_service.py — when `template` field specified in create request, copy template files to artifact disk_path
- [ ] T038 [US4] Extend NewArtifactDialog with template picker in frontend/src/components/artifacts/NewArtifactDialog.tsx — dropdown/cards for available templates (Blank, Text Factory, Image Factory)

**Checkpoint**: Content Factory artifact created from template, pipeline stages configured through the UI

---

## Phase 6: Test Button & Preview (US5) [P2]

**Goal**: Users click Test to generate one piece of content and see an interactive preview

**Independent Test**: Configure a text_generation stage with OpenRouter, click Test, see rendered text in preview panel

### Implementation

- [ ] T039 [US5] Implement test execution in Content Factory backend in artifacts/content-factory/backend/main.py — handle("test") runs pipeline for one item, returns all stage outputs with progress events
- [ ] T040 [US5] Implement progress reporting via harness notifications in artifacts/content-factory/backend/pipeline.py — emit `_type: notification` for each stage start/complete/error
- [ ] T041 [US5] Implement preview panel in Content Factory frontend in artifacts/content-factory/frontend/app.js — text: rendered markdown with word count; image: full-size preview with metadata; cost display
- [ ] T042 [US5] Implement iteration controls in Content Factory frontend in artifacts/content-factory/frontend/app.js — Approve/Reject/Edit&Retry/Variations buttons, history navigation (prev/next)
- [ ] T043 [US5] Implement test history storage in Content Factory backend in artifacts/content-factory/backend/main.py — store each test result in `.vlt/storage/test_history/{id}.json` with all stage outputs

**Checkpoint**: Full test → preview → iterate loop working with real OpenRouter API calls

---

## Phase 7: QC Gates & Auto-Approve (US6) + Cost Controls (US8) [P2]

**Goal**: QC stages score content and auto-approve above threshold; cost limits prevent overspend

**Independent Test**: Add a text_review QC stage with auto_approve_threshold=8.0, generate content that scores above/below threshold, verify auto-approve behavior. Set a $0.01 cost limit, generate until limit hit, verify error.

### Implementation

- [ ] T044 [P] [US6] Implement vision_review stage executor in artifacts/content-factory/backend/pipeline.py — send image/video to vision model connector, parse score + feedback from response
- [ ] T045 [P] [US6] Implement text_review stage executor in artifacts/content-factory/backend/pipeline.py — send text to LLM with scoring criteria prompt, parse score + feedback
- [ ] T046 [US6] Implement auto-approve logic in Content Factory backend in artifacts/content-factory/backend/queue.py — after QC stage, if score >= auto_approve_threshold, automatically approve and emit content.approved event
- [ ] T047 [US6] Implement QC score display in Content Factory frontend in artifacts/content-factory/frontend/app.js — quality bar, score number, feedback text, auto-approved badge
- [ ] T048 [US8] Implement cost limit configuration in Content Factory frontend in artifacts/content-factory/frontend/app.js — per-connector daily USD input in pipeline config panel
- [ ] T049 [US8] Implement cost display widget in Content Factory frontend in artifacts/content-factory/frontend/app.js — spent/limit per connector, total daily spend
- [ ] T050 [US8] Wire cost_tracker into manifest cost_limits in packages/vlt-cli/src/vlt/daemon/harness_dispatcher.py — read cost_limits from artifact manifest, pass to cost_tracker for enforcement

**Checkpoint**: Content auto-approved when QC score exceeds threshold; cost limits enforced with clear error messaging

---

## Phase 8: Content Queue (US10) [P2]

**Goal**: Queue of generated content with status tracking and approval management

**Independent Test**: Generate several content items, see them in queue with statuses, bulk approve, verify IPC event emission

### Implementation

- [ ] T051 [US10] Implement content queue data model in artifacts/content-factory/backend/queue.py — ContentItem with states: generating, generated, qc_pending, approved, auto_approved, rejected, scheduled, emitted
- [ ] T052 [US10] Implement queue management in Content Factory backend in artifacts/content-factory/backend/queue.py — add_item, approve, reject, bulk_approve, list_by_status, get_item
- [ ] T053 [US10] Implement IPC event emission on approval in artifacts/content-factory/backend/queue.py — emit `content.approved` via harness `_type: event` with content payload
- [ ] T054 [US10] Implement queue UI in Content Factory frontend in artifacts/content-factory/frontend/app.js — content item cards with status badges, filter by status, bulk actions, detail view on click
- [ ] T055 [US10] Implement queue persistence in artifacts/content-factory/backend/queue.py — store queue items in `.vlt/storage/content/{id}.json` with all stage outputs, scores, costs

**Checkpoint**: Full content lifecycle working: generate → QC → auto-approve/manual → emit event

---

## Phase 9: Prompt Versioning (US7) [P3]

**Goal**: Prompt edits tracked in a version tree with branch/compare/rollback

**Independent Test**: Edit a prompt, see version history, branch to new version, compare two versions' outputs side-by-side, rollback

### Implementation

- [ ] T056 [US7] Add prompt_versions table to database schema in backend/src/services/database.py — reuse context_nodes pattern: id, parent_id, artifact_id, prompt_text, model, test_result_id, created_at
- [ ] T057 [US7] Implement prompt version service in artifacts/content-factory/backend/main.py — save_prompt_version, list_versions, get_version, branch_from, rollback_to
- [ ] T058 [US7] Implement prompt version tree UI in artifacts/content-factory/frontend/app.js — version list with tree visualization, branch button, compare button (side-by-side outputs), rollback button
- [ ] T059 [US7] Link test results to prompt versions in artifacts/content-factory/backend/main.py — each test run records which prompt version was used, enabling version-to-output comparison

**Checkpoint**: Prompt versioning working with branch/compare/rollback capabilities

---

## Phase 10: Runbooks & Automation (US9) [P3]

**Goal**: Save pipeline config as runbook, schedule via cronban, trigger via webhook

**Independent Test**: Tune pipeline, save as runbook, create cronban schedule, verify auto-execution produces content

### Implementation

- [ ] T060 [US9] Implement runbook save/load in artifacts/content-factory/backend/runbook.py — save_runbook (freeze pipeline config + prompts + thresholds), load_runbook, list_runbooks, delete_runbook
- [ ] T061 [US9] Implement runbook execution in artifacts/content-factory/backend/runbook.py — run_runbook executes saved pipeline, applies auto-approve thresholds, emits events
- [ ] T062 [US9] Add runbook REST endpoints to daemon in packages/vlt-cli/src/vlt/daemon/artifact_routes.py — POST /{id}/runbook/{name}/run, GET /{id}/runbooks, POST /{id}/runbooks (save)
- [ ] T063 [US9] Implement cronban runbook trigger type in packages/vlt-cli/src/vlt/daemon/cronban_routes.py — new trigger type "artifact_runbook" that calls artifact backend run_runbook action
- [ ] T064 [US9] Implement webhook trigger endpoint in packages/vlt-cli/src/vlt/daemon/artifact_routes.py — POST /{id}/webhook/{runbook_name} fires runbook with optional params from request body
- [ ] T065 [US9] Implement event-triggered runbook in packages/vlt-cli/src/vlt/daemon/artifact_service.py — IPC event subscription that fires a configured runbook when event matches
- [ ] T066 [US9] Implement runbook UI in Content Factory frontend in artifacts/content-factory/frontend/app.js — save current config as runbook button, runbook list, manual run button, schedule link to cronban

**Checkpoint**: Full automation loop: tune → save runbook → schedule → auto-generate → auto-approve → emit events

---

## Phase 11: Artifact Templates (US11) [P3]

**Goal**: New artifacts can be created from pre-built templates

**Independent Test**: Create new artifact, select "Text Factory" template, verify pipeline config and prompt files are scaffolded

### Implementation

- [ ] T067 [P] [US11] Create blank template in packages/vlt-cli/src/vlt/daemon/artifact_templates/blank/ — minimal manifest.json, empty index.html, empty style.css
- [ ] T068 [P] [US11] Create text factory template in packages/vlt-cli/src/vlt/daemon/artifact_templates/text_factory/ — manifest with text_generation + text_review stages, full frontend with pipeline editor, backend with pipeline executor, sample blog prompt
- [ ] T069 [P] [US11] Create image factory template in packages/vlt-cli/src/vlt/daemon/artifact_templates/image_factory/ — manifest with text_gen + image_gen + vision_review stages, image preview frontend, sample image prompt
- [ ] T070 [US11] Implement template listing endpoint in packages/vlt-cli/src/vlt/daemon/artifact_routes.py — GET /api/artifacts/templates returns available templates with descriptions
- [ ] T071 [US11] Wire template picker into NewArtifactDialog in frontend/src/components/artifacts/NewArtifactDialog.tsx — template selection cards, pass template name in create request

**Checkpoint**: Users can create Content Factory artifacts from templates with working pipeline configs out of the box

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Integration testing, cleanup, and documentation

- [ ] T072 End-to-end integration test: create Content Factory from template → configure OpenRouter → test generate → approve → verify IPC event emission
- [ ] T073 [P] Update CLAUDE.md with ContentForge architecture notes (connector list, harness protocol, runbook pattern)
- [ ] T074 [P] Update artifact MCP tools for runbook operations in packages/vlt-cli/src/vlt/mcp/artifact_tools.py — vlt_artifact_run_runbook, vlt_artifact_list_runbooks
- [ ] T075 [P] Error handling audit across all new code — ensure cost limit errors, connector failures, and harness timeouts produce clear user-facing messages
- [ ] T076 Performance check: verify connector call latency through harness proxy is < 100ms overhead vs direct call

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **BLOCKS all other phases**
- **Phase 3 (Connectors, US2)**: Depends on Phase 2
- **Phase 4 (Settings UI, US3)**: Depends on Phase 3 (needs connector health check)
- **Phase 5 (Pipeline Config, US4)**: Depends on Phase 2 + Phase 3
- **Phase 6 (Test Button, US5)**: Depends on Phase 5
- **Phase 7 (QC + Cost, US6/US8)**: Depends on Phase 6
- **Phase 8 (Queue, US10)**: Depends on Phase 6
- **Phase 9 (Prompt Versioning, US7)**: Depends on Phase 6 (can parallelize with Phase 7/8)
- **Phase 10 (Runbooks, US9)**: Depends on Phase 7 + Phase 8
- **Phase 11 (Templates, US11)**: Depends on Phase 5 (can parallelize with Phase 6+)
- **Phase 12 (Polish)**: Depends on all phases

### User Story Dependencies

- **US1 (Harness)**: No user story deps — foundational
- **US2 (Connectors)**: Depends on US1
- **US3 (Settings)**: Depends on US2
- **US4 (Pipeline Config)**: Depends on US1 + US2
- **US5 (Test Button)**: Depends on US4
- **US6 (QC Gates)**: Depends on US5
- **US7 (Prompt Versioning)**: Depends on US5, parallelizable with US6/US8/US10
- **US8 (Cost Controls)**: Depends on US1, parallelizable with US6
- **US9 (Runbooks)**: Depends on US6 + US10
- **US10 (Queue)**: Depends on US5
- **US11 (Templates)**: Depends on US4, parallelizable with US5+

### Parallel Opportunities

```
After Phase 2 (Foundational):
  ├── Phase 3: AI Connectors (T017-T022 all parallel)
  │   └── Phase 4: Settings UI (T027-T028 parallel)
  └── (blocked until connectors done)

After Phase 5 (Pipeline Config):
  ├── Phase 6: Test Button
  │   ├── Phase 7: QC + Cost (T044-T045 parallel)
  │   ├── Phase 8: Queue
  │   └── Phase 9: Prompt Versioning (parallel with 7+8)
  │       └── Phase 10: Runbooks (after 7+8)
  └── Phase 11: Templates (parallel with 6+)
```

---

## Parallel Example: Phase 3 — AI Connectors

```bash
# All six connector implementations are fully independent:
Task: T017 "Implement OpenRouter connector in .../openrouter.py"
Task: T018 "Implement z.ai connector in .../zai.py"
Task: T019 "Implement Custom connector in .../custom_openai.py"
Task: T020 "Implement HuggingFace Inference connector in .../huggingface_inference.py"
Task: T021 "Implement Google Gemini connector in .../gemini.py"
Task: T022 "Implement ElevenLabs connector in .../elevenlabs.py"
```

---

## Implementation Strategy

### MVP First (Phases 1-6 only)

1. Phase 1: Setup
2. Phase 2: Bidirectional harness (**CRITICAL** — blocks everything)
3. Phase 3: OpenRouter connector only (T017 — skip others initially)
4. Phase 4: Settings UI for OpenRouter only
5. Phase 5: Content Factory text pipeline
6. Phase 6: Test button with text preview
7. **STOP and VALIDATE**: SC1 + SC3 — can generate text through Test button

### Incremental Delivery

1. **MVP**: Harness + OpenRouter + Text Factory + Test Button = generate text content
2. **+QC**: Add QC gates + auto-approve = automated quality control
3. **+Queue**: Content queue + IPC events = ready for distribution
4. **+Automation**: Runbooks + cronban = fully automated generation
5. **+Media**: Image/video/audio connectors = full multimodal pipeline
6. **+Templates**: Out-of-box experience for new users

---

## Notes

- Total tasks: 78
- Phase 2 (harness bidir) is the critical path — ~30% new code on 70% existing ANS infrastructure
- 6 connectors are fully parallelizable (Phase 3)
- Content Factory is both an artifact AND uses artifact infrastructure — it's the first real "eating our own dogfood" test
- Cost tracker should use optimistic estimates (model pricing from OpenRouter API) since exact costs aren't known until billing
- The Content Factory artifact lives in the artifact storage directory, not in the repo — it's user data, not code
