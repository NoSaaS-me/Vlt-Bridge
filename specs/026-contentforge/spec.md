# ContentForge — Feature Specification

## Overview

A modular content automation pipeline built as artifact sandbox plugins. Configurable content generation with prompt engineering workbench, automated QC gates, runbook automation, and multi-provider AI model connector support.

## User Stories

### US1 — Bidirectional Harness Communication [P1]
**As an** artifact backend developer,
**I want** my backend process to initiate connector calls, storage ops, and event emissions through the harness,
**So that** artifact backends can call AI model APIs and persist data without direct network access.

**Acceptance Criteria:**
- Backend can send `_type: connector_call` messages on stdout and receive responses on stdin
- Backend can send `_type: storage` messages for key-value operations
- Backend can send `_type: event` messages to emit IPC events
- Backend can send `_type: notification` messages that route through ANS
- Harness validates connector calls against artifact manifest declarations
- Harness enforces cost limits before proxying connector calls
- Existing harness request/response protocol continues to work unchanged

### US2 — AI Model Connectors [P1]
**As a** content creator,
**I want** to configure AI model providers (OpenRouter, z.ai, HuggingFace, Gemini, ElevenLabs, custom endpoints),
**So that** I can generate text, images, video, and audio through a unified connector interface.

**Acceptance Criteria:**
- OpenRouter connector with chat_completion, generate_text, generate_image, generate_video, generate_audio, list_models actions
- z.ai connector with chat_completion, generate_text actions
- Custom/OpenAI-compatible connector with configurable endpoint URL (multi-instance)
- HuggingFace Inference connector with text_to_image, image_to_image actions
- Google Gemini connector with generate_text, generate_image, generate_video, analyze_image actions
- ElevenLabs connector with text_to_speech, list_voices actions
- z.ai Vision connector wrapping z.ai Vision MCP Server (GLM-4.6V) — 8 vision tools: image_analysis, video_analysis, ui_diff_check, extract_text, diagnose_error, understand_diagram, analyze_visualization, ui_to_artifact. Free with z.ai subscription (zero per-call cost)
- Vision QC fallback: if configured model doesn't support vision, fall back to z.ai Vision; if z.ai not configured, skip with warning
- All API keys stored Fernet-encrypted
- Connection health check per provider

### US3 — AI Provider Settings UI [P1]
**As a** user,
**I want** to configure AI provider API keys and default models in the Settings page,
**So that** my artifacts can use these providers without embedding credentials.

**Acceptance Criteria:**
- Settings page section for AI Providers
- Per-provider: API key input (masked), default model selector, connection test button
- Support for: OpenRouter, z.ai, HuggingFace, Gemini, ElevenLabs
- Custom endpoint configuration (URL + optional API key + model name)
- Health check status indicator (green/red dot) per provider
- Encrypted storage using existing Fernet infrastructure

### US4 — Content Factory Pipeline Configuration [P2]
**As a** content creator,
**I want** to configure a multi-stage content generation pipeline with model selection, prompt editing, and parameters,
**So that** I can define exactly how my content is generated.

**Acceptance Criteria:**
- Pipeline is a sequence of stages (text_generation, image_generation, video_generation, audio_generation, vision_review, text_review, composite)
- Each stage specifies: connector, model, prompt_file, parameters
- Stages can reference previous stage output via `input_from` or `prompt_source`
- Pipeline config stored in artifact manifest
- Frontend shows stage list with add/remove/reorder controls
- Per-stage model selector filtered by connector capabilities

### US5 — Test Button & Preview [P2]
**As a** content creator,
**I want** to click a Test button that generates one piece of content and shows a preview,
**So that** I can iterate on my prompts and parameters until the output quality is satisfactory.

**Acceptance Criteria:**
- Test button executes pipeline for ONE content item
- Each stage shows progress indicator (generating → complete)
- Text output: rendered markdown with word count
- Image output: full-size preview with metadata (model, seed, params)
- Video output: embedded player
- Composite: multi-panel view of each component
- Approve/Reject/Edit&Retry/Variations controls
- History navigation (prev/next through test attempts)
- Cost display per generation
- No auto-generation on prompt edit — explicit button click only

### US6 — QC Gates & Auto-Approve [P2]
**As a** content creator,
**I want** QC stages that score content quality and optionally auto-approve above a threshold,
**So that** my pipeline can run autonomously while maintaining quality standards.

**Acceptance Criteria:**
- vision_review stage: sends image/video to vision model, returns score + feedback
- text_review stage: sends text to LLM, returns score + feedback against criteria
- Configurable `auto_approve_threshold` per QC stage (0 = disabled, require human)
- Content scoring on 0-10 scale
- Auto-approved content tagged with score and "auto-approved" flag
- QC stage can use any connector (OpenRouter vision, Gemini, custom Qwen 3.5)

### US7 — Prompt Versioning [P3]
**As a** content creator,
**I want** my prompt edits tracked in a version tree with branch/compare/rollback,
**So that** I can experiment with different prompt strategies and revert to what worked.

**Acceptance Criteria:**
- Each prompt save creates a new node in a version tree (reuse Oracle context_nodes pattern)
- Branch: create a new prompt variation from any previous version
- Compare: side-by-side view of two prompt versions' outputs
- Rollback: revert to a previous prompt version
- Version tree visualization in prompt editor

### US8 — Cost Controls [P2]
**As a** user,
**I want** to set per-connector daily cost limits on my artifacts,
**So that** I don't accidentally burn through my API credits.

**Acceptance Criteria:**
- Per-connector daily USD limit in artifact manifest cost_limits
- 0 = unlimited (no enforcement)
- Harness checks cumulative daily cost before each connector call
- When limit exceeded: connector call returns error, artifact enters error state
- Cost tracking: each connector call logs estimated cost in artifact storage
- Daily reset at midnight UTC
- Cost display in artifact UI (spent / limit per connector)

### US9 — Runbooks & Automation [P3]
**As a** content creator,
**I want** to save my tuned pipeline as a runbook and schedule it via cronban,
**So that** content is generated automatically on a schedule without my intervention.

**Acceptance Criteria:**
- Save current pipeline config as named runbook (prompts, models, params, thresholds)
- Runbooks stored in `.vlt/runbooks/{name}.json`
- Manual execution: "Run Runbook" button generates one batch
- Cronban integration: `vlt cron create --artifact {id} --runbook {name} --schedule "cron"`
- Webhook trigger: daemon endpoint to fire a runbook externally
- Event trigger: IPC event from another artifact can fire a runbook
- Runbook versioning (snapshots)

### US10 — Content Queue [P2]
**As a** content creator,
**I want** a queue of generated content with status tracking,
**So that** I can review, approve, and manage content before distribution.

**Acceptance Criteria:**
- Content items with states: generating, generated, qc_pending, approved, auto-approved, rejected, scheduled, emitted
- Queue UI showing all items with status badges
- Bulk approve/reject
- Content item detail view (all stage outputs, scores, costs)
- Filter by status
- IPC event emission on content approval (for Distribution Hub)

### US11 — Artifact Templates [P3]
**As a** user,
**I want** to create new artifacts from pre-built templates,
**So that** I can quickly start with a working pipeline configuration.

**Acceptance Criteria:**
- Template picker in New Artifact dialog (Blank, Text Factory, Image Factory, Video Factory)
- Templates include: manifest.json, scaffold frontend files, scaffold backend files, sample prompts
- Templates stored in daemon artifact_templates/ directory
- Template applies pipeline config, connector declarations, sample prompt files

## Success Criteria

- SC1: Artifact backend can call OpenRouter text generation through harness and display result
- SC2: User can configure OpenRouter API key in Settings and verify connection
- SC3: Content Factory generates text from a prompt via Test button and shows preview
- SC4: QC gate scores content and auto-approves above threshold
- SC5: Saved runbook executes on cronban schedule
- SC6: Cost limits prevent overspend when threshold is hit
