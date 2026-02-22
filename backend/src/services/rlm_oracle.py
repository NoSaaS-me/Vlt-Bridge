"""RLM Oracle — Recursive Language Model inference-time harness.

Implements FR-001 (persistent REPL session), FR-002 (constant-size metadata),
FR-003 (metadata-only history), FR-004 (Final sentinel loop), FR-005 (sub_oracle),
FR-006 (programmatic sub-oracle), FR-019 (unchanged API), FR-021 (ANS events),
FR-022 (REPL stdout as SSE progress), FR-023 (Final streams as content).

Components:
    RLMSession         — Per-query session state (ephemeral, not persisted).
    RLMPromptBuilder   — Builds the 5-part system prompt and iteration messages.
    SubOracleCallable  — Callable injected as `sub_oracle` in REPL namespace.
    RLMOracleWrapper   — Drop-in replacement for OracleBTWrapper; drives the loop.

Part of 022-rlm-oracle (RLM Oracle replacement).
"""
from __future__ import annotations
