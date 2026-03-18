"""Content Factory — pipeline stage executor.

Executes a sequence of pipeline stages, passing outputs forward.
Each stage calls a connector through the artifact harness.

Stage types:
  text_generation  — LLM text output
  text_review      — LLM-based scoring of previous text
  image_generation — Image generation from prompt or previous text
  vision_review    — Vision model scoring of generated image
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# harness_call is available because the harness adds the backend dir to sys.path
# and registers itself. Import at call-time to avoid module-load-order issues.
try:
    from artifact_harness import harness_call  # type: ignore[import]
except ImportError:
    def harness_call(call_type: str, payload: dict) -> dict:  # type: ignore[misc]
        raise RuntimeError("artifact_harness not available")

_BACKEND_DIR = Path(__file__).parent
_PROMPTS_DIR = _BACKEND_DIR.parent / "prompts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notify(message: str, severity: str = "info") -> None:
    """Emit a notification line on stdout (consumed by the harness)."""
    sys.stdout.write(json.dumps({
        "_type": "notification",
        "severity": severity,
        "message": message,
    }) + "\n")
    sys.stdout.flush()


def _call_connector(connector: str, action: str, params: dict) -> dict:
    """Invoke a connector through the harness bidirectional protocol.

    Returns the unwrapped connector result dict.  The full response chain is:
      harness_call → {"_id", "_type", "result": <connector_result>}
      connector_result → {"success": bool, "data": {...}} | {"success": False, "error": "..."}

    This function unwraps both layers and returns the inner data dict directly,
    or raises on failure.
    """
    raw = harness_call("connector_call", {
        "connector": connector,
        "action": action,
        "params": params,
    })

    # Layer 1: harness response envelope — extract "result" field
    connector_result = raw.get("result", raw)

    # Layer 2: connector envelope — {"success": bool, "data": {...}}
    if isinstance(connector_result, dict):
        if connector_result.get("success") is False:
            error = connector_result.get("error", "Unknown connector error")
            raise RuntimeError(f"Connector {connector}.{action} failed: {error}")
        # Unwrap data if present
        data = connector_result.get("data")
        if data is not None:
            return data
    return connector_result


def _load_prompt_file(filename: str) -> str:
    """Load a prompt template. Resolves relative to artifact root first, then prompts/ dir."""
    path = _BACKEND_DIR.parent / filename
    if path.exists():
        return path.read_text()
    path = _PROMPTS_DIR / filename
    if path.exists():
        return path.read_text()
    raise FileNotFoundError(f"Prompt file not found: {filename}")


def _apply_topic(text: str, topic: str) -> str:
    """Replace {{topic}} placeholder in prompt templates."""
    return text.replace("{{topic}}", topic)


def _extract_text_output(stage_result: dict) -> str:
    """Pull the canonical text output from a stage result dict."""
    return stage_result.get("output", "")


def _extract_image_output(stage_result: dict) -> str:
    """Pull the image URL/base64 from a stage result dict."""
    return stage_result.get("output", "")


def _parse_score_from_text(text: str) -> tuple[float, str]:
    """Attempt to parse a JSON score block from LLM review output.

    Returns (score, feedback). Falls back gracefully.
    """
    # Try JSON block first
    try:
        # Strip markdown code fences if present
        clean = text.strip()
        if clean.startswith("```"):
            lines = clean.splitlines()
            clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        data = json.loads(clean)
        score = float(data.get("score", 0))
        feedback = str(data.get("feedback", ""))
        return score, feedback
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: scan for "score: N" or "Score: N" patterns
    import re
    match = re.search(r"[Ss]core[:\s]+([0-9]+(?:\.[0-9]+)?)", text)
    score = float(match.group(1)) if match else 0.0
    return score, text


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------

def _execute_text_generation(stage: dict, results: dict, topic: str) -> dict:
    """Call an LLM connector to generate text."""
    connector = stage.get("connector", "openrouter")
    action = stage.get("action", "generate_text")
    model = stage.get("model", "openai/gpt-4o-mini")

    # Build system prompt: inline system_prompt overrides prompt_file
    inline_prompt = (stage.get("system_prompt") or "").strip()
    prompt_file = stage.get("prompt_file")
    if inline_prompt:
        system_prompt = _apply_topic(inline_prompt, topic)
    elif prompt_file:
        system_prompt = _apply_topic(_load_prompt_file(prompt_file), topic)
    else:
        system_prompt = _apply_topic(stage.get("prompt", ""), topic)

    # Build user message; prepend context from a previous stage if requested
    user_message = stage.get("user_message", topic or "Generate content.")
    input_from = stage.get("input_from")
    if input_from and input_from in results:
        prior_text = _extract_text_output(results[input_from])
        if prior_text:
            user_message = f"Context:\n{prior_text}\n\n{user_message}"

    # Stage-level params or nested params dict (manifest uses params.temperature etc.)
    stage_params = stage.get("params", {})

    params: dict[str, Any] = {
        "model": model,
        "prompt": user_message,
        "system_prompt": system_prompt,
    }
    max_tokens = stage.get("max_tokens") or stage_params.get("max_tokens")
    if max_tokens:
        params["max_tokens"] = str(max_tokens)
    temperature = stage.get("temperature") or stage_params.get("temperature")
    if temperature is not None:
        params["temperature"] = str(temperature)

    data = _call_connector(connector, action, params)

    # data is the unwrapped connector response, e.g.:
    #   {"content": "...", "finish_reason": "stop", "model": "...", "usage": {...}}
    if isinstance(data, str):
        output_text = data
        tokens = 0
    elif isinstance(data, dict):
        output_text = (
            data.get("content")
            or data.get("text")
            or data.get("output")
            or str(data)
        )
        usage = data.get("usage", {})
        tokens = usage.get("total_tokens", 0) if isinstance(usage, dict) else 0
    else:
        output_text = str(data)
        tokens = 0

    return {
        "output": output_text,
        "model": model,
        "tokens": tokens,
    }


def _execute_text_review(stage: dict, results: dict, topic: str) -> dict:
    """Score a previous stage's text output using an LLM reviewer."""
    connector = stage.get("connector", "openrouter")
    action = stage.get("action", "generate_text")
    model = stage.get("model", "openai/gpt-4o-mini")
    threshold = float(stage.get("threshold", 7.0))
    # Also check auto_approve_threshold (manifest uses this name)
    if "auto_approve_threshold" in stage:
        threshold = float(stage["auto_approve_threshold"])

    # Find source text
    input_from = stage.get("input_from")
    if input_from and input_from in results:
        source_text = _extract_text_output(results[input_from])
    else:
        # Fall back to last stage that produced text output
        source_text = ""
        for r in reversed(list(results.values())):
            if "output" in r and isinstance(r["output"], str):
                source_text = r["output"]
                break

    criteria = stage.get("criteria", "quality, clarity, engagement")
    if isinstance(criteria, list):
        criteria = ", ".join(criteria)

    # Load review prompt from file if specified, otherwise use inline
    review_prompt_file = stage.get("review_prompt_file")
    if review_prompt_file:
        system_prompt = _load_prompt_file(review_prompt_file)
    else:
        system_prompt = (
            "You are a strict content quality reviewer. "
            "Evaluate the provided content and respond ONLY with valid JSON.\n"
            "Format: {\"score\": <number 0-10>, \"feedback\": \"<2-3 sentence explanation>\"}\n"
            "Do NOT include any text outside the JSON object."
        )

    user_prompt = (
        f"Score this content 0-10 on: {criteria}.\n\n"
        f"---BEGIN CONTENT---\n{source_text}\n---END CONTENT---"
    )

    # Use generate_text action (prompt + system_prompt), NOT messages
    data = _call_connector(connector, action, {
        "model": model,
        "prompt": user_prompt,
        "system_prompt": system_prompt,
        "temperature": str(stage.get("temperature", 0.3)),
    })

    # Extract text from unwrapped connector data
    if isinstance(data, dict):
        raw_text = (
            data.get("content")
            or data.get("text")
            or data.get("output")
            or json.dumps(data)
        )
    else:
        raw_text = str(data)

    score, feedback = _parse_score_from_text(raw_text)
    auto_approved = score >= threshold

    return {
        "output": raw_text,  # preserve review text for downstream
        "score": score,
        "feedback": feedback,
        "model": model,
        "auto_approved": auto_approved,
        "threshold": threshold,
    }


def _execute_image_generation(stage: dict, results: dict, topic: str) -> dict:
    """Generate an image, optionally using a previous stage's text as prompt."""
    connector = stage.get("connector", "openrouter")
    action = stage.get("action", "generate_image")
    model = stage.get("model", "openai/dall-e-3")

    prompt_source = stage.get("prompt_source")
    if prompt_source == "previous_stage" or prompt_source in results:
        source_key = prompt_source if prompt_source in results else None
        if source_key:
            image_prompt = _extract_text_output(results[source_key])
        else:
            # Use last text-producing stage
            image_prompt = ""
            for r in reversed(list(results.values())):
                if "output" in r and isinstance(r["output"], str):
                    image_prompt = r["output"]
                    break
    else:
        image_prompt = _apply_topic(stage.get("prompt", ""), topic) or topic

    params: dict[str, Any] = {
        "model": model,
        "prompt": image_prompt,
    }
    if "size" in stage:
        params["size"] = stage["size"]
    if "quality" in stage:
        params["quality"] = stage["quality"]

    data = _call_connector(connector, action, params)

    if isinstance(data, dict):
        # Image connectors return {"images": [...], "model": "..."}
        images = data.get("images", [])
        if images and isinstance(images, list):
            first = images[0]
            output = first.get("url") or first.get("b64_json") or str(first)
        else:
            output = data.get("url") or data.get("b64_json") or data.get("output") or str(data)
        seed = data.get("seed")
    else:
        output = str(data)
        seed = None

    return {
        "output": output,
        "model": model,
        "seed": seed,
        "prompt_used": image_prompt,
    }


def _execute_vision_review(stage: dict, results: dict, topic: str) -> dict:
    """Score an image from a previous stage using a vision model."""
    connector = stage.get("connector", "openrouter")
    action = stage.get("action", "analyze_image")
    model = stage.get("model", "openai/gpt-4o")
    threshold = float(stage.get("threshold", 7.0))
    criteria = stage.get("criteria", "quality, composition, relevance")

    # Find source image
    input_from = stage.get("input_from")
    if input_from and input_from in results:
        image_data = _extract_image_output(results[input_from])
    else:
        image_data = ""
        for r in reversed(list(results.values())):
            if "output" in r and "model" in r:
                candidate = r["output"]
                # Heuristic: image outputs are URLs or base64 (long strings)
                if isinstance(candidate, str) and (
                    candidate.startswith("http") or len(candidate) > 200
                ):
                    image_data = candidate
                    break

    review_prompt = (
        f"Review this image and score it 0-10 on: {criteria}.\n"
        f"Respond ONLY with valid JSON: "
        f'{{\"score\": <number 0-10>, \"feedback\": \"<brief explanation>\"}}'
    )

    params: dict[str, Any] = {
        "model": model,
        "prompt": review_prompt,
        "image": image_data,
        "max_tokens": stage.get("max_tokens", 256),
    }

    data = _call_connector(connector, action, params)

    if isinstance(data, dict):
        raw_text = (
            data.get("content")
            or data.get("text")
            or data.get("output")
            or json.dumps(data)
        )
    else:
        raw_text = str(data)

    score, feedback = _parse_score_from_text(raw_text)
    auto_approved = score >= threshold

    return {
        "output": raw_text,
        "score": score,
        "feedback": feedback,
        "model": model,
        "auto_approved": auto_approved,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Stage dispatch table
# ---------------------------------------------------------------------------

_STAGE_EXECUTORS = {
    "text_generation": _execute_text_generation,
    "text_review": _execute_text_review,
    "image_generation": _execute_image_generation,
    "vision_review": _execute_vision_review,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_stage(stage: dict, results: dict, topic: str = "") -> dict:
    """Execute a single pipeline stage.

    Args:
        stage:   Stage config dict (must have "type" key).
        results: Accumulated results from previous stages (keyed by stage id).
        topic:   Topic string passed through to prompt templates.

    Returns:
        Stage result dict.
    """
    stage_type = stage.get("type", "")
    executor = _STAGE_EXECUTORS.get(stage_type)
    if executor is None:
        raise ValueError(
            f"Unknown stage type: '{stage_type}'. "
            f"Available: {sorted(_STAGE_EXECUTORS)}"
        )
    return executor(stage, results, topic)


def execute_pipeline(stages: list[dict], topic: str = "") -> dict:
    """Execute all stages in sequence, passing outputs forward.

    Args:
        stages: Ordered list of stage config dicts.
        topic:  Topic string for prompt template substitution.

    Returns:
        Dict mapping stage_id → stage result.
    """
    results: dict[str, dict] = {}
    for stage in stages:
        stage_id = stage.get("id") or stage.get("type", f"stage_{len(results)}")
        result = execute_stage(stage, results, topic)
        results[stage_id] = result
        _notify(f"Stage '{stage_id}' complete", severity="info")
    return results
