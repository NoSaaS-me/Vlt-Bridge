"""z.ai Vision connector — multimodal image/video analysis via GLM-4.6V."""
from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any

import httpx

from ..base import ActionConnector
from ..models import ConnectorAction, ConnectorParam, CredentialField

logger = logging.getLogger(__name__)

ZAI_API_BASE = "https://api.z.ai/v1"
VISION_MODEL = "glm-4.6v"

# Magic-byte → MIME type map for common image formats
_MAGIC: list[tuple[bytes, str]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),   # checked alongside b"WEBP" at offset 8
    (b"\x00\x00\x00\x0cjP  ", "image/jp2"),
]

_SYSTEM_PROMPTS: dict[str, str] = {
    "image_analysis": (
        "You are a vision analysis assistant. Analyze the provided image thoroughly "
        "according to the user's instructions."
    ),
    "video_analysis": (
        "You are a video analysis assistant. Analyze the provided video content "
        "according to the user's instructions, describing key scenes, actions, and insights."
    ),
    "ui_diff_check": (
        "You are a UI visual regression expert. Compare the two screenshots provided. "
        "Identify any visual differences, layout shifts, color changes, missing elements, "
        "or other regressions. Be specific about location and nature of each difference."
    ),
    "extract_text": (
        "Extract all visible text from this image exactly as it appears. "
        "Preserve formatting, line breaks, and structure where possible. "
        "Return only the extracted text, nothing else."
    ),
    "diagnose_error": (
        "Analyze this screenshot of an error or problem. "
        "1) Identify the error type and message. "
        "2) Explain what it means in plain language. "
        "3) List concrete steps to fix it."
    ),
    "understand_diagram": (
        "You are a software architecture expert. Analyze this diagram carefully. "
        "Identify all components, relationships, data flows, and key design decisions. "
        "Provide a clear explanation of what the diagram represents."
    ),
    "analyze_visualization": (
        "You are a data analyst. Examine this chart, graph, or dashboard carefully. "
        "Extract the key data points, trends, and insights. "
        "Summarize the most important findings in a structured way."
    ),
    "ui_to_artifact": (
        "Analyze this UI screenshot and convert it to the requested output format. "
        "For 'code': produce clean, well-structured HTML/CSS/JS that replicates the UI. "
        "For 'spec': write a detailed product specification describing all UI elements and behavior. "
        "For 'description': write a clear, comprehensive natural-language description of the interface."
    ),
}


def _detect_mime(data: bytes) -> str:
    """Detect MIME type from magic bytes; fall back to image/png."""
    for magic, mime in _MAGIC:
        if data[:len(magic)] == magic:
            # WebP special case: "RIFF....WEBP"
            if mime == "image/webp" and data[8:12] != b"WEBP":
                continue
            return mime
    return "image/png"


def _load_image_as_base64(image: str) -> tuple[str, str]:
    """Return (base64_data, mime_type) for a file path or existing base64 string.

    Accepts:
    - Absolute/relative file path
    - Data URL (data:image/...;base64,...)
    - Raw base64 string
    """
    # Data URL passthrough
    if image.startswith("data:"):
        # e.g. data:image/png;base64,iVBORw...
        header, _, b64 = image.partition(",")
        mime = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
        return b64, mime

    # File path
    path = Path(image)
    if path.exists() and path.is_file():
        raw = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or _detect_mime(raw)
        return base64.b64encode(raw).decode(), mime

    # Assume raw base64 — detect from decoded magic bytes
    try:
        raw = base64.b64decode(image)
        mime = _detect_mime(raw)
        return image, mime
    except Exception:
        pass

    # Last resort: treat as-is with unknown type
    return image, "image/png"


class ZaiVisionConnector(ActionConnector):
    name = "zai_vision"
    display_name = "z.ai Vision"
    description = (
        "Multimodal image and video analysis powered by z.ai's GLM-4.6V vision model. "
        "Supports image analysis, OCR, UI diffing, error diagnosis, diagram understanding, "
        "chart analysis, and UI-to-code conversion."
    )

    credential_fields = [
        CredentialField(
            name="api_key",
            label="Z_AI_API_KEY",
            secret=True,
            placeholder="zai-...",
        ),
    ]

    actions = [
        ConnectorAction(
            name="image_analysis",
            description="General-purpose image understanding and analysis.",
            params=[
                ConnectorParam(
                    name="image",
                    description="File path, base64 string, or data URL of the image",
                    required=True,
                ),
                ConnectorParam(name="prompt", description="What to analyze or look for", required=True),
            ],
        ),
        ConnectorAction(
            name="video_analysis",
            description="Video content analysis (≤8 MB MP4/MOV, file path or URL).",
            params=[
                ConnectorParam(
                    name="video",
                    description="File path or URL of the video (≤8 MB MP4/MOV)",
                    required=True,
                ),
                ConnectorParam(name="prompt", description="What to analyze in the video", required=True),
            ],
        ),
        ConnectorAction(
            name="ui_diff_check",
            description="Compare two screenshots for visual drift or regressions.",
            params=[
                ConnectorParam(
                    name="image_a",
                    description="First screenshot (file path, base64, or data URL)",
                    required=True,
                ),
                ConnectorParam(
                    name="image_b",
                    description="Second screenshot (file path, base64, or data URL)",
                    required=True,
                ),
                ConnectorParam(
                    name="prompt",
                    description="Optional specific aspects to focus on",
                    required=False,
                ),
            ],
        ),
        ConnectorAction(
            name="extract_text",
            description="OCR — extract all visible text from a screenshot or document image.",
            params=[
                ConnectorParam(
                    name="image",
                    description="File path, base64 string, or data URL of the image",
                    required=True,
                ),
            ],
        ),
        ConnectorAction(
            name="diagnose_error",
            description="Analyze an error screenshot to identify the problem and suggest fixes.",
            params=[
                ConnectorParam(
                    name="image",
                    description="Screenshot of the error",
                    required=True,
                ),
                ConnectorParam(
                    name="context",
                    description="Optional context about the system or what was being done",
                    required=False,
                ),
            ],
        ),
        ConnectorAction(
            name="understand_diagram",
            description="Interpret architecture diagrams, UML, flowcharts, and other technical diagrams.",
            params=[
                ConnectorParam(
                    name="image",
                    description="File path, base64 string, or data URL of the diagram",
                    required=True,
                ),
                ConnectorParam(
                    name="prompt",
                    description="Optional specific question about the diagram",
                    required=False,
                ),
            ],
        ),
        ConnectorAction(
            name="analyze_visualization",
            description="Extract insights and data from charts, graphs, and dashboards.",
            params=[
                ConnectorParam(
                    name="image",
                    description="File path, base64 string, or data URL of the visualization",
                    required=True,
                ),
                ConnectorParam(
                    name="prompt",
                    description="Optional specific question about the data",
                    required=False,
                ),
            ],
        ),
        ConnectorAction(
            name="ui_to_artifact",
            description="Convert a UI screenshot into code, a product spec, or a description.",
            params=[
                ConnectorParam(
                    name="image",
                    description="Screenshot of the UI",
                    required=True,
                ),
                ConnectorParam(
                    name="output_format",
                    description="Output format: 'code', 'spec', or 'description' (default: description)",
                    required=False,
                    default="description",
                ),
            ],
        ),
    ]

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        match action:
            case "image_analysis":
                return await self._single_image(action, params, credentials)
            case "video_analysis":
                return await self._video_analysis(params, credentials)
            case "ui_diff_check":
                return await self._ui_diff_check(params, credentials)
            case "extract_text":
                return await self._single_image(action, params, credentials)
            case "diagnose_error":
                return await self._single_image(action, params, credentials)
            case "understand_diagram":
                return await self._single_image(action, params, credentials)
            case "analyze_visualization":
                return await self._single_image(action, params, credentials)
            case "ui_to_artifact":
                return await self._ui_to_artifact(params, credentials)
            case _:
                return {"success": False, "error": f"Unknown action: {action}"}

    # ── action implementations ────────────────────────────────────────────────

    async def _single_image(
        self,
        action: str,
        params: dict[str, Any],
        credentials: dict[str, str],
    ) -> dict:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            return {"success": False, "error": "api_key is required"}

        image_param = params.get("image", "").strip()
        if not image_param:
            return {"success": False, "error": "image is required"}

        try:
            b64_data, mime = _load_image_as_base64(image_param)
        except Exception as exc:
            return {"success": False, "error": f"Failed to load image: {exc}"}

        system_prompt = _SYSTEM_PROMPTS[action]

        # Build user message content
        user_content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}},
        ]

        # Action-specific user text
        if action == "extract_text":
            user_content.append({"type": "text", "text": "Extract all visible text from this image."})
        elif action == "diagnose_error":
            context = params.get("context", "").strip()
            text = "Diagnose this error screenshot."
            if context:
                text += f" Context: {context}"
            user_content.append({"type": "text", "text": text})
        else:
            prompt = params.get("prompt", "").strip()
            if prompt:
                user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return await self._call_api(api_key, messages)

    async def _ui_diff_check(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            return {"success": False, "error": "api_key is required"}

        image_a = params.get("image_a", "").strip()
        image_b = params.get("image_b", "").strip()
        if not image_a:
            return {"success": False, "error": "image_a is required"}
        if not image_b:
            return {"success": False, "error": "image_b is required"}

        try:
            b64_a, mime_a = _load_image_as_base64(image_a)
            b64_b, mime_b = _load_image_as_base64(image_b)
        except Exception as exc:
            return {"success": False, "error": f"Failed to load image(s): {exc}"}

        prompt = params.get("prompt", "").strip()
        user_text = "Compare these two screenshots and identify all visual differences."
        if prompt:
            user_text += f" Focus on: {prompt}"

        user_content: list[dict] = [
            {"type": "text", "text": "Screenshot A:"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_a};base64,{b64_a}"}},
            {"type": "text", "text": "Screenshot B:"},
            {"type": "image_url", "image_url": {"url": f"data:{mime_b};base64,{b64_b}"}},
            {"type": "text", "text": user_text},
        ]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPTS["ui_diff_check"]},
            {"role": "user", "content": user_content},
        ]

        return await self._call_api(api_key, messages)

    async def _ui_to_artifact(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            return {"success": False, "error": "api_key is required"}

        image_param = params.get("image", "").strip()
        if not image_param:
            return {"success": False, "error": "image is required"}

        output_format = (params.get("output_format") or "description").strip().lower()
        if output_format not in {"code", "spec", "description"}:
            return {"success": False, "error": "output_format must be 'code', 'spec', or 'description'"}

        try:
            b64_data, mime = _load_image_as_base64(image_param)
        except Exception as exc:
            return {"success": False, "error": f"Failed to load image: {exc}"}

        user_content: list[dict] = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64_data}"}},
            {"type": "text", "text": f"Convert this UI screenshot to: {output_format}"},
        ]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPTS["ui_to_artifact"]},
            {"role": "user", "content": user_content},
        ]

        return await self._call_api(api_key, messages, extra={"output_format": output_format})

    async def _video_analysis(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        """Video analysis — loads small local files as base64, passes URLs directly."""
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            return {"success": False, "error": "api_key is required"}

        video_param = params.get("video", "").strip()
        if not video_param:
            return {"success": False, "error": "video is required"}

        prompt = params.get("prompt", "").strip()
        if not prompt:
            return {"success": False, "error": "prompt is required"}

        # Determine how to pass the video to the model
        video_content: dict
        path = Path(video_param)
        if path.exists() and path.is_file():
            size = path.stat().st_size
            if size > 8 * 1024 * 1024:
                return {"success": False, "error": "Video file exceeds 8 MB limit"}
            raw = path.read_bytes()
            b64 = base64.b64encode(raw).decode()
            mime = mimetypes.guess_type(str(path))[0] or "video/mp4"
            video_content = {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{b64}"}}
        elif video_param.startswith(("http://", "https://")):
            video_content = {"type": "video_url", "video_url": {"url": video_param}}
        else:
            return {"success": False, "error": "video must be a valid file path or HTTP(S) URL"}

        user_content: list[dict] = [
            video_content,
            {"type": "text", "text": prompt},
        ]

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPTS["video_analysis"]},
            {"role": "user", "content": user_content},
        ]

        return await self._call_api(api_key, messages)

    # ── HTTP ─────────────────────────────────────────────────────────────────

    async def _call_api(
        self,
        api_key: str,
        messages: list[dict],
        *,
        extra: dict | None = None,
    ) -> dict:
        payload: dict = {
            "model": VISION_MODEL,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{ZAI_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out (60s)"}
        except httpx.RequestError as exc:
            return {"success": False, "error": f"Request error: {exc}"}

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                detail = resp.text[:300]
            logger.warning("z.ai Vision API error %s: %s", resp.status_code, detail)
            return {"success": False, "error": f"z.ai Vision API error {resp.status_code}: {detail}"}

        try:
            body = resp.json()
        except Exception as exc:
            return {"success": False, "error": f"Invalid JSON response: {exc}"}

        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = body.get("usage", {})

        result: dict[str, Any] = {
            "success": True,
            "data": {
                "content": content,
                "model": body.get("model", VISION_MODEL),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
            },
        }
        if extra:
            result["data"].update(extra)

        return result
