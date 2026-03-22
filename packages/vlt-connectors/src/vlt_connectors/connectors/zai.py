"""z.ai text generation connector (OpenAI-compatible API)."""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..base import ActionConnector
from ..models import ConnectorAction, ConnectorParam, CredentialField

logger = logging.getLogger(__name__)

ZAI_API_BASE = "https://api.z.ai/v1"


class ZaiConnector(ActionConnector):
    name = "zai"
    display_name = "z.ai"
    description = "Text generation via z.ai's OpenAI-compatible API."

    credential_fields = [
        CredentialField(
            name="api_key",
            label="API Key",
            secret=True,
            placeholder="zai-...",
        ),
    ]

    actions = [
        ConnectorAction(
            name="chat_completion",
            description="Send a chat completion request using the z.ai API.",
            params=[
                ConnectorParam(name="model", description="Model ID to use", required=True),
                ConnectorParam(
                    name="messages",
                    description="JSON string of message array, e.g. [{\"role\":\"user\",\"content\":\"Hello\"}]",
                    required=True,
                ),
                ConnectorParam(
                    name="temperature",
                    description="Sampling temperature (0.0–2.0)",
                    required=False,
                    default="0.7",
                ),
                ConnectorParam(
                    name="max_tokens",
                    description="Maximum tokens to generate",
                    required=False,
                    default="4096",
                ),
            ],
        ),
        ConnectorAction(
            name="generate_text",
            description="Generate text from a simple prompt (wraps chat_completion with a single user message).",
            params=[
                ConnectorParam(name="model", description="Model ID to use", required=True),
                ConnectorParam(name="prompt", description="User prompt text", required=True),
                ConnectorParam(
                    name="system_prompt",
                    description="Optional system prompt",
                    required=False,
                ),
                ConnectorParam(
                    name="temperature",
                    description="Sampling temperature (0.0–2.0)",
                    required=False,
                    default="0.7",
                ),
            ],
        ),
    ]

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        match action:
            case "chat_completion":
                return await self._chat_completion(params, credentials)
            case "generate_text":
                return await self._generate_text(params, credentials)
            case _:
                return {"success": False, "error": f"Unknown action: {action}"}

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _chat_completion(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            return {"success": False, "error": "api_key is required in connector credentials"}

        model = params.get("model", "").strip()
        if not model:
            return {"success": False, "error": "model is required"}

        messages_raw = params.get("messages", "")
        if not messages_raw:
            return {"success": False, "error": "messages is required"}

        if isinstance(messages_raw, str):
            try:
                messages = json.loads(messages_raw)
            except json.JSONDecodeError as exc:
                return {"success": False, "error": f"messages must be valid JSON: {exc}"}
        else:
            messages = messages_raw

        temperature = float(params.get("temperature") or 0.7)
        max_tokens = int(params.get("max_tokens") or 4096)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        return await self._post_completion(api_key, payload)

    async def _generate_text(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            return {"success": False, "error": "api_key is required in connector credentials"}

        model = params.get("model", "").strip()
        if not model:
            return {"success": False, "error": "model is required"}

        prompt = params.get("prompt", "").strip()
        if not prompt:
            return {"success": False, "error": "prompt is required"}

        messages: list[dict] = []
        system_prompt = params.get("system_prompt", "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temperature = float(params.get("temperature") or 0.7)

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        return await self._post_completion(api_key, payload)

    async def _post_completion(self, api_key: str, payload: dict) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{ZAI_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out"}
        except httpx.RequestError as exc:
            return {"success": False, "error": f"Request error: {exc}"}

        if resp.status_code >= 400:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text[:300])
            except Exception:
                detail = resp.text[:300]
            logger.warning("z.ai API error %s: %s", resp.status_code, detail)
            return {"success": False, "error": f"z.ai API error {resp.status_code}: {detail}"}

        try:
            body = resp.json()
        except Exception as exc:
            return {"success": False, "error": f"Invalid JSON response: {exc}"}

        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = body.get("usage", {})

        return {
            "success": True,
            "data": {
                "content": content,
                "model": body.get("model", payload.get("model")),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                },
            },
        }
