"""OpenRouter AI connector.

Unified gateway for text, image, and audio generation models.
Supports chat completions, simple text generation, image generation,
and model discovery.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..base import ActionConnector
from ..models import ConnectorAction, ConnectorParam, CredentialField

logger = logging.getLogger(__name__)

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# Timeouts
_GENERATION_TIMEOUT = 60.0
_LIST_TIMEOUT = 10.0


class OpenRouterConnector(ActionConnector):
    name = "openrouter"
    display_name = "OpenRouter"
    description = (
        "Unified AI gateway for text, image, and audio generation. "
        "Access hundreds of models (GPT-4o, Claude, Gemini, Llama, etc.) "
        "through a single API key."
    )

    credential_fields = [
        CredentialField(
            name="api_key",
            label="API Key",
            secret=True,
            placeholder="sk-or-...",
        ),
        CredentialField(
            name="site_url",
            label="Site URL (optional, for attribution)",
            secret=False,
            placeholder="https://yourapp.com",
        ),
        CredentialField(
            name="app_name",
            label="App Name (optional, for attribution)",
            secret=False,
            placeholder="ContentForge",
        ),
    ]

    actions = [
        ConnectorAction(
            name="chat_completion",
            description="Send a chat completion request with a full message array.",
            params=[
                ConnectorParam(name="model", description="Model ID (e.g. openai/gpt-4o, anthropic/claude-3-5-sonnet)", required=True),
                ConnectorParam(name="messages", description="JSON string of message array: [{\"role\":\"user\",\"content\":\"...\"}]", required=True),
                ConnectorParam(name="temperature", description="Sampling temperature 0.0–2.0 (default 0.7)", required=False, default="0.7"),
                ConnectorParam(name="max_tokens", description="Maximum tokens to generate (default 4096)", required=False, default="4096"),
            ],
        ),
        ConnectorAction(
            name="generate_text",
            description="Simple text generation from a single prompt. Wraps chat_completion.",
            params=[
                ConnectorParam(name="model", description="Model ID (e.g. openai/gpt-4o)", required=True),
                ConnectorParam(name="prompt", description="User prompt text", required=True),
                ConnectorParam(name="system_prompt", description="Optional system prompt", required=False),
                ConnectorParam(name="temperature", description="Sampling temperature 0.0–2.0 (default 0.7)", required=False, default="0.7"),
            ],
        ),
        ConnectorAction(
            name="generate_image",
            description="Generate images from a text prompt.",
            params=[
                ConnectorParam(name="model", description="Image model ID (e.g. openai/dall-e-3)", required=True),
                ConnectorParam(name="prompt", description="Image description prompt", required=True),
                ConnectorParam(name="size", description="Image dimensions (default 1024x1024)", required=False, default="1024x1024"),
                ConnectorParam(name="n", description="Number of images to generate (default 1)", required=False, default="1"),
            ],
        ),
        ConnectorAction(
            name="list_models",
            description="Discover all models available via OpenRouter.",
            params=[],
        ),
    ]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _headers(self, credentials: dict[str, str]) -> dict[str, str]:
        api_key = credentials.get("api_key", "").strip()
        if not api_key:
            raise ValueError("api_key is required in connector credentials")
        site_url = credentials.get("site_url", "").strip() or "https://contentforge.ai"
        app_name = credentials.get("app_name", "").strip() or "ContentForge"
        return {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": site_url,
            "X-Title": app_name,
            "Content-Type": "application/json",
        }

    # ── action dispatch ───────────────────────────────────────────────────────

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        match action:
            case "chat_completion":
                return await self._chat_completion(params, credentials)
            case "generate_text":
                return await self._generate_text(params, credentials)
            case "generate_image":
                return await self._generate_image(params, credentials)
            case "list_models":
                return await self._list_models(credentials)
            case _:
                return {"success": False, "error": f"Unknown action: {action}"}

    # ── actions ───────────────────────────────────────────────────────────────

    async def _chat_completion(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        model = params.get("model", "").strip()
        messages_raw = params.get("messages", "").strip()

        if not model:
            return {"success": False, "error": "param 'model' is required"}
        if not messages_raw:
            return {"success": False, "error": "param 'messages' is required"}

        try:
            messages = json.loads(messages_raw)
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"'messages' is not valid JSON: {exc}"}

        if not isinstance(messages, list):
            return {"success": False, "error": "'messages' must be a JSON array"}

        temperature = float(params.get("temperature") or 0.7)
        max_tokens = int(params.get("max_tokens") or 4096)

        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=_GENERATION_TIMEOUT) as client:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers=self._headers(credentials),
                    json=body,
                )
        except httpx.TimeoutException:
            logger.warning("[OpenRouterConnector] chat_completion timed out (model=%s)", model)
            return {"success": False, "error": "Request timed out after 60s"}
        except httpx.RequestError as exc:
            logger.error("[OpenRouterConnector] Network error: %s", exc)
            return {"success": False, "error": f"Network error: {exc}"}

        if resp.status_code >= 400:
            _err = _extract_api_error(resp)
            logger.warning("[OpenRouterConnector] API error %d: %s", resp.status_code, _err)
            return {"success": False, "error": f"OpenRouter error {resp.status_code}: {_err}"}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return {"success": False, "error": "OpenRouter returned no choices"}

        choice = choices[0]
        message = choice.get("message", {})
        return {
            "success": True,
            "data": {
                "content": message.get("content", ""),
                "finish_reason": choice.get("finish_reason", "stop"),
                "model": data.get("model", model),
                "usage": data.get("usage", {}),
            },
        }

    async def _generate_text(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        model = params.get("model", "").strip()
        prompt = params.get("prompt", "").strip()

        if not model:
            return {"success": False, "error": "param 'model' is required"}
        if not prompt:
            return {"success": False, "error": "param 'prompt' is required"}

        messages: list[dict] = []
        system_prompt = params.get("system_prompt", "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_params = {
            "model": model,
            "messages": json.dumps(messages),
            "temperature": params.get("temperature", "0.7"),
        }
        return await self._chat_completion(chat_params, credentials)

    async def _generate_image(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        model = params.get("model", "").strip()
        prompt = params.get("prompt", "").strip()

        if not model:
            return {"success": False, "error": "param 'model' is required"}
        if not prompt:
            return {"success": False, "error": "param 'prompt' is required"}

        size = params.get("size", "1024x1024").strip() or "1024x1024"
        n = int(params.get("n") or 1)

        body = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "n": n,
        }

        try:
            async with httpx.AsyncClient(timeout=_GENERATION_TIMEOUT) as client:
                resp = await client.post(
                    f"{OPENROUTER_BASE}/images/generations",
                    headers=self._headers(credentials),
                    json=body,
                )
        except httpx.TimeoutException:
            logger.warning("[OpenRouterConnector] generate_image timed out (model=%s)", model)
            return {"success": False, "error": "Request timed out after 60s"}
        except httpx.RequestError as exc:
            logger.error("[OpenRouterConnector] Network error: %s", exc)
            return {"success": False, "error": f"Network error: {exc}"}

        if resp.status_code >= 400:
            _err = _extract_api_error(resp)
            logger.warning("[OpenRouterConnector] API error %d: %s", resp.status_code, _err)
            return {"success": False, "error": f"OpenRouter error {resp.status_code}: {_err}"}

        data = resp.json()
        return {
            "success": True,
            "data": {
                "images": data.get("data", []),
                "model": model,
            },
        }

    async def _list_models(self, credentials: dict[str, str]) -> dict:
        try:
            async with httpx.AsyncClient(timeout=_LIST_TIMEOUT) as client:
                resp = await client.get(
                    f"{OPENROUTER_BASE}/models",
                    headers=self._headers(credentials),
                )
        except httpx.TimeoutException:
            return {"success": False, "error": "Request timed out after 10s"}
        except httpx.RequestError as exc:
            logger.error("[OpenRouterConnector] Network error: %s", exc)
            return {"success": False, "error": f"Network error: {exc}"}

        if resp.status_code >= 400:
            _err = _extract_api_error(resp)
            return {"success": False, "error": f"OpenRouter error {resp.status_code}: {_err}"}

        data = resp.json()
        models = data.get("data", [])

        # Return lightweight summary — ID, name, context length, pricing
        summaries = [
            {
                "id": m.get("id", ""),
                "name": m.get("name", ""),
                "context_length": m.get("context_length"),
                "pricing": m.get("pricing", {}),
            }
            for m in models
        ]
        return {
            "success": True,
            "data": {
                "models": summaries,
                "count": len(summaries),
            },
        }


# ── module-level helper ────────────────────────────────────────────────────────

def _extract_api_error(resp: httpx.Response) -> str:
    """Pull the most informative error string from an error response."""
    try:
        body = resp.json()
        return (
            body.get("error", {}).get("message")
            or body.get("message")
            or resp.text[:300]
        )
    except Exception:
        return resp.text[:300]
