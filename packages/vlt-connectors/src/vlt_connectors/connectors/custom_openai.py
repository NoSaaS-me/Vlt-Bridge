"""Custom / OpenAI-compatible connector.

For self-hosted models: vLLM, Ollama, LiteLLM, Qwen, LocalAI, etc.
Supports multiple named instances (custom:gpu-box-1, custom:gpu-box-2)
via credential-based endpoint configuration.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any

import httpx

from ..base import ActionConnector
from ..models import ConnectorAction, ConnectorParam, CredentialField

logger = logging.getLogger(__name__)

# Timeouts
_GENERATION_TIMEOUT = 60.0
_LIST_TIMEOUT = 10.0


class CustomOpenAIConnector(ActionConnector):
    name = "custom_openai"
    display_name = "Custom / OpenAI-compatible"
    description = (
        "Connect to any OpenAI-compatible API endpoint: vLLM, Ollama, "
        "LiteLLM, LocalAI, Qwen, or any self-hosted model server. "
        "Supports chat completions and vision analysis."
    )

    credential_fields = [
        CredentialField(
            name="endpoint_url",
            label="Endpoint URL",
            secret=False,
            placeholder="http://localhost:8080/v1",
        ),
        CredentialField(
            name="api_key",
            label="API Key (optional)",
            secret=True,
            placeholder="Leave blank if no auth required",
        ),
        CredentialField(
            name="model_name",
            label="Default Model Name (optional)",
            secret=False,
            placeholder="mistral-7b-instruct",
        ),
    ]

    actions = [
        ConnectorAction(
            name="chat_completion",
            description="OpenAI-compatible chat completion.",
            params=[
                ConnectorParam(name="model", description="Model name (overrides configured default)", required=False),
                ConnectorParam(name="messages", description="JSON string of message array: [{\"role\":\"user\",\"content\":\"...\"}]", required=True),
                ConnectorParam(name="temperature", description="Sampling temperature 0.0–2.0 (default 0.7)", required=False, default="0.7"),
                ConnectorParam(name="max_tokens", description="Maximum tokens to generate (default 4096)", required=False, default="4096"),
            ],
        ),
        ConnectorAction(
            name="generate_text",
            description="Simple text generation from a single prompt. Wraps chat_completion.",
            params=[
                ConnectorParam(name="model", description="Model name (overrides configured default)", required=False),
                ConnectorParam(name="prompt", description="User prompt text", required=True),
                ConnectorParam(name="system_prompt", description="Optional system prompt", required=False),
                ConnectorParam(name="temperature", description="Sampling temperature 0.0–2.0 (default 0.7)", required=False, default="0.7"),
            ],
        ),
        ConnectorAction(
            name="analyze_image",
            description="Vision analysis — send an image (base64 or URL) with a text prompt. Requires a vision-capable model.",
            params=[
                ConnectorParam(name="model", description="Vision-capable model name (overrides configured default)", required=False),
                ConnectorParam(name="image", description="Image as base64-encoded bytes or a public URL", required=True),
                ConnectorParam(name="prompt", description="Question or instruction about the image", required=True),
            ],
        ),
    ]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_endpoint(self, credentials: dict[str, str]) -> str:
        url = credentials.get("endpoint_url", "").strip().rstrip("/")
        if not url:
            raise ValueError("endpoint_url is required in connector credentials")
        return url

    def _resolve_model(self, params: dict[str, Any], credentials: dict[str, str]) -> str:
        """Param-level model overrides credential-level default."""
        model = (params.get("model") or "").strip()
        if model:
            return model
        default = credentials.get("model_name", "").strip()
        if default:
            return default
        raise ValueError(
            "No model specified. Provide 'model' param or set 'model_name' credential."
        )

    def _headers(self, credentials: dict[str, str]) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = credentials.get("api_key", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    # ── action dispatch ───────────────────────────────────────────────────────

    async def invoke(self, action: str, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        match action:
            case "chat_completion":
                return await self._chat_completion(params, credentials)
            case "generate_text":
                return await self._generate_text(params, credentials)
            case "analyze_image":
                return await self._analyze_image(params, credentials)
            case _:
                return {"success": False, "error": f"Unknown action: {action}"}

    # ── actions ───────────────────────────────────────────────────────────────

    async def _chat_completion(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        messages_raw = params.get("messages", "").strip()
        if not messages_raw:
            return {"success": False, "error": "param 'messages' is required"}

        try:
            messages = json.loads(messages_raw)
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"'messages' is not valid JSON: {exc}"}

        if not isinstance(messages, list):
            return {"success": False, "error": "'messages' must be a JSON array"}

        try:
            endpoint = self._resolve_endpoint(credentials)
            model = self._resolve_model(params, credentials)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

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
                    f"{endpoint}/chat/completions",
                    headers=self._headers(credentials),
                    json=body,
                )
        except httpx.TimeoutException:
            logger.warning("[CustomOpenAIConnector] chat_completion timed out (endpoint=%s, model=%s)", endpoint, model)
            return {"success": False, "error": "Request timed out after 60s"}
        except httpx.RequestError as exc:
            logger.error("[CustomOpenAIConnector] Network error: %s", exc)
            return {"success": False, "error": f"Network error: {exc}"}

        if resp.status_code >= 400:
            _err = _extract_api_error(resp)
            logger.warning("[CustomOpenAIConnector] API error %d: %s", resp.status_code, _err)
            return {"success": False, "error": f"Endpoint error {resp.status_code}: {_err}"}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return {"success": False, "error": "Endpoint returned no choices"}

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
        prompt = params.get("prompt", "").strip()
        if not prompt:
            return {"success": False, "error": "param 'prompt' is required"}

        messages: list[dict] = []
        system_prompt = params.get("system_prompt", "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        chat_params = {
            "model": params.get("model", ""),
            "messages": json.dumps(messages),
            "temperature": params.get("temperature", "0.7"),
        }
        return await self._chat_completion(chat_params, credentials)

    async def _analyze_image(self, params: dict[str, Any], credentials: dict[str, str]) -> dict:
        image_input = params.get("image", "").strip()
        prompt = params.get("prompt", "").strip()

        if not image_input:
            return {"success": False, "error": "param 'image' is required"}
        if not prompt:
            return {"success": False, "error": "param 'prompt' is required"}

        try:
            endpoint = self._resolve_endpoint(credentials)
            model = self._resolve_model(params, credentials)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        # Build the image content part: URL or base64 data URI
        if image_input.startswith(("http://", "https://")):
            image_content = {"type": "image_url", "image_url": {"url": image_input}}
        else:
            # Assume raw base64; detect JPEG vs PNG by magic bytes
            try:
                raw = base64.b64decode(image_input[:16])
                mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
            except Exception:
                mime = "image/jpeg"
            data_uri = f"data:{mime};base64,{image_input}"
            image_content = {"type": "image_url", "image_url": {"url": data_uri}}

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        body = {
            "model": model,
            "messages": messages,
        }

        try:
            async with httpx.AsyncClient(timeout=_GENERATION_TIMEOUT) as client:
                resp = await client.post(
                    f"{endpoint}/chat/completions",
                    headers=self._headers(credentials),
                    json=body,
                )
        except httpx.TimeoutException:
            logger.warning("[CustomOpenAIConnector] analyze_image timed out (endpoint=%s, model=%s)", endpoint, model)
            return {"success": False, "error": "Request timed out after 60s"}
        except httpx.RequestError as exc:
            logger.error("[CustomOpenAIConnector] Network error: %s", exc)
            return {"success": False, "error": f"Network error: {exc}"}

        if resp.status_code >= 400:
            _err = _extract_api_error(resp)
            logger.warning("[CustomOpenAIConnector] API error %d: %s", resp.status_code, _err)
            return {"success": False, "error": f"Endpoint error {resp.status_code}: {_err}"}

        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return {"success": False, "error": "Endpoint returned no choices"}

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


# ── module-level helper ────────────────────────────────────────────────────────

def _extract_api_error(resp: httpx.Response) -> str:
    """Pull the most informative error string from an error response."""
    try:
        body = resp.json()
        return (
            body.get("error", {}).get("message")
            or body.get("message")
            or body.get("detail")
            or resp.text[:300]
        )
    except Exception:
        return resp.text[:300]
