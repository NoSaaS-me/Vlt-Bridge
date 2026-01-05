"""LLM service for Deep Research orchestration.

Provides a simplified interface for making LLM calls during research,
supporting both OpenRouter and Google (Gemini) providers.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from ...models.settings import ModelProvider
from ..user_settings import UserSettingsService

logger = logging.getLogger(__name__)


class ResearchLLMService:
    """Service for making LLM calls during research.

    Provides a simple interface for research behaviors to call LLMs
    without worrying about provider-specific details.
    """

    OPENROUTER_BASE = "https://openrouter.ai/api/v1"
    GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        user_id: str,
        user_settings: Optional[UserSettingsService] = None,
    ):
        """Initialize the LLM service.

        Args:
            user_id: User identifier for fetching API keys and settings
            user_settings: Optional user settings service
        """
        self.user_id = user_id
        self.user_settings = user_settings or UserSettingsService()
        self._cached_settings = None

    def _get_settings(self):
        """Get cached user settings."""
        if self._cached_settings is None:
            self._cached_settings = self.user_settings.get_settings(self.user_id)
        return self._cached_settings

    def _get_api_keys(self) -> tuple[Optional[str], Optional[str]]:
        """Get OpenRouter and Google API keys.

        Returns:
            Tuple of (openrouter_key, google_key)
        """
        openrouter_key = self.user_settings.get_openrouter_api_key(self.user_id)
        if not openrouter_key:
            openrouter_key = os.getenv("OPENROUTER_API_KEY")

        google_key = os.getenv("GOOGLE_API_KEY")

        return openrouter_key, google_key

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[ModelProvider] = None,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate text from an LLM.

        Args:
            prompt: The user prompt to send
            model: Optional model override (uses user's subagent model by default)
            provider: Optional provider override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            system_prompt: Optional system prompt

        Returns:
            The generated text content

        Raises:
            ValueError: If no API key is configured
            httpx.HTTPError: If the API call fails
        """
        settings = self._get_settings()

        # Use subagent settings by default (research is a background task)
        model = model or settings.subagent_model
        provider = provider or settings.subagent_provider

        openrouter_key, google_key = self._get_api_keys()

        # Fall back to available provider if configured one isn't available
        if provider == ModelProvider.GOOGLE and not google_key:
            if openrouter_key:
                logger.info(f"No Google API key, falling back to OpenRouter for research")
                provider = ModelProvider.OPENROUTER
                if model.startswith("gemini") or model.startswith("models/"):
                    model = "deepseek/deepseek-chat"
            else:
                raise ValueError("No API key configured for research LLM calls")
        elif provider == ModelProvider.OPENROUTER and not openrouter_key:
            if google_key:
                logger.info(f"No OpenRouter API key, falling back to Google for research")
                provider = ModelProvider.GOOGLE
                model = "gemini-2.0-flash-exp"
            else:
                raise ValueError("No API key configured for research LLM calls")

        if provider == ModelProvider.GOOGLE:
            return await self._call_google(
                prompt, model, max_tokens, temperature, system_prompt, google_key
            )
        else:
            return await self._call_openrouter(
                prompt, model, max_tokens, temperature, system_prompt, openrouter_key
            )

    async def generate_json(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[ModelProvider] = None,
        max_tokens: int = 4000,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate and parse JSON from an LLM.

        Args:
            prompt: The user prompt (should ask for JSON output)
            model: Optional model override
            provider: Optional provider override
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (lower for JSON)
            system_prompt: Optional system prompt

        Returns:
            Parsed JSON dictionary

        Raises:
            json.JSONDecodeError: If the response is not valid JSON
        """
        response = await self.generate(
            prompt=prompt,
            model=model,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        # Try to extract JSON from the response
        # Sometimes models wrap JSON in markdown code blocks
        content = response.strip()

        # Remove markdown code block if present
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        return json.loads(content)

    async def _call_openrouter(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        api_key: str,
    ) -> str:
        """Call OpenRouter API."""
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://vlt.ai/research",
            "X-Title": "Vlt Research",
            "Content-Type": "application/json",
        }

        request_body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.OPENROUTER_BASE}/chat/completions",
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]

            # Log token usage
            usage = data.get("usage", {})
            logger.debug(
                "OpenRouter research call completed",
                extra={
                    "model": model,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            )

            return content

    async def _call_google(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str],
        api_key: str,
    ) -> str:
        """Call Google Gemini API."""
        # Map model names if needed
        gemini_model = model
        if not model.startswith("models/"):
            gemini_model = f"models/{model}"

        url = f"{self.GOOGLE_API_BASE}/{gemini_model}:generateContent"

        # Build contents
        contents = []

        # For Gemini, we combine system prompt with user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        contents.append({
            "parts": [{"text": full_prompt}]
        })

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                url,
                params={"key": api_key},
                json={
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": max_tokens,
                        "temperature": temperature,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No response from Gemini")

            content = candidates[0]["content"]["parts"][0]["text"]

            # Log token usage
            usage = data.get("usageMetadata", {})
            logger.debug(
                "Gemini research call completed",
                extra={
                    "model": model,
                    "prompt_tokens": usage.get("promptTokenCount"),
                    "completion_tokens": usage.get("candidatesTokenCount"),
                }
            )

            return content


__all__ = ["ResearchLLMService"]
