"""Librarian Service - Server-side thread summarization.

This service provides LLM-powered summarization of threads that were previously
handled by the vlt-cli directly. By moving summarization to the server, we:
1. Remove the need for users to configure OpenRouter API keys in the CLI
2. Centralize LLM operations and billing
3. Enable richer summarization with access to more context (vault, code, etc.)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional
from datetime import datetime

import httpx

from ..models.thread import ThreadEntry
from ..models.settings import ModelProvider
from .user_settings import UserSettingsService

logger = logging.getLogger(__name__)


# Summarization prompt - same as the original CLI librarian
LIBRARIAN_PROMPT = """
You are the 'Librarian' for an AI Agent's long-term memory.
Your goal is to maintain a structured "State Object" that allows the agent to resume work immediately after amnesia.

INPUTS:
1. Current State:
{current_summary}

2. New Thoughts (The Delta):
{new_content}

INSTRUCTIONS:
Update the State to reflect the New Thoughts.
DO NOT just append a log. SYNTHESIZE the information.

REQUIRED OUTPUT FORMAT (Markdown):

# Status: [Active Goal/Phase]
**Focus:** [What is the agent doing RIGHT NOW?]

## Context & Architecture (The "Truth")
[Bulleted list of *current* facts, decisions, and architectural truths. Prune obsolete info.]
- Key: Value

## Pivot Log (Last 3 Major Decisions)
[Only list critical changes in direction or approach. Drop minor task completions.]
- Decision: ...

## Next Steps
[Immediate actionable tasks]
1. ...
"""


class LibrarianService:
    """Server-side thread summarization service."""

    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, user_settings_service: Optional[UserSettingsService] = None):
        """
        Initialize the librarian service.

        Args:
            user_settings_service: Optional user settings service for fetching API keys
        """
        self.user_settings = user_settings_service or UserSettingsService()

    async def summarize_thread(
        self,
        user_id: str,
        entries: List[ThreadEntry],
        current_summary: Optional[str] = None,
        model_override: Optional[str] = None,
        provider_override: Optional[ModelProvider] = None,
    ) -> dict:
        """
        Generate a summary of thread entries.

        Args:
            user_id: User identifier for fetching settings
            entries: List of thread entries to summarize
            current_summary: Existing summary to update (None for first summarization)
            model_override: Override the user's configured model
            provider_override: Override the user's configured provider

        Returns:
            Dict with:
                - summary: The generated summary text
                - model: Model used for generation
                - tokens_used: Approximate token count
                - success: Whether summarization succeeded
                - error: Error message if failed
        """
        if not entries:
            return {
                "summary": current_summary or "No entries to summarize.",
                "model": None,
                "tokens_used": 0,
                "success": True,
                "error": None,
            }

        # Get user settings - use subagent settings (not oracle)
        # The librarian is a background task, so it uses the subagent model
        settings = self.user_settings.get_settings(user_id)
        model = model_override or settings.subagent_model
        provider = provider_override or settings.subagent_provider

        # Check if the required API key is available for the chosen provider
        # If not, try to fall back to the other provider
        google_key = os.getenv("GOOGLE_API_KEY")
        openrouter_key = self.user_settings.get_openrouter_api_key(user_id) or os.getenv("OPENROUTER_API_KEY")

        if provider == ModelProvider.GOOGLE and not google_key:
            if openrouter_key:
                logger.info(f"No Google API key, falling back to OpenRouter for user {user_id}")
                provider = ModelProvider.OPENROUTER
                # Use a default OpenRouter model if the current model is Google-specific
                if model.startswith("gemini") or model.startswith("models/"):
                    model = "deepseek/deepseek-chat"  # Free, capable model
            else:
                return {
                    "summary": current_summary or "Summarization unavailable.",
                    "model": None,
                    "tokens_used": 0,
                    "success": False,
                    "error": "No API key configured. Set GOOGLE_API_KEY or configure OpenRouter in settings.",
                }
        elif provider == ModelProvider.OPENROUTER and not openrouter_key:
            if google_key:
                logger.info(f"No OpenRouter API key, falling back to Google for user {user_id}")
                provider = ModelProvider.GOOGLE
                model = "gemini-2.0-flash-exp"
            else:
                return {
                    "summary": current_summary or "Summarization unavailable.",
                    "model": None,
                    "tokens_used": 0,
                    "success": False,
                    "error": "No API key configured. Set GOOGLE_API_KEY or configure OpenRouter in settings.",
                }

        # Format new content from entries
        new_content = "\n".join([
            f"- [{e.author}] {e.content}"
            for e in entries
        ])

        # Build prompt
        prompt = LIBRARIAN_PROMPT.format(
            current_summary=current_summary or "No prior state. This is the first summarization.",
            new_content=new_content,
        )

        try:
            if provider == ModelProvider.GOOGLE:
                result = await self._call_google(user_id, model, prompt)
            else:
                result = await self._call_openrouter(user_id, model, prompt)

            return {
                "summary": result["content"],
                "model": model,
                "tokens_used": result.get("tokens_used", 0),
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Summarization failed for user {user_id}: {e}")
            return {
                "summary": current_summary or "Summarization failed.",
                "model": model,
                "tokens_used": 0,
                "success": False,
                "error": str(e),
            }

    async def _call_openrouter(
        self,
        user_id: str,
        model: str,
        prompt: str,
    ) -> dict:
        """Call OpenRouter API for summarization."""
        # Get user's API key or fall back to system key
        api_key = self.user_settings.get_openrouter_api_key(user_id)
        if not api_key:
            api_key = os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            raise ValueError("No OpenRouter API key configured. Please set your API key in settings.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://vlt.cli",
            "X-Title": "Vlt-Bridge Librarian",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.OPENROUTER_API_BASE}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)

            return {
                "content": content,
                "tokens_used": tokens_used,
            }

    async def _call_google(
        self,
        user_id: str,
        model: str,
        prompt: str,
    ) -> dict:
        """Call Google Gemini API for summarization."""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No Google API key configured. Set GOOGLE_API_KEY environment variable.")

        # Map model names if needed
        gemini_model = model
        if not model.startswith("models/"):
            gemini_model = f"models/{model}"

        url = f"{self.GOOGLE_API_BASE}/{gemini_model}:generateContent"

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                url,
                params={"key": api_key},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 2000,
                        "temperature": 0.7,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

            # Extract content from Gemini response
            candidates = data.get("candidates", [])
            if not candidates:
                raise ValueError("No response from Gemini")

            content = candidates[0]["content"]["parts"][0]["text"]

            # Estimate tokens (Gemini doesn't always return usage)
            usage = data.get("usageMetadata", {})
            tokens_used = usage.get("totalTokenCount", len(prompt.split()) + len(content.split()))

            return {
                "content": content,
                "tokens_used": tokens_used,
            }

    async def get_thread_summary(
        self,
        user_id: str,
        thread_id: str,
    ) -> Optional[str]:
        """
        Get the current summary for a thread (from cache/database).

        This retrieves a previously generated summary. Use summarize_thread()
        to generate a new summary.

        Args:
            user_id: User identifier
            thread_id: Thread identifier

        Returns:
            The cached summary or None if not found
        """
        # TODO: Implement summary storage in database
        # For now, this would need to be stored in a new table
        # or as part of the thread_sync_status table
        return None


# Singleton instance
_librarian_service: Optional[LibrarianService] = None


def get_librarian_service() -> LibrarianService:
    """Get or create the librarian service singleton."""
    global _librarian_service
    if _librarian_service is None:
        _librarian_service = LibrarianService()
    return _librarian_service
