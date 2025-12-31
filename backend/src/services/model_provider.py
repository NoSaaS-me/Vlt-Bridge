"""Service for fetching and managing AI model providers."""

from __future__ import annotations

import logging
from typing import List, Optional
import httpx
from functools import lru_cache

from ..models.settings import ModelInfo, ModelProvider

logger = logging.getLogger(__name__)

# Hardcoded Google models
GOOGLE_MODELS = [
    ModelInfo(
        id="gemini-2.0-flash-exp",
        name="Gemini 2.0 Flash (Experimental)",
        provider=ModelProvider.GOOGLE,
        is_free=True,
        supports_thinking=False,
        context_length=1000000,
        description="Latest experimental Gemini model with 1M token context"
    ),
    ModelInfo(
        id="gemini-1.5-pro",
        name="Gemini 1.5 Pro",
        provider=ModelProvider.GOOGLE,
        is_free=False,
        supports_thinking=False,
        context_length=2000000,
        description="Advanced Gemini model with 2M token context"
    ),
    ModelInfo(
        id="gemini-1.5-flash",
        name="Gemini 1.5 Flash",
        provider=ModelProvider.GOOGLE,
        is_free=True,
        supports_thinking=False,
        context_length=1000000,
        description="Fast and efficient Gemini model"
    ),
]



class ModelProviderService:
    """Service for fetching models from various providers."""

    OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
    CACHE_TTL_SECONDS = 300  # 5 minutes

    def __init__(self, openrouter_api_key: Optional[str] = None):
        """
        Initialize the model provider service.

        Args:
            openrouter_api_key: Optional OpenRouter API key for authenticated requests
        """
        self.openrouter_api_key = openrouter_api_key

    async def get_openrouter_models(self) -> List[ModelInfo]:
        """
        Fetch available models from OpenRouter API.

        Returns:
            List of ModelInfo objects for free models and priority models
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {}
                if self.openrouter_api_key:
                    headers["Authorization"] = f"Bearer {self.openrouter_api_key}"

                response = await client.get(
                    f"{self.OPENROUTER_API_BASE}/models",
                    headers=headers
                )
                response.raise_for_status()
                data = response.json()

                models = []
                for model_data in data.get("data", []):
                    model_id = model_data.get("id", "")

                    # Check actual pricing to determine if model is free
                    # Free models have :free suffix OR both prompt and completion cost of "0"
                    pricing = model_data.get("pricing", {})
                    prompt_price = str(pricing.get("prompt", "1"))
                    completion_price = str(pricing.get("completion", "1"))

                    is_free = (
                        ":free" in model_id.lower()
                        or (prompt_price == "0" and completion_price == "0")
                    )

                    # Check if model supports thinking/reasoning mode
                    supports_thinking = (
                        ":thinking" in model_id.lower()
                        or "thinking" in model_data.get("name", "").lower()
                        or "-r1" in model_id.lower()
                        or "/r1" in model_id.lower()
                        or "/o1" in model_id.lower()
                        or "/o3" in model_id.lower()
                    )

                    models.append(ModelInfo(
                        id=model_id,
                        name=model_data.get("name", model_id),
                        provider=ModelProvider.OPENROUTER,
                        is_free=is_free,
                        supports_thinking=supports_thinking,
                        context_length=model_data.get("context_length"),
                        description=model_data.get("description")
                    ))

                # Sort: free models first, then by name
                models.sort(
                    key=lambda m: (
                        not m.is_free,  # Free first
                        m.name.lower()
                    )
                )

                logger.info(f"Fetched {len(models)} models from OpenRouter")
                return models

        except httpx.HTTPError as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching OpenRouter models: {e}")
            return []

    def get_google_models(self) -> List[ModelInfo]:
        """
        Get hardcoded Google models.

        Returns:
            List of Google ModelInfo objects
        """
        return GOOGLE_MODELS.copy()

    async def get_all_models(self) -> List[ModelInfo]:
        """
        Get all available models from all providers.

        Returns:
            Combined list of ModelInfo objects from all providers
        """
        google_models = self.get_google_models()
        openrouter_models = await self.get_openrouter_models()

        # Combine and deduplicate (Google takes priority)
        all_models = google_models + openrouter_models

        # Remove duplicates based on provider+id
        seen = set()
        unique_models = []
        for model in all_models:
            key = (model.provider, model.id)
            if key not in seen:
                seen.add(key)
                unique_models.append(model)

        return unique_models

    def apply_thinking_suffix(self, model_id: str, enabled: bool) -> str:
        """
        Apply or remove :thinking suffix from model ID.

        Args:
            model_id: Base model ID
            enabled: Whether thinking mode is enabled

        Returns:
            Model ID with or without :thinking suffix
        """
        base_id = model_id.replace(":thinking", "")

        if enabled and not model_id.endswith(":thinking"):
            return f"{base_id}:thinking"
        elif not enabled and model_id.endswith(":thinking"):
            return base_id

        return model_id


@lru_cache(maxsize=1)
def get_model_provider_service() -> ModelProviderService:
    """Get cached instance of ModelProviderService."""
    import os
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    return ModelProviderService(openrouter_api_key=openrouter_key)
