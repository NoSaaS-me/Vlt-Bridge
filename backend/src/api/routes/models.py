"""API routes for model selection and user settings."""

from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from ..middleware import AuthContext, get_auth_context
from ...models.settings import (
    AgentConfig,
    AgentConfigUpdate,
    ModelInfo,
    ModelSettings,
    ModelSettingsUpdateRequest,
    ModelsListResponse,
    ModelProvider
)
from ...services.model_provider import ModelProviderService, get_model_provider_service
from ...services.user_settings import UserSettingsService, get_user_settings_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["models"])


@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    auth: AuthContext = Depends(get_auth_context),
    provider_service: ModelProviderService = Depends(get_model_provider_service)
):
    """
    Get all available models from all providers.

    Returns a combined list of models from Google AI and OpenRouter.
    """
    try:
        models = await provider_service.get_all_models()
        return ModelsListResponse(models=models)
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch models: {str(e)}"
        )


@router.get("/models/openrouter", response_model=ModelsListResponse)
async def list_openrouter_models(
    auth: AuthContext = Depends(get_auth_context),
    provider_service: ModelProviderService = Depends(get_model_provider_service)
):
    """
    Get available free models from OpenRouter API.

    Fetches models from OpenRouter's /api/v1/models endpoint and filters for:
    - Free models (pricing.prompt = "0")
    - Priority models (DeepSeek, Grok, Gemini, etc.)
    """
    try:
        models = await provider_service.get_openrouter_models()
        return ModelsListResponse(models=models)
    except Exception as e:
        logger.error(f"Failed to fetch OpenRouter models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch OpenRouter models: {str(e)}"
        )


@router.get("/models/google", response_model=ModelsListResponse)
async def list_google_models(
    auth: AuthContext = Depends(get_auth_context),
    provider_service: ModelProviderService = Depends(get_model_provider_service)
):
    """
    Get available Google AI models.

    Returns hardcoded list of supported Google Gemini models.
    """
    try:
        models = provider_service.get_google_models()
        return ModelsListResponse(models=models)
    except Exception as e:
        logger.error(f"Failed to fetch Google models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch Google models: {str(e)}"
        )


@router.get("/settings/models", response_model=ModelSettings)
async def get_model_settings(
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service)
):
    """
    Get user's current model preferences.

    Returns the user's saved model settings or defaults if not set.
    """
    try:
        settings = settings_service.get_settings(auth.user_id)
        return settings
    except Exception as e:
        logger.error(f"Failed to get model settings for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model settings: {str(e)}"
        )


@router.put("/settings/models", response_model=ModelSettings)
async def update_model_settings(
    request: ModelSettingsUpdateRequest,
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service)
):
    """
    Update user's model preferences.

    Allows partial updates - only provided fields will be updated.
    Returns the updated settings (API key is never returned, only openrouter_api_key_set flag).
    """
    try:
        updated_settings = settings_service.update_settings(
            user_id=auth.user_id,
            oracle_model=request.oracle_model,
            oracle_provider=request.oracle_provider,
            subagent_model=request.subagent_model,
            subagent_provider=request.subagent_provider,
            thinking_enabled=request.thinking_enabled,
            reasoning_effort=request.reasoning_effort,
            chat_center_mode=request.chat_center_mode,
            librarian_timeout=request.librarian_timeout,
            openrouter_api_key=request.openrouter_api_key
        )
        return updated_settings
    except Exception as e:
        logger.error(f"Failed to update model settings for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model settings: {str(e)}"
        )


@router.get("/settings/agent-config", response_model=AgentConfig)
async def get_agent_config(
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service)
):
    """Get user's agent configuration for turn control."""
    try:
        config = settings_service.get_agent_config(auth.user_id)
        return config
    except Exception as e:
        logger.error(f"Failed to get agent config for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get agent config: {str(e)}"
        )


@router.put("/settings/agent-config", response_model=AgentConfig)
async def update_agent_config(
    request: AgentConfigUpdate,
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service)
):
    """Update user's agent configuration for turn control."""
    try:
        config = settings_service.update_agent_config(
            user_id=auth.user_id,
            max_iterations=request.max_iterations,
            soft_warning_percent=request.soft_warning_percent,
            token_budget=request.token_budget,
            token_warning_percent=request.token_warning_percent,
            timeout_seconds=request.timeout_seconds,
            max_tool_calls_per_turn=request.max_tool_calls_per_turn,
            max_parallel_tools=request.max_parallel_tools,
        )
        return config
    except Exception as e:
        logger.error(f"Failed to update agent config for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update agent config: {str(e)}"
        )


@router.post("/settings/agent-config/reset", response_model=AgentConfig)
async def reset_agent_config(
    auth: AuthContext = Depends(get_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service)
):
    """Reset agent configuration to defaults."""
    try:
        # Reset all fields to defaults by explicitly setting them
        # Defaults match AgentConfig in settings.py
        config = settings_service.update_agent_config(
            user_id=auth.user_id,
            max_iterations=15,
            soft_warning_percent=70,
            token_budget=50000,
            token_warning_percent=80,
            timeout_seconds=120,
            max_tool_calls_per_turn=100,  # Increased from 5 to allow complex multi-step queries
            max_parallel_tools=3,
        )
        return config
    except Exception as e:
        logger.error(f"Failed to reset agent config for user {auth.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset agent config: {str(e)}"
        )
