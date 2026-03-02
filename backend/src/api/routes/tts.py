"""HTTP API routes for ElevenLabs text-to-speech."""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from ..middleware import AuthContext, require_auth_context
from ...models.settings import TtsSettings, TtsSettingsUpdate, VoiceInfo, VoiceListResponse
from ...services.user_settings import UserSettingsService, get_user_settings_service

router = APIRouter()

ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
DEFAULT_MODEL = "eleven_multilingual_v2"
# ElevenLabs docs mention a hard limit; keep a conservative cap for safety.
MAX_TEXT_LENGTH = 4800
ELEVENLABS_VOICES_URL = "https://api.elevenlabs.io/v1/voices"


class TtsRequest(BaseModel):
    """Payload for synthesizing speech."""

    text: str = Field(..., min_length=1, description="Plaintext to convert to speech")
    voice_id: str | None = Field(
        default=None,
        description="Override voice id; falls back to ELEVENLABS_VOICE_ID",
    )
    model: str | None = Field(
        default=None,
        description="Model override; defaults to ELEVENLABS_MODEL or a safe default",
    )


async def _call_elevenlabs(
    api_key: str, voice_id: str, model: str, text: str
) -> httpx.Response:
    """Invoke ElevenLabs TTS API and return the raw response."""
    headers = {
        "xi-api-key": api_key,
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        return await client.post(
            f"{ELEVENLABS_API_URL}/{voice_id}",
            headers=headers,
            json=payload,
        )


def _resolve_api_key(settings_service: UserSettingsService, user_id: str) -> str | None:
    """Get ElevenLabs API key: user setting first, then env var fallback."""
    return settings_service.get_elevenlabs_api_key(user_id) or os.getenv("ELEVENLABS_API_KEY")


@router.post("/api/tts")
async def synthesize_tts(
    payload: TtsRequest,
    auth: AuthContext = Depends(require_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """Synthesize speech for the provided text using ElevenLabs."""
    api_key = _resolve_api_key(settings_service, auth.user_id)
    default_voice = os.getenv("ELEVENLABS_VOICE_ID")
    default_model = os.getenv("ELEVENLABS_MODEL") or DEFAULT_MODEL

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "tts_not_configured",
                "message": "ElevenLabs API key is not configured. Set it in Settings > TTS or via ELEVENLABS_API_KEY env var.",
            },
        )

    voice_id = payload.voice_id or default_voice
    if not voice_id:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "voice_required",
                "message": "Voice ID is required. Set ELEVENLABS_VOICE_ID or pass voice_id in the request.",
            },
        )

    text = (payload.text or "").strip()
    if not text:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "empty_text",
                "message": "Text is empty.",
            },
        )

    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    try:
        response = await _call_elevenlabs(
            api_key, voice_id, payload.model or default_model, text
        )
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "tts_timeout",
                "message": "ElevenLabs request timed out.",
            },
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "tts_http_error",
                "message": f"ElevenLabs request failed: {str(exc)}",
            },
        ) from exc

    if response.status_code >= 400:
        try:
            error_payload = response.json()
            message = (
                error_payload.get("detail")
                or error_payload.get("message")
                or "Failed to synthesize speech."
            )
        except Exception:
            message = response.text[:200] or "Failed to synthesize speech."

        raise HTTPException(
            status_code=response.status_code,
            detail={
                "error": "tts_failed",
                "message": message,
            },
        )

    return Response(
        content=response.content,
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/api/voices", response_model=VoiceListResponse)
async def list_voices(
    auth: AuthContext = Depends(require_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
):
    """List available ElevenLabs voices (proxied to keep API key server-side)."""
    api_key = _resolve_api_key(settings_service, auth.user_id)
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "tts_not_configured",
                "message": "ElevenLabs API key is not configured. Set it in Settings > TTS or via ELEVENLABS_API_KEY env var.",
            },
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                ELEVENLABS_VOICES_URL,
                headers={"xi-api-key": api_key},
            )
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "voices_fetch_failed",
                "message": f"Failed to fetch voices: {exc}",
            },
        ) from exc

    if resp.status_code >= 400:
        raise HTTPException(
            status_code=resp.status_code,
            detail={
                "error": "voices_api_error",
                "message": resp.text[:200],
            },
        )

    data = resp.json()
    voices = []
    for v in data.get("voices", []):
        voices.append(
            VoiceInfo(
                voice_id=v["voice_id"],
                name=v.get("name", "Unknown"),
                category=v.get("category", "unknown"),
                labels=v.get("labels", {}),
                preview_url=v.get("preview_url", ""),
            )
        )
    return VoiceListResponse(voices=voices)


@router.get("/api/settings/tts", response_model=TtsSettings)
async def get_tts_settings(
    auth: AuthContext = Depends(require_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
) -> TtsSettings:
    """Get the user's TTS voice and model preferences."""
    data = settings_service.get_tts_settings(auth.user_id)
    return TtsSettings(voice_id=data["voice_id"], model=data["model"], api_key_set=data["api_key_set"])


@router.put("/api/settings/tts", response_model=TtsSettings)
async def update_tts_settings(
    body: TtsSettingsUpdate,
    auth: AuthContext = Depends(require_auth_context),
    settings_service: UserSettingsService = Depends(get_user_settings_service),
) -> TtsSettings:
    """Update the user's TTS voice and model preferences."""
    settings_service.set_tts_settings(auth.user_id, body.voice_id, body.model, body.api_key)
    data = settings_service.get_tts_settings(auth.user_id)
    return TtsSettings(voice_id=data["voice_id"], model=data["model"], api_key_set=data["api_key_set"])
