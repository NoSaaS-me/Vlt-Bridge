"""Ollama local LLM service connector."""
from __future__ import annotations
from typing import ClassVar, Optional
import httpx
from ..base import ServiceConnector
from ..models import CredentialField

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"


class OllamaConnector(ServiceConnector):
    name = "ollama"
    display_name = "Ollama"
    description = "Local LLM inference via Ollama. Supports embeddings, text generation, and chat completions."
    connector_type: ClassVar[str] = "service"
    available_contexts: ClassVar[list[str]] = ["backend", "daemon", "cli"]

    credential_fields = [
        CredentialField(
            name="base_url",
            label="Ollama Base URL",
            secret=False,
            placeholder=DEFAULT_BASE_URL,
        ),
        CredentialField(
            name="default_model",
            label="Default Model",
            secret=False,
            placeholder=DEFAULT_MODEL,
        ),
        CredentialField(
            name="default_embedding_model",
            label="Default Embedding Model",
            secret=False,
            placeholder=DEFAULT_EMBEDDING_MODEL,
        ),
    ]

    def _base_url(self, credentials: dict[str, str]) -> str:
        return (credentials.get("base_url") or DEFAULT_BASE_URL).rstrip("/")

    async def embed(
        self,
        texts: list[str],
        credentials: dict[str, str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Uses POST /api/embed with batch input. Returns list of float vectors.
        """
        base = self._base_url(credentials)
        model_id = model or credentials.get("default_embedding_model", DEFAULT_EMBEDDING_MODEL)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{base}/api/embed",
                json={"model": model_id, "input": texts},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama embed error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        # Ollama /api/embed returns {"embeddings": [[...], ...]}
        return data["embeddings"]

    async def generate(
        self,
        prompt: str,
        credentials: dict[str, str],
        model: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """Generate text from a prompt.

        Uses POST /api/generate (non-streaming). Returns the response text string.
        """
        base = self._base_url(credentials)
        model_id = model or credentials.get("default_model", DEFAULT_MODEL)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base}/api/generate",
                json={"model": model_id, "prompt": prompt, "stream": False},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama generate error {resp.status_code}: {resp.text[:200]}")

        return resp.json()["response"]

    async def chat(
        self,
        messages: list[dict],
        credentials: dict[str, str],
        model: Optional[str] = None,
    ) -> str:
        """Chat completion via POST /api/chat.

        messages format: [{"role": "user", "content": "..."}]
        Returns the assistant message content string.
        """
        base = self._base_url(credentials)
        model_id = model or credentials.get("default_model", DEFAULT_MODEL)

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{base}/api/chat",
                json={"model": model_id, "messages": messages, "stream": False},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama chat error {resp.status_code}: {resp.text[:200]}")

        return resp.json()["message"]["content"]

    async def list_models(self, credentials: dict[str, str]) -> list[str]:
        """List available models via GET /api/tags.

        Returns a list of model name strings.
        """
        base = self._base_url(credentials)

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{base}/api/tags")

        if resp.status_code >= 400:
            raise RuntimeError(f"Ollama list_models error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
