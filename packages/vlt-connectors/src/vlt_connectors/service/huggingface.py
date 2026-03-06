"""HuggingFace Inference API service connector."""
from __future__ import annotations
from typing import Any, ClassVar, Optional
import httpx
from ..base import ServiceConnector
from ..models import CredentialField

HF_INFERENCE_BASE = "https://api-inference.huggingface.co"


class HuggingFaceInferenceConnector(ServiceConnector):
    name = "huggingface_inference"
    display_name = "HuggingFace Inference"
    description = "ML model inference via HuggingFace serverless or dedicated endpoints."
    connector_type: ClassVar[str] = "service"
    available_contexts: ClassVar[list[str]] = ["backend", "daemon", "cli"]

    credential_fields = [
        CredentialField(name="api_key", label="HF API Token", secret=True, placeholder="hf_..."),
        CredentialField(
            name="default_embedding_model",
            label="Default Embedding Model",
            secret=False,
            placeholder="BAAI/bge-small-en-v1.5",
        ),
        CredentialField(
            name="default_classification_model",
            label="Default Classification Model (prompt injection detection)",
            secret=False,
            placeholder="deepset/deberta-v3-base-injection",
        ),
        CredentialField(
            name="endpoint_url",
            label="Custom Endpoint URL (leave empty for HuggingFace serverless)",
            secret=False,
            placeholder="https://your-endpoint.endpoints.huggingface.cloud",
        ),
    ]

    async def embed(
        self,
        texts: list[str],
        credentials: dict[str, str],
        model: Optional[str] = None,
    ) -> list[list[float]]:
        """Generate embeddings. Returns list of float vectors."""
        api_key = credentials.get("api_key", "").strip()
        model_id = model or credentials.get("default_embedding_model", "BAAI/bge-small-en-v1.5")
        endpoint = credentials.get("endpoint_url") or f"{HF_INFERENCE_BASE}/models/{model_id}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                endpoint,
                json={"inputs": texts},
                headers={"Authorization": f"Bearer {api_key}"},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    async def classify(
        self,
        text: str,
        credentials: dict[str, str],
        model: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run text classification (e.g., prompt injection detection)."""
        api_key = credentials.get("api_key", "").strip()
        model_id = model or credentials.get("default_classification_model", "deepset/deberta-v3-base-injection")
        endpoint = credentials.get("endpoint_url") or f"{HF_INFERENCE_BASE}/models/{model_id}"

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                endpoint,
                json={"inputs": text},
                headers={"Authorization": f"Bearer {api_key}"},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    async def text_generation(
        self,
        prompt: str,
        credentials: dict[str, str],
        model: Optional[str] = None,
        max_new_tokens: int = 256,
    ) -> str:
        """Generate text completion."""
        api_key = credentials.get("api_key", "").strip()
        model_id = model or "mistralai/Mistral-7B-Instruct-v0.1"
        endpoint = credentials.get("endpoint_url") or f"{HF_INFERENCE_BASE}/models/{model_id}"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                endpoint,
                json={"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}},
                headers={"Authorization": f"Bearer {api_key}"},
            )

        if resp.status_code >= 400:
            raise RuntimeError(f"HuggingFace API error {resp.status_code}: {resp.text[:200]}")
        result = resp.json()
        if isinstance(result, list) and result:
            return result[0].get("generated_text", "")
        return str(result)
