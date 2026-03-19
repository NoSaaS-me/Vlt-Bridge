"""Content Factory — output destination executor.

Fires after content is approved (manually or auto-approved).
Collects multimodal outputs from pipeline stages and routes them
to configured destinations.

Output types:
  vault_note         — save as markdown note with embedded assets in vault
  file_save          — write files to artifact working directory
  connector_publish  — push through a connector (API, CMS, etc.)

Each output receives the full stage results dict and extracts content
by type (text, image, audio, metadata).
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from artifact_harness import harness_call  # type: ignore[import]
except ImportError:
    def harness_call(call_type: str, payload: dict) -> dict:  # type: ignore[misc]
        raise RuntimeError("artifact_harness not available")


_BACKEND_DIR = Path(__file__).parent
_ARTIFACT_DIR = _BACKEND_DIR.parent


# ---------------------------------------------------------------------------
# Multimodal content collector
# ---------------------------------------------------------------------------

class ContentPackage:
    """Collected outputs from a pipeline run, classified by media type."""

    def __init__(self, stages: dict, topic: str = "", metadata: dict | None = None):
        self.topic = topic
        self.texts: list[dict] = []      # {stage_id, content, tokens}
        self.images: list[dict] = []     # {stage_id, data, format, prompt_used}
        self.reviews: list[dict] = []    # {stage_id, score, feedback, auto_approved}
        self.metadata = metadata or {}
        self.all_stages = stages

        self._classify(stages)

    def _classify(self, stages: dict) -> None:
        """Walk stage results and classify by content type."""
        for stage_id, result in stages.items():
            if not isinstance(result, dict):
                continue

            output = result.get("output", "")

            # Review stages (have score field)
            if "score" in result:
                self.reviews.append({
                    "stage_id": stage_id,
                    "score": result["score"],
                    "feedback": result.get("feedback", ""),
                    "auto_approved": result.get("auto_approved", False),
                    "threshold": result.get("threshold"),
                })
                continue

            # Image detection: URLs or base64 blobs
            if isinstance(output, str) and (
                output.startswith(("http://", "https://", "data:image/"))
                or (len(output) > 500 and _looks_like_base64(output))
            ):
                fmt = "png"
                if "jpeg" in output[:50] or "jpg" in output[:50]:
                    fmt = "jpg"
                self.images.append({
                    "stage_id": stage_id,
                    "data": output,
                    "format": fmt,
                    "prompt_used": result.get("prompt_used", ""),
                    "model": result.get("model", ""),
                    "seed": result.get("seed"),
                })
                continue

            # Everything else is text
            if isinstance(output, str) and output.strip():
                self.texts.append({
                    "stage_id": stage_id,
                    "content": output,
                    "tokens": result.get("tokens", 0),
                    "model": result.get("model", ""),
                    "source_length": result.get("source_length"),
                    "output_length": result.get("output_length"),
                })

    @property
    def primary_text(self) -> str:
        """Last text output (the final product after cleanup stages)."""
        return self.texts[-1]["content"] if self.texts else ""

    @property
    def primary_score(self) -> float | None:
        """Score from the last review stage."""
        return self.reviews[-1]["score"] if self.reviews else None

    @property
    def primary_feedback(self) -> str:
        """Feedback from the last review stage."""
        return self.reviews[-1]["feedback"] if self.reviews else ""

    @property
    def slug(self) -> str:
        """URL-safe slug from topic."""
        s = self.topic.lower().strip()
        s = re.sub(r"[^a-z0-9]+", "-", s)
        return s.strip("-")[:60] or "untitled"

    @property
    def total_tokens(self) -> int:
        return sum(t.get("tokens", 0) for t in self.texts)

    @property
    def total_cost_usd(self) -> float:
        return round(self.total_tokens * 0.000002, 6)


def _looks_like_base64(s: str) -> bool:
    """Heuristic: string is likely base64-encoded binary."""
    clean = s.strip()
    if len(clean) < 100:
        return False
    sample = clean[:200]
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r")
    return all(c in valid_chars for c in sample)


# ---------------------------------------------------------------------------
# Template variable expansion
# ---------------------------------------------------------------------------

def _expand_template(template: str, pkg: ContentPackage, extra: dict | None = None) -> str:
    """Replace {{var}} placeholders with values from the content package."""
    now = datetime.now(timezone.utc)
    variables = {
        "topic": pkg.topic,
        "slug": pkg.slug,
        "date": now.strftime("%Y-%m-%d"),
        "datetime": now.isoformat()[:19],
        "year": now.strftime("%Y"),
        "month": now.strftime("%m"),
        "day": now.strftime("%d"),
        "time": now.strftime("%H%M"),
        "content": pkg.primary_text,
        "score": f"{pkg.primary_score:.1f}" if pkg.primary_score is not None else "",
        "feedback": pkg.primary_feedback,
        "tokens": str(pkg.total_tokens),
        "cost": f"{pkg.total_cost_usd:.6f}",
        "image_count": str(len(pkg.images)),
    }
    if extra:
        variables.update(extra)

    result = template
    for key, value in variables.items():
        result = result.replace("{{" + key + "}}", str(value))
    return result


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _download_image(url: str) -> tuple[bytes, str]:
    """Download an image URL, return (bytes, extension)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ContentFactory/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "image/png")
            ext = "png"
            if "jpeg" in content_type or "jpg" in content_type:
                ext = "jpg"
            elif "webp" in content_type:
                ext = "webp"
            elif "gif" in content_type:
                ext = "gif"
            return data, ext
    except Exception as e:
        raise RuntimeError(f"Failed to download image from {url[:100]}: {e}")


def _decode_base64_image(data: str) -> tuple[bytes, str]:
    """Decode a base64 image string, return (bytes, extension)."""
    # Handle data URI format: data:image/png;base64,...
    if data.startswith("data:"):
        header, encoded = data.split(",", 1)
        ext = "png"
        if "jpeg" in header or "jpg" in header:
            ext = "jpg"
        elif "webp" in header:
            ext = "webp"
        return base64.b64decode(encoded), ext

    # Raw base64
    raw = base64.b64decode(data)
    # Detect by magic bytes
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return raw, "png"
    elif raw[:3] == b"\xff\xd8\xff":
        return raw, "jpg"
    return raw, "png"


def _resolve_image(image_data: str) -> tuple[bytes, str]:
    """Resolve an image (URL or base64) to (bytes, extension)."""
    if image_data.startswith(("http://", "https://")):
        return _download_image(image_data)
    return _decode_base64_image(image_data)


# ---------------------------------------------------------------------------
# Output executors
# ---------------------------------------------------------------------------

def _execute_vault_note(config: dict, pkg: ContentPackage) -> dict:
    """Save content as a markdown note with embedded assets in the vault.

    Config:
        folder:           Target folder pattern (default: "ContentForge/{{date}}")
        title:            Note title pattern (default: "{{topic}}")
        include_metadata: Include score/cost/model info (default: true)
        embed_images:     Embed images inline (default: true)
    """
    folder = _expand_template(config.get("folder", "ContentForge/{{date}}"), pkg)
    title = _expand_template(config.get("title", "{{topic}}"), pkg)
    include_metadata = config.get("include_metadata", True)
    embed_images = config.get("embed_images", True)

    # Build note content
    lines: list[str] = []
    lines.append(f"# {title}\n")

    if include_metadata and pkg.primary_score is not None:
        lines.append(f"> **Score:** {pkg.primary_score:.1f}/10 | "
                     f"**Cost:** ${pkg.total_cost_usd:.4f} | "
                     f"**Tokens:** {pkg.total_tokens}\n")

    lines.append(pkg.primary_text)

    # Handle images
    saved_assets: list[str] = []
    if embed_images and pkg.images:
        lines.append("\n\n---\n")
        for i, img in enumerate(pkg.images):
            filename = f"{pkg.slug}_img{i+1}.{img['format']}"
            asset_path = f"{folder}/{filename}"

            try:
                img_bytes, ext = _resolve_image(img["data"])
                # Write image directly to vault path (artifact is inside the vault tree)
                vault_root = _find_vault_root()
                if vault_root:
                    full_path = vault_root / folder / filename
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_bytes(img_bytes)
                    saved_assets.append(asset_path)
                    lines.append(f"\n[[embed:{filename}]]")
                    if img.get("prompt_used"):
                        lines.append(f"*Prompt: {img['prompt_used'][:100]}*\n")
            except Exception as e:
                lines.append(f"\n*Image {i+1} failed: {e}*\n")

    if include_metadata and pkg.primary_feedback:
        lines.append(f"\n\n---\n**Review feedback:** {pkg.primary_feedback}")

    note_content = "\n".join(lines)

    # Write note to vault
    note_filename = f"{pkg.slug}.md"
    note_path = f"{folder}/{note_filename}"
    vault_root = _find_vault_root()
    if vault_root:
        full_note_path = vault_root / folder / note_filename
        full_note_path.parent.mkdir(parents=True, exist_ok=True)
        full_note_path.write_text(note_content, encoding="utf-8")
    else:
        # Fallback: write to artifact's own directory
        fallback = _ARTIFACT_DIR / "outputs" / folder / note_filename
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(note_content, encoding="utf-8")
        note_path = str(fallback.relative_to(_ARTIFACT_DIR))

    return {
        "type": "vault_note",
        "success": True,
        "note_path": note_path,
        "assets": saved_assets,
        "note_length": len(note_content),
    }


def _execute_file_save(config: dict, pkg: ContentPackage) -> dict:
    """Save outputs as files in the artifact's working directory.

    Config:
        output_dir:        Directory pattern (default: "outputs/{{date}}")
        filename:          Filename pattern (default: "{{slug}}")
        format:            Output format: md, txt, html, json (default: "md")
        include_metadata:  Write .json sidecar (default: true)
        save_images:       Save images as separate files (default: true)
    """
    output_dir = _expand_template(config.get("output_dir", "outputs/{{date}}"), pkg)
    filename = _expand_template(config.get("filename", "{{slug}}"), pkg)
    fmt = config.get("format", "md")
    include_metadata = config.get("include_metadata", True)
    save_images = config.get("save_images", True)

    base_dir = _ARTIFACT_DIR / output_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    files_written: list[dict] = []

    # Write main content
    if fmt == "json":
        content_data = {
            "topic": pkg.topic,
            "content": pkg.primary_text,
            "score": pkg.primary_score,
            "feedback": pkg.primary_feedback,
            "images": [{"stage_id": img["stage_id"], "format": img["format"],
                        "prompt_used": img.get("prompt_used", "")} for img in pkg.images],
            "metadata": pkg.metadata,
        }
        content_str = json.dumps(content_data, indent=2)
    elif fmt == "html":
        content_str = f"<html><head><title>{pkg.topic}</title></head><body>\n"
        content_str += f"<h1>{pkg.topic}</h1>\n"
        # Convert markdown-ish content to basic HTML paragraphs
        for para in pkg.primary_text.split("\n\n"):
            para = para.strip()
            if para.startswith("# "):
                content_str += f"<h1>{para[2:]}</h1>\n"
            elif para.startswith("## "):
                content_str += f"<h2>{para[3:]}</h2>\n"
            elif para.startswith("### "):
                content_str += f"<h3>{para[4:]}</h3>\n"
            elif para:
                content_str += f"<p>{para}</p>\n"
        content_str += "</body></html>"
    else:
        # md or txt
        content_str = pkg.primary_text

    ext = fmt if fmt in ("md", "txt", "html", "json") else "md"
    content_path = base_dir / f"{filename}.{ext}"
    content_path.write_text(content_str, encoding="utf-8")
    files_written.append({
        "path": str(content_path.relative_to(_ARTIFACT_DIR)),
        "type": f"text/{ext}",
        "size": len(content_str),
    })

    # Write metadata sidecar
    if include_metadata:
        meta = {
            "topic": pkg.topic,
            "score": pkg.primary_score,
            "feedback": pkg.primary_feedback,
            "tokens": pkg.total_tokens,
            "cost_usd": pkg.total_cost_usd,
            "image_count": len(pkg.images),
            "stages": {sid: {"model": s.get("model", ""), "tokens": s.get("tokens", 0)}
                       for sid, s in pkg.all_stages.items() if isinstance(s, dict)},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = base_dir / f"{filename}.meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        files_written.append({
            "path": str(meta_path.relative_to(_ARTIFACT_DIR)),
            "type": "application/json",
            "size": meta_path.stat().st_size,
        })

    # Save images
    if save_images and pkg.images:
        for i, img in enumerate(pkg.images):
            try:
                img_bytes, ext = _resolve_image(img["data"])
                img_path = base_dir / f"{filename}_img{i+1}.{ext}"
                img_path.write_bytes(img_bytes)
                files_written.append({
                    "path": str(img_path.relative_to(_ARTIFACT_DIR)),
                    "type": f"image/{ext}",
                    "size": len(img_bytes),
                })
            except Exception as e:
                files_written.append({
                    "path": f"(failed) {filename}_img{i+1}",
                    "type": "error",
                    "error": str(e),
                })

    return {
        "type": "file_save",
        "success": True,
        "files": files_written,
        "output_dir": output_dir,
    }


def _execute_connector_publish(config: dict, pkg: ContentPackage) -> dict:
    """Publish content through a connector.

    Config:
        connector:  Connector name (e.g. "custom_openai", "openrouter")
        action:     Action to invoke (e.g. "chat_completion", "generate_text")
        params:     Dict of params with {{template}} variables
                    Special vars: {{content}}, {{title}}, {{image}}, {{image_base64}}
    """
    connector = config.get("connector", "")
    action = config.get("action", "")
    param_templates = config.get("params", {})

    if not connector or not action:
        return {"type": "connector_publish", "success": False,
                "error": "connector and action are required"}

    # Build extra template vars for connector params
    extra: dict[str, str] = {"title": pkg.topic}

    # First image as template variable
    if pkg.images:
        img = pkg.images[0]
        extra["image"] = img["data"]  # URL or base64
        if img["data"].startswith(("http://", "https://")):
            extra["image_url"] = img["data"]
            extra["image_base64"] = ""
        else:
            extra["image_url"] = ""
            extra["image_base64"] = img["data"]
    else:
        extra["image"] = ""
        extra["image_url"] = ""
        extra["image_base64"] = ""

    # Expand templates in all param values
    params: dict[str, Any] = {}
    for key, value in param_templates.items():
        if isinstance(value, str):
            params[key] = _expand_template(value, pkg, extra)
        else:
            params[key] = value

    try:
        raw = harness_call("connector_call", {
            "connector": connector,
            "action": action,
            "params": params,
        })

        # Unwrap response (same as pipeline._call_connector)
        connector_result = raw.get("result", raw)
        if isinstance(connector_result, dict):
            if connector_result.get("success") is False:
                return {"type": "connector_publish", "success": False,
                        "error": connector_result.get("error", "Unknown error"),
                        "connector": connector, "action": action}
            data = connector_result.get("data", connector_result)
        else:
            data = connector_result

        return {
            "type": "connector_publish",
            "success": True,
            "connector": connector,
            "action": action,
            "response": _truncate(data, 500),
        }
    except Exception as e:
        return {"type": "connector_publish", "success": False,
                "error": str(e), "connector": connector, "action": action}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_OUTPUT_EXECUTORS = {
    "vault_note": _execute_vault_note,
    "file_save": _execute_file_save,
    "connector_publish": _execute_connector_publish,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def execute_outputs(
    outputs_config: list[dict],
    stages_result: dict,
    topic: str = "",
    metadata: dict | None = None,
) -> list[dict]:
    """Execute all configured outputs for approved content.

    Args:
        outputs_config: List of output config dicts (each has "type" key).
        stages_result:  Full stage results from execute_pipeline().
        topic:          Original topic string.
        metadata:       Extra metadata (item_id, pipeline_version, etc.).

    Returns:
        List of output result dicts (one per configured output).
    """
    pkg = ContentPackage(stages_result, topic=topic, metadata=metadata)
    results: list[dict] = []

    for output_config in outputs_config:
        output_type = output_config.get("type", "")
        executor = _OUTPUT_EXECUTORS.get(output_type)

        if executor is None:
            results.append({
                "type": output_type,
                "success": False,
                "error": f"Unknown output type: '{output_type}'. "
                         f"Available: {sorted(_OUTPUT_EXECUTORS)}",
            })
            continue

        try:
            result = executor(output_config, pkg)
            results.append(result)
            _notify(f"Output '{output_type}' complete", severity="info")
        except Exception as e:
            results.append({
                "type": output_type,
                "success": False,
                "error": str(e),
            })
            _notify(f"Output '{output_type}' failed: {e}", severity="error")

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_vault_root() -> Path | None:
    """Walk up from artifact dir to find the vault root (contains the user's vault).

    Artifact layout: data/vaults/{user_id}/{project_id}/Artifacts/{name}_{id}/
    Vault root:      data/vaults/{user_id}/{project_id}/
    """
    # Go up from artifact dir: Artifacts/{name}_{id} → {project_id}
    candidate = _ARTIFACT_DIR.parent.parent
    # Sanity check: should be inside a 'vaults' tree
    if "vaults" in str(candidate):
        return candidate
    return None


def _notify(message: str, severity: str = "info") -> None:
    """Emit a notification line on stdout (consumed by the harness)."""
    sys.stdout.write(json.dumps({
        "_type": "notification",
        "severity": severity,
        "message": message,
    }) + "\n")
    sys.stdout.flush()


def _truncate(data: Any, max_len: int = 500) -> Any:
    """Truncate large values for storage in output results."""
    if isinstance(data, str) and len(data) > max_len:
        return data[:max_len] + "..."
    if isinstance(data, dict):
        return {k: _truncate(v, max_len) for k, v in data.items()}
    return data
