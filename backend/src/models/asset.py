"""Pydantic models for asset (binary/non-markdown) file operations."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

OcrStatus = Literal["skipped", "pending", "running", "done", "failed"]


class AssetSummary(BaseModel):
    asset_path: str
    mime_type: str
    file_size: int
    ocr_status: OcrStatus
    updated: str


class AssetMetadata(AssetSummary):
    created: str
    ocr_text: Optional[str] = None


class AssetUploadResponse(BaseModel):
    asset_path: str
    mime_type: str
    file_size: int
    ocr_status: OcrStatus
    created: str


class AssetMoveRequest(BaseModel):
    new_path: str


class AssetSearchResult(BaseModel):
    asset_path: str
    mime_type: str
    filename: str
    snippet: str
    score: float
