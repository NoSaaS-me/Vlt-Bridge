export type OcrStatus = 'skipped' | 'pending' | 'running' | 'done' | 'failed';

export interface AssetSummary {
  asset_path: string;
  mime_type: string;
  file_size: number;
  ocr_status: OcrStatus;
  updated: string;
}

export interface AssetMetadata extends AssetSummary {
  created: string;
  ocr_text?: string;
}

export interface AssetUploadResponse {
  asset_path: string;
  mime_type: string;
  file_size: number;
  ocr_status: OcrStatus;
  created: string;
}
