/**
 * TypeScript types for CodeRAG API
 *
 * Based on spec: specs/011-coderag-project-init/contracts/coderag-api.yaml
 */

/**
 * CodeRAG index status values
 */
export type CodeRAGIndexStatus =
  | 'not_initialized'
  | 'indexing'
  | 'ready'
  | 'failed'
  | 'stale';

/**
 * Indexing job status values
 */
export type JobStatus =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

/**
 * Init job status (queued vs immediately started)
 */
export type InitJobStatus = 'queued' | 'started';

/**
 * Summary of an active indexing job
 */
export interface JobSummary {
  job_id: string;
  progress_percent: number;
  files_processed: number;
  files_total: number;
  started_at: string; // ISO 8601 datetime
}

/**
 * Response from GET /api/coderag/status
 */
export interface CodeRAGStatusResponse {
  project_id: string;
  status: CodeRAGIndexStatus;
  file_count: number;
  chunk_count: number;
  last_indexed_at: string | null; // ISO 8601 datetime
  error_message: string | null;
  active_job: JobSummary | null;
}

/**
 * Request payload for POST /api/coderag/init
 */
export interface InitCodeRAGRequest {
  project_id: string;
  target_path: string;
  force?: boolean;
  background?: boolean;
}

/**
 * Response from POST /api/coderag/init
 */
export interface InitCodeRAGResponse {
  job_id: string;
  status: InitJobStatus;
  message: string;
}

/**
 * Response from GET /api/coderag/jobs/{job_id}
 */
export interface JobStatusResponse {
  job_id: string;
  project_id: string;
  status: JobStatus;
  progress_percent: number;
  files_total: number;
  files_processed: number;
  chunks_created: number;
  started_at: string | null; // ISO 8601 datetime
  completed_at: string | null; // ISO 8601 datetime
  error_message: string | null;
  duration_seconds: number | null;
}

/**
 * Error response format
 */
export interface CodeRAGErrorResponse {
  error: string;
  message: string;
  detail?: string;
}
