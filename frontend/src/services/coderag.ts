/**
 * CodeRAG API service
 *
 * Provides functions for interacting with CodeRAG index management endpoints.
 * Based on spec: specs/011-coderag-project-init/contracts/coderag-api.yaml
 */

import { apiFetch } from './api';
import type {
  CodeRAGStatusResponse,
  InitCodeRAGRequest,
  InitCodeRAGResponse,
  JobStatusResponse,
} from '@/types/coderag';

/**
 * Get CodeRAG index status for a project
 *
 * @param projectId - Project identifier
 * @returns CodeRAG status including index state and active job info
 */
export async function getCodeRAGStatus(
  projectId: string
): Promise<CodeRAGStatusResponse> {
  const params = new URLSearchParams({ project_id: projectId });
  return apiFetch<CodeRAGStatusResponse>(`/api/coderag/status?${params.toString()}`);
}

/**
 * Initialize CodeRAG indexing for a project
 *
 * @param projectId - Project to associate with index
 * @param targetPath - Directory path to index
 * @param force - Force re-index even if index exists (default: false)
 * @param background - Run indexing in background (default: true)
 * @returns Init response with job ID and status
 */
export async function initCodeRAG(
  projectId: string,
  targetPath: string,
  force: boolean = false,
  background: boolean = true
): Promise<InitCodeRAGResponse> {
  const request: InitCodeRAGRequest = {
    project_id: projectId,
    target_path: targetPath,
    force,
    background,
  };

  return apiFetch<InitCodeRAGResponse>('/api/coderag/init', {
    method: 'POST',
    body: JSON.stringify(request),
  });
}

/**
 * Get detailed status of an indexing job
 *
 * @param jobId - Job identifier (UUID)
 * @returns Detailed job status including progress metrics
 */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return apiFetch<JobStatusResponse>(`/api/coderag/jobs/${encodeURIComponent(jobId)}`);
}

/**
 * Cancel an in-progress or pending indexing job
 *
 * @param jobId - Job identifier (UUID)
 * @returns Cancellation confirmation
 */
export async function cancelJob(
  jobId: string
): Promise<{ status: string; message: string }> {
  return apiFetch<{ status: string; message: string }>(
    `/api/coderag/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: 'POST' }
  );
}
