/**
 * Project API service for multi-project architecture.
 * Handles CRUD operations for projects.
 */

import { apiFetch } from './api';
import type { Project, ProjectCreate, ProjectUpdate, ProjectStats } from '@/types/project';

interface ProjectListResponse {
  projects: Project[];
  total: number;
}

/**
 * Fetch all projects for the current user.
 */
export async function fetchProjects(): Promise<Project[]> {
  const response = await apiFetch<ProjectListResponse>('/api/projects');
  return response.projects;
}

/**
 * Get a single project by ID.
 */
export async function fetchProject(projectId: string): Promise<Project> {
  return apiFetch<Project>(`/api/projects/${encodeURIComponent(projectId)}`);
}

/**
 * Create a new project.
 */
export async function createProject(data: ProjectCreate): Promise<Project> {
  return apiFetch<Project>('/api/projects', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

/**
 * Update an existing project.
 */
export async function updateProject(projectId: string, data: ProjectUpdate): Promise<Project> {
  return apiFetch<Project>(`/api/projects/${encodeURIComponent(projectId)}`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
}

/**
 * Delete a project.
 */
export async function deleteProject(projectId: string): Promise<void> {
  await apiFetch<void>(`/api/projects/${encodeURIComponent(projectId)}`, {
    method: 'DELETE',
  });
}

/**
 * Get project statistics (note count, thread count, etc).
 */
export async function fetchProjectStats(projectId: string): Promise<ProjectStats> {
  return apiFetch<ProjectStats>(`/api/projects/${encodeURIComponent(projectId)}/stats`);
}
