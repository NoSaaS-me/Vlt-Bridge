/**
 * Model selection and settings service
 */
import { apiFetch } from './api';
import type { ModelInfo, ModelSettings, ModelsResponse } from '@/types/models';

/**
 * Fetch available models from the backend
 */
export async function getModels(): Promise<ModelInfo[]> {
  const response = await apiFetch<ModelsResponse>('/api/models');
  return response.models;
}

/**
 * Get user's current model settings
 */
export async function getModelSettings(): Promise<ModelSettings> {
  return apiFetch<ModelSettings>('/api/settings/models');
}

/**
 * Save user's model settings
 */
export async function saveModelSettings(settings: ModelSettings): Promise<ModelSettings> {
  return apiFetch<ModelSettings>('/api/settings/models', {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}

export interface TavilyTestResponse {
  success: boolean;
  message: string;
  results_count?: number;
}

/**
 * Test Tavily API key connection
 */
export async function testTavilyConnection(): Promise<TavilyTestResponse> {
  return apiFetch<TavilyTestResponse>('/api/settings/test-tavily', {
    method: 'POST',
  });
}
