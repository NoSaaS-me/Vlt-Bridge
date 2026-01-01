/**
 * Model provider types and model selection configuration
 * Note: Field names match backend snake_case (Pydantic default)
 */

export type ModelProvider = 'openrouter' | 'google';

export interface ModelInfo {
  id: string;
  name: string;
  provider: ModelProvider;
  is_free: boolean;
  supports_thinking: boolean;
  context_length: number | null;
  description?: string;
}

export interface ModelSettings {
  oracle_model: string;
  oracle_provider: ModelProvider;
  subagent_model: string;
  subagent_provider: ModelProvider;
  thinking_enabled: boolean;
  chat_center_mode: boolean;
  librarian_timeout: number;
  openrouter_api_key?: string | null;
  openrouter_api_key_set: boolean;
}

export interface ModelsResponse {
  models: ModelInfo[];
}
