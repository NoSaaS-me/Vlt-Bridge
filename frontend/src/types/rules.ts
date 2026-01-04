/**
 * TypeScript types for Rules API (T067)
 *
 * These types match the backend Pydantic models and OpenAPI schema
 * for the Oracle Plugin System rules management.
 */

/**
 * Agent lifecycle hook points for rule triggers
 */
export type HookPoint =
  | 'on_query_start'
  | 'on_turn_start'
  | 'on_turn_end'
  | 'on_tool_call'
  | 'on_tool_complete'
  | 'on_tool_failure'
  | 'on_session_end';

/**
 * Types of actions a rule can execute
 */
export type ActionType = 'notify_self' | 'log' | 'set_state' | 'emit_event';

/**
 * Notification priority levels
 */
export type Priority = 'low' | 'normal' | 'high' | 'critical';

/**
 * Where in the agent flow to inject notifications
 */
export type InjectionPoint = 'immediate' | 'turn_start' | 'after_tool' | 'turn_end';

/**
 * Action configuration for a rule
 */
export interface RuleAction {
  type: ActionType;
  message?: string | null;
  category?: string | null;
  priority: Priority;
  deliver_at: InjectionPoint;
}

/**
 * Summary information about a rule for list responses
 */
export interface RuleInfo {
  id: string;
  name: string;
  description?: string | null;
  trigger: HookPoint;
  enabled: boolean;
  core: boolean;
  priority: number;
  plugin_id?: string | null;
}

/**
 * Detailed information about a rule including condition and action
 */
export interface RuleDetail extends RuleInfo {
  version: string;
  condition?: string | null;
  script?: string | null;
  action?: RuleAction | null;
  source_path: string;
}

/**
 * Response for listing rules
 */
export interface RuleListResponse {
  rules: RuleInfo[];
  total: number;
}

/**
 * Request body for toggling a rule
 */
export interface RuleToggleRequest {
  enabled: boolean;
}

/**
 * Request body for testing a rule
 */
export interface RuleTestRequest {
  context_override?: Record<string, unknown> | null;
}

/**
 * Response from testing a rule
 */
export interface RuleTestResponse {
  condition_result: boolean;
  action_would_execute: boolean;
  evaluation_time_ms: number;
  error?: string | null;
}

/**
 * Human-readable labels for hook points
 */
export const HOOK_POINT_LABELS: Record<HookPoint, string> = {
  on_query_start: 'Query Start',
  on_turn_start: 'Turn Start',
  on_turn_end: 'Turn End',
  on_tool_call: 'Tool Call',
  on_tool_complete: 'Tool Complete',
  on_tool_failure: 'Tool Failure',
  on_session_end: 'Session End',
};

/**
 * Human-readable labels for action types
 */
export const ACTION_TYPE_LABELS: Record<ActionType, string> = {
  notify_self: 'Notify Agent',
  log: 'Log Message',
  set_state: 'Set State',
  emit_event: 'Emit Event',
};

/**
 * Human-readable labels for priorities
 */
export const PRIORITY_LABELS: Record<Priority, string> = {
  low: 'Low',
  normal: 'Normal',
  high: 'High',
  critical: 'Critical',
};

// ============================================================================
// Plugin Types (T086)
// ============================================================================

/**
 * Schema for a plugin setting
 */
export interface PluginSettingSchema {
  name: string;
  type: 'integer' | 'float' | 'string' | 'boolean';
  default: number | string | boolean;
  description: string;
  min_value?: number | null;
  max_value?: number | null;
  options?: string[] | null;
}

/**
 * Summary information about a plugin for list responses
 */
export interface PluginInfo {
  id: string;
  name: string;
  version: string;
  description: string;
  rule_count: number;
  enabled: boolean;
}

/**
 * Detailed information about a plugin
 */
export interface PluginDetail extends PluginInfo {
  rules: RuleInfo[];
  requires: string[];
  settings_schema: Record<string, PluginSettingSchema>;
  source_dir: string;
}

/**
 * Response for listing plugins
 */
export interface PluginListResponse {
  plugins: PluginInfo[];
}

/**
 * Request body for updating plugin settings
 */
export interface PluginSettingsUpdateRequest {
  settings: Record<string, number | string | boolean>;
}
