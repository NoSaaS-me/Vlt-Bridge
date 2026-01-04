/**
 * T062: Notification System Types
 * Types for subscriber management and notification settings
 */

/**
 * Information about a notification subscriber
 */
export interface SubscriberInfo {
  /** Unique subscriber identifier (e.g., "tool_failure", "budget_warning") */
  id: string;
  /** Human-readable subscriber name */
  name: string;
  /** Description of what this subscriber does */
  description: string;
  /** Subscriber version */
  version: string;
  /** List of event types this subscriber handles (e.g., ["tool.call.failure", "tool.call.timeout"]) */
  events: string[];
  /** If true, this subscriber cannot be disabled (core system functionality) */
  is_core: boolean;
  /** Current enabled state */
  enabled: boolean;
}

/**
 * User's notification preferences
 */
export interface NotificationSettings {
  /** List of disabled subscriber IDs */
  disabled_subscribers: string[];
}

/**
 * Response from toggling a subscriber
 */
export interface ToggleSubscriberResponse {
  /** The subscriber ID that was toggled */
  id: string;
  /** New enabled state */
  enabled: boolean;
}
