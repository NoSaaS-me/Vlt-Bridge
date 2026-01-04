/**
 * T063: Notifications API Client
 * Handles subscriber management and notification settings
 */
import { apiFetch } from './api';
import type {
  SubscriberInfo,
  NotificationSettings,
  ToggleSubscriberResponse,
} from '@/types/notifications';

/**
 * Response wrapper from the backend
 */
interface SubscriberListResponse {
  subscribers: SubscriberInfo[];
}

/**
 * Get all registered notification subscribers
 */
export async function getSubscribers(): Promise<SubscriberInfo[]> {
  const response = await apiFetch<SubscriberListResponse>('/api/notifications/subscribers');
  return response.subscribers;
}

/**
 * Toggle a subscriber's enabled state
 * @param id The subscriber ID to toggle
 * @param enabled The new enabled state
 * @returns The new enabled state
 */
export async function toggleSubscriber(id: string, enabled: boolean): Promise<ToggleSubscriberResponse> {
  return apiFetch<ToggleSubscriberResponse>(
    `/api/notifications/subscribers/${encodeURIComponent(id)}/toggle`,
    {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }
  );
}

/**
 * Get notification settings for the current user
 */
export async function getNotificationSettings(): Promise<NotificationSettings> {
  return apiFetch<NotificationSettings>('/api/settings/notifications');
}

/**
 * Update notification settings
 * @param settings The settings to update
 */
export async function updateNotificationSettings(
  settings: NotificationSettings
): Promise<NotificationSettings> {
  return apiFetch<NotificationSettings>('/api/settings/notifications', {
    method: 'PUT',
    body: JSON.stringify(settings),
  });
}
