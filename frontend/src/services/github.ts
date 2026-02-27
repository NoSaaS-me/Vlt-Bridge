/**
 * GitHub integration API service
 */

import { apiFetch, getAuthToken } from './api';
import type { GitHubStatus, GitHubDisconnectResponse } from '@/types/github';

/**
 * Get GitHub connection status for current user
 */
export async function getGitHubStatus(): Promise<GitHubStatus> {
  return apiFetch<GitHubStatus>('/api/auth/github/status');
}

/**
 * Disconnect GitHub account from user
 */
export async function disconnectGitHub(): Promise<GitHubDisconnectResponse> {
  return apiFetch<GitHubDisconnectResponse>('/api/auth/github', {
    method: 'DELETE',
  });
}

/**
 * Get GitHub OAuth connect URL
 * Note: This redirects the browser, not an API call
 *
 * The JWT token is passed as a query parameter because browser navigation
 * (window.location.href) cannot include custom headers like Authorization.
 * The backend accepts the token from either the Authorization header or
 * the 'token' query parameter.
 */
export function getGitHubConnectUrl(): string {
  // Use current origin for the API endpoint
  const baseUrl = window.API_BASE_URL || window.location.origin;
  const token = getAuthToken();

  // Include token as query parameter since browser navigation can't set headers
  if (token) {
    return `${baseUrl}/api/auth/github?token=${encodeURIComponent(token)}`;
  }

  // If no token, return URL without token (will result in 401 error)
  return `${baseUrl}/api/auth/github`;
}
