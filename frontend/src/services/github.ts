/**
 * GitHub integration API service
 */

import { apiFetch } from './api';
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
 * Note: This redirects the browser, not an API call.
 *
 * Security: The JWT must NOT be placed in the URL query string because:
 * - URLs appear in browser history, server access logs, and Referrer headers
 * - This constitutes a credential leak (CWE-598)
 *
 * Instead, the token is stored in sessionStorage under a one-time-use key
 * that the backend can retrieve via a pre-auth lookup endpoint, or the
 * backend uses the existing session cookie for the OAuth flow.
 *
 * If the backend requires the token for the GitHub connect redirect, it
 * should be fetched server-side from the active session, not passed in the URL.
 *
 * For backwards compatibility, the URL is constructed without the token.
 * The backend /api/auth/github endpoint must use the session or a separate
 * pre-auth token exchange to authenticate the OAuth initiation.
 */
export function getGitHubConnectUrl(): string {
  // Only allow the current origin or a validated API_BASE_URL to prevent
  // open redirect via a tampered window.API_BASE_URL.
  const baseUrl = window.location.origin;

  // Do NOT append the JWT as a query parameter — it leaks via logs/history.
  // The backend must authenticate this request via session or cookie.
  return `${baseUrl}/api/auth/github`;
}
