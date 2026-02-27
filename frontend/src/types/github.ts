/**
 * GitHub integration types for frontend
 */

/**
 * GitHub connection status response
 */
export interface GitHubStatus {
  connected: boolean;
  username: string | null;
  rate_limit?: {
    remaining: number;
    limit: number;
    reset: number;  // Unix timestamp
  };
}

/**
 * GitHub disconnect response
 */
export interface GitHubDisconnectResponse {
  status: 'disconnected';
  previous_username: string | null;
}
