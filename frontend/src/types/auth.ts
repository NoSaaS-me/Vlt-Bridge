/**
 * Token response returned after authentication.
 */
export interface TokenResponse {
  token: string;
  token_type: "bearer";
  expires_at: string; // ISO 8601 timestamp
}

/**
 * Standard API error envelope.
 */
export interface APIError {
  error: string;
  message: string;
  detail?: Record<string, unknown>;
}

