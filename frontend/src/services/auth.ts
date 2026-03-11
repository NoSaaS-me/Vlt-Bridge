/**
 * Authentication service for GitHub OAuth and token management
 */
import type { User } from '@/types/user';
import type { TokenResponse } from '@/types/auth';

const AUTH_TOKEN_KEY = 'auth_token';
const AUTH_TOKEN_EXPIRES_KEY = 'auth_token_expires_at';
export const AUTH_TOKEN_CHANGED_EVENT = 'auth-token-changed';

const API_BASE = '';

function notifyTokenChange(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(AUTH_TOKEN_CHANGED_EVENT));
  }
}

function storeAuthToken(token: string, expiresAt?: string): void {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  if (expiresAt) {
    localStorage.setItem(AUTH_TOKEN_EXPIRES_KEY, expiresAt);
  } else {
    localStorage.removeItem(AUTH_TOKEN_EXPIRES_KEY);
  }
  notifyTokenChange();
}

function clearStoredAuthToken(): void {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_TOKEN_EXPIRES_KEY);
  // Clean up legacy keys
  localStorage.removeItem('auth_token_source');
  notifyTokenChange();
}

/**
 * Redirect to GitHub OAuth login
 */
export function login(): void {
  window.location.href = '/auth/login';
}

/**
 * Logout - clear token and redirect
 */
export function logout(): void {
  clearStoredAuthToken();
  window.location.href = '/';
}

/**
 * Get current authenticated user
 */
export async function getCurrentUser(): Promise<User> {
  const { apiFetch } = await import('@/services/api');
  return apiFetch<User>('/api/me');
}

/**
 * Generate new API token for MCP access
 */
export async function getToken(): Promise<TokenResponse> {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);

  const response = await fetch(`${API_BASE}/api/tokens`, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error('Failed to generate token');
  }

  const tokenResponse: TokenResponse = await response.json();

  // Store the new token
  storeAuthToken(tokenResponse.token, tokenResponse.expires_at);

  return tokenResponse;
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!localStorage.getItem(AUTH_TOKEN_KEY);
}

/**
 * Get stored token
 */
export function getStoredToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

/**
 * Extract JWT token from URL hash after OAuth callback.
 * URL format: /#token=<jwt>
 * Returns true if token was found and saved.
 */
export function setAuthTokenFromHash(): boolean {
  const hash = window.location.hash;
  if (hash.startsWith('#token=')) {
    const token = hash.substring(7); // Remove '#token='
    if (token) {
      storeAuthToken(token);
      // Clean up the URL
      window.history.replaceState(null, '', window.location.pathname);
      return true;
    }
  }
  return false;
}
