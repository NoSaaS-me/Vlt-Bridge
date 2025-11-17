/**
 * T105-T107: Authentication service for HF OAuth and token management
 */
import type { User } from '@/types/user';
import type { TokenResponse } from '@/types/auth';

const API_BASE = '';

/**
 * T105: Redirect to HF OAuth login
 */
export function login(): void {
  window.location.href = '/auth/login';
}

/**
 * Logout - clear token and redirect
 */
export function logout(): void {
  localStorage.removeItem('auth_token');
  window.location.href = '/';
}

/**
 * T106: Get current authenticated user
 */
export async function getCurrentUser(): Promise<User> {
  const token = localStorage.getItem('auth_token');
  
  const response = await fetch(`${API_BASE}/api/me`, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    throw new Error('Failed to get current user');
  }

  return response.json();
}

/**
 * T107: Generate new API token for MCP access
 */
export async function getToken(): Promise<TokenResponse> {
  const token = localStorage.getItem('auth_token');
  
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
  localStorage.setItem('auth_token', tokenResponse.token);
  
  return tokenResponse;
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!localStorage.getItem('auth_token');
}

/**
 * Get stored token
 */
export function getStoredToken(): string | null {
  return localStorage.getItem('auth_token');
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
      localStorage.setItem('auth_token', token);
      // Clean up the URL
      window.history.replaceState(null, '', window.location.pathname);
      return true;
    }
  }
  return false;
}

