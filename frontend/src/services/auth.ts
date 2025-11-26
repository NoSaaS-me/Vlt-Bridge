/**
 * T105-T107: Authentication service for HF OAuth and token management
 */
import type { User } from '@/types/user';
import type { TokenResponse } from '@/types/auth';
import type { DemoTokenResponse } from '@/services/api';
import { getDemoToken } from '@/services/api';

const AUTH_TOKEN_KEY = 'auth_token';
const AUTH_TOKEN_SOURCE_KEY = 'auth_token_source';
const AUTH_TOKEN_EXPIRES_KEY = 'auth_token_expires_at';
export const AUTH_TOKEN_CHANGED_EVENT = 'auth-token-changed';

type TokenSource = 'user' | 'demo';

const API_BASE = '';

function notifyTokenChange(): void {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent(AUTH_TOKEN_CHANGED_EVENT));
  }
}

function storeAuthToken(token: string, source: TokenSource, expiresAt?: string): void {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  localStorage.setItem(AUTH_TOKEN_SOURCE_KEY, source);
  if (expiresAt) {
    localStorage.setItem(AUTH_TOKEN_EXPIRES_KEY, expiresAt);
  } else {
    localStorage.removeItem(AUTH_TOKEN_EXPIRES_KEY);
  }
  notifyTokenChange();
}

function clearStoredAuthToken(): void {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_TOKEN_SOURCE_KEY);
  localStorage.removeItem(AUTH_TOKEN_EXPIRES_KEY);
  notifyTokenChange();
}

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
  clearStoredAuthToken();
  window.location.href = '/';
}

/**
 * T106: Get current authenticated user
 */
export async function getCurrentUser(): Promise<User> {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  
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
  storeAuthToken(tokenResponse.token, 'user', tokenResponse.expires_at);
  
  return tokenResponse;
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (!token) {
    return false;
  }
  if (isDemoSession()) {
    return !demoTokenExpired();
  }
  return true;
}

/**
 * Get stored token
 */
export function getStoredToken(): string | null {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}

export function isDemoSession(): boolean {
  return localStorage.getItem(AUTH_TOKEN_SOURCE_KEY) === 'demo';
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
      storeAuthToken(token, 'user');
      // Clean up the URL
      window.history.replaceState(null, '', window.location.pathname);
      return true;
    }
  }
  return false;
}

function demoTokenExpired(): boolean {
  const expiresAt = localStorage.getItem(AUTH_TOKEN_EXPIRES_KEY);
  if (!expiresAt) {
    return false;
  }
  const now = Date.now();
  return new Date(expiresAt).getTime() <= now;
}

async function requestDemoToken(): Promise<DemoTokenResponse> {
  return getDemoToken();
}

export async function ensureDemoToken(): Promise<boolean> {
  // If we already have a user token, nothing to do
  if (isAuthenticated() && !isDemoSession()) {
    return true;
  }

  // If we have a demo token that hasn't expired, reuse it
  if (isAuthenticated() && isDemoSession() && !demoTokenExpired()) {
    return true;
  }

  try {
    const demoResponse = await requestDemoToken();
    storeAuthToken(demoResponse.token, 'demo', demoResponse.expires_at);
    return true;
  } catch (error) {
    console.warn('Failed to obtain demo token', error);
    clearStoredAuthToken();
    return false;
  }
}

