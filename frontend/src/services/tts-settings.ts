/**
 * API client for TTS settings and voice listing.
 */

export interface TtsSettingsResponse {
  voice_id: string | null;
  model: string;
}

export interface Voice {
  voice_id: string;
  name: string;
  category: string;
  labels: Record<string, string>;
  preview_url: string;
}

interface VoiceListResponse {
  voices: Voice[];
}

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('auth_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function getTtsSettings(): Promise<TtsSettingsResponse> {
  const resp = await fetch('/api/settings/tts', {
    headers: authHeaders(),
  });
  if (!resp.ok) throw new Error(`Failed to load TTS settings (${resp.status})`);
  return resp.json();
}

export async function saveTtsSettings(settings: { voice_id: string; model: string }): Promise<TtsSettingsResponse> {
  const resp = await fetch('/api/settings/tts', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(settings),
  });
  if (!resp.ok) throw new Error(`Failed to save TTS settings (${resp.status})`);
  return resp.json();
}

export async function getVoices(): Promise<Voice[]> {
  const resp = await fetch('/api/voices', {
    headers: authHeaders(),
  });
  if (!resp.ok) throw new Error(`Failed to load voices (${resp.status})`);
  const data: VoiceListResponse = await resp.json();
  return data.voices;
}
