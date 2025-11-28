/**
 * ElevenLabs TTS client hitting the backend proxy.
 */

export interface TtsOptions {
  voiceId?: string;
  model?: string;
  signal?: AbortSignal;
}

export async function synthesizeTts(text: string, options: TtsOptions = {}): Promise<Blob> {
  const token = localStorage.getItem('auth_token');
  const response = await fetch('/api/tts', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    signal: options.signal,
    body: JSON.stringify({
      text,
      voice_id: options.voiceId,
      model: options.model,
    }),
  });

  if (!response.ok) {
    let message = `TTS failed (HTTP ${response.status})`;
    try {
      const data = await response.json();
      message = data?.message || data?.detail?.message || message;
    } catch {
      // ignore parse errors
    }
    throw new Error(message);
  }

  return response.blob();
}
