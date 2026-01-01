import type { OracleRequest, OracleStreamChunk, OracleMessage } from '@/types/oracle';
import { getAuthToken } from './api';

/**
 * Stream Oracle response using Server-Sent Events (SSE)
 *
 * @param request Oracle query request
 * @param onChunk Callback for each stream chunk
 * @param signal Optional AbortSignal for cancellation
 */
export async function streamOracle(
  request: OracleRequest,
  onChunk: (chunk: OracleStreamChunk) => void,
  signal?: AbortSignal
): Promise<void> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Handle absolute URL injection for widget mode
  let url = '/api/oracle/stream';
  if (window.API_BASE_URL) {
    url = `${window.API_BASE_URL}${url}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({
      error: 'Unknown error',
      message: `HTTP ${response.status}: ${response.statusText}`,
    }));
    throw new Error(errorData.message || `HTTP ${response.status}`);
  }

  if (!response.body) {
    throw new Error('Response body is null');
  }

  // Read SSE stream
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let chunkCounter = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        console.debug('[SSE] Stream complete');
        break;
      }

      // Append to buffer and process complete lines
      const decoded = decoder.decode(value, { stream: true });
      buffer += decoded;
      const lines = buffer.split('\n');

      // Keep incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();

        // Skip empty lines and SSE comments
        if (!trimmed || trimmed.startsWith(':')) {
          continue;
        }

        // Parse SSE data field
        if (trimmed.startsWith('data: ')) {
          const data = trimmed.substring(6);

          try {
            const chunk = JSON.parse(data) as OracleStreamChunk;
            chunkCounter++;
            // Debug logging to trace chunk duplication issue
            console.debug(
              `[SSE #${chunkCounter}] type=${chunk.type} content_preview=${
                chunk.content?.substring(0, 50) || 'N/A'
              }`
            );
            onChunk(chunk);
          } catch (err) {
            console.error('Failed to parse SSE chunk:', data, err);
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/**
 * Cancel the active Oracle session for the current user.
 * This stops all running agents (Oracle + any subagents).
 *
 * @returns Promise resolving to the cancellation status
 */
export async function cancelOracle(): Promise<{ status: 'cancelled' | 'no_active_session' }> {
  const token = getAuthToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  // Handle absolute URL injection for widget mode
  let url = '/api/oracle/cancel';
  if (window.API_BASE_URL) {
    url = `${window.API_BASE_URL}${url}`;
  }

  const response = await fetch(url, {
    method: 'POST',
    headers,
  });

  if (!response.ok) {
    throw new Error(`Failed to cancel Oracle: HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Clear conversation history (not implemented server-side yet)
 * For now, this is a client-side operation
 */
export async function clearHistory(): Promise<void> {
  // TODO: Implement server-side endpoint if needed
  // For now, clearing is handled client-side in ChatPanel
  return Promise.resolve();
}

/**
 * Compact conversation history by summarizing
 * This would call a server endpoint to compress the conversation
 */
export async function compactHistory(messages: OracleMessage[]): Promise<OracleMessage[]> {
  // TODO: Implement server-side /api/oracle/compact endpoint
  // For MVP, return messages as-is
  console.warn('compactHistory not yet implemented server-side');
  return messages;
}

/**
 * Export conversation as markdown
 */
export function exportConversationAsMarkdown(messages: OracleMessage[]): string {
  const lines: string[] = [
    '# Oracle Conversation Export',
    `**Exported**: ${new Date().toISOString()}`,
    '',
    '---',
    '',
  ];

  for (const message of messages) {
    const role = message.role === 'user' ? 'User' : 'Assistant';
    lines.push(`## ${role} (${new Date(message.timestamp).toLocaleString()})`);
    lines.push('');
    lines.push(message.content);
    lines.push('');

    if (message.thinking) {
      lines.push('### Thinking');
      lines.push('');
      lines.push(message.thinking);
      lines.push('');
    }

    if (message.sources && message.sources.length > 0) {
      lines.push('### Sources');
      lines.push('');
      for (const source of message.sources) {
        lines.push(`- **${source.source_type}**: ${source.source_path}${source.score != null ? ` (score: ${source.score.toFixed(3)})` : ''}`);
        if (source.content) {
          lines.push(`  > ${source.content.substring(0, 150)}...`);
        }
      }
      lines.push('');
    }

    if (message.model) {
      lines.push(`*Model: ${message.model}*`);
      lines.push('');
    }

    lines.push('---');
    lines.push('');
  }

  return lines.join('\n');
}

/**
 * Download a string as a file
 */
export function downloadAsFile(content: string, filename: string, mimeType = 'text/markdown'): void {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
