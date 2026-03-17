/**
 * ArtifactViewer — Renders the artifact's frontend inside a sandboxed iframe.
 *
 * Architecture:
 *   - ArtifactStateBar at top (state machine controls)
 *   - Sandboxed iframe below (served from daemon /api/artifacts/{id}/frontend/)
 *   - VltBridgeHost handles postMessage routing from iframe → daemon REST
 *   - WebSocket keeps artifact state in sync
 */
import { useEffect, useRef, useCallback } from 'react';
import { type ArtifactData, transitionState, exportArtifact, getArtifactStateWsUrl } from '@/services/artifact-api';
import { createBridgeHost } from '@/lib/vlt-bridge-host';
import { ArtifactStateBar } from './ArtifactStateBar';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ArtifactViewerProps {
  artifact: ArtifactData;
  onStateChange?: (artifact: ArtifactData) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ArtifactViewer({ artifact, onStateChange }: ArtifactViewerProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const bridgeCleanupRef = useRef<(() => void) | null>(null);

  // Attach bridge host when iframe mounts / artifact changes
  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    // Cleanup previous bridge
    bridgeCleanupRef.current?.();

    const cleanup = createBridgeHost(artifact.id, iframe);
    bridgeCleanupRef.current = cleanup;

    return () => {
      cleanup();
      bridgeCleanupRef.current = null;
    };
  }, [artifact.id]);

  // WebSocket for state change notifications
  useEffect(() => {
    if (!onStateChange) return;

    const wsUrl = getArtifactStateWsUrl(artifact.id);
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string);
        if (msg.type === 'state_change' && msg.artifact) {
          onStateChange(msg.artifact as ArtifactData);
        }
      } catch {
        // ignore malformed frames
      }
    };

    ws.onerror = () => {
      // WebSocket errors are non-fatal; polling handles recovery
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [artifact.id, onStateChange]);

  const handleTransition = useCallback(
    async (targetState: string) => {
      try {
        const updated = await transitionState(artifact.id, targetState, 'user');
        onStateChange?.(updated);
      } catch (err) {
        console.error('State transition failed:', err);
      }
    },
    [artifact.id, onStateChange],
  );

  const handleExport = useCallback(async () => {
    try {
      const blob = await exportArtifact(artifact.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${artifact.name.replace(/\s+/g, '_')}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
    }
  }, [artifact.id, artifact.name]);

  // The iframe src points to the daemon via the Vite proxy
  const iframeSrc = `/vlt/api/artifacts/${encodeURIComponent(artifact.id)}/frontend/index.html`;

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <ArtifactStateBar
        artifact={artifact}
        onTransition={handleTransition}
        onExport={handleExport}
      />
      <iframe
        ref={iframeRef}
        src={iframeSrc}
        className="flex-1 w-full border-0 bg-background"
        // sandbox allows scripts (the artifact may need them) but restricts top navigation
        sandbox="allow-scripts allow-same-origin allow-forms allow-modals allow-popups"
        title={`Artifact: ${artifact.name}`}
      />
    </div>
  );
}

export default ArtifactViewer;
