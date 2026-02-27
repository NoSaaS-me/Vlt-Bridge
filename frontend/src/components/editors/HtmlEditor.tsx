/**
 * HtmlEditor — three-way (Source / Split / Preview) HTML file editor
 * Uses CodeMirror for editing and a sandboxed blob-URL iframe for preview.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { html } from '@codemirror/lang-html';
import { vscodeDark } from '@uiw/codemirror-theme-vscode';
import { Save, Code2, Columns2, Eye, AlertCircle, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';

interface HtmlEditorProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

type ViewMode = 'source' | 'split' | 'preview';

// Resolve the API base safely (matches pattern in services/api.ts)
// Empty string = relative URL → goes through Vite proxy in dev.
function getApiBase(): string {
  return (window as unknown as { API_BASE_URL?: string }).API_BASE_URL || '';
}

function getAuthToken(): string {
  return localStorage.getItem('auth_token') || '';
}

// Build a blob URL from HTML content, revoke previous one
function makeBlobUrl(html: string): string {
  const blob = new Blob([html], { type: 'text/html' });
  return URL.createObjectURL(blob);
}

export function HtmlEditor({ assetPath, projectId, fileName }: HtmlEditorProps) {
  const [content, setContent] = useState('');
  const [savedContent, setSavedContent] = useState('');
  const [viewMode, setViewMode] = useState<ViewMode>('split');
  const [loadState, setLoadState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  const iframeRef = useRef<HTMLIFrameElement>(null);
  const prevBlobUrlRef = useRef<string | null>(null);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const displayName = fileName ?? assetPath.split('/').pop() ?? assetPath;
  const isDirty = content !== savedContent;

  // ── Load HTML from backend ───────────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    setLoadState('loading');
    setError(null);

    const base = getApiBase();
    const token = getAuthToken();
    const qs = projectId ? `?project_id=${encodeURIComponent(projectId)}` : '';

    fetch(`${base}/api/assets/${encodeURIComponent(assetPath)}${qs}`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.text();
      })
      .then((text) => {
        if (cancelled) return;
        setContent(text);
        setSavedContent(text);
        setLoadState('ready');
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Failed to load file: ${msg}`);
        setLoadState('error');
      });

    return () => {
      cancelled = true;
    };
  }, [assetPath, projectId]);

  // ── Blob URL management ──────────────────────────────────────────────────
  const refreshPreview = useCallback((html: string) => {
    const newUrl = makeBlobUrl(html);
    setBlobUrl(newUrl);
    // Revoke the previous URL after state has updated
    const oldUrl = prevBlobUrlRef.current;
    if (oldUrl) URL.revokeObjectURL(oldUrl);
    prevBlobUrlRef.current = newUrl;
  }, []);

  // Initial preview when content loads
  useEffect(() => {
    if (loadState === 'ready' && content) {
      refreshPreview(content);
    }
  }, [loadState]); // eslint-disable-line react-hooks/exhaustive-deps

  // Debounced update in split mode; immediate in preview mode
  const handleContentChange = useCallback(
    (value: string) => {
      setContent(value);

      if (viewMode === 'preview') {
        // Immediate update
        refreshPreview(value);
      } else if (viewMode === 'split') {
        // Debounce 300ms
        if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
        debounceTimerRef.current = setTimeout(() => {
          refreshPreview(value);
        }, 300);
      }
      // In source mode: no preview update needed until mode switch
    },
    [viewMode, refreshPreview]
  );

  // When switching to preview or split, immediately refresh blob
  useEffect(() => {
    if (viewMode !== 'source' && loadState === 'ready') {
      refreshPreview(content);
    }
  }, [viewMode]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (prevBlobUrlRef.current) URL.revokeObjectURL(prevBlobUrlRef.current);
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    };
  }, []);

  // ── Save ─────────────────────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    if (!isDirty || isSaving) return;
    setIsSaving(true);
    setError(null);

    const base = getApiBase();
    const token = getAuthToken();
    const url = projectId
      ? `${base}/api/assets/${encodeURIComponent(assetPath)}?project_id=${encodeURIComponent(projectId)}`
      : `${base}/api/assets/${encodeURIComponent(assetPath)}`;

    try {
      const res = await fetch(url, {
        method: 'PUT',
        headers: {
          'Content-Type': 'text/html; charset=utf-8',
          Authorization: `Bearer ${token}`,
        },
        body: content,
      });
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
      }
      setSavedContent(content);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Save failed: ${msg}`);
    } finally {
      setIsSaving(false);
    }
  }, [content, isDirty, isSaving, assetPath, projectId]);

  // ── Keyboard shortcut ────────────────────────────────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleSave]);

  // ── Render helpers ───────────────────────────────────────────────────────
  const editorPanel = (
    <CodeMirror
      value={content}
      height="100%"
      extensions={[html()]}
      theme={vscodeDark}
      onChange={handleContentChange}
      style={{ height: '100%', fontSize: '13px' }}
    />
  );

  const previewPanel = (
    <iframe
      ref={iframeRef}
      src={blobUrl ?? undefined}
      sandbox="allow-scripts"
      title="HTML preview"
      className="w-full h-full border-0 bg-white"
    />
  );

  // ── Loading / error states ───────────────────────────────────────────────
  if (loadState === 'loading') {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" />
        <span className="text-sm">Loading {displayName}…</span>
      </div>
    );
  }

  if (loadState === 'error') {
    return (
      <div className="h-full flex items-center justify-center p-6">
        <Alert variant="destructive" className="max-w-md">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  // ── Main render ──────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full bg-background">
      {/* ── Toolbar ─────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2 flex-shrink-0">
        {/* File info */}
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium truncate text-foreground">{displayName}</span>
          <Badge variant="secondary" className="flex-shrink-0 text-xs">HTML</Badge>
          {isDirty && (
            <span
              className="text-amber-500 font-bold text-base leading-none flex-shrink-0"
              title="Unsaved changes"
            >
              •
            </span>
          )}
        </div>

        {/* View toggle + Save */}
        <div className="flex items-center gap-1 flex-shrink-0">
          {/* View mode toggle — three buttons acting as a group */}
          <div className="flex items-center rounded-md border border-border overflow-hidden">
            <Button
              variant={viewMode === 'source' ? 'default' : 'ghost'}
              size="sm"
              className="rounded-none h-7 px-2.5 text-xs"
              onClick={() => setViewMode('source')}
              title="Source only"
            >
              <Code2 className="h-3.5 w-3.5 mr-1" />
              Source
            </Button>
            <Button
              variant={viewMode === 'split' ? 'default' : 'ghost'}
              size="sm"
              className="rounded-none h-7 px-2.5 text-xs border-x border-border"
              onClick={() => setViewMode('split')}
              title="Split view"
            >
              <Columns2 className="h-3.5 w-3.5 mr-1" />
              Split
            </Button>
            <Button
              variant={viewMode === 'preview' ? 'default' : 'ghost'}
              size="sm"
              className="rounded-none h-7 px-2.5 text-xs"
              onClick={() => setViewMode('preview')}
              title="Preview only"
            >
              <Eye className="h-3.5 w-3.5 mr-1" />
              Preview
            </Button>
          </div>

          <Button
            size="sm"
            onClick={handleSave}
            disabled={!isDirty || isSaving}
            className="ml-2 h-7 px-3 text-xs"
            title="Save (Ctrl/Cmd+S)"
          >
            {isSaving ? (
              <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
            ) : (
              <Save className="h-3.5 w-3.5 mr-1" />
            )}
            {isSaving ? 'Saving…' : 'Save'}
          </Button>
        </div>
      </div>

      {/* ── Error banner (save errors) ────────────────────────────── */}
      {error && loadState === 'ready' && (
        <div className="px-4 py-2 flex-shrink-0">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        </div>
      )}

      {/* ── Main editing area ─────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden">
        {viewMode === 'source' && (
          <div className="h-full overflow-auto">{editorPanel}</div>
        )}

        {viewMode === 'preview' && (
          <div className="h-full">{previewPanel}</div>
        )}

        {viewMode === 'split' && (
          <ResizablePanelGroup direction="horizontal" className="h-full">
            <ResizablePanel defaultSize={50} minSize={25}>
              <div className="h-full flex flex-col">
                <div className="px-3 py-1 border-b border-border flex-shrink-0">
                  <span className="text-xs text-muted-foreground font-medium">Source</span>
                </div>
                <div className="flex-1 overflow-auto">{editorPanel}</div>
              </div>
            </ResizablePanel>

            <ResizableHandle withHandle />

            <ResizablePanel defaultSize={50} minSize={25}>
              <div className="h-full flex flex-col">
                <div className="px-3 py-1 border-b border-border flex-shrink-0">
                  <span className="text-xs text-muted-foreground font-medium">Preview</span>
                </div>
                <div className="flex-1">{previewPanel}</div>
              </div>
            </ResizablePanel>
          </ResizablePanelGroup>
        )}
      </div>

      {/* ── Footer hint ──────────────────────────────────────────── */}
      <div className="border-t border-border px-4 py-1.5 text-xs text-muted-foreground flex-shrink-0">
        <kbd className="px-1.5 py-0.5 bg-muted rounded">Ctrl/Cmd+S</kbd> to save
        {viewMode === 'split' && (
          <span className="ml-3 text-muted-foreground/70">Preview auto-refreshes after 300ms</span>
        )}
      </div>
    </div>
  );
}
