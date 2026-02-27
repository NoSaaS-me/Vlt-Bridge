/**
 * FileViewer — type-routing wrapper that selects the appropriate viewer/editor
 * based on the file extension of assetPath.
 *
 * Categories handled:
 *  pdf         → PdfViewer
 *  image       → ImageViewer
 *  audio       → AudioPlayer
 *  video       → VideoPlayer
 *  html        → HtmlEditor
 *  spreadsheet → SpreadsheetEditor
 *  text|unknown → TextViewer (inline, raw fetch into <pre>)
 *  markdown    → null (should not reach here; handled by NoteViewer in MainApp)
 */
import { useState, useEffect } from 'react';
import { Loader2, FileText, AlertCircle } from 'lucide-react';
import { getFileCategory } from '@/lib/fileTypes';
import { getAssetUrl } from '@/services/api';
import { PdfViewer } from '@/components/viewers/PdfViewer';
import { ImageViewer } from '@/components/viewers/ImageViewer';
import { AudioPlayer } from '@/components/viewers/AudioPlayer';
import { VideoPlayer } from '@/components/viewers/VideoPlayer';
import { HtmlEditor } from '@/components/editors/HtmlEditor';
import { SpreadsheetEditor } from '@/components/editors/SpreadsheetEditor';

interface FileViewerProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

// ── Inline plain-text viewer ────────────────────────────────────────────────

function TextViewer({ assetPath, projectId, fileName }: FileViewerProps) {
  const [content, setContent] = useState<string>('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const displayName = fileName ?? assetPath.split('/').pop() ?? assetPath;

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    setContent('');

    // Use the same URL construction as other viewers (token in query string)
    const url = getAssetUrl(assetPath, projectId);

    fetch(url)
      .then(async (res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        return res.text();
      })
      .then((text) => {
        if (cancelled) return;
        setContent(text);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg = err instanceof Error ? err.message : String(err);
        setError(`Failed to load file: ${msg}`);
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [assetPath, projectId]);

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-2 flex-shrink-0">
        <FileText className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        <span className="text-sm font-medium truncate text-foreground">{displayName}</span>
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-auto p-4">
        {loading ? (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            <Loader2 className="h-6 w-6 animate-spin mr-2" />
            <span className="text-sm">Loading {displayName}…</span>
          </div>
        ) : error ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground gap-3">
            <AlertCircle className="h-10 w-10" />
            <span className="text-sm font-medium">{error}</span>
          </div>
        ) : (
          <pre className="text-sm font-mono whitespace-pre-wrap break-words text-foreground leading-relaxed">
            {content}
          </pre>
        )}
      </div>
    </div>
  );
}

// ── Main router component ───────────────────────────────────────────────────

export function FileViewer({ assetPath, projectId, fileName }: FileViewerProps) {
  const category = getFileCategory(assetPath);

  switch (category) {
    case 'pdf':
      return <PdfViewer assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'image':
      return <ImageViewer assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'audio':
      return <AudioPlayer assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'video':
      return <VideoPlayer assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'html':
      return <HtmlEditor assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'spreadsheet':
      return <SpreadsheetEditor assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'text':
    case 'unknown':
      return <TextViewer assetPath={assetPath} projectId={projectId} fileName={fileName} />;

    case 'markdown':
      // Markdown files are handled by NoteViewer in MainApp — should not reach here.
      return null;

    default:
      return <TextViewer assetPath={assetPath} projectId={projectId} fileName={fileName} />;
  }
}
