/**
 * PdfViewer: react-pdf v10 based PDF viewer with pagination, zoom, and OCR status badge.
 * Uses getAssetUrl for file access and polls getAssetMetadata for OCR status.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw, Loader2, FileX } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { getAssetUrl, getAssetMetadata } from '@/services/api';
import type { OcrStatus } from '@/types/asset';

// react-pdf v10 worker setup — must use the pdfjs-dist version bundled inside
// react-pdf (5.4.296), not any top-level pdfjs-dist install (may differ).
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'react-pdf/node_modules/pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url
).toString();

interface PdfViewerProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

const MIN_SCALE = 0.5;
const MAX_SCALE = 3.0;
const SCALE_STEP = 0.25;
const DEFAULT_SCALE = 1.0;
const OCR_POLL_INTERVAL_MS = 3000;

/** Returns true if OCR is in a terminal state (no more polling needed). */
function isOcrTerminal(status: OcrStatus | null): boolean {
  return status === 'done' || status === 'failed' || status === 'skipped' || status === null;
}

export function PdfViewer({ assetPath, projectId, fileName }: PdfViewerProps) {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [scale, setScale] = useState<number>(DEFAULT_SCALE);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [isDocLoading, setIsDocLoading] = useState<boolean>(true);

  const [ocrStatus, setOcrStatus] = useState<OcrStatus | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Fetch OCR metadata and optionally poll
  const fetchOcrStatus = useCallback(async () => {
    try {
      const meta = await getAssetMetadata(assetPath, projectId);
      setOcrStatus(meta.ocr_status);
      if (isOcrTerminal(meta.ocr_status)) {
        if (pollTimerRef.current !== null) {
          clearInterval(pollTimerRef.current);
          pollTimerRef.current = null;
        }
      }
    } catch {
      // Silently ignore metadata fetch errors — OCR badge is non-critical
    }
  }, [assetPath, projectId]);

  // Start polling on mount; stop when terminal
  useEffect(() => {
    fetchOcrStatus();

    pollTimerRef.current = setInterval(() => {
      fetchOcrStatus();
    }, OCR_POLL_INTERVAL_MS);

    return () => {
      if (pollTimerRef.current !== null) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [fetchOcrStatus]);

  // Stop polling once terminal state reached
  useEffect(() => {
    if (isOcrTerminal(ocrStatus) && pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, [ocrStatus]);

  // Reset page/zoom when file changes
  useEffect(() => {
    setCurrentPage(1);
    setScale(DEFAULT_SCALE);
    setLoadError(null);
    setIsDocLoading(true);
    setNumPages(0);
  }, [assetPath, projectId]);

  const onDocumentLoadSuccess = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setIsDocLoading(false);
    setLoadError(null);
  };

  const onDocumentLoadError = (error: Error) => {
    setLoadError(error.message || 'Failed to load PDF.');
    setIsDocLoading(false);
  };

  const goToPrevPage = () => setCurrentPage((p) => Math.max(1, p - 1));
  const goToNextPage = () => setCurrentPage((p) => Math.min(numPages, p + 1));

  const zoomIn = () => setScale((s) => Math.min(MAX_SCALE, parseFloat((s + SCALE_STEP).toFixed(2))));
  const zoomOut = () => setScale((s) => Math.max(MIN_SCALE, parseFloat((s - SCALE_STEP).toFixed(2))));
  const resetZoom = () => setScale(DEFAULT_SCALE);

  const fileUrl = getAssetUrl(assetPath, projectId);
  const displayName = fileName ?? assetPath.split('/').pop() ?? 'document.pdf';

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Controls bar */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-2 bg-card shrink-0 flex-wrap">
        {/* File name */}
        <span className="text-sm font-medium text-foreground mr-2 truncate max-w-[200px]" title={displayName}>
          {displayName}
        </span>

        <div className="flex items-center gap-1 border-r border-border pr-3">
          {/* Pagination */}
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={goToPrevPage}
            disabled={currentPage <= 1 || numPages === 0}
            aria-label="Previous page"
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          <span className="text-sm text-muted-foreground min-w-[80px] text-center select-none">
            {numPages > 0 ? `Page ${currentPage} / ${numPages}` : '—'}
          </span>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={goToNextPage}
            disabled={currentPage >= numPages || numPages === 0}
            aria-label="Next page"
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        {/* Zoom controls */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={zoomOut}
            disabled={scale <= MIN_SCALE}
            aria-label="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>

          <button
            className="text-sm text-muted-foreground min-w-[52px] text-center tabular-nums hover:text-foreground transition-colors"
            onClick={resetZoom}
            title="Reset zoom"
            aria-label="Reset zoom"
          >
            {Math.round(scale * 100)}%
          </button>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={zoomIn}
            disabled={scale >= MAX_SCALE}
            aria-label="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={resetZoom}
            disabled={scale === DEFAULT_SCALE}
            aria-label="Reset zoom"
            title="Reset zoom"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </Button>
        </div>

        {/* OCR status badge */}
        <div className="ml-auto">
          {ocrStatus === 'pending' || ocrStatus === 'running' ? (
            <Badge variant="secondary" className="flex items-center gap-1 text-xs">
              <Loader2 className="h-3 w-3 animate-spin" />
              OCR processing...
            </Badge>
          ) : ocrStatus === 'done' ? (
            <Badge className="bg-green-600 hover:bg-green-600 text-white text-xs">
              Searchable
            </Badge>
          ) : ocrStatus === 'failed' ? (
            <Badge variant="outline" className="border-yellow-500 text-yellow-600 text-xs">
              OCR failed
            </Badge>
          ) : null}
        </div>
      </div>

      {/* PDF canvas area */}
      <div className="flex-1 overflow-auto flex items-start justify-center bg-muted/30 p-4">
        {loadError ? (
          <div className="flex flex-col items-center justify-center gap-3 text-muted-foreground mt-16">
            <FileX className="h-12 w-12 opacity-40" />
            <p className="text-sm font-medium">Failed to load PDF</p>
            <p className="text-xs max-w-[300px] text-center opacity-70">{loadError}</p>
          </div>
        ) : (
          <div className="relative">
            {isDocLoading && (
              <div className="flex items-center justify-center w-[600px] h-[800px]">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            )}
            <Document
              file={fileUrl}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={null}
              className={isDocLoading ? 'invisible' : 'visible'}
            >
              <Page
                pageNumber={currentPage}
                scale={scale}
                renderTextLayer={true}
                renderAnnotationLayer={true}
                className="shadow-md"
              />
            </Document>
          </div>
        )}
      </div>
    </div>
  );
}
