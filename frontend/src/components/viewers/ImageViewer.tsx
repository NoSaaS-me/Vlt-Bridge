import { useState, useRef, useCallback, useEffect } from 'react';
import { Download, ZoomIn, ZoomOut, RotateCcw, ImageOff, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { getAssetUrl } from '@/services/api';

interface ImageViewerProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

interface Dimensions {
  width: number;
  height: number;
}

interface PanState {
  x: number;
  y: number;
}

interface DragState {
  isDragging: boolean;
  startX: number;
  startY: number;
  panAtStart: PanState;
}

const MIN_SCALE = 0.1;
const MAX_SCALE = 5;
const ZOOM_STEP = 0.15;

export function ImageViewer({ assetPath, projectId, fileName }: ImageViewerProps) {
  const [scale, setScale] = useState(1);
  const [pan, setPan] = useState<PanState>({ x: 0, y: 0 });
  const [dimensions, setDimensions] = useState<Dimensions | null>(null);
  const [loadState, setLoadState] = useState<'loading' | 'loaded' | 'error'>('loading');

  const dragRef = useRef<DragState>({
    isDragging: false,
    startX: 0,
    startY: 0,
    panAtStart: { x: 0, y: 0 },
  });
  const containerRef = useRef<HTMLDivElement>(null);

  const imageUrl = getAssetUrl(assetPath, projectId);
  const displayName = fileName ?? assetPath.split('/').pop() ?? assetPath;
  const ext = assetPath.slice(assetPath.lastIndexOf('.') + 1).toUpperCase();

  // Reset view when asset changes
  useEffect(() => {
    setScale(1);
    setPan({ x: 0, y: 0 });
    setLoadState('loading');
    setDimensions(null);
  }, [assetPath]);

  const resetView = useCallback(() => {
    setScale(1);
    setPan({ x: 0, y: 0 });
  }, []);

  const clampScale = (s: number) => Math.min(MAX_SCALE, Math.max(MIN_SCALE, s));

  // Wheel zoom centred on cursor
  const handleWheel = useCallback((e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP;
    setScale((prev) => clampScale(prev + delta));
  }, []);

  // Pan via mouse drag
  const handleMouseDown = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    e.preventDefault();
    dragRef.current = {
      isDragging: true,
      startX: e.clientX,
      startY: e.clientY,
      panAtStart: pan,
    };
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!dragRef.current.isDragging) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setPan({
      x: dragRef.current.panAtStart.x + dx,
      y: dragRef.current.panAtStart.y + dy,
    });
  }, []);

  const handleMouseUp = useCallback(() => {
    dragRef.current.isDragging = false;
  }, []);

  const handleImageLoad = useCallback((e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    setLoadState('loaded');
  }, []);

  const handleImageError = useCallback(() => {
    setLoadState('error');
  }, []);

  const isDragging = scale > 1;

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2 flex-shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium truncate text-foreground">{displayName}</span>
          <Badge variant="secondary" className="flex-shrink-0 text-xs">{ext}</Badge>
          {dimensions && (
            <span className="text-xs text-muted-foreground flex-shrink-0">
              {dimensions.width} × {dimensions.height}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1 flex-shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setScale((s) => clampScale(s - ZOOM_STEP))}
            disabled={scale <= MIN_SCALE}
            title="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs text-muted-foreground w-12 text-center select-none">
            {Math.round(scale * 100)}%
          </span>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setScale((s) => clampScale(s + ZOOM_STEP))}
            disabled={scale >= MAX_SCALE}
            title="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={resetView}
            title="Reset view"
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="sm" asChild>
            <a href={imageUrl} download={displayName} title="Download">
              <Download className="h-4 w-4 mr-1" />
              Download
            </a>
          </Button>
        </div>
      </div>

      {/* Image canvas */}
      <div
        ref={containerRef}
        className="flex-1 overflow-hidden relative"
        style={{ cursor: isDragging ? (dragRef.current.isDragging ? 'grabbing' : 'grab') : 'default' }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        {/* Loading state */}
        {loadState === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="text-sm">Loading image…</span>
            </div>
          </div>
        )}

        {/* Error state */}
        {loadState === 'error' && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3 text-muted-foreground">
              <ImageOff className="h-12 w-12" />
              <span className="text-sm font-medium">Failed to load image</span>
              <span className="text-xs">{displayName}</span>
            </div>
          </div>
        )}

        {/* Image with transform */}
        <div
          className="absolute inset-0 flex items-center justify-center"
          style={{ visibility: loadState === 'loaded' ? 'visible' : 'hidden' }}
        >
          <img
            src={imageUrl}
            alt={displayName}
            draggable={false}
            onLoad={handleImageLoad}
            onError={handleImageError}
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})`,
              transformOrigin: 'center center',
              transition: dragRef.current.isDragging ? 'none' : 'transform 0.1s ease-out',
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              userSelect: 'none',
            }}
          />
        </div>
      </div>
    </div>
  );
}
