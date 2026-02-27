import { useRef, useState, useCallback, useEffect } from 'react';
import { Video, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { getAssetUrl } from '@/services/api';

interface VideoPlayerProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

export function VideoPlayer({ assetPath, projectId, fileName }: VideoPlayerProps) {
  const [hasError, setHasError] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  const videoUrl = getAssetUrl(assetPath, projectId);
  const displayName = fileName ?? assetPath.split('/').pop() ?? assetPath;
  const ext = assetPath.slice(assetPath.lastIndexOf('.') + 1).toUpperCase();

  // Reset error when asset changes
  useEffect(() => {
    setHasError(false);
  }, [assetPath]);

  const handleError = useCallback(() => {
    setHasError(true);
  }, []);

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-border px-4 py-2 flex-shrink-0">
        <Video className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        <span className="text-sm font-medium truncate text-foreground">{displayName}</span>
        <Badge variant="secondary" className="flex-shrink-0 text-xs">{ext}</Badge>
      </div>

      {/* Video area */}
      <div className="flex-1 flex flex-col items-center justify-center p-4 overflow-hidden">
        {hasError ? (
          <div className="flex flex-col items-center gap-3 text-muted-foreground">
            <AlertCircle className="h-12 w-12" />
            <span className="text-sm font-medium">Failed to load video</span>
            <span className="text-xs">{displayName}</span>
          </div>
        ) : (
          <div className="w-full flex flex-col items-center gap-3">
            <video
              ref={videoRef}
              controls
              src={videoUrl}
              preload="metadata"
              onError={handleError}
              className="w-full max-h-[70vh] rounded-md bg-black"
            >
              Your browser does not support the video element.
            </video>
            <p className="text-xs text-muted-foreground self-start">{displayName}</p>
          </div>
        )}
      </div>
    </div>
  );
}
