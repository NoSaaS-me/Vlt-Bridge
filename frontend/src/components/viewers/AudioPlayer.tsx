import { useRef, useState, useCallback, useEffect } from 'react';
import { Music, AlertCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { getAssetUrl } from '@/services/api';

interface AudioPlayerProps {
  assetPath: string;
  projectId?: string;
  fileName?: string;
}

export function AudioPlayer({ assetPath, projectId, fileName }: AudioPlayerProps) {
  const [hasError, setHasError] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);

  const audioUrl = getAssetUrl(assetPath, projectId);
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
        <Music className="h-4 w-4 text-muted-foreground flex-shrink-0" />
        <span className="text-sm font-medium truncate text-foreground">{displayName}</span>
        <Badge variant="secondary" className="flex-shrink-0 text-xs">{ext}</Badge>
      </div>

      {/* Player area */}
      <div className="flex-1 flex items-center justify-center p-8">
        {hasError ? (
          <div className="flex flex-col items-center gap-3 text-muted-foreground">
            <AlertCircle className="h-12 w-12" />
            <span className="text-sm font-medium">Failed to load audio</span>
            <span className="text-xs">{displayName}</span>
          </div>
        ) : (
          <div className="w-full max-w-lg flex flex-col items-center gap-6">
            {/* Decorative icon */}
            <div className="rounded-full bg-accent p-8">
              <Music className="h-12 w-12 text-accent-foreground" />
            </div>

            <div className="text-center">
              <p className="text-sm font-medium text-foreground">{displayName}</p>
              <p className="text-xs text-muted-foreground mt-1">{ext} audio file</p>
            </div>

            {/* Native audio controls */}
            <audio
              ref={audioRef}
              controls
              src={audioUrl}
              onError={handleError}
              className="w-full"
              preload="metadata"
            >
              Your browser does not support the audio element.
            </audio>
          </div>
        )}
      </div>
    </div>
  );
}
