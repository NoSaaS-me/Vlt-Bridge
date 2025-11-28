import { SourceReference } from '@/types/rag';
import { FileText, ExternalLink } from 'lucide-react';
import { useState } from 'react';
import { cn } from '@/lib/utils';

interface SourceListProps {
  sources: SourceReference[];
  onSourceClick: (path: string) => void;
}

export function SourceList({ sources, onSourceClick }: SourceListProps) {
  const [expanded, setExpanded] = useState(false);

  if (!sources || sources.length === 0) return null;

  const displayedSources = expanded ? sources : sources.slice(0, 3);
  const hasMore = sources.length > 3;

  return (
    <div className="mt-2 text-sm">
      <p className="text-xs font-semibold text-muted-foreground mb-1 flex items-center gap-1">
        <FileText className="h-3 w-3" />
        Sources
      </p>
      <div className="flex flex-wrap gap-2">
        {displayedSources.map((source, i) => (
          <button
            key={i}
            onClick={() => onSourceClick(source.path)}
            className={cn(
              "flex items-center gap-1 px-2 py-1 rounded-md border border-border bg-muted/50 hover:bg-muted transition-colors text-xs max-w-[200px]",
              "text-left truncate"
            )}
            title={source.title}
          >
            <span className="truncate">{source.title}</span>
            <ExternalLink className="h-3 w-3 opacity-50 flex-shrink-0" />
          </button>
        ))}
        {hasMore && !expanded && (
          <button
            onClick={() => setExpanded(true)}
            className="px-2 py-1 text-xs text-muted-foreground hover:text-foreground"
          >
            +{sources.length - 3} more
          </button>
        )}
      </div>
    </div>
  );
}
