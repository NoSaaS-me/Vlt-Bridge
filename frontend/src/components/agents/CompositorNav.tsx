/**
 * CompositorNav — Prev/Next buttons + dot indicators for terminal compositor.
 */
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { cn } from '@/lib/utils';

export function CompositorNav({
  paneCount,
  focusedIndex,
  visibleIndices,
  onPrev,
  onNext,
  onDotClick,
}: {
  paneCount: number;
  focusedIndex: number;
  visibleIndices: Set<number>;
  onPrev: () => void;
  onNext: () => void;
  onDotClick: (index: number) => void;
}) {
  if (paneCount <= 1) return null;

  return (
    <div className="flex items-center justify-center gap-3 px-3 py-1.5 shrink-0">
      <button
        onClick={onPrev}
        disabled={focusedIndex === 0}
        className={cn(
          'p-1 rounded transition-colors',
          focusedIndex === 0
            ? 'text-muted-foreground/20 cursor-not-allowed'
            : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
        )}
        title="Previous terminal (Alt+←)"
      >
        <ChevronLeft className="h-4 w-4" />
      </button>

      <div className="flex items-center gap-1.5">
        {Array.from({ length: paneCount }).map((_, i) => (
          <button
            key={i}
            onClick={() => onDotClick(i)}
            className={cn(
              'rounded-full transition-all duration-200',
              i === focusedIndex
                ? 'w-5 h-2 bg-blue-400'
                : visibleIndices.has(i)
                  ? 'w-2 h-2 bg-muted-foreground/40 hover:bg-muted-foreground/60'
                  : 'w-2 h-2 bg-muted-foreground/20 hover:bg-muted-foreground/40',
            )}
            title={`Terminal ${i + 1}`}
          />
        ))}
      </div>

      <button
        onClick={onNext}
        disabled={focusedIndex === paneCount - 1}
        className={cn(
          'p-1 rounded transition-colors',
          focusedIndex === paneCount - 1
            ? 'text-muted-foreground/20 cursor-not-allowed'
            : 'text-muted-foreground hover:text-foreground hover:bg-muted/50',
        )}
        title="Next terminal (Alt+→)"
      >
        <ChevronRight className="h-4 w-4" />
      </button>
    </div>
  );
}
