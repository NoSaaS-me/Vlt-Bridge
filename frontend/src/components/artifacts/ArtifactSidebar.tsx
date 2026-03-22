/**
 * ArtifactSidebar — Left panel listing artifacts for the current project.
 *
 * Follows the same structure as SessionSidebar.tsx.
 */
import { useRef } from 'react';
import { Plus, Upload, Box } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type ArtifactData } from '@/services/artifact-api';
import { Separator } from '@/components/ui/separator';
import { Button } from '@/components/ui/button';

// ---------------------------------------------------------------------------
// State badge helpers
// ---------------------------------------------------------------------------

type ArtifactState = ArtifactData['state'];

const STATE_COLORS: Record<ArtifactState, string> = {
  draft: 'bg-muted-foreground',
  building: 'bg-yellow-400',
  testing: 'bg-blue-400',
  reviewing: 'bg-purple-400',
  approved: 'bg-green-400',
  deployed: 'bg-emerald-400',
  error: 'bg-red-400',
  update_pending: 'bg-orange-400',
};

function StateDot({ state }: { state: ArtifactState }) {
  return (
    <span
      className={cn('h-1.5 w-1.5 rounded-full shrink-0', STATE_COLORS[state] ?? 'bg-muted-foreground')}
      title={state}
    />
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ArtifactSidebarProps {
  artifacts: ArtifactData[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete?: (id: string) => void;
  onImport?: (file: File) => void;
  isLoading: boolean;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ArtifactSidebar({
  artifacts,
  selectedId,
  onSelect,
  onCreate,
  onDelete,
  onImport,
  isLoading,
}: ArtifactSidebarProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleImportClick() {
    fileInputRef.current?.click();
  }

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file && onImport) {
      onImport(file);
    }
    // Reset so the same file can be re-imported
    e.target.value = '';
  }

  return (
    <div className="h-full flex flex-col bg-background/50 border-r border-border">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Box className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Artifacts
          </span>
        </div>
        <div className="flex items-center gap-1">
          {onImport && (
            <Button
              variant="ghost"
              size="sm"
              className="h-5 w-5 p-0"
              title="Import artifact"
              onClick={handleImportClick}
            >
              <Upload className="h-3 w-3" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-5 w-5 p-0"
            title="New artifact"
            onClick={onCreate}
          >
            <Plus className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {/* Hidden file input for import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        className="hidden"
        onChange={handleFileChange}
      />

      <Separator />

      {/* Artifact list */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="p-3 space-y-2">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="h-10 rounded border border-border bg-muted/30 animate-pulse"
              />
            ))}
          </div>
        ) : artifacts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 px-3 text-center">
            <Box className="h-6 w-6 text-muted-foreground/30 mb-2" />
            <p className="text-[10px] text-muted-foreground">No artifacts yet</p>
            <p className="mt-1 text-[9px] text-muted-foreground/60">
              Use + to create one
            </p>
          </div>
        ) : (
          <div className="p-2 space-y-0.5">
            {artifacts.map((a) => (
              <ArtifactRow
                key={a.id}
                artifact={a}
                selected={a.id === selectedId}
                onSelect={() => onSelect(a.id)}
                onDelete={onDelete ? () => onDelete(a.id) : undefined}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Row
// ---------------------------------------------------------------------------

function ArtifactRow({
  artifact,
  selected,
  onSelect,
  onDelete,
}: {
  artifact: ArtifactData;
  selected: boolean;
  onSelect: () => void;
  onDelete?: () => void;
}) {
  return (
    <div
      className={cn(
        'group flex items-center gap-2 rounded px-2 py-1.5 cursor-pointer transition-colors duration-100',
        selected
          ? 'bg-blue-500/15 text-foreground'
          : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
      )}
      onClick={onSelect}
    >
      <StateDot state={artifact.state} />
      <span className="flex-1 text-xs truncate min-w-0">{artifact.name}</span>
      {onDelete && (
        <button
          className="hidden group-hover:flex h-4 w-4 items-center justify-center rounded text-muted-foreground/50 hover:text-destructive transition-colors"
          onClick={(e) => {
            e.stopPropagation();
            onDelete();
          }}
          title="Delete artifact"
        >
          <span className="text-[10px] leading-none">✕</span>
        </button>
      )}
    </div>
  );
}

export default ArtifactSidebar;
