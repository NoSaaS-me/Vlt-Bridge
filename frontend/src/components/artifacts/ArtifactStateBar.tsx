/**
 * ArtifactStateBar — State machine display bar rendered above the artifact iframe.
 *
 * Shows the artifact name, current state badge, valid transition buttons, and export.
 */
import { Download } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type ArtifactData } from '@/services/artifact-api';
import { Button } from '@/components/ui/button';

// ---------------------------------------------------------------------------
// State machine: valid transitions per state
// ---------------------------------------------------------------------------

type ArtifactState = ArtifactData['state'];

const TRANSITIONS: Record<ArtifactState, ArtifactState[]> = {
  draft: ['building'],
  building: ['testing', 'error'],
  testing: ['reviewing', 'error'],
  reviewing: ['approved', 'building'],
  approved: ['deployed', 'building'],
  deployed: ['update_pending'],
  error: ['draft', 'building'],
  update_pending: ['building'],
};

const STATE_LABEL: Record<ArtifactState, string> = {
  draft: 'Draft',
  building: 'Building',
  testing: 'Testing',
  reviewing: 'Reviewing',
  approved: 'Approved',
  deployed: 'Deployed',
  error: 'Error',
  update_pending: 'Update Pending',
};

const STATE_BADGE_CLASS: Record<ArtifactState, string> = {
  draft: 'bg-muted text-muted-foreground',
  building: 'bg-yellow-500/20 text-yellow-300',
  testing: 'bg-blue-500/20 text-blue-300',
  reviewing: 'bg-purple-500/20 text-purple-300',
  approved: 'bg-green-500/20 text-green-300',
  deployed: 'bg-emerald-500/20 text-emerald-300',
  error: 'bg-red-500/20 text-red-300',
  update_pending: 'bg-orange-500/20 text-orange-300',
};

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ArtifactStateBarProps {
  artifact: ArtifactData;
  onTransition: (targetState: string) => void;
  onExport?: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ArtifactStateBar({ artifact, onTransition, onExport }: ArtifactStateBarProps) {
  const validTransitions = TRANSITIONS[artifact.state] ?? [];

  const updatedAt = new Date(artifact.updated_at);
  const timeLabel = updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  return (
    <div className="flex items-center gap-3 px-3 py-1.5 border-b border-border bg-background/70 shrink-0">
      {/* Name */}
      <span className="text-sm font-medium truncate min-w-0 max-w-[200px]">{artifact.name}</span>

      {/* State badge */}
      <span
        className={cn(
          'inline-flex items-center rounded px-2 py-0.5 text-xs font-medium shrink-0',
          STATE_BADGE_CLASS[artifact.state] ?? 'bg-muted text-muted-foreground',
        )}
      >
        {STATE_LABEL[artifact.state] ?? artifact.state}
      </span>

      {/* Transition buttons */}
      {validTransitions.length > 0 && (
        <div className="flex items-center gap-1 shrink-0">
          {validTransitions.map((target) => (
            <Button
              key={target}
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[11px]"
              onClick={() => onTransition(target)}
            >
              → {STATE_LABEL[target] ?? target}
            </Button>
          ))}
        </div>
      )}

      <div className="flex-1" />

      {/* Last updated */}
      <span className="text-[10px] text-muted-foreground shrink-0">
        v{artifact.version} · {timeLabel}
      </span>

      {/* Export */}
      {onExport && (
        <Button
          variant="ghost"
          size="sm"
          className="h-6 w-6 p-0 shrink-0"
          title="Export artifact as zip"
          onClick={onExport}
        >
          <Download className="h-3.5 w-3.5" />
        </Button>
      )}
    </div>
  );
}

export default ArtifactStateBar;
