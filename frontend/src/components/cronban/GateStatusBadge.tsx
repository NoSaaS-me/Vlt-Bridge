/**
 * GateStatusBadge — compact indicator for a gate entry's last check result.
 *
 * States:
 *   - null result + not checking: dim gray "No check" with Clock icon
 *   - isChecking: spinning loader "Checking..."
 *   - met: true — emerald "Passed" with CheckCircle
 *   - met: false — amber "Not met" with XCircle
 *
 * Shows gate_last_result.reasoning as a tooltip on hover.
 * Clicking the badge triggers onRunCheck (if provided).
 */
import { Clock, CheckCircle, XCircle, RefreshCw, Brain } from 'lucide-react';
import { cn } from '@/lib/utils';
import type { CronbanEntry } from '@/services/cronban-api';

export interface GateStatusBadgeProps {
  entry: CronbanEntry;
  onRunCheck?: (entryId: string) => void;
  isChecking?: boolean;
}

export function GateStatusBadge({ entry, onRunCheck, isChecking = false }: GateStatusBadgeProps) {
  const result = entry.gate_last_result;
  const tooltip = result?.reasoning ?? undefined;

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (onRunCheck && !isChecking) {
      onRunCheck(entry.id);
    }
  };

  const baseClass =
    'inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium border transition-colors select-none';

  // Helper Claude is evaluating (triggered automatically after agent turn ends)
  if (entry.gate_eval_pending) {
    return (
      <span
        title={`Helper ${entry.eval_model} is evaluating…`}
        className={cn(baseClass, 'bg-violet-500/15 border-violet-500/30 text-violet-400 cursor-not-allowed')}
      >
        <Brain className="h-3 w-3 animate-pulse" />
        Evaluating…
      </span>
    );
  }

  if (isChecking) {
    return (
      <span
        title="Running gate check..."
        className={cn(baseClass, 'bg-muted/40 border-border text-muted-foreground cursor-not-allowed')}
      >
        <RefreshCw className="h-3 w-3 animate-spin" />
        Checking…
      </span>
    );
  }

  if (result === null) {
    return (
      <span
        title={tooltip ?? 'No gate check has run yet'}
        onClick={onRunCheck ? handleClick : undefined}
        className={cn(
          baseClass,
          'bg-muted/30 border-border text-muted-foreground',
          onRunCheck && 'cursor-pointer hover:bg-muted/60',
        )}
      >
        <Clock className="h-3 w-3" />
        No check
      </span>
    );
  }

  if (result.met) {
    return (
      <span
        title={tooltip}
        onClick={onRunCheck ? handleClick : undefined}
        className={cn(
          baseClass,
          'bg-emerald-500/15 border-emerald-500/30 text-emerald-400',
          onRunCheck && 'cursor-pointer hover:bg-emerald-500/25',
        )}
      >
        <CheckCircle className="h-3 w-3" />
        Passed
      </span>
    );
  }

  return (
    <span
      title={tooltip}
      onClick={onRunCheck ? handleClick : undefined}
      className={cn(
        baseClass,
        'bg-amber-500/15 border-amber-500/30 text-amber-400',
        onRunCheck && 'cursor-pointer hover:bg-amber-500/25',
      )}
    >
      <XCircle className="h-3 w-3" />
      Not met
    </span>
  );
}
