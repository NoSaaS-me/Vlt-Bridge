/**
 * KanbanCard — one PipelineCard displayed within a pipeline stage column.
 *
 * Shows title, status line (fire count, last fired, session status),
 * and hover actions (fire+advance / delete). Expanding shows an activity
 * feed pulled from fire logs + gate logs.
 */
import { useState, useEffect, useCallback } from 'react';
import {
  ChevronDown, ChevronUp, Trash2, Play, Loader2,
  ExternalLink, Zap, ShieldCheck,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { GateStatusBadge } from './GateStatusBadge';
import type { PipelineCard, PipelineStage } from '@/services/cronban-api';
import { getFireLogs, getGateLogs } from '@/services/cronban-api';

/** Relative time string, e.g. "2m ago", "3h ago" */
function timeAgo(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  if (ms < 0) return 'just now'; // future timestamps (clock skew)
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.floor(ms / 3_600_000)}h ago`;
  return `${Math.floor(ms / 86_400_000)}d ago`;
}

interface FireLogEntry {
  id: string;
  fired_at: string;
  trigger_type: string;
  target_session_id: string | null;
  result: string;
  error_message: string | null;
}

interface GateLogEntry {
  id: string;
  checked_at: string;
  met: boolean;
  reasoning: string | null;
  eval_model: string | null;
}

export interface KanbanCardProps {
  card: PipelineCard;
  stages: PipelineStage[];
  isAdvancing: boolean;
  onAdvance: (cardId: string, stageId?: string) => void;
  onDelete: (cardId: string) => void;
  onFire: (cardId: string) => Promise<void>;
  /** Session status lookup: session_id → status string */
  sessionStatuses?: Record<string, string>;
  /** Called when user wants to view a session */
  onViewSession?: (sessionId: string) => void;
}

export function KanbanCard({
  card, stages, isAdvancing, onAdvance, onDelete, onFire,
  sessionStatuses = {}, onViewSession,
}: KanbanCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [firing, setFiring] = useState(false);
  const [fireLogs, setFireLogs] = useState<FireLogEntry[]>([]);
  const [gateLogs, setGateLogs] = useState<GateLogEntry[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);

  const borderColor = card.color ?? '#3b82f6';

  // Current + next stage for advance
  const currentStage = stages.find((s) => s.id === card.current_stage_id);
  const nextStage = stages.find(
    (s) => currentStage && s.stage_order === currentStage.stage_order + 1,
  );
  const isTerminal = currentStage?.is_terminal ?? false;

  // Session status dot
  const sessionStatus = card.target_session_id
    ? sessionStatuses[card.target_session_id] ?? null
    : null;

  // Last fire error (from card data — quick check without logs)
  const lastFireFailed = card.status === 'failed';

  // Load logs when expanded
  const loadLogs = useCallback(async () => {
    setLogsLoading(true);
    try {
      const [fires, gates] = await Promise.all([
        getFireLogs(card.id, 10) as Promise<FireLogEntry[]>,
        getGateLogs(card.id, 10) as Promise<GateLogEntry[]>,
      ]);
      setFireLogs(fires);
      setGateLogs(gates);
    } catch (e) {
      console.error('Failed to load card logs:', e);
    } finally {
      setLogsLoading(false);
    }
  }, [card.id]);

  useEffect(() => {
    if (expanded) loadLogs();
  }, [expanded, loadLogs]);

  // Fire + advance handler
  const handleFireAndAdvance = async (e: React.MouseEvent) => {
    e.stopPropagation();
    setFiring(true);
    try {
      await onFire(card.id);
      // Auto-advance to next stage after successful fire
      if (nextStage && !isTerminal) {
        onAdvance(card.id, nextStage.id);
      }
    } finally {
      setFiring(false);
    }
  };

  // Merge fire + gate logs into unified timeline
  const timeline = [
    ...fireLogs.map((l) => ({
      type: 'fire' as const,
      time: l.fired_at,
      ...l,
    })),
    ...gateLogs.map((l) => ({
      type: 'gate' as const,
      time: l.checked_at,
      ...l,
    })),
  ].sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime());

  return (
    <div
      draggable
      onDragStart={(e) => {
        e.dataTransfer.setData('text/plain', card.id);
        e.dataTransfer.effectAllowed = 'move';
        (e.currentTarget as HTMLElement).style.opacity = '0.5';
      }}
      onDragEnd={(e) => {
        (e.currentTarget as HTMLElement).style.opacity = '1';
      }}
      className={cn(
        'relative rounded-md border border-border bg-card text-card-foreground',
        'shadow-sm transition-shadow hover:shadow-md group',
        'overflow-hidden cursor-grab active:cursor-grabbing',
        lastFireFailed && 'border-destructive/40',
      )}
      style={{ borderLeftColor: borderColor, borderLeftWidth: 3 }}
    >
      {/* Main body */}
      <div className="p-3">
        {/* Title row */}
        <div className="flex items-start justify-between gap-2 mb-1">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex-1 text-left"
          >
            <span className="text-sm font-semibold leading-snug line-clamp-2 group-hover:text-foreground">
              {card.title}
            </span>
          </button>
          <button
            onClick={() => setExpanded((v) => !v)}
            className="shrink-0 text-muted-foreground hover:text-foreground mt-0.5"
          >
            {expanded
              ? <ChevronUp className="h-3.5 w-3.5" />
              : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        </div>

        {/* Always-visible status line */}
        <div className="space-y-1 text-[10px] text-muted-foreground mb-1.5">
          {/* Config summary — what this card does */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {card.has_prompt && (
              <span className="px-1.5 py-0 rounded bg-muted/50">prompt</span>
            )}
            {card.skill_id && (
              <span className="px-1.5 py-0 rounded bg-indigo-500/15 text-indigo-400">skill</span>
            )}
            {card.gate_id && (
              <span className="px-1.5 py-0 rounded bg-purple-500/15 text-purple-400">gated</span>
            )}
            {card.use_helper_session && (
              <span className="px-1.5 py-0 rounded bg-cyan-500/15 text-cyan-400">helper pool</span>
            )}
            {!card.use_helper_session && !card.target_session_id && (
              <span className="px-1.5 py-0 rounded bg-muted/50">new session</span>
            )}
            {card.target_session_id && (
              <span className="px-1.5 py-0 rounded bg-blue-500/15 text-blue-400">
                session {card.target_session_id.slice(0, 6)}
              </span>
            )}
          </div>

          {/* Fire stats + session status */}
          <div className="flex items-center gap-2 flex-wrap">
            {/* Fire count + last fired */}
            {card.fire_count > 0 && (
              <span className="flex items-center gap-0.5">
                <Zap className="h-2.5 w-2.5" />
                {card.fire_count}x
                {card.last_fired_at && (
                  <span className="opacity-70">· {timeAgo(card.last_fired_at)}</span>
                )}
              </span>
            )}
            {card.fire_count === 0 && (
              <span className="opacity-50 italic">never fired</span>
            )}

            {/* Session status indicator */}
            {card.target_session_id && (
              <span className="flex items-center gap-1">
                <span className={cn(
                  'inline-block h-1.5 w-1.5 rounded-full',
                  sessionStatus === 'thinking' && 'bg-amber-400 animate-pulse',
                  sessionStatus === 'executing' && 'bg-blue-400 animate-pulse',
                  sessionStatus === 'idle' && 'bg-emerald-400',
                  sessionStatus === 'dead' && 'bg-neutral-500',
                  !sessionStatus && 'bg-neutral-600',
                )} />
                <span className="opacity-70">
                  {sessionStatus ?? 'unknown'}
                </span>
                {onViewSession && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onViewSession(card.target_session_id!);
                    }}
                    className="hover:text-foreground transition-colors"
                    title="View session"
                  >
                    <ExternalLink className="h-2.5 w-2.5" />
                  </button>
                )}
              </span>
            )}

            {/* Card status badge (non-active) */}
            {card.status !== 'active' && (
              <span className={cn(
                'px-1.5 py-0 rounded',
                card.status === 'completed' && 'bg-emerald-500/20 text-emerald-400',
                card.status === 'paused' && 'bg-amber-500/20 text-amber-400',
                card.status === 'failed' && 'bg-destructive/20 text-destructive',
              )}>
                {card.status}
              </span>
            )}
          </div>

          {/* Last fire error inline (without expanding) */}
          {lastFireFailed && card.fire_count > 0 && (
            <div className="flex items-start gap-1 text-destructive/80 bg-destructive/10 rounded px-1.5 py-1">
              <ShieldCheck className="h-3 w-3 mt-0.5 shrink-0" />
              <span>Last fire failed — expand card for details</span>
            </div>
          )}
        </div>

        {/* Bottom row: gate badge + actions */}
        <div className="flex items-center justify-between gap-2">
          <GateStatusBadge card={card} />

          {/* Action buttons */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {/* Fire + Advance button */}
            {(card.has_prompt || card.skill_id) && !isTerminal && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-1.5 text-[10px] gap-1 text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/10"
                onClick={handleFireAndAdvance}
                disabled={firing || isAdvancing}
                title={nextStage ? `Fire & advance to "${nextStage.name}"` : 'Fire card'}
              >
                {firing
                  ? <Loader2 className="h-3 w-3 animate-spin" />
                  : <Play className="h-3 w-3" />}
                {nextStage ? 'Run' : 'Fire'}
              </Button>
            )}

            {/* Fire only (for terminal stage or no next stage) */}
            {(card.has_prompt || card.skill_id) && isTerminal && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-1.5 text-[10px] gap-1 text-emerald-400 hover:text-emerald-300 hover:bg-emerald-500/10"
                onClick={async (e) => {
                  e.stopPropagation();
                  setFiring(true);
                  try { await onFire(card.id); } finally { setFiring(false); }
                }}
                disabled={firing}
                title="Fire card"
              >
                {firing
                  ? <Loader2 className="h-3 w-3 animate-spin" />
                  : <Play className="h-3 w-3" />}
              </Button>
            )}

            {/* Delete */}
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-1.5 text-[10px] text-muted-foreground hover:text-destructive"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(card.id);
              }}
              title="Delete card"
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>

      {/* Expanded: Activity feed */}
      {expanded && (
        <div className="border-t border-border px-3 py-2.5 bg-muted/20 space-y-2 max-h-64 overflow-y-auto">
          {/* Gate reasoning (current) */}
          {card.gate_last_result?.reasoning && (
            <div className="rounded border border-border/50 bg-background/50 px-2 py-1.5">
              <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide mb-0.5">
                Latest gate
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {card.gate_last_result.reasoning}
              </p>
            </div>
          )}

          {/* Activity timeline */}
          {logsLoading ? (
            <div className="flex items-center gap-1.5 text-[10px] text-muted-foreground py-2">
              <Loader2 className="h-3 w-3 animate-spin" /> Loading activity…
            </div>
          ) : timeline.length === 0 ? (
            <p className="text-[10px] text-muted-foreground/50 italic py-1">
              No activity yet
            </p>
          ) : (
            <div className="space-y-1">
              <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide">
                Activity
              </p>
              {timeline.slice(0, 15).map((entry) => (
                <div
                  key={entry.type + '-' + entry.id}
                  className="flex items-start gap-1.5 text-[10px] text-muted-foreground"
                >
                  {/* Icon */}
                  {entry.type === 'fire' ? (
                    <Zap className={cn(
                      'h-3 w-3 mt-0.5 shrink-0',
                      (entry as FireLogEntry & { type: 'fire' }).result === 'success'
                        ? 'text-emerald-400'
                        : 'text-destructive',
                    )} />
                  ) : (
                    <ShieldCheck className={cn(
                      'h-3 w-3 mt-0.5 shrink-0',
                      (entry as GateLogEntry & { type: 'gate' }).met
                        ? 'text-emerald-400'
                        : 'text-amber-400',
                    )} />
                  )}

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    {entry.type === 'fire' ? (
                      <>
                        <span className="font-medium">
                          {(entry as FireLogEntry & { type: 'fire' }).result === 'success' ? 'Fired' : 'Fire failed'}
                        </span>
                        {(entry as FireLogEntry & { type: 'fire' }).target_session_id && (
                          <span className="opacity-60 ml-1">
                            → {(entry as FireLogEntry & { type: 'fire' }).target_session_id!.slice(0, 8)}
                          </span>
                        )}
                        {(entry as FireLogEntry & { type: 'fire' }).error_message && (
                          <p className="text-destructive/80 truncate">
                            {(entry as FireLogEntry & { type: 'fire' }).error_message}
                          </p>
                        )}
                      </>
                    ) : (
                      <>
                        <span className="font-medium">
                          Gate {(entry as GateLogEntry & { type: 'gate' }).met ? 'met ✓' : 'not met'}
                        </span>
                        {(entry as GateLogEntry & { type: 'gate' }).reasoning && (
                          <p className="opacity-70 truncate">
                            {(entry as GateLogEntry & { type: 'gate' }).reasoning}
                          </p>
                        )}
                      </>
                    )}
                  </div>

                  {/* Timestamp */}
                  <span className="opacity-50 shrink-0 tabular-nums">
                    {timeAgo(entry.time)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Target session link */}
          {card.target_session_id && onViewSession && (
            <button
              className="flex items-center gap-1.5 text-[10px] text-blue-400 hover:text-blue-300 transition-colors pt-1"
              onClick={() => onViewSession(card.target_session_id!)}
            >
              <ExternalLink className="h-3 w-3" />
              Open live session
            </button>
          )}
        </div>
      )}
    </div>
  );
}
