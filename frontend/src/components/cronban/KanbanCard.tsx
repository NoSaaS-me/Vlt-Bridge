/**
 * KanbanCard — one PipelineCard displayed within a pipeline stage column.
 *
 * Shows title, gate evaluation status, and hover actions (advance / delete).
 */
import { useState } from 'react';
import { ChevronDown, ChevronUp, ChevronRight, Trash2, Play, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { GateStatusBadge } from './GateStatusBadge';
import type { PipelineCard, PipelineStage } from '@/services/cronban-api';

export interface KanbanCardProps {
  card: PipelineCard;
  stages: PipelineStage[];          // all stages of this pipeline (for move menu)
  isAdvancing: boolean;
  onAdvance: (cardId: string, stageId?: string) => void;
  onDelete: (cardId: string) => void;
  onFire: (cardId: string) => Promise<void>;
}

export function KanbanCard({ card, stages, isAdvancing, onAdvance, onDelete, onFire }: KanbanCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [showMoveMenu, setShowMoveMenu] = useState(false);
  const [firing, setFiring] = useState(false);

  const borderColor = card.color ?? '#3b82f6';

  // Stages we can manually advance to (higher stage_order, not current)
  const currentStage = stages.find((s) => s.id === card.current_stage_id);
  const advanceableStages = stages.filter(
    (s) => currentStage && s.stage_order > currentStage.stage_order,
  );

  return (
    <div
      className={cn(
        'relative rounded-md border border-border bg-card text-card-foreground',
        'shadow-sm transition-shadow hover:shadow-md group',
        'overflow-hidden',
      )}
      style={{ borderLeftColor: borderColor, borderLeftWidth: 3 }}
    >
      {/* Main body */}
      <div className="p-3">
        {/* Title row */}
        <div className="flex items-start justify-between gap-2 mb-1.5">
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

        {/* Status indicator */}
        {card.status !== 'active' && (
          <span className={cn(
            'inline-block text-[10px] px-1.5 py-0.5 rounded mb-1.5',
            card.status === 'completed' && 'bg-emerald-500/20 text-emerald-400',
            card.status === 'paused' && 'bg-amber-500/20 text-amber-400',
            card.status === 'failed' && 'bg-destructive/20 text-destructive',
          )}>
            {card.status}
          </span>
        )}

        {/* Bottom row: gate badge + actions */}
        <div className="flex items-center justify-between gap-2">
          <GateStatusBadge card={card} />

          {/* Hover action row */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            {/* Fire (play) button — dispatches prompt to target session */}
            {(card.has_prompt || card.skill_id) && (
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
                title="Fire card — send prompt to session"
              >
                {firing
                  ? <Loader2 className="h-3 w-3 animate-spin" />
                  : <Play className="h-3 w-3" />}
              </Button>
            )}

            {/* Advance button */}
            {advanceableStages.length > 0 && (
              <div className="relative">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 px-1.5 text-[10px] gap-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    if (advanceableStages.length === 1) {
                      onAdvance(card.id, advanceableStages[0].id);
                    } else {
                      setShowMoveMenu((v) => !v);
                    }
                  }}
                  disabled={isAdvancing || card.gate_eval_pending}
                  title="Advance to next stage"
                >
                  <ChevronRight className="h-3 w-3" />
                  Advance
                </Button>
                {showMoveMenu && (
                  <div
                    className="absolute right-0 bottom-full mb-1 z-50 min-w-40 rounded-md border border-border bg-popover shadow-lg py-1"
                    onMouseLeave={() => setShowMoveMenu(false)}
                  >
                    {advanceableStages.map((s) => (
                      <button
                        key={s.id}
                        className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 transition-colors flex items-center gap-2"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowMoveMenu(false);
                          onAdvance(card.id, s.id);
                        }}
                      >
                        {s.is_terminal && <span className="text-emerald-400 text-[9px]">✓</span>}
                        {s.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
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

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-border px-3 py-2.5 bg-muted/20 space-y-2">
          {/* Gate reasoning */}
          {card.gate_last_result?.reasoning && (
            <div>
              <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide mb-1">
                Gate reasoning
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {card.gate_last_result.reasoning}
              </p>
            </div>
          )}

          {/* Target session */}
          {card.target_session_id && (
            <div>
              <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide mb-0.5">
                Session
              </p>
              <p className="text-[10px] text-muted-foreground font-mono">
                {card.target_session_id.slice(0, 16)}…
              </p>
            </div>
          )}

          {/* Stats row */}
          <div className="flex gap-3 text-[10px] text-muted-foreground pt-0.5">
            {card.gate_last_checked_at && (
              <span>
                Last checked:{' '}
                {new Date(card.gate_last_checked_at).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            )}
            {card.gate_consecutive_not_met > 0 && (
              <span className="text-amber-400/70">
                {card.gate_consecutive_not_met}x not met
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
