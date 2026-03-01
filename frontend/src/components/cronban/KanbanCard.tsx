/**
 * KanbanCard — one gate-type CronbanEntry displayed as a kanban card.
 *
 * Layout:
 *   - Colored left border (entry.color, default blue)
 *   - Title (bold, truncated)
 *   - Skill name or "Inline prompt" label (small, muted)
 *   - Lock icon if has_eval is set, dashed lock if not
 *   - GateStatusBadge (bottom right)
 *   - Hover: "Run Gate Check" button + move hint
 *   - Click to expand detail popover with full title, prompt preview, gate reasoning
 */
import { useState } from 'react';
import { Lock, Unlock, ChevronDown, ChevronUp, Play, MoveRight } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { GateStatusBadge } from './GateStatusBadge';
import type { CronbanEntry, KanbanColumn } from '@/services/cronban-api';

export interface KanbanCardProps {
  entry: CronbanEntry;
  onRunGateCheck: (entryId: string) => void;
  isCheckingGate: boolean;
  onMoveCard: (entryId: string, columnId: string) => void;
  columns: KanbanColumn[];
  /** Optional: skill name resolved externally so card doesn't need to fetch it */
  skillName?: string;
}

export function KanbanCard({
  entry,
  onRunGateCheck,
  isCheckingGate,
  onMoveCard,
  columns,
  skillName,
}: KanbanCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [showMoveMenu, setShowMoveMenu] = useState(false);

  const borderColor = entry.color ?? '#3b82f6'; // fallback: blue-500

  // Columns we can move to (exclude current)
  const otherColumns = columns.filter((c) => c.id !== entry.kanban_column_id);

  const promptPreview = entry.prompt_text
    ? entry.prompt_text.slice(0, 200) + (entry.prompt_text.length > 200 ? '…' : '')
    : null;

  return (
    <div
      className={cn(
        'relative rounded-md border border-border bg-card text-card-foreground',
        'shadow-sm transition-shadow hover:shadow-md group',
        'pl-0 overflow-hidden',
      )}
      style={{ borderLeftColor: borderColor, borderLeftWidth: 3 }}
    >
      {/* Main card body */}
      <div className="p-3 pl-3">
        {/* Title row */}
        <div className="flex items-start justify-between gap-2 mb-1.5">
          <button
            onClick={() => setExpanded((v) => !v)}
            className="flex-1 text-left"
          >
            <span className="text-sm font-semibold leading-snug line-clamp-2 group-hover:text-foreground">
              {entry.title}
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

        {/* Skill / prompt label row */}
        <div className="flex items-center gap-1.5 mb-2">
          {entry.has_eval ? (
            <Lock className="h-3 w-3 text-muted-foreground shrink-0" title="Eval criterion is set" />
          ) : (
            <Unlock
              className="h-3 w-3 text-muted-foreground/50 shrink-0 stroke-dashed"
              title="No eval criterion set"
              strokeDasharray="2 1"
            />
          )}
          <span className="text-[10px] text-muted-foreground truncate">
            {skillName
              ? skillName
              : entry.skill_id
              ? `Skill: ${entry.skill_id.slice(0, 8)}…`
              : 'Inline prompt'}
          </span>
        </div>

        {/* Bottom: gate badge + hover actions */}
        <div className="flex items-center justify-between gap-2">
          <GateStatusBadge
            entry={entry}
            onRunCheck={onRunGateCheck}
            isChecking={isCheckingGate}
          />

          {/* Hover action row */}
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              size="sm"
              variant="ghost"
              className="h-6 px-1.5 text-[10px] gap-1"
              onClick={(e) => {
                e.stopPropagation();
                onRunGateCheck(entry.id);
              }}
              disabled={isCheckingGate}
              title="Run gate check"
            >
              <Play className="h-3 w-3" />
              Check
            </Button>

            {otherColumns.length > 0 && (
              <div className="relative">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-6 px-1.5 text-[10px] gap-1"
                  onClick={(e) => {
                    e.stopPropagation();
                    setShowMoveMenu((v) => !v);
                  }}
                  title="Move to column"
                >
                  <MoveRight className="h-3 w-3" />
                  Move
                </Button>
                {showMoveMenu && (
                  <div
                    className="absolute right-0 bottom-full mb-1 z-50 min-w-36 rounded-md border border-border bg-popover shadow-lg py-1"
                    onMouseLeave={() => setShowMoveMenu(false)}
                  >
                    {otherColumns.map((col) => (
                      <button
                        key={col.id}
                        className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 transition-colors"
                        onClick={(e) => {
                          e.stopPropagation();
                          setShowMoveMenu(false);
                          onMoveCard(entry.id, col.id);
                        }}
                      >
                        {col.name}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Expanded detail panel */}
      {expanded && (
        <div className="border-t border-border px-3 py-2.5 bg-muted/20 space-y-2">
          {/* Prompt preview */}
          {promptPreview && (
            <div>
              <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide mb-1">
                Prompt
              </p>
              <p className="text-xs text-muted-foreground font-mono leading-relaxed whitespace-pre-wrap">
                {promptPreview}
              </p>
            </div>
          )}

          {/* Gate reasoning */}
          {entry.gate_last_result?.reasoning && (
            <div>
              <p className="text-[9px] font-medium text-muted-foreground uppercase tracking-wide mb-1">
                Gate reasoning
              </p>
              <p className="text-xs text-muted-foreground leading-relaxed">
                {entry.gate_last_result.reasoning}
              </p>
            </div>
          )}

          {/* Stats row */}
          <div className="flex gap-3 text-[10px] text-muted-foreground pt-0.5">
            {entry.gate_last_checked_at && (
              <span>
                Last checked:{' '}
                {new Date(entry.gate_last_checked_at).toLocaleString(undefined, {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            )}
            {entry.gate_consecutive_not_met > 0 && (
              <span className="text-amber-400/70">
                {entry.gate_consecutive_not_met}x not met
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
