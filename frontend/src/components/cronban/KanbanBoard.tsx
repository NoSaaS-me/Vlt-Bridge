/**
 * KanbanBoard — main board view for gate-type CronbanEntries.
 *
 * Layout:
 *   - Horizontal scrolling flex row of KanbanColumns
 *   - Each column: header (name, count badge, add-card button), cards, empty state
 *   - "New Column" button at the end of the row
 *
 * Data:
 *   - listColumns(projectId) for columns
 *   - listEntries({ entry_type: 'gate', project_id }) for entries
 *   - runGateCheck(entryId) — shows result; if met, prompts to move card
 *   - moveKanbanCard(entryId, columnId) — handles requires_gate_check warning
 *   - createColumn() for new columns
 *   - updateColumn() for rename
 *   - deleteColumn() with optional move_to
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import { Plus, MoreVertical, Pencil, Trash2, Check, X, GraduationCap } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { KanbanCard } from './KanbanCard';
import {
  type CronbanEntry,
  type KanbanColumn,
  listColumns,
  listEntries,
  createColumn,
  updateColumn,
  deleteColumn,
  runGateCheck,
  moveKanbanCard,
} from '@/services/cronban-api';

// ---------------------------------------------------------------------------
// Column header with inline rename + kebab menu
// ---------------------------------------------------------------------------
function ColumnHeader({
  column,
  count,
  columns,
  onNewEntry,
  onRename,
  onDelete,
  onUpdateGraduation,
}: {
  column: KanbanColumn;
  count: number;
  columns: KanbanColumn[];
  onNewEntry: () => void;
  onRename: (newName: string) => void;
  onDelete: () => void;
  onUpdateGraduation: (auto_graduate: boolean, graduation_column_id: string | null) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(column.name);
  const [menuOpen, setMenuOpen] = useState(false);
  const [gradOpen, setGradOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close menu on outside click
  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpen]);

  const commitRename = () => {
    const name = draft.trim();
    if (name && name !== column.name) onRename(name);
    setEditing(false);
  };

  const startEdit = () => {
    setDraft(column.name);
    setEditing(true);
    setMenuOpen(false);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  // Other columns available for graduation target
  const otherCols = columns.filter((c) => c.id !== column.id);

  return (
    <>
    <div className="flex items-center gap-1.5 px-3 py-2.5 border-b border-border">
      {editing ? (
        <div className="flex items-center gap-1 flex-1">
          <Input
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitRename();
              if (e.key === 'Escape') { setEditing(false); setDraft(column.name); }
            }}
            className="h-6 text-xs flex-1"
          />
          <button onClick={commitRename} className="text-emerald-400 hover:text-emerald-300">
            <Check className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => { setEditing(false); setDraft(column.name); }} className="text-muted-foreground hover:text-foreground">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <>
          <span
            className="text-xs font-semibold flex-1 truncate cursor-pointer hover:text-foreground text-foreground/80"
            onDoubleClick={startEdit}
            title="Double-click to rename"
          >
            {column.name}
          </span>
          <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 shrink-0">
            {count}
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground shrink-0"
            onClick={onNewEntry}
            title="Add card to this column"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
          {/* Kebab menu */}
          <div className="relative" ref={menuRef}>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground shrink-0"
              onClick={() => setMenuOpen((v) => !v)}
            >
              <MoreVertical className="h-3.5 w-3.5" />
            </Button>
            {menuOpen && (
              <div className="absolute right-0 top-full mt-1 z-50 min-w-44 rounded-md border border-border bg-popover shadow-lg py-1">
                <button
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 flex items-center gap-2 transition-colors"
                  onClick={startEdit}
                >
                  <Pencil className="h-3 w-3" /> Rename column
                </button>
                <button
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 flex items-center gap-2 transition-colors"
                  onClick={() => { setMenuOpen(false); setGradOpen(true); }}
                >
                  <GraduationCap className="h-3 w-3" /> Graduation settings
                </button>
                <button
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 flex items-center gap-2 text-destructive hover:text-destructive transition-colors"
                  onClick={() => { setMenuOpen(false); onDelete(); }}
                >
                  <Trash2 className="h-3 w-3" /> Delete column
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>

    {/* Graduation settings panel */}
    {gradOpen && (
      <div className="px-3 py-2.5 border-b border-border bg-muted/30 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-wide text-muted-foreground flex items-center gap-1">
            <GraduationCap className="h-3 w-3" /> Graduation
          </span>
          <button onClick={() => setGradOpen(false)} className="text-muted-foreground hover:text-foreground">
            <X className="h-3 w-3" />
          </button>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={column.auto_graduate}
            onChange={(e) => onUpdateGraduation(e.target.checked, column.graduation_column_id)}
            className="h-3 w-3"
          />
          <span className="text-xs">Auto-move card when gate passes</span>
        </label>
        {column.auto_graduate && (
          <div className="space-y-1">
            <p className="text-[10px] text-muted-foreground">Target column</p>
            <select
              className="w-full h-7 rounded border border-border bg-background text-xs px-2"
              value={column.graduation_column_id ?? ''}
              onChange={(e) =>
                onUpdateGraduation(column.auto_graduate, e.target.value || null)
              }
            >
              <option value="">Next column (by order)</option>
              {otherCols.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </div>
        )}
      </div>
    )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Single kanban column
// ---------------------------------------------------------------------------
function KanbanColumnView({
  column,
  entries,
  columns,
  checkingIds,
  onNewEntry,
  onRunGateCheck,
  onMoveCard,
  onRename,
  onDelete,
  onUpdateGraduation,
}: {
  column: KanbanColumn;
  entries: CronbanEntry[];
  columns: KanbanColumn[];
  checkingIds: Set<string>;
  onNewEntry: (columnId: string) => void;
  onRunGateCheck: (entryId: string) => void;
  onMoveCard: (entryId: string, columnId: string) => void;
  onRename: (columnId: string, newName: string) => void;
  onDelete: (columnId: string) => void;
  onUpdateGraduation: (columnId: string, auto_graduate: boolean, graduation_column_id: string | null) => void;
}) {
  return (
    <div className="flex flex-col w-72 shrink-0 rounded-lg border border-border bg-muted/20 overflow-hidden">
      <ColumnHeader
        column={column}
        count={entries.length}
        columns={columns}
        onNewEntry={() => onNewEntry(column.id)}
        onRename={(name) => onRename(column.id, name)}
        onDelete={() => onDelete(column.id)}
        onUpdateGraduation={(ag, gcid) => onUpdateGraduation(column.id, ag, gcid)}
      />

      <div className="flex-1 overflow-y-auto p-2 space-y-2 min-h-24">
        {entries.length === 0 ? (
          <div className="border-2 border-dashed border-border rounded-md p-4 text-center text-muted-foreground text-xs">
            No cards
          </div>
        ) : (
          entries.map((entry) => (
            <KanbanCard
              key={entry.id}
              entry={entry}
              onRunGateCheck={onRunGateCheck}
              isCheckingGate={checkingIds.has(entry.id)}
              onMoveCard={onMoveCard}
              columns={columns}
            />
          ))
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// "New Column" inline creator
// ---------------------------------------------------------------------------
function NewColumnInput({
  projectId,
  onCreated,
}: {
  projectId?: string;
  onCreated: (col: KanbanColumn) => void;
}) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [saving, setSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleOpen = () => {
    setOpen(true);
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const handleCreate = async () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    setSaving(true);
    try {
      const col = await createColumn({
        project_id: projectId ?? '',
        name: trimmed,
        col_order: 9999,
        color: null,
        is_terminal: false,
      });
      onCreated(col);
      setName('');
      setOpen(false);
    } finally {
      setSaving(false);
    }
  };

  if (!open) {
    return (
      <Button
        variant="outline"
        size="sm"
        className="h-8 gap-1.5 shrink-0 text-xs border-dashed"
        onClick={handleOpen}
      >
        <Plus className="h-3.5 w-3.5" />
        New Column
      </Button>
    );
  }

  return (
    <div className="flex items-center gap-1.5 shrink-0">
      <Input
        ref={inputRef}
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleCreate();
          if (e.key === 'Escape') { setOpen(false); setName(''); }
        }}
        placeholder="Column name…"
        className="h-8 text-xs w-40"
        disabled={saving}
      />
      <Button size="sm" className="h-8 px-2" onClick={handleCreate} disabled={saving || !name.trim()}>
        {saving ? '…' : <Check className="h-3.5 w-3.5" />}
      </Button>
      <Button size="sm" variant="ghost" className="h-8 px-2" onClick={() => { setOpen(false); setName(''); }}>
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Default columns seed
// ---------------------------------------------------------------------------
const DEFAULT_COLUMNS = [
  { name: 'To Do', col_order: 0, color: 'blue', is_terminal: false },
  { name: 'In Progress', col_order: 1, color: 'amber', is_terminal: false },
  { name: 'In Review', col_order: 2, color: 'purple', is_terminal: false },
  { name: 'Done', col_order: 3, color: 'emerald', is_terminal: true },
] as const;

// ---------------------------------------------------------------------------
// Move confirmation toast/banner (simple inline approach)
// ---------------------------------------------------------------------------
interface MoveWarning {
  entryId: string;
  columnId: string;
  message: string;
}

// ---------------------------------------------------------------------------
// Main KanbanBoard
// ---------------------------------------------------------------------------
export function KanbanBoard({
  projectId,
  onNewEntry,
  refreshKey = 0,
}: {
  projectId?: string;
  onNewEntry: (columnId: string) => void;
  refreshKey?: number;
}) {
  const [columns, setColumns] = useState<KanbanColumn[]>([]);
  const [entries, setEntries] = useState<CronbanEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [checkingIds, setCheckingIds] = useState<Set<string>>(new Set());
  const [moveWarning, setMoveWarning] = useState<MoveWarning | null>(null);
  const hasSeededRef = useRef(false);

  // ---------------------------------------------------------------------------
  // Load data
  // ---------------------------------------------------------------------------
  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [cols, ents] = await Promise.all([
        listColumns(projectId),
        listEntries({ entry_type: 'gate', project_id: projectId }),
      ]);
      // Sort columns by col_order
      setColumns(cols.slice().sort((a, b) => a.col_order - b.col_order));
      setEntries(ents.slice().sort((a, b) => a.kanban_order - b.kanban_order));
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load board');
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => { loadData(); }, [loadData, refreshKey]);

  // Auto-seed standard columns on first load if board is empty
  const seedDefaultColumns = useCallback(async () => {
    try {
      const created = await Promise.all(
        DEFAULT_COLUMNS.map((col) =>
          createColumn({ project_id: projectId ?? '', ...col })
        )
      );
      setColumns(created.sort((a, b) => a.col_order - b.col_order));
    } catch (e) {
      console.error('Failed to seed default columns:', e);
    }
  }, [projectId]);

  useEffect(() => {
    if (!loading && columns.length === 0 && !hasSeededRef.current) {
      hasSeededRef.current = true;
      seedDefaultColumns();
    }
  }, [loading, columns.length, seedDefaultColumns]);

  // ---------------------------------------------------------------------------
  // Gate check + auto-graduation
  // ---------------------------------------------------------------------------
  const handleRunGateCheck = useCallback(async (entryId: string) => {
    setCheckingIds((prev) => new Set(prev).add(entryId));
    try {
      const result = await runGateCheck(entryId);
      // Update the entry's gate_last_result in local state
      setEntries((prev) => {
        const updated = prev.map((e) =>
          e.id === entryId
            ? {
                ...e,
                gate_last_result: { met: result.met ?? false, reasoning: result.reasoning },
                gate_last_checked_at: new Date().toISOString(),
                gate_consecutive_not_met: result.met ? 0 : e.gate_consecutive_not_met + 1,
              }
            : e,
        );
        return updated;
      });

      // Auto-graduate: if gate passed, find the source column and move to graduation target
      if (result.met) {
        setEntries((prev) => {
          const entry = prev.find((e) => e.id === entryId);
          if (!entry) return prev;
          setColumns((cols) => {
            const srcCol = cols.find((c) => c.id === entry.kanban_column_id);
            if (!srcCol || !srcCol.auto_graduate) return cols;
            // Find target: explicit graduation_column_id or next by col_order
            const sortedCols = cols.slice().sort((a, b) => a.col_order - b.col_order);
            const targetId = srcCol.graduation_column_id
              ?? sortedCols.find((c) => c.col_order > srcCol.col_order)?.id
              ?? null;
            if (targetId && targetId !== entry.kanban_column_id) {
              // Fire-and-forget move
              moveKanbanCard(entryId, targetId, true).then(() => {
                setEntries((e) =>
                  e.map((x) => x.id === entryId ? { ...x, kanban_column_id: targetId } : x)
                );
              }).catch((err) => console.error('Auto-graduate move failed:', err));
            }
            return cols;
          });
          return prev;
        });
      }
    } catch (e) {
      console.error('Gate check failed:', e);
    } finally {
      setCheckingIds((prev) => {
        const next = new Set(prev);
        next.delete(entryId);
        return next;
      });
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Move card
  // ---------------------------------------------------------------------------
  const handleMoveCard = useCallback(async (entryId: string, columnId: string, force = false) => {
    try {
      const res = await moveKanbanCard(entryId, columnId, force);
      if (res.requires_gate_check && !force) {
        setMoveWarning({
          entryId,
          columnId,
          message: res.message ?? 'This column requires a passing gate check before moving.',
        });
        return;
      }
      if (res.moved) {
        setEntries((prev) =>
          prev.map((e) => (e.id === entryId ? { ...e, kanban_column_id: columnId } : e)),
        );
      }
    } catch (e) {
      console.error('Move card failed:', e);
    }
  }, []);

  const handleForceMove = useCallback(() => {
    if (!moveWarning) return;
    handleMoveCard(moveWarning.entryId, moveWarning.columnId, true);
    setMoveWarning(null);
  }, [moveWarning, handleMoveCard]);

  // ---------------------------------------------------------------------------
  // Column management
  // ---------------------------------------------------------------------------
  const handleColumnCreated = useCallback((col: KanbanColumn) => {
    setColumns((prev) => [...prev, col].sort((a, b) => a.col_order - b.col_order));
  }, []);

  const handleRenameColumn = useCallback(async (columnId: string, newName: string) => {
    try {
      const updated = await updateColumn(columnId, { name: newName });
      setColumns((prev) => prev.map((c) => (c.id === columnId ? updated : c)));
    } catch (e) {
      console.error('Rename column failed:', e);
    }
  }, []);

  const handleDeleteColumn = useCallback(async (columnId: string) => {
    // Move entries to first other column if any exist
    const hasCards = entries.some((e) => e.kanban_column_id === columnId);
    const otherCol = columns.find((c) => c.id !== columnId);
    if (hasCards && !otherCol) {
      alert('Cannot delete the only column that has cards.');
      return;
    }
    try {
      await deleteColumn(columnId, hasCards && otherCol ? otherCol.id : undefined);
      setColumns((prev) => prev.filter((c) => c.id !== columnId));
      if (hasCards && otherCol) {
        setEntries((prev) =>
          prev.map((e) =>
            e.kanban_column_id === columnId ? { ...e, kanban_column_id: otherCol.id } : e,
          ),
        );
      }
    } catch (e) {
      console.error('Delete column failed:', e);
    }
  }, [columns, entries]);

  // ---------------------------------------------------------------------------
  // Graduation settings
  // ---------------------------------------------------------------------------
  const handleUpdateGraduation = useCallback(async (
    columnId: string,
    auto_graduate: boolean,
    graduation_column_id: string | null,
  ) => {
    try {
      const updated = await updateColumn(columnId, { auto_graduate, graduation_column_id });
      setColumns((prev) => prev.map((c) => (c.id === columnId ? updated : c)));
    } catch (e) {
      console.error('Update graduation settings failed:', e);
    }
  }, []);

  // ---------------------------------------------------------------------------
  // Group entries by column
  // ---------------------------------------------------------------------------
  const entriesByColumn = useCallback(
    (columnId: string) => entries.filter((e) => e.kanban_column_id === columnId),
    [entries],
  );

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
        Loading board…
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-2">
          <p className="text-sm text-destructive">{error}</p>
          <Button size="sm" variant="outline" onClick={loadData}>
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Move warning banner */}
      {moveWarning && (
        <div className="shrink-0 mx-4 mt-3 rounded-md border border-amber-500/30 bg-amber-500/10 px-4 py-2.5 flex items-center gap-3 text-sm">
          <span className="flex-1 text-amber-300/90 text-xs">{moveWarning.message}</span>
          <Button
            size="sm"
            variant="outline"
            className="h-7 text-xs border-amber-500/40 hover:bg-amber-500/20"
            onClick={handleForceMove}
          >
            Move anyway
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 text-xs"
            onClick={() => setMoveWarning(null)}
          >
            Cancel
          </Button>
        </div>
      )}

      {/* Board scroll area */}
      <div className="flex-1 overflow-x-auto overflow-y-hidden p-4">
        <div className="flex gap-3 h-full items-start">
          {columns.length === 0 ? (
            <div className="flex items-center justify-center flex-1 text-muted-foreground text-sm">
              <div className="text-center space-y-3">
                <p>No columns yet. Create one to get started.</p>
                <NewColumnInput projectId={projectId} onCreated={handleColumnCreated} />
              </div>
            </div>
          ) : (
            <>
              {columns.map((col) => (
                <KanbanColumnView
                  key={col.id}
                  column={col}
                  entries={entriesByColumn(col.id)}
                  columns={columns}
                  checkingIds={checkingIds}
                  onNewEntry={onNewEntry}
                  onRunGateCheck={handleRunGateCheck}
                  onMoveCard={handleMoveCard}
                  onRename={handleRenameColumn}
                  onDelete={handleDeleteColumn}
                  onUpdateGraduation={handleUpdateGraduation}
                />
              ))}
              {/* New column creator */}
              <div className="shrink-0 flex items-start pt-1">
                <NewColumnInput projectId={projectId} onCreated={handleColumnCreated} />
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
