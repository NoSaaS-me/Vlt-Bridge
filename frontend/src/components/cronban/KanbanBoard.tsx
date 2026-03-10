/**
 * KanbanBoard — Pipeline workflow view.
 *
 * Layout:
 *   - Pipeline selector (dropdown at top-left, or "Create pipeline" prompt)
 *   - Horizontal columns = pipeline stages
 *   - Cards within each stage
 *   - "New Card" button in each stage, "New Stage" at the end
 *   - Pipeline can be renamed/deleted via header menu
 */
import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Plus, Check, X, Pencil, Trash2, ChevronDown, MoreVertical, Flag } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { KanbanCard } from './KanbanCard';
import {
  type Pipeline,
  type PipelineStage,
  type PipelineCard,
  type CronbanSkill,
  type CronbanGate,
  listPipelines,
  getPipeline,
  createPipeline,
  updatePipeline,
  deletePipeline,
  addStage,
  updateStage,
  deleteStage,
  listCards,
  createCard,
  advanceCard,
  deleteCard,
  fireCard,
  listSkills,
  listGates,
} from '@/services/cronban-api';
import { type AgentSession, listSessions } from '@/services/daemon-api';
import { type ChainItem, assembleChain, PromptChainBuilder } from './PromptChainBuilder';

// ---------------------------------------------------------------------------
// New pipeline prompt
// ---------------------------------------------------------------------------
function NewPipelinePrompt({
  projectId,
  onCreated,
}: {
  projectId?: string;
  onCreated: (p: Pipeline) => void;
}) {
  const [name, setName] = useState('');
  const [creating, setCreating] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleCreate = async () => {
    const trimmed = name.trim();
    if (!trimmed) return;
    setCreating(true);
    try {
      const p = await createPipeline({ project_id: projectId, name: trimmed });
      // Add default stages
      const stages = [
        { name: 'To Do', stage_order: 0, auto_advance: true, is_terminal: false },
        { name: 'In Progress', stage_order: 1, auto_advance: true, is_terminal: false },
        { name: 'Done', stage_order: 2, auto_advance: false, is_terminal: true },
      ];
      for (const s of stages) {
        await addStage(p.id, s);
      }
      onCreated(await getPipeline(p.id));
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center space-y-4 max-w-sm">
        <div className="space-y-1">
          <p className="text-sm font-medium">No pipelines yet</p>
          <p className="text-xs text-muted-foreground">
            A pipeline defines stages for your work items to move through.
          </p>
        </div>
        <div className="flex gap-2 justify-center">
          <Input
            ref={inputRef}
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
            placeholder="Pipeline name (e.g. Bug Fix)"
            className="h-8 text-xs w-48"
            disabled={creating}
          />
          <Button
            size="sm"
            className="h-8"
            onClick={handleCreate}
            disabled={creating || !name.trim()}
          >
            {creating ? '…' : 'Create'}
          </Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pipeline selector toolbar
// ---------------------------------------------------------------------------
function PipelineSelector({
  pipelines,
  selected,
  onSelect,
  onNewPipeline,
}: {
  pipelines: Pipeline[];
  selected: Pipeline | null;
  onSelect: (p: Pipeline) => void;
  onNewPipeline: () => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <Button
        size="sm"
        variant="outline"
        className="h-7 gap-1.5 text-xs max-w-48"
        onClick={() => setOpen((v) => !v)}
      >
        <span className="truncate">{selected?.name ?? 'Select pipeline'}</span>
        <ChevronDown className="h-3 w-3 shrink-0" />
      </Button>
      {open && (
        <div className="absolute left-0 top-full mt-1 z-50 min-w-48 rounded-md border border-border bg-popover shadow-lg py-1">
          {pipelines.map((p) => (
            <button
              key={p.id}
              className={cn(
                'w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 transition-colors truncate',
                selected?.id === p.id && 'bg-muted/30 font-medium',
              )}
              onClick={() => { onSelect(p); setOpen(false); }}
            >
              {p.name}
            </button>
          ))}
          <div className="border-t border-border mt-1 pt-1">
            <button
              className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 transition-colors flex items-center gap-1.5 text-muted-foreground"
              onClick={() => { onNewPipeline(); setOpen(false); }}
            >
              <Plus className="h-3 w-3" /> New pipeline
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage column header
// ---------------------------------------------------------------------------
function StageHeader({
  stage,
  count,
  onNewCard,
  onRename,
  onDelete,
  onToggleTerminal,
}: {
  stage: PipelineStage;
  count: number;
  onNewCard: () => void;
  onRename: (name: string) => void;
  onDelete: () => void;
  onToggleTerminal: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(stage.name);
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setMenuOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [menuOpen]);

  const commitRename = () => {
    const n = draft.trim();
    if (n && n !== stage.name) onRename(n);
    setEditing(false);
  };

  return (
    <div className="flex items-center gap-1.5 px-3 py-2.5 border-b border-border">
      {editing ? (
        <div className="flex items-center gap-1 flex-1">
          <Input
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') commitRename();
              if (e.key === 'Escape') { setEditing(false); setDraft(stage.name); }
            }}
            className="h-6 text-xs flex-1"
          />
          <button onClick={commitRename} className="text-emerald-400 hover:text-emerald-300">
            <Check className="h-3.5 w-3.5" />
          </button>
          <button onClick={() => { setEditing(false); setDraft(stage.name); }} className="text-muted-foreground hover:text-foreground">
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      ) : (
        <>
          <span
            className="text-xs font-semibold flex-1 truncate cursor-pointer hover:text-foreground text-foreground/80"
            onDoubleClick={() => {
              setDraft(stage.name);
              setEditing(true);
              setTimeout(() => inputRef.current?.focus(), 0);
            }}
            title="Double-click to rename"
          >
            {stage.name}
          </span>
          {stage.is_terminal && (
            <Flag className="h-3 w-3 text-emerald-400 shrink-0" aria-label="Terminal stage" />
          )}
          <Badge variant="secondary" className="text-[9px] px-1.5 py-0 h-4 shrink-0">
            {count}
          </Badge>
          <Button
            size="sm"
            variant="ghost"
            className="h-6 w-6 p-0 text-muted-foreground hover:text-foreground shrink-0"
            onClick={onNewCard}
            title="Add card to this stage"
          >
            <Plus className="h-3.5 w-3.5" />
          </Button>
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
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 flex items-center gap-2"
                  onClick={() => {
                    setMenuOpen(false);
                    setDraft(stage.name);
                    setEditing(true);
                    setTimeout(() => inputRef.current?.focus(), 0);
                  }}
                >
                  <Pencil className="h-3 w-3" /> Rename stage
                </button>
                <button
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 flex items-center gap-2"
                  onClick={() => { setMenuOpen(false); onToggleTerminal(); }}
                >
                  <Flag className="h-3 w-3" />
                  {stage.is_terminal ? 'Unmark terminal' : 'Mark as terminal'}
                </button>
                <button
                  className="w-full text-left px-3 py-1.5 text-xs hover:bg-muted/50 flex items-center gap-2 text-destructive"
                  onClick={() => { setMenuOpen(false); onDelete(); }}
                >
                  <Trash2 className="h-3 w-3" /> Delete stage
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// New Card inline creator
// ---------------------------------------------------------------------------
interface NewCardData {
  title: string;
  prompt_text?: string;
  skill_id?: string;
  gate_id?: string;
  target_session_id?: string;
  target_cwd?: string;
  use_helper_session?: boolean;
}

const DIALOG_TABS = ['Action', 'Gate', 'Session'] as const;
type DialogTab = typeof DIALOG_TABS[number];

function NewCardDialog({
  open,
  onClose,
  onCreate,
  projectId,
}: {
  open: boolean;
  onClose: () => void;
  onCreate: (data: NewCardData) => void;
  projectId?: string;
}) {
  const [title, setTitle] = useState('');
  const [chainItems, setChainItems] = useState<ChainItem[]>([]);
  const [gateId, setGateId] = useState('');
  const [skills, setSkills] = useState<CronbanSkill[]>([]);
  const [gates, setGates] = useState<CronbanGate[]>([]);
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  // Session mode: 'new' | 'helper' | 'specific'
  const [sessionMode, setSessionMode] = useState<'new' | 'helper' | 'specific'>('new');
  const [targetSessionId, setTargetSessionId] = useState('');
  const [targetCwd, setTargetCwd] = useState('');
  const [tab, setTab] = useState<DialogTab>('Action');
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    setTitle('');
    setChainItems([]);
    setGateId('');
    setSessionMode('new');
    setTargetSessionId('');
    setTargetCwd('');
    setTab('Action');
    setTimeout(() => inputRef.current?.focus(), 0);
    Promise.all([listSkills(projectId), listGates(projectId), listSessions()])
      .then(([s, g, sess]) => {
        setSkills(s);
        setGates(g);
        // Sort: relay first, then managed/hook; prefer sessions matching projectId cwd
        const alive = sess.filter((s) => s.status !== 'dead');
        alive.sort((a, b) => {
          const aRelay = a.source === 'relay' ? 0 : 1;
          const bRelay = b.source === 'relay' ? 0 : 1;
          if (aRelay !== bRelay) return aRelay - bRelay;
          // Project match (cwd contains project cwd path heuristic)
          if (projectId) {
            const aMatch = a.project_id === projectId ? 0 : 1;
            const bMatch = b.project_id === projectId ? 0 : 1;
            if (aMatch !== bMatch) return aMatch - bMatch;
          }
          return (b.last_activity ?? '').localeCompare(a.last_activity ?? '');
        });
        setSessions(alive);
      })
      .catch(console.error);
  }, [open, projectId]);

  if (!open) return null;

  const assembledPrompt = assembleChain(chainItems, skills);

  const handleCreate = () => {
    const t = title.trim();
    if (!t) return;
    // If chain has a single pure-skill item, use skill_id; else use assembled text
    const singleSkill = chainItems.length === 1 && chainItems[0].type === 'skill'
      ? chainItems[0].skillId : undefined;
    onCreate({
      title: t,
      prompt_text: singleSkill ? undefined : (assembledPrompt || undefined),
      skill_id: singleSkill,
      gate_id: gateId || undefined,
      target_session_id: sessionMode === 'specific' ? (targetSessionId || undefined) : undefined,
      target_cwd: sessionMode === 'new' ? (targetCwd.trim() || undefined) : undefined,
      use_helper_session: sessionMode === 'helper' ? true : undefined,
    });
    onClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-md rounded-lg border border-border bg-popover shadow-2xl flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="flex items-center justify-between px-4 pt-4 pb-0">
          <h2 className="text-sm font-semibold">New Card</h2>
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground">
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Title */}
        <div className="px-4 pt-3 pb-2">
          <Input
            ref={inputRef}
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Escape') onClose(); }}
            placeholder="What needs to be done?"
            className="h-8 text-sm"
          />
        </div>

        {/* Tabs */}
        <div className="flex border-b border-border px-4">
          {DIALOG_TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                'px-3 py-2 text-xs font-medium border-b-2 -mb-px transition-colors',
                tab === t
                  ? 'border-primary text-foreground'
                  : 'border-transparent text-muted-foreground hover:text-foreground',
              )}
            >
              {t}
              {t === 'Gate' && gateId && <span className="ml-1 text-[9px] text-purple-400">●</span>}
              {t === 'Session' && (sessionMode === 'helper' || (sessionMode === 'specific' && targetSessionId) || (sessionMode === 'new' && targetCwd.trim())) && <span className="ml-1 text-[9px] text-blue-400">●</span>}
            </button>
          ))}
        </div>

        {/* Tab body */}
        <div className="px-4 py-3 overflow-y-auto flex-1">
          {tab === 'Action' && (
            <div className="space-y-1.5">
              <p className="text-[10px] text-muted-foreground">
                Build the prompt Claude receives when this card fires.
              </p>
              <PromptChainBuilder
                items={chainItems}
                skills={skills}
                onChange={setChainItems}
              />
            </div>
          )}

          {tab === 'Gate' && (
            <div className="space-y-2">
              <p className="text-[10px] text-muted-foreground">
                A gate runs a helper Claude session to check the work after the agent goes idle.
              </p>
              {gates.length === 0 ? (
                <p className="text-xs text-muted-foreground/60 italic">
                  No gates defined — create some in the Gates tab.
                </p>
              ) : (
                <select
                  value={gateId}
                  onChange={(e) => setGateId(e.target.value)}
                  className="w-full h-8 text-xs bg-background border border-border rounded px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                >
                  <option value="">— No gate —</option>
                  {gates.map((g) => (
                    <option key={g.id} value={g.id}>{g.name}</option>
                  ))}
                </select>
              )}
              {gateId && gates.find((g) => g.id === gateId)?.description && (
                <p className="text-[10px] text-muted-foreground/70 italic">
                  {gates.find((g) => g.id === gateId)!.description}
                </p>
              )}
            </div>
          )}

          {tab === 'Session' && (
            <div className="space-y-3">
              {/* Mode selector — three buttons side by side */}
              <div className="grid grid-cols-3 gap-1.5">
                {(
                  [
                    { mode: 'new', label: 'New Session', desc: 'Spawn a fresh Claude Code session' },
                    { mode: 'helper', label: 'Helper Session', desc: 'System-managed helper pool' },
                    { mode: 'specific', label: 'Specific Session', desc: 'Target a running relay/managed session' },
                  ] as const
                ).map(({ mode, label, desc }) => (
                  <button
                    key={mode}
                    onClick={() => setSessionMode(mode)}
                    className={cn(
                      'flex flex-col items-center justify-center gap-0.5 rounded-md border px-2 py-2 text-center transition-colors',
                      sessionMode === mode
                        ? 'border-primary bg-primary/10 text-foreground'
                        : 'border-border text-muted-foreground hover:border-border/80 hover:bg-muted/30',
                    )}
                  >
                    <span className="text-[11px] font-medium leading-tight">{label}</span>
                    <span className="text-[9px] leading-tight opacity-70">{desc}</span>
                  </button>
                ))}
              </div>

              {/* Mode-specific content */}
              {sessionMode === 'new' && (
                <div className="space-y-1.5">
                  <label className="text-[10px] text-muted-foreground">
                    Working directory <span className="opacity-60">(optional — defaults to daemon cwd)</span>
                  </label>
                  <Input
                    value={targetCwd}
                    onChange={(e) => setTargetCwd(e.target.value)}
                    placeholder="/home/user/my-project"
                    className="h-8 text-xs font-mono"
                  />
                </div>
              )}

              {sessionMode === 'helper' && (
                <p className="text-[10px] text-muted-foreground rounded-md border border-border/50 bg-muted/20 px-3 py-2 leading-relaxed">
                  System will pick an idle helper session or spawn one automatically. No further configuration needed.
                </p>
              )}

              {sessionMode === 'specific' && (
                <div className="space-y-1.5">
                  <label className="text-[10px] text-muted-foreground">
                    Running session <span className="opacity-60">(relay sessions inject directly)</span>
                  </label>
                  {sessions.length === 0 ? (
                    <p className="text-[10px] text-muted-foreground/60 italic">No active sessions found.</p>
                  ) : (
                    <select
                      value={targetSessionId}
                      onChange={(e) => setTargetSessionId(e.target.value)}
                      className="w-full h-8 text-xs bg-background border border-border rounded px-2 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
                    >
                      <option value="">— Pick a session —</option>
                      {sessions.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.source === 'relay' ? '⚡ ' : ''}{s.name || s.id.slice(0, 8)}
                          {s.project_id === projectId ? ' ★' : ''} [{s.source}]
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              )}
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 px-4 py-3 border-t border-border">
          <Button size="sm" variant="outline" onClick={onClose}>Cancel</Button>
          <Button size="sm" onClick={handleCreate} disabled={!title.trim()}>Create</Button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stage column — owns newCardOpen state so header + and bottom link both work
// ---------------------------------------------------------------------------
function StageColumn({
  stage,
  stageCards,
  stages,
  advancingIds,
  projectId,
  onNewCard,
  onAdvanceCard,
  onDeleteCard,
  onFireCard,
  onRenameStage,
  onDeleteStage,
  onToggleTerminal,
  sessionStatuses,
  onViewSession,
}: {
  stage: PipelineStage;
  stageCards: PipelineCard[];
  stages: PipelineStage[];
  advancingIds: Set<string>;
  projectId?: string;
  onNewCard: (stageId: string, data: NewCardData) => void;
  onAdvanceCard: (cardId: string, stageId?: string) => void;
  onDeleteCard: (cardId: string) => void;
  onFireCard: (cardId: string) => Promise<void>;
  onRenameStage: (stageId: string, name: string) => void;
  onDeleteStage: (stageId: string) => void;
  onToggleTerminal: (stageId: string, isTerminal: boolean) => void;
  sessionStatuses: Record<string, string>;
  onViewSession?: (sessionId: string) => void;
}) {
  const [newCardOpen, setNewCardOpen] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        const cardId = e.dataTransfer.getData('text/plain');
        if (cardId) onAdvanceCard(cardId, stage.id);
      }}
      className={cn(
        'flex flex-col w-72 shrink-0 rounded-lg border border-border overflow-hidden transition-colors',
        stage.is_terminal ? 'bg-emerald-500/5' : 'bg-muted/20',
        dragOver && 'border-blue-400/60 bg-blue-500/5',
      )}
    >
      <NewCardDialog
        open={newCardOpen}
        onClose={() => setNewCardOpen(false)}
        onCreate={(data) => onNewCard(stage.id, data)}
        projectId={projectId}
      />
      <StageHeader
        stage={stage}
        count={stageCards.length}
        onNewCard={() => setNewCardOpen(true)}
        onRename={(name) => onRenameStage(stage.id, name)}
        onDelete={() => onDeleteStage(stage.id)}
        onToggleTerminal={() => onToggleTerminal(stage.id, stage.is_terminal)}
      />
      <div className="flex-1 overflow-y-auto p-2 space-y-2 min-h-24">
        {stageCards.length === 0 && (
          <div className="border-2 border-dashed border-border rounded-md p-3 text-center text-muted-foreground text-xs">
            No cards
          </div>
        )}
        {stageCards.map((card) => (
          <KanbanCard
            key={card.id}
            card={card}
            stages={stages}
            isAdvancing={advancingIds.has(card.id)}
            onAdvance={onAdvanceCard}
            onDelete={onDeleteCard}
            onFire={onFireCard}
            sessionStatuses={sessionStatuses}
            onViewSession={onViewSession}
          />
        ))}
        {!stage.is_terminal && (
          <button
            className="w-full text-left px-2 py-1.5 text-[10px] text-muted-foreground hover:text-foreground flex items-center gap-1.5 rounded hover:bg-muted/30 transition-colors"
            onClick={() => setNewCardOpen(true)}
          >
            <Plus className="h-3 w-3" /> Add card
          </button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// New Stage inline creator (at end of board)
// ---------------------------------------------------------------------------
function NewStageInput({ onCreate }: { onCreate: (name: string) => void }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');

  const handleCreate = () => {
    const n = name.trim();
    if (!n) return;
    onCreate(n);
    setName('');
    setOpen(false);
  };

  if (!open) {
    return (
      <Button
        variant="outline"
        size="sm"
        className="h-8 gap-1.5 shrink-0 text-xs border-dashed"
        onClick={() => setOpen(true)}
      >
        <Plus className="h-3.5 w-3.5" /> New Stage
      </Button>
    );
  }

  return (
    <div className="flex items-center gap-1.5 shrink-0">
      <Input
        autoFocus
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter') handleCreate();
          if (e.key === 'Escape') { setOpen(false); setName(''); }
        }}
        placeholder="Stage name…"
        className="h-8 text-xs w-36"
      />
      <Button size="sm" className="h-8 px-2" onClick={handleCreate} disabled={!name.trim()}>
        <Check className="h-3.5 w-3.5" />
      </Button>
      <Button size="sm" variant="ghost" className="h-8 px-2" onClick={() => { setOpen(false); setName(''); }}>
        <X className="h-3.5 w-3.5" />
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main KanbanBoard
// ---------------------------------------------------------------------------
export function KanbanBoard({
  projectId,
  refreshKey = 0,
  onViewSession,
}: {
  projectId?: string;
  onNewEntry?: (columnId: string) => void; // kept for backwards compat, unused
  refreshKey?: number;
  onViewSession?: (sessionId: string) => void;
}) {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [selected, setSelected] = useState<Pipeline | null>(null);
  const [cards, setCards] = useState<PipelineCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [advancingIds, setAdvancingIds] = useState<Set<string>>(new Set());
  const [showNewPipeline, setShowNewPipeline] = useState(false);
  const [sessionStatuses, setSessionStatuses] = useState<Record<string, string>>({});

  // Poll session statuses for cards that have target_session_id
  const trackedSessionIds = useMemo(
    () => [...new Set(cards.map((c) => c.target_session_id).filter(Boolean) as string[])],
    [cards],
  );

  useEffect(() => {
    if (trackedSessionIds.length === 0) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const sessions = await listSessions();
        if (cancelled) return;
        const map: Record<string, string> = {};
        for (const s of sessions) {
          if (trackedSessionIds.includes(s.id)) {
            map[s.id] = s.status;
          }
        }
        setSessionStatuses(map);
      } catch { /* daemon may be down */ }
    };
    poll();
    const iv = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(iv); };
  }, [trackedSessionIds]);

  // ─── Load pipelines ────────────────────────────────────────────────────
  const loadPipelines = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const ps = await listPipelines(projectId);
      setPipelines(ps);
      if (ps.length > 0 && !selected) {
        setSelected(ps[0]);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load pipelines');
    } finally {
      setLoading(false);
    }
  }, [projectId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { loadPipelines(); }, [loadPipelines, refreshKey]);

  // ─── Load cards for selected pipeline ─────────────────────────────────
  const loadCards = useCallback(async () => {
    if (!selected) return;
    try {
      const cs = await listCards({ pipeline_id: selected.id });
      setCards(cs);
    } catch (e) {
      console.error('Failed to load cards:', e);
    }
  }, [selected]);

  useEffect(() => { loadCards(); }, [loadCards]);

  // ─── Pipeline operations ───────────────────────────────────────────────
  const handlePipelineCreated = useCallback(async (p: Pipeline) => {
    const ps = await listPipelines(projectId);
    setPipelines(ps);
    setSelected(ps.find((x) => x.id === p.id) ?? p);
    setShowNewPipeline(false);
  }, [projectId]);

  const handleRenamePipeline = useCallback(async (name: string) => {
    if (!selected) return;
    try {
      const updated = await updatePipeline(selected.id, { name });
      setPipelines((prev) => prev.map((p) => p.id === selected.id ? { ...p, name } : p));
      setSelected((prev) => prev ? { ...prev, name: updated.name } : prev);
    } catch (e) {
      console.error('Rename pipeline failed:', e);
    }
  }, [selected]);

  const handleDeletePipeline = useCallback(async () => {
    if (!selected) return;
    if (!confirm(`Delete pipeline "${selected.name}" and all its cards?`)) return;
    try {
      await deletePipeline(selected.id);
      const ps = await listPipelines(projectId);
      setPipelines(ps);
      setSelected(ps[0] ?? null);
      setCards([]);
    } catch (e) {
      console.error('Delete pipeline failed:', e);
    }
  }, [selected, projectId]);

  // ─── Stage operations ──────────────────────────────────────────────────
  const stages = selected?.stages ?? [];
  const sortedStages = [...stages].sort((a, b) => a.stage_order - b.stage_order);

  const handleAddStage = useCallback(async (name: string) => {
    if (!selected) return;
    const maxOrder = Math.max(...sortedStages.map((s) => s.stage_order), -1);
    try {
      const stage = await addStage(selected.id, {
        name,
        stage_order: maxOrder + 1,
        auto_advance: true,
        is_terminal: false,
      });
      setSelected((prev) => prev ? { ...prev, stages: [...prev.stages, stage] } : prev);
    } catch (e) {
      console.error('Add stage failed:', e);
    }
  }, [selected, sortedStages]);

  const handleRenameStage = useCallback(async (stageId: string, name: string) => {
    try {
      await updateStage(stageId, { name });
      setSelected((prev) => prev ? {
        ...prev,
        stages: prev.stages.map((s) => s.id === stageId ? { ...s, name } : s),
      } : prev);
    } catch (e) {
      console.error('Rename stage failed:', e);
    }
  }, []);

  const handleDeleteStage = useCallback(async (stageId: string) => {
    const hasCards = cards.some((c) => c.current_stage_id === stageId);
    if (hasCards) {
      alert('Cannot delete a stage that has cards. Move or delete the cards first.');
      return;
    }
    try {
      await deleteStage(stageId);
      setSelected((prev) => prev ? {
        ...prev,
        stages: prev.stages.filter((s) => s.id !== stageId),
      } : prev);
    } catch (e) {
      console.error('Delete stage failed:', e);
    }
  }, [cards]);

  const handleToggleTerminal = useCallback(async (stageId: string, isTerminal: boolean) => {
    try {
      await updateStage(stageId, { is_terminal: !isTerminal });
      setSelected((prev) => prev ? {
        ...prev,
        stages: prev.stages.map((s) => s.id === stageId ? { ...s, is_terminal: !isTerminal } : s),
      } : prev);
    } catch (e) {
      console.error('Toggle terminal failed:', e);
    }
  }, []);

  // ─── Card operations ───────────────────────────────────────────────────
  const handleNewCard = useCallback(async (stageId: string, data: NewCardData) => {
    if (!selected) return;
    try {
      const card = await createCard({
        pipeline_id: selected.id,
        project_id: projectId,
        title: data.title,
        stage_id: stageId,
        prompt_text: data.prompt_text,
        skill_id: data.skill_id,
        gate_id: data.gate_id,
        target_session_id: data.target_session_id,
        target_cwd: data.target_cwd,
        use_helper_session: data.use_helper_session,
      });
      setCards((prev) => [...prev, card]);
    } catch (e) {
      console.error('Create card failed:', e);
    }
  }, [selected, projectId]);

  const handleAdvanceCard = useCallback(async (cardId: string, stageId?: string) => {
    setAdvancingIds((prev) => new Set(prev).add(cardId));
    try {
      const updated = await advanceCard(cardId, stageId);
      setCards((prev) => prev.map((c) => c.id === cardId ? updated : c));
    } catch (e) {
      console.error('Advance card failed:', e);
    } finally {
      setAdvancingIds((prev) => {
        const next = new Set(prev);
        next.delete(cardId);
        return next;
      });
    }
  }, []);

  const handleDeleteCard = useCallback(async (cardId: string) => {
    try {
      await deleteCard(cardId);
      setCards((prev) => prev.filter((c) => c.id !== cardId));
    } catch (e) {
      console.error('Delete card failed:', e);
    }
  }, []);

  const handleFireCard = useCallback(async (cardId: string) => {
    try {
      const result = await fireCard(cardId);
      // Refresh card data to pick up fire_count, last_fired_at, target_session_id
      if (selected) {
        const cs = await listCards({ pipeline_id: selected.id });
        setCards(cs);
      } else if (result.session_id) {
        setCards((prev) => prev.map((c) =>
          c.id === cardId ? { ...c, target_session_id: result.session_id } : c
        ));
      }
    } catch (e) {
      console.error('Fire card failed:', e);
    }
  }, [selected]);

  // ─── Pipeline rename inline ────────────────────────────────────────────
  const [renamingPipeline, setRenamingPipeline] = useState(false);
  const [pipelineDraft, setPipelineDraft] = useState('');

  const commitPipelineRename = async () => {
    const n = pipelineDraft.trim();
    if (n && n !== selected?.name) await handleRenamePipeline(n);
    setRenamingPipeline(false);
  };

  // ─── Render ────────────────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground text-sm">
        Loading…
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-2">
          <p className="text-sm text-destructive">{error}</p>
          <Button size="sm" variant="outline" onClick={loadPipelines}>Retry</Button>
        </div>
      </div>
    );
  }

  if (pipelines.length === 0 || showNewPipeline) {
    return (
      <NewPipelinePrompt
        projectId={projectId}
        onCreated={handlePipelineCreated}
      />
    );
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* Toolbar */}
      <div className="shrink-0 flex items-center gap-2 px-4 py-2 border-b border-border">
        <PipelineSelector
          pipelines={pipelines}
          selected={selected}
          onSelect={setSelected}
          onNewPipeline={() => setShowNewPipeline(true)}
        />

        {selected && (
          <>
            {renamingPipeline ? (
              <div className="flex items-center gap-1">
                <Input
                  autoFocus
                  value={pipelineDraft}
                  onChange={(e) => setPipelineDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitPipelineRename();
                    if (e.key === 'Escape') setRenamingPipeline(false);
                  }}
                  className="h-7 text-xs w-40"
                />
                <Button size="sm" variant="ghost" className="h-7 px-2" onClick={commitPipelineRename}>
                  <Check className="h-3 w-3" />
                </Button>
                <Button size="sm" variant="ghost" className="h-7 px-2" onClick={() => setRenamingPipeline(false)}>
                  <X className="h-3 w-3" />
                </Button>
              </div>
            ) : (
              <div className="flex items-center gap-1">
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground"
                  onClick={() => { setPipelineDraft(selected.name); setRenamingPipeline(true); }}
                  title="Rename pipeline"
                >
                  <Pencil className="h-3 w-3" />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                  onClick={handleDeletePipeline}
                  title="Delete pipeline"
                >
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            )}
          </>
        )}
      </div>

      {/* Board */}
      <div className="flex-1 overflow-x-auto overflow-y-hidden p-4">
        <div className="flex gap-3 h-full items-start">
          {sortedStages.map((stage) => (
            <StageColumn
              key={stage.id}
              stage={stage}
              stageCards={cards.filter((c) => c.current_stage_id === stage.id)}
              stages={sortedStages}
              advancingIds={advancingIds}
              projectId={projectId}
              onNewCard={handleNewCard}
              onAdvanceCard={handleAdvanceCard}
              onDeleteCard={handleDeleteCard}
              onFireCard={handleFireCard}
              onRenameStage={handleRenameStage}
              onDeleteStage={handleDeleteStage}
              onToggleTerminal={handleToggleTerminal}
              sessionStatuses={sessionStatuses}
              onViewSession={onViewSession}
            />
          ))}

          {/* New stage button */}
          <div className="shrink-0 flex items-start pt-1">
            <NewStageInput onCreate={handleAddStage} />
          </div>
        </div>
      </div>
    </div>
  );
}
