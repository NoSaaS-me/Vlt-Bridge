/**
 * TranscriptOverlay — Dialog overlay for reading/searching/editing
 * Claude Code JSONL session transcripts.
 *
 * Opened when clicking a discovered (non-relay) session in the sidebar.
 */
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Search, ChevronDown, ChevronRight, Pencil, Save, X,
  User, Bot, Monitor, Wrench, Eye, FileText,
} from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import {
  type AgentSession,
  type TranscriptEntry,
  type ContentBlock,
  fetchTranscript,
  saveTranscript,
} from '@/services/daemon-api';
import { shortPath, timeAgo } from './utils';

// ---------------------------------------------------------------------------
// Content rendering helpers
// ---------------------------------------------------------------------------

function extractText(content: string | ContentBlock[] | undefined): string {
  if (!content) return '';
  if (typeof content === 'string') return content;
  return content
    .map((b) => {
      if (b.type === 'text') return b.text ?? '';
      if (b.type === 'thinking') return `[thinking] ${(b.thinking ?? '').slice(0, 200)}`;
      if (b.type === 'tool_use') return `[tool: ${b.name}]`;
      if (b.type === 'tool_result') {
        const c = b.content;
        const text = typeof c === 'string' ? c : JSON.stringify(c);
        return `[result] ${text.slice(0, 300)}`;
      }
      return '';
    })
    .filter(Boolean)
    .join(' ');
}

/** Role badge color + icon */
function roleMeta(type: string, role?: string) {
  if (type === 'user' || role === 'user')
    return { label: 'User', color: 'bg-blue-500/20 text-blue-400', Icon: User };
  if (type === 'assistant' || role === 'assistant')
    return { label: 'Assistant', color: 'bg-emerald-500/20 text-emerald-400', Icon: Bot };
  if (type === 'system')
    return { label: 'System', color: 'bg-amber-500/20 text-amber-400', Icon: Monitor };
  return { label: type, color: 'bg-muted text-muted-foreground', Icon: FileText };
}

// ---------------------------------------------------------------------------
// ContentBlockView — renders a single content block
// ---------------------------------------------------------------------------

function ContentBlockView({ block }: { block: ContentBlock }) {
  const [expanded, setExpanded] = useState(false);

  if (block.type === 'text') {
    return <p className="text-sm whitespace-pre-wrap break-words">{block.text}</p>;
  }

  if (block.type === 'thinking') {
    return (
      <div className="text-xs text-muted-foreground/60">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 hover:text-muted-foreground transition-colors"
        >
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          <Eye className="h-3 w-3" />
          <span className="italic">Thinking...</span>
        </button>
        {expanded && (
          <pre className="mt-1 ml-5 p-2 rounded bg-muted/30 text-[11px] whitespace-pre-wrap max-h-64 overflow-y-auto">
            {block.thinking}
          </pre>
        )}
      </div>
    );
  }

  if (block.type === 'tool_use') {
    return (
      <div className="text-xs">
        <div className="flex items-center gap-1.5 text-amber-400">
          <Wrench className="h-3 w-3" />
          <span className="font-semibold">{block.name}</span>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-0.5 text-muted-foreground/60 hover:text-muted-foreground text-[10px] flex items-center gap-1"
        >
          {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
          input
        </button>
        {expanded && (
          <pre className="mt-1 ml-5 p-2 rounded bg-muted/30 text-[11px] whitespace-pre-wrap max-h-48 overflow-y-auto">
            {typeof block.input === 'string' ? block.input : JSON.stringify(block.input, null, 2)}
          </pre>
        )}
      </div>
    );
  }

  if (block.type === 'tool_result') {
    const text = typeof block.content === 'string'
      ? block.content
      : JSON.stringify(block.content, null, 2);
    const truncated = text.length > 500;
    return (
      <div className="text-xs">
        <pre className={cn(
          'p-2 rounded bg-muted/30 text-[11px] whitespace-pre-wrap overflow-y-auto',
          expanded ? 'max-h-96' : 'max-h-24',
        )}>
          {expanded ? text : text.slice(0, 500)}
        </pre>
        {truncated && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-[10px] text-blue-400 hover:underline mt-0.5"
          >
            {expanded ? 'Show less' : `Show all (${text.length.toLocaleString()} chars)`}
          </button>
        )}
      </div>
    );
  }

  // Unknown block type — show raw
  return (
    <pre className="text-[10px] text-muted-foreground/50 whitespace-pre-wrap">
      {JSON.stringify(block, null, 2)}
    </pre>
  );
}

// ---------------------------------------------------------------------------
// EntryCard — a single transcript entry
// ---------------------------------------------------------------------------

function EntryCard({
  entry,
  isEditing,
  editJson,
  onStartEdit,
  onEditChange,
  onCancelEdit,
  onApplyEdit,
}: {
  entry: TranscriptEntry;
  isEditing: boolean;
  editJson: string;
  onStartEdit: () => void;
  onEditChange: (json: string) => void;
  onCancelEdit: () => void;
  onApplyEdit: () => void;
}) {
  const { label, color, Icon } = roleMeta(entry.type, entry.message?.role);
  const content = entry.message?.content;
  const ts = entry.timestamp
    ? new Date(entry.timestamp).toLocaleTimeString('en', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    })
    : null;

  const [rawExpanded, setRawExpanded] = useState(false);

  return (
    <div className="group rounded-md border border-border/50 hover:border-border transition-colors">
      {/* Entry header */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/20">
        <span className={cn('flex items-center gap-1 text-[10px] font-bold uppercase px-1.5 py-0.5 rounded', color)}>
          <Icon className="h-3 w-3" />
          {label}
        </span>
        {ts && <span className="text-[10px] text-muted-foreground/50 tabular-nums">{ts}</span>}
        <span className="text-[10px] text-muted-foreground/30">#{entry.lineIndex}</span>
        <div className="flex-1" />
        <button
          onClick={() => setRawExpanded(!rawExpanded)}
          className="text-[10px] text-muted-foreground/40 hover:text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity"
          title="View raw JSON"
        >
          {'{ }'}
        </button>
        <button
          onClick={onStartEdit}
          className="text-muted-foreground/40 hover:text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity"
          title="Edit entry"
        >
          <Pencil className="h-3 w-3" />
        </button>
      </div>

      {/* Content or editor */}
      <div className="px-3 py-2 space-y-1.5">
        {isEditing ? (
          <div className="space-y-2">
            <textarea
              value={editJson}
              onChange={(e) => onEditChange(e.target.value)}
              className="w-full h-48 p-2 rounded border border-border bg-background font-mono text-[11px] resize-y focus:outline-none focus:ring-1 focus:ring-blue-500"
              spellCheck={false}
            />
            <div className="flex gap-1.5 justify-end">
              <Button size="sm" variant="ghost" onClick={onCancelEdit} className="h-6 text-[10px] px-2">
                Cancel
              </Button>
              <Button size="sm" onClick={onApplyEdit} className="h-6 text-[10px] px-2">
                Apply
              </Button>
            </div>
          </div>
        ) : (
          <>
            {typeof content === 'string' ? (
              <p className="text-sm whitespace-pre-wrap break-words">{content}</p>
            ) : Array.isArray(content) ? (
              content.map((block, i) => (
                <ContentBlockView key={i} block={block} />
              ))
            ) : (
              <p className="text-xs text-muted-foreground/50 italic">No message content</p>
            )}
          </>
        )}

        {/* Raw JSON (collapsible) */}
        {rawExpanded && !isEditing && (
          <>
            <Separator className="my-1" />
            <pre className="text-[10px] text-muted-foreground/50 whitespace-pre-wrap max-h-48 overflow-y-auto bg-muted/20 p-2 rounded">
              {JSON.stringify(entry.raw, null, 2)}
            </pre>
          </>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// TranscriptOverlay — main component
// ---------------------------------------------------------------------------

export function TranscriptOverlay({
  session,
  open,
  onClose,
}: {
  session: AgentSession;
  open: boolean;
  onClose: () => void;
}) {
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);
  const [allRawEntries, setAllRawEntries] = useState<Record<string, unknown>[]>([]);
  const [totalLines, setTotalLines] = useState(0);
  const [transcriptPath, setTranscriptPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editJson, setEditJson] = useState('');
  const [modifiedIndices, setModifiedIndices] = useState<Set<number>>(new Set());
  const [isSaving, setIsSaving] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  const isDirty = modifiedIndices.size > 0;

  // Load transcript when opened
  useEffect(() => {
    if (!open) return;
    setIsLoading(true);
    setError(null);
    setSearchQuery('');
    setEditingIndex(null);
    setModifiedIndices(new Set());

    fetchTranscript(session.id)
      .then((data) => {
        setEntries(data.entries);
        setAllRawEntries(data.entries.map((e) => e.raw));
        setTotalLines(data.total_lines);
        setTranscriptPath(data.path);
      })
      .catch((err) => setError(err.message))
      .finally(() => setIsLoading(false));
  }, [open, session.id]);

  // Filtered entries for search
  const filteredEntries = useMemo(() => {
    if (!searchQuery.trim()) return entries;
    const q = searchQuery.toLowerCase();
    return entries.filter((e) => {
      const text = extractText(e.message?.content);
      return text.toLowerCase().includes(q)
        || e.type.toLowerCase().includes(q)
        || (e.timestamp ?? '').includes(q);
    });
  }, [entries, searchQuery]);

  // Edit handlers
  const startEdit = useCallback((idx: number) => {
    const entry = entries[idx];
    setEditingIndex(idx);
    setEditJson(JSON.stringify(entry.raw, null, 2));
  }, [entries]);

  const cancelEdit = useCallback(() => {
    setEditingIndex(null);
    setEditJson('');
  }, []);

  const applyEdit = useCallback(() => {
    if (editingIndex === null) return;
    try {
      const parsed = JSON.parse(editJson);
      // Update both entries and allRawEntries
      setEntries((prev) => {
        const next = [...prev];
        next[editingIndex] = {
          ...next[editingIndex],
          raw: parsed,
          type: parsed.type ?? next[editingIndex].type,
          message: parsed.message ?? next[editingIndex].message,
          timestamp: parsed.timestamp ?? next[editingIndex].timestamp,
        };
        return next;
      });
      setAllRawEntries((prev) => {
        const next = [...prev];
        next[editingIndex] = parsed;
        return next;
      });
      setModifiedIndices((prev) => new Set(prev).add(editingIndex));
      setEditingIndex(null);
      setEditJson('');
    } catch {
      // Invalid JSON — don't apply
    }
  }, [editingIndex, editJson]);

  // Save handler
  const handleSave = useCallback(async () => {
    setIsSaving(true);
    try {
      await saveTranscript(session.id, allRawEntries);
      setModifiedIndices(new Set());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed');
    } finally {
      setIsSaving(false);
    }
  }, [session.id, allRawEntries]);

  const handleClose = useCallback(() => {
    if (isDirty && !confirm('You have unsaved changes. Discard them?')) return;
    onClose();
  }, [isDirty, onClose]);

  return (
    <Dialog open={open} onOpenChange={(isOpen) => { if (!isOpen) handleClose(); }}>
      <DialogContent
        className="max-w-[92vw] w-[92vw] h-[88vh] flex flex-col p-0 gap-0"
        onInteractOutside={(e) => { if (isDirty) e.preventDefault(); }}
      >
        {/* Header */}
        <div className="px-5 pt-4 pb-3 space-y-2 shrink-0">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-sm">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <span className="font-mono">{session.name || session.id.slice(0, 12)}</span>
              <span className="text-xs text-muted-foreground font-normal">
                {shortPath(session.cwd)}
              </span>
            </DialogTitle>
            <DialogDescription className="text-[11px] flex items-center gap-3">
              <span>{totalLines} total lines</span>
              <span>{entries.length} conversation entries</span>
              <span className="text-muted-foreground/50">{timeAgo(session.last_activity)}</span>
              {transcriptPath && (
                <span className="text-muted-foreground/30 font-mono truncate max-w-[300px]" title={transcriptPath}>
                  {transcriptPath}
                </span>
              )}
            </DialogDescription>
          </DialogHeader>

          {/* Search */}
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground/40" />
            <Input
              ref={searchRef}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search transcript..."
              className="pl-8 h-8 text-xs"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery('')}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground/40 hover:text-muted-foreground"
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>

        <Separator />

        {/* Entry list */}
        <ScrollArea className="flex-1 min-h-0">
          <div className="p-4 space-y-2">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="h-6 w-6 border-2 border-muted-foreground/20 border-t-muted-foreground rounded-full animate-spin" />
                <p className="mt-3 text-xs text-muted-foreground">Loading transcript...</p>
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <p className="text-sm text-red-400 mb-1">Failed to load transcript</p>
                <p className="text-xs text-muted-foreground">{error}</p>
              </div>
            ) : filteredEntries.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-20 text-center">
                <p className="text-xs text-muted-foreground">
                  {searchQuery ? 'No entries match your search' : 'No conversation entries found'}
                </p>
              </div>
            ) : (
              filteredEntries.map((entry) => {
                // Find the real index in the entries array
                const realIdx = entries.indexOf(entry);
                return (
                  <EntryCard
                    key={entry.uuid ?? entry.lineIndex}
                    entry={entry}
                    isEditing={editingIndex === realIdx}
                    editJson={editingIndex === realIdx ? editJson : ''}
                    onStartEdit={() => startEdit(realIdx)}
                    onEditChange={setEditJson}
                    onCancelEdit={cancelEdit}
                    onApplyEdit={applyEdit}
                  />
                );
              })
            )}
          </div>
        </ScrollArea>

        {/* Footer */}
        <Separator />
        <div className="px-5 py-3 flex items-center justify-between shrink-0">
          <div className="text-[10px] text-muted-foreground/50">
            {searchQuery && `${filteredEntries.length} / ${entries.length} entries`}
            {isDirty && (
              <span className="ml-2 text-amber-400">
                {modifiedIndices.size} modified {modifiedIndices.size === 1 ? 'entry' : 'entries'}
              </span>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={handleClose}>
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!isDirty || isSaving}
              className={cn(!isDirty && 'opacity-50')}
            >
              {isSaving ? (
                <span className="flex items-center gap-1.5">
                  <div className="h-3 w-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Saving...
                </span>
              ) : (
                <span className="flex items-center gap-1.5">
                  <Save className="h-3 w-3" />
                  Save
                </span>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
