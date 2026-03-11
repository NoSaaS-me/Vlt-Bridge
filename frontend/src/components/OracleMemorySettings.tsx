/**
 * OracleMemorySettings — Memory tab content for the Settings page.
 *
 * Two sections:
 *   1. Conversation Threads — list past Oracle threads, view history, delete
 *   2. Memory Search — query Graphiti facts stored across sessions
 */
import { useState, useEffect, useCallback } from 'react';
import {
  Brain,
  MessageSquare,
  Search,
  Trash2,
  ChevronDown,
  ChevronRight,
  Clock,
  Loader2,
  AlertCircle,
  FolderOpen,
  RefreshCw,
  Info,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  listOracleThreads,
  getOracleThreadHistory,
  deleteOracleThread,
  searchOracleMemory,
  type OracleThreadSummary,
  type OracleThreadHistoryMessage,
} from '@/services/api';
import { useProjectContext } from '@/contexts/ProjectContext';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatRelativeTime(isoStr: string): string {
  try {
    const date = new Date(isoStr);
    const diffMs = Date.now() - date.getTime();
    const diffMin = Math.floor(diffMs / 60_000);
    if (diffMin < 1) return 'just now';
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffH = Math.floor(diffMin / 60);
    if (diffH < 24) return `${diffH}h ago`;
    const diffD = Math.floor(diffH / 24);
    if (diffD < 7) return `${diffD}d ago`;
    return date.toLocaleDateString();
  } catch {
    return isoStr;
  }
}

// ---------------------------------------------------------------------------
// Thread history drawer
// ---------------------------------------------------------------------------

function ThreadHistoryView({ threadId, onClose }: { threadId: string; onClose: () => void }) {
  const [messages, setMessages] = useState<OracleThreadHistoryMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getOracleThreadHistory(threadId)
      .then((r) => setMessages(r.messages))
      .catch((e) => setError(String(e?.message ?? e)))
      .finally(() => setLoading(false));
  }, [threadId]);

  return (
    <div className="mt-3 border rounded-md bg-muted/30">
      <div className="flex items-center justify-between px-3 py-2 border-b">
        <span className="text-xs font-medium text-muted-foreground">Conversation history</span>
        <Button variant="ghost" size="sm" className="h-6 text-xs" onClick={onClose}>
          Close
        </Button>
      </div>
      {loading && (
        <div className="flex items-center gap-2 p-4 text-sm text-muted-foreground">
          <Loader2 className="h-3 w-3 animate-spin" /> Loading…
        </div>
      )}
      {error && (
        <div className="p-3 text-xs text-destructive">{error}</div>
      )}
      {!loading && !error && messages.length === 0 && (
        <div className="p-3 text-xs text-muted-foreground">No messages stored in checkpoint.</div>
      )}
      {!loading && !error && messages.length > 0 && (
        <ScrollArea className="max-h-64">
          <div className="p-3 space-y-2">
            {messages.map((msg, i) => (
              <div key={i} className={`text-xs rounded p-2 ${
                msg.role === 'user'
                  ? 'bg-primary/10 border border-primary/20 ml-4'
                  : 'bg-background border mr-4'
              }`}>
                <p className="font-medium mb-1 text-muted-foreground capitalize">{msg.role}</p>
                <p className="whitespace-pre-wrap break-words leading-relaxed">{msg.content}</p>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Thread row
// ---------------------------------------------------------------------------

function ThreadRow({
  thread,
  onDeleted,
}: {
  thread: OracleThreadSummary;
  onDeleted: (id: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [deleting, setDeleting] = useState(false);

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!confirm('Delete this conversation thread? This cannot be undone.')) return;
    setDeleting(true);
    try {
      await deleteOracleThread(thread.thread_id);
      onDeleted(thread.thread_id);
    } catch (err) {
      alert(`Failed to delete: ${err}`);
      setDeleting(false);
    }
  };

  return (
    <div className="border rounded-md">
      <div
        className="flex items-center gap-3 px-3 py-2.5 cursor-pointer hover:bg-muted/40 transition-colors"
        onClick={() => setExpanded((v) => !v)}
      >
        <span className="text-muted-foreground">
          {expanded ? (
            <ChevronDown className="h-3.5 w-3.5" />
          ) : (
            <ChevronRight className="h-3.5 w-3.5" />
          )}
        </span>
        <MessageSquare className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        <span className="flex-1 text-sm truncate">
          {thread.title || <span className="italic text-muted-foreground">Untitled conversation</span>}
        </span>
        {thread.project_id && thread.project_id !== 'default' && (
          <Badge variant="outline" className="text-xs shrink-0">{thread.project_id}</Badge>
        )}
        <span className="text-xs text-muted-foreground shrink-0 flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {formatRelativeTime(thread.last_active_at)}
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0 text-muted-foreground hover:text-destructive"
          onClick={handleDelete}
          disabled={deleting}
        >
          {deleting ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
        </Button>
      </div>
      {expanded && (
        <div className="px-3 pb-3">
          <ThreadHistoryView
            threadId={thread.thread_id}
            onClose={() => setExpanded(false)}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function OracleMemorySettings() {
  const { selectedProjectId } = useProjectContext();

  // Threads
  const [threads, setThreads] = useState<OracleThreadSummary[]>([]);
  const [threadsLoading, setThreadsLoading] = useState(true);
  const [threadsError, setThreadsError] = useState<string | null>(null);
  const [threadsTotal, setThreadsTotal] = useState(0);

  // Memory search
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<string[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchMessage, setSearchMessage] = useState<string | null>(null);
  const [memoryAvailable, setMemoryAvailable] = useState<boolean | null>(null);

  const loadThreads = useCallback(async () => {
    setThreadsLoading(true);
    setThreadsError(null);
    try {
      const res = await listOracleThreads(50, 0);
      setThreads(res.threads);
      setThreadsTotal(res.total);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setThreadsError(msg);
    } finally {
      setThreadsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadThreads();
  }, [loadThreads]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setSearchLoading(true);
    setSearchResults([]);
    setSearchMessage(null);
    try {
      const res = await searchOracleMemory(
        searchQuery,
        selectedProjectId ?? 'default',
        20,
      );
      setMemoryAvailable(res.available);
      setSearchResults(res.facts);
      if (res.message) setSearchMessage(res.message);
      if (res.error) setSearchMessage(`Error: ${res.error}`);
    } catch (e: unknown) {
      setSearchMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setSearchLoading(false);
    }
  };

  const handleThreadDeleted = (id: string) => {
    setThreads((prev) => prev.filter((t) => t.thread_id !== id));
    setThreadsTotal((n) => Math.max(0, n - 1));
  };

  return (
    <div className="space-y-6">
      {/* ---- Thread history ---- */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5 text-muted-foreground" />
              <CardTitle>Conversation Threads</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              {!threadsLoading && (
                <span className="text-xs text-muted-foreground">
                  {threadsTotal} thread{threadsTotal !== 1 ? 's' : ''}
                </span>
              )}
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1"
                onClick={loadThreads}
                disabled={threadsLoading}
              >
                <RefreshCw className={`h-3 w-3 ${threadsLoading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
            </div>
          </div>
          <CardDescription>
            Past Oracle V2 conversations stored as durable LangGraph checkpoints.
            Click a thread to view its message history. Delete to free checkpoint storage.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {threadsLoading && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground py-6 justify-center">
              <Loader2 className="h-4 w-4 animate-spin" /> Loading threads…
            </div>
          )}
          {threadsError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{threadsError}</AlertDescription>
            </Alert>
          )}
          {!threadsLoading && !threadsError && threads.length === 0 && (
            <div className="flex flex-col items-center gap-2 py-8 text-muted-foreground">
              <FolderOpen className="h-8 w-8 opacity-40" />
              <p className="text-sm">No conversation threads yet.</p>
              <p className="text-xs opacity-70">
                Start chatting in the Oracle panel — threads appear here automatically.
              </p>
            </div>
          )}
          {!threadsLoading && !threadsError && threads.length > 0 && (
            <div className="space-y-2">
              {threads.map((t) => (
                <ThreadRow key={t.thread_id} thread={t} onDeleted={handleThreadDeleted} />
              ))}
              {threadsTotal > threads.length && (
                <p className="text-xs text-center text-muted-foreground pt-1">
                  Showing {threads.length} of {threadsTotal} threads
                </p>
              )}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ---- Memory fact search ---- */}
      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-muted-foreground" />
            <CardTitle>Cross-Session Memory</CardTitle>
          </div>
          <CardDescription>
            Search facts the Oracle has stored in its long-term memory (Graphiti knowledge graph).
            Facts are recalled automatically at the start of each turn when relevant.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {memoryAvailable === false && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription>
                Graphiti memory is not connected. Start FalkorDB and set{' '}
                <code className="text-xs">FALKORDB_URL</code> to enable cross-session memory.
              </AlertDescription>
            </Alert>
          )}

          <div className="flex gap-2">
            <Input
              placeholder="Search stored facts… e.g. 'rate limiter', 'auth middleware'"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1"
            />
            <Button onClick={handleSearch} disabled={searchLoading || !searchQuery.trim()} size="sm">
              {searchLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Search className="h-4 w-4" />
              )}
              <span className="ml-1.5">Search</span>
            </Button>
          </div>

          {searchMessage && !searchLoading && (
            <p className="text-xs text-muted-foreground">{searchMessage}</p>
          )}

          {searchResults.length > 0 && (
            <>
              <Separator />
              <div className="space-y-2">
                <p className="text-xs font-medium text-muted-foreground">
                  {searchResults.length} fact{searchResults.length !== 1 ? 's' : ''} recalled
                </p>
                <ScrollArea className="max-h-72">
                  <div className="space-y-2 pr-2">
                    {searchResults.map((fact, i) => (
                      <div
                        key={i}
                        className="text-sm border rounded-md px-3 py-2 bg-muted/30 leading-relaxed"
                      >
                        <span className="text-xs text-muted-foreground mr-2 font-mono">{i + 1}.</span>
                        {fact}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </>
          )}

          {!searchLoading && searchQuery && searchResults.length === 0 && memoryAvailable !== false && searchMessage === null && (
            <p className="text-sm text-muted-foreground text-center py-2">
              No facts found for this query.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
