/**
 * AgentsPage — Dashboard for Claude Code terminal session management.
 *
 * Layout:
 *   Left sidebar (48px) — nav icons (Agents, Cronban, Connectors)
 *   Main area          — 2×2 panel grid (Sessions, Terminal, Tasks, Events)
 *
 * Polls GET /vlt/api/sessions every 5s and GET /vlt/api/hooks/recent every 3s.
 * Terminal streams via WebSocket at /vlt/ws/sessions/{id}.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  Bot,
  Clock,
  Plug,
  Terminal,
  Activity,
  RefreshCw,
  WifiOff,
  Folder,
} from 'lucide-react';
import {
  listSessions,
  listHookEvents,
  parseDaemonTs,
  type AgentSession,
  type HookEvent,
} from '@/services/daemon-api';
import { cn } from '@/lib/utils';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';

// ---------------------------------------------------------------------------
// Nav
// ---------------------------------------------------------------------------

type NavSection = 'agents' | 'cronban' | 'connectors';

const NAV_ITEMS: { id: NavSection; icon: React.ElementType; label: string }[] = [
  { id: 'agents', icon: Bot, label: 'Agents' },
  { id: 'cronban', icon: Clock, label: 'Cronban' },
  { id: 'connectors', icon: Plug, label: 'Connectors' },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusColor(status: AgentSession['status']): string {
  switch (status) {
    case 'thinking':  return 'text-blue-400 bg-blue-500/15 border-blue-500/30';
    case 'executing': return 'text-amber-400 bg-amber-500/15 border-amber-500/30';
    case 'idle':      return 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30';
    case 'dead':      return 'text-muted-foreground bg-muted/50 border-border';
    default:          return 'text-muted-foreground bg-muted/50 border-border';
  }
}

function isPulsing(s: AgentSession['status']) {
  return s === 'thinking' || s === 'executing';
}

function shortPath(cwd: string): string {
  if (!cwd) return '—';
  const parts = cwd.split('/');
  return parts.slice(-2).join('/') || cwd;
}

function timeAgo(isoStr: string): string {
  const diff = Date.now() - parseDaemonTs(isoStr).getTime();
  const secs = Math.floor(Math.abs(diff) / 1000);
  const sign = diff < 0 ? '+' : '';  // future-dated shouldn't happen but handle gracefully
  if (secs < 60) return `${sign}${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${sign}${mins}m ago`;
  return `${sign}${Math.floor(mins / 60)}h ago`;
}

// ---------------------------------------------------------------------------
// Session Card
// ---------------------------------------------------------------------------

function SessionCard({
  session,
  selected,
  onSelect,
}: {
  session: AgentSession;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={cn(
        'w-full text-left rounded-md border p-3 space-y-2 transition-all duration-200',
        selected
          ? 'border-blue-500/60 bg-blue-500/8 ring-1 ring-blue-500/30'
          : 'border-border hover:border-border/80 hover:bg-muted/30',
        session.status === 'dead' && 'opacity-50',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Bot className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <span className="text-sm font-medium truncate">{session.name || `pid:${session.pid}`}</span>
        </div>
        <div className={cn(
          'flex items-center gap-1 text-xs px-1.5 py-0.5 rounded border font-medium shrink-0',
          statusColor(session.status),
        )}>
          {isPulsing(session.status) && (
            <span className="h-1.5 w-1.5 rounded-full bg-current animate-pulse" />
          )}
          {session.status.charAt(0).toUpperCase() + session.status.slice(1)}
        </div>
      </div>

      <div className="flex items-center gap-1.5 text-xs text-muted-foreground font-mono">
        <Folder className="h-3 w-3 shrink-0" />
        <span className="truncate" title={session.cwd}>{shortPath(session.cwd)}</span>
      </div>

      <div className="flex items-center gap-2">
        {session.model && (
          <span className="text-[10px] text-muted-foreground font-mono truncate flex-1">
            {session.model.split('/').pop()}
          </span>
        )}
        {session.ctx_pct != null && (
          <div className="flex items-center gap-1 shrink-0">
            <div className="h-1 w-16 rounded-full bg-muted overflow-hidden">
              <div
                className={cn(
                  'h-full rounded-full transition-all duration-500',
                  session.ctx_pct > 80 ? 'bg-red-500' :
                  session.ctx_pct > 60 ? 'bg-amber-500' : 'bg-blue-500',
                )}
                style={{ width: `${Math.min(100, session.ctx_pct)}%` }}
              />
            </div>
            <span className="text-[10px] text-muted-foreground tabular-nums">
              {Math.round(session.ctx_pct)}%
            </span>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between text-[10px] text-muted-foreground">
        <span className="font-mono">⬡ {session.source}</span>
        <span>{timeAgo(session.last_activity)}</span>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Sessions Panel
// ---------------------------------------------------------------------------

function SessionsPanel({
  sessions,
  isLoading,
  daemonOnline,
  selectedId,
  onSelect,
}: {
  sessions: AgentSession[];
  isLoading: boolean;
  daemonOnline: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const active = sessions.filter((s) => s.status !== 'dead');

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 pt-3 pb-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Bot className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sessions</span>
        </div>
        <div className="flex items-center gap-2">
          {daemonOnline ? (
            <span className="flex items-center gap-1 text-[10px] text-emerald-400">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
              daemon live
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
              <WifiOff className="h-3 w-3" />
              daemon offline
            </span>
          )}
          <Badge variant="secondary" className="text-[10px] h-4 px-1">{active.length}</Badge>
        </div>
      </div>
      <Separator />
      <ScrollArea className="flex-1">
        <div className="p-3 space-y-2">
          {isLoading ? (
            Array.from({ length: 2 }).map((_, i) => (
              <div key={i} className="h-20 rounded-md border border-border bg-muted/30 animate-pulse" />
            ))
          ) : !daemonOnline ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <WifiOff className="h-8 w-8 text-muted-foreground/40 mb-3" />
              <p className="text-xs text-muted-foreground">Daemon not running</p>
              <div className="mt-2 space-y-1 text-[10px] text-muted-foreground/60">
                <code className="block font-mono bg-muted px-1.5 py-1 rounded text-foreground/70">vlt daemon start</code>
                <code className="block font-mono bg-muted px-1.5 py-1 rounded text-foreground/70">vlt session-relay</code>
              </div>
            </div>
          ) : active.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <Bot className="h-8 w-8 text-muted-foreground/40 mb-3" />
              <p className="text-xs text-muted-foreground">No active sessions</p>
              <p className="text-[10px] text-muted-foreground/60 mt-1 max-w-[180px]">
                <code className="font-mono bg-muted px-0.5 rounded">vlt session-relay</code>
              </p>
            </div>
          ) : (
            active.map((s) => (
              <SessionCard
                key={s.id}
                session={s}
                selected={s.id === selectedId}
                onSelect={() => onSelect(s.id)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Terminal Panel (xterm.js)
// ---------------------------------------------------------------------------

function TerminalPanel({
  sessions,
  selectedId,
  onSelectId,
}: {
  sessions: AgentSession[];
  selectedId: string | null;
  onSelectId: (id: string) => void;
}) {
  const termRef = useRef<HTMLDivElement>(null);
  const xtermRef = useRef<import('@xterm/xterm').Terminal | null>(null);
  const fitRef = useRef<import('@xterm/addon-fit').FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const activeSessionId = useRef<string | null>(null);
  const [xtermReady, setXtermReady] = useState(false);

  const relaySessions = sessions.filter((s) => s.source === 'relay' && s.status !== 'dead');
  const selectedSession = sessions.find((s) => s.id === selectedId);

  // Initialize xterm once — termRef is always mounted so this always succeeds
  useEffect(() => {
    if (!termRef.current) return;

    import('@xterm/xterm').then(({ Terminal }) => {
      import('@xterm/addon-fit').then(({ FitAddon }) => {
        if (xtermRef.current) return; // already initialized

        const term = new Terminal({
          theme: {
            background: '#0a0f1a',
            foreground: '#c9d1d9',
            cursor: '#58a6ff',
            black: '#0d1117',
            brightBlack: '#6e7681',
            red: '#f85149',
            brightRed: '#ff7b72',
            green: '#3fb950',
            brightGreen: '#56d364',
            yellow: '#d29922',
            brightYellow: '#e3b341',
            blue: '#58a6ff',
            brightBlue: '#79c0ff',
            magenta: '#bc8cff',
            brightMagenta: '#d2a8ff',
            cyan: '#39c5cf',
            brightCyan: '#56d4dd',
            white: '#b1bac4',
            brightWhite: '#cdd9e5',
          },
          fontFamily: '"Cascadia Code", "Fira Code", "JetBrains Mono", monospace',
          fontSize: 13,
          lineHeight: 1.2,
          cursorBlink: true,
          allowProposedApi: true,
        });

        const fitAddon = new FitAddon();
        term.loadAddon(fitAddon);
        term.open(termRef.current!);
        fitAddon.fit();

        xtermRef.current = term;
        fitRef.current = fitAddon;
        setXtermReady(true);

        // Handle user input → send via WebSocket
        term.onData((data) => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'inject', data }));
          }
        });
      });
    });

    // Resize observer
    const observer = new ResizeObserver(() => fitRef.current?.fit());
    observer.observe(termRef.current);
    return () => {
      observer.disconnect();
    };
  }, []);

  // Connect/disconnect WebSocket when selectedId changes or xterm becomes ready
  useEffect(() => {
    if (!selectedId || !xtermReady || !xtermRef.current) return;
    if (activeSessionId.current === selectedId) return;

    // Close existing WS
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    activeSessionId.current = selectedId;
    const term = xtermRef.current;
    term.reset();
    term.writeln(`\x1b[2m\x1b[36m── connecting to session ${selectedId.slice(0, 8)} ──\x1b[0m`);

    // Build WebSocket URL — proxy strips /vlt, so /vlt/ws/... → ws://localhost:8765/ws/...
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${window.location.host}/vlt/ws/sessions/${selectedId}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      term.writeln(`\x1b[2m\x1b[32m── connected ──\x1b[0m`);
      fitRef.current?.fit();
      // Send current terminal dimensions — triggers SIGWINCH in Claude → TUI redraws
      const dims = fitRef.current?.proposeDimensions();
      if (dims) {
        ws.send(JSON.stringify({ type: 'resize', cols: dims.cols, rows: dims.rows }));
      }
    };

    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        const bytes = new Uint8Array(ev.data);
        term.write(bytes);
      } else if (typeof ev.data === 'string') {
        // Could be JSON control messages
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'data') term.write(msg.data);
        } catch {
          term.write(ev.data);
        }
      }
    };

    ws.onerror = () => {
      term.writeln(`\x1b[31m── WebSocket error ──\x1b[0m`);
    };

    ws.onclose = (ev) => {
      if (activeSessionId.current === selectedId) {
        term.writeln(`\x1b[2m── disconnected (${ev.code}) ──\x1b[0m`);
      }
    };

    return () => {
      ws.close();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, xtermReady]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      xtermRef.current?.dispose();
    };
  }, []);

  return (
    <div className="h-full flex flex-col bg-[#0a0f1a]">
      {/* Header */}
      <div className="px-3 pt-2.5 pb-2 flex items-center justify-between shrink-0 bg-background border-b border-border">
        <div className="flex items-center gap-2">
          <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Terminal</span>
        </div>
        {/* Session selector chips */}
        <div className="flex items-center gap-1 overflow-x-auto max-w-[55%]">
          {relaySessions.length === 0 ? (
            <span className="text-[10px] text-muted-foreground">no relay sessions</span>
          ) : (
            relaySessions.map((s) => (
              <button
                key={s.id}
                onClick={() => onSelectId(s.id)}
                className={cn(
                  'flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono shrink-0 transition-colors',
                  selectedId === s.id
                    ? 'bg-blue-500/25 text-blue-300 border border-blue-500/40'
                    : 'bg-muted/40 text-muted-foreground hover:bg-muted border border-transparent',
                )}
              >
                <span className={cn(
                  'h-1.5 w-1.5 rounded-full shrink-0',
                  isPulsing(s.status) ? 'bg-amber-400 animate-pulse' : 'bg-emerald-400',
                )} />
                {s.name || s.id.slice(0, 8)}
              </button>
            ))
          )}
        </div>
      </div>

      {/* Terminal area — always in DOM so xterm can mount on first render */}
      <div className="flex-1 relative overflow-hidden">
        <div ref={termRef} className="absolute inset-0 p-1" />
        {/* Empty state overlay — shown when no relay session is active */}
        {(!selectedId || relaySessions.length === 0) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-4 bg-[#0a0f1a]">
            <Terminal className="h-10 w-10 text-muted-foreground/20 mb-4" />
            {relaySessions.length === 0 ? (
              <>
                <p className="text-xs text-muted-foreground mb-1">No relay sessions</p>
                <p className="text-[10px] text-muted-foreground/60 max-w-[200px]">
                  Start with <code className="font-mono bg-muted/50 px-0.5 rounded">vlt session-relay</code> to stream live output
                </p>
              </>
            ) : (
              <p className="text-xs text-muted-foreground">Select a session to connect</p>
            )}
          </div>
        )}
      </div>

      {/* Status bar */}
      {selectedSession && relaySessions.length > 0 && (
        <div className="px-3 py-1 flex items-center gap-2 text-[10px] text-muted-foreground border-t border-border/50 bg-background shrink-0">
          <span className="font-mono truncate">{selectedSession.cwd || '~'}</span>
          <span className="ml-auto shrink-0">{selectedSession.model?.split('/').pop() ?? ''}</span>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Task Queue (Cronban placeholder)
// ---------------------------------------------------------------------------

function TaskQueuePanel() {
  return (
    <div className="h-full flex flex-col">
      <div className="px-3 pt-3 pb-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Clock className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Task Queue</span>
        </div>
        <Badge variant="secondary" className="text-[10px] h-4 px-1">0</Badge>
      </div>
      <Separator />
      <div className="flex-1 flex flex-col items-center justify-center p-4 text-center">
        <Clock className="h-10 w-10 text-muted-foreground/30 mb-4" />
        <p className="text-xs font-medium text-muted-foreground mb-1">Cronban</p>
        <p className="text-[10px] text-muted-foreground/60 max-w-[200px]">
          Schedule recurring agent tasks — commits, tests, reviews, and more. Coming soon.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Events Panel
// ---------------------------------------------------------------------------

const EVENT_LABEL: Record<string, string> = {
  UserPromptSubmit: 'prompt',
  PostToolUse: 'tool',
  PreToolUse: 'pre-tool',
  Stop: 'stop',
  SessionStart: 'start',
  SessionEnd: 'end',
  SubagentStart: 'subagent',
};

function EventsPanel({ events }: { events: HookEvent[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const isAtBottom = useRef(true);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    isAtBottom.current = scrollHeight - scrollTop - clientHeight < 20;
  };

  useEffect(() => {
    if (isAtBottom.current && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [events]);

  return (
    <div className="h-full flex flex-col">
      <div className="px-3 pt-3 pb-2 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <Activity className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Events</span>
        </div>
        <Badge variant="secondary" className="text-[10px] h-4 px-1">{events.length}</Badge>
      </div>
      <Separator />
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto"
        onScroll={handleScroll}
      >
        {events.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full p-4 text-center">
            <Activity className="h-10 w-10 text-muted-foreground/30 mb-4" />
            <p className="text-xs font-medium text-muted-foreground mb-1">No events</p>
            <p className="text-[10px] text-muted-foreground/60 max-w-[200px]">
              Claude Code hook events appear here in real-time.
            </p>
          </div>
        ) : (
          <div className="px-2 py-1 space-y-0 font-mono text-[10px]">
            {events.map((ev) => {
              const d = parseDaemonTs(ev.ts);
              const t = d.toLocaleTimeString('en', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
              const label = EVENT_LABEL[ev.event] ?? ev.event.toLowerCase().replace('_', '-');
              return (
                <div key={ev.id} className="flex gap-2 py-0.5 leading-snug hover:bg-muted/20 rounded px-1">
                  <span className="text-muted-foreground/50 shrink-0 tabular-nums">{t}</span>
                  <span className={cn(
                    'shrink-0 px-1 rounded text-[9px] uppercase font-bold',
                    ev.event === 'UserPromptSubmit' ? 'bg-blue-500/20 text-blue-400' :
                    ev.event === 'PostToolUse' ? 'bg-amber-500/20 text-amber-400' :
                    ev.event === 'Stop' ? 'bg-red-500/20 text-red-400' :
                    ev.event === 'SessionStart' ? 'bg-emerald-500/20 text-emerald-400' :
                    'bg-muted text-muted-foreground',
                  )}>
                    {label}
                  </span>
                  <span className="text-muted-foreground truncate">
                    {ev.cwd ? shortPath(ev.cwd) : ev.session_id.slice(0, 8)}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2×2 Grid
// ---------------------------------------------------------------------------

function AgentsGrid({
  sessions,
  isLoading,
  daemonOnline,
  events,
  selectedId,
  onSelectId,
}: {
  sessions: AgentSession[];
  isLoading: boolean;
  daemonOnline: boolean;
  events: HookEvent[];
  selectedId: string | null;
  onSelectId: (id: string) => void;
}) {
  return (
    <div className="h-full grid grid-cols-2 grid-rows-2 divide-x divide-y divide-border">
      <div className="overflow-hidden">
        <SessionsPanel
          sessions={sessions}
          isLoading={isLoading}
          daemonOnline={daemonOnline}
          selectedId={selectedId}
          onSelect={onSelectId}
        />
      </div>
      <div className="overflow-hidden">
        <TerminalPanel
          sessions={sessions}
          selectedId={selectedId}
          onSelectId={onSelectId}
        />
      </div>
      <div className="overflow-hidden">
        <TaskQueuePanel />
      </div>
      <div className="overflow-hidden">
        <EventsPanel events={events} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Other sections
// ---------------------------------------------------------------------------

function CronbanView() {
  return (
    <div className="h-full flex flex-col items-center justify-center p-8 text-center">
      <Clock className="h-14 w-14 text-muted-foreground/30 mb-5" />
      <h2 className="text-base font-semibold mb-2">Cronban</h2>
      <p className="text-sm text-muted-foreground max-w-sm">
        Schedule AI agents on a cron schedule — daily reviews, tests, doc generation. Coming soon.
      </p>
    </div>
  );
}

function ConnectorsView() {
  const connectors = [
    { name: 'GitHub',  status: 'connected',    detail: 'PR reviews, issue tracking' },
    { name: 'Slack',   status: 'disconnected',  detail: 'Notification dispatch' },
    { name: 'Linear',  status: 'disconnected',  detail: 'Task management' },
    { name: 'Discord', status: 'disconnected',  detail: 'Alert broadcasting' },
  ];
  return (
    <div className="h-full flex flex-col">
      <div className="px-4 pt-4 pb-3 flex items-center gap-2 border-b border-border">
        <Plug className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-semibold">Connectors</span>
      </div>
      <div className="p-4 grid grid-cols-2 gap-3 content-start">
        {connectors.map((c) => (
          <div key={c.name} className="rounded-md border border-border p-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm">{c.name}</span>
              <span className={cn(
                'text-[10px] px-1.5 py-0.5 rounded border',
                c.status === 'connected'
                  ? 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30'
                  : 'text-muted-foreground bg-muted/50 border-border',
              )}>
                {c.status}
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground">{c.detail}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main AgentsPage
// ---------------------------------------------------------------------------

export function AgentsPage() {
  const [activeSection, setActiveSection] = useState<NavSection>('agents');
  const [sessions, setSessions] = useState<AgentSession[]>([]);
  const [events, setEvents] = useState<HookEvent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [daemonOnline, setDaemonOnline] = useState(false);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);
  const eventPollRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const fetchSessions = useCallback(async () => {
    try {
      const data = await listSessions();
      setSessions(data);
      setDaemonOnline(true);
      setLastRefresh(new Date());
      // Auto-select first relay session if none selected
      setSelectedId((prev) => {
        if (prev) return prev;
        const relay = data.find((s) => s.source === 'relay' && s.status !== 'dead');
        return relay?.id ?? null;
      });
    } catch {
      setDaemonOnline(false);
      setSessions([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const fetchEvents = useCallback(async () => {
    try {
      const data = await listHookEvents(200);
      setEvents(data);
    } catch {
      // daemon offline — silently ignore
    }
  }, []);

  useEffect(() => {
    fetchSessions();
    fetchEvents();
    pollRef.current = setInterval(fetchSessions, 5000);
    eventPollRef.current = setInterval(fetchEvents, 3000);
    return () => {
      clearInterval(pollRef.current);
      clearInterval(eventPollRef.current);
    };
  }, [fetchSessions, fetchEvents]);

  return (
    <div className="h-full flex">
      {/* Left nav sidebar */}
      <aside className="w-12 border-r border-border flex flex-col items-center pt-3 pb-2 gap-1 shrink-0">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveSection(item.id)}
            className={cn(
              'w-8 h-8 rounded flex items-center justify-center transition-colors duration-150',
              activeSection === item.id
                ? 'bg-blue-500/20 text-blue-400'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
            )}
            title={item.label}
          >
            <item.icon className="h-4 w-4" />
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={() => { fetchSessions(); fetchEvents(); }}
          className="w-8 h-8 rounded flex items-center justify-center text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
          title={`Last refresh: ${lastRefresh.toLocaleTimeString()}`}
        >
          <RefreshCw className="h-3.5 w-3.5" />
        </button>
      </aside>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeSection === 'agents' && (
          <AgentsGrid
            sessions={sessions}
            isLoading={isLoading}
            daemonOnline={daemonOnline}
            events={events}
            selectedId={selectedId}
            onSelectId={setSelectedId}
          />
        )}
        {activeSection === 'cronban' && <CronbanView />}
        {activeSection === 'connectors' && <ConnectorsView />}
      </div>
    </div>
  );
}
