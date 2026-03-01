/**
 * AgentsPage — Niri-style terminal compositor for Claude Code sessions.
 *
 * Layout:
 *   Left nav (48px) — icons (Agents, Cronban, Connectors)
 *   Session sidebar   — relay + discovery session list
 *   Compositor        — horizontal scrolling terminal panes
 *   Events ticker     — compact event strip at bottom
 */
import { useState, useCallback, useMemo } from 'react';
import { Bot, Clock, Plug, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession, dismissSession, spawnSession } from '@/services/daemon-api';
import { useSessionPolling } from '@/hooks/useSessionPolling';
import { useProjectContext } from '@/contexts/ProjectContext';
import { SessionSidebar } from '@/components/agents/SessionSidebar';
import { TerminalCompositor } from '@/components/agents/TerminalCompositor';
import { EventsTicker } from '@/components/agents/EventsTicker';
import { LiveSessionPanel } from '@/components/agents/LiveSessionPanel';

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
// Placeholder sections (unchanged from original)
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
// Agents Compositor Layout
// ---------------------------------------------------------------------------

function AgentsCompositorView({
  polling,
  focusedSessionId,
  liveSession,
  selectedProjectId,
  spawnCwd,
  onSelectSession,
  onSelectDiscoveredSession,
  onDismissSession,
  onCloseLiveSession,
  onStartFresh,
  onResumeSession,
}: {
  polling: ReturnType<typeof useSessionPolling>;
  focusedSessionId: string | null;
  liveSession: AgentSession | null;
  selectedProjectId: string | null;
  spawnCwd: string;
  onSelectSession: (id: string) => void;
  onSelectDiscoveredSession: (session: AgentSession) => void;
  onDismissSession: (id: string) => void;
  onCloseLiveSession: () => void;
  onStartFresh: (prompt: string) => void;
  onResumeSession: (session: AgentSession) => void;
}) {
  const hasRelaySessions = useMemo(
    () => polling.sessions.some((s) => s.source === 'relay' && s.status !== 'dead'),
    [polling.sessions],
  );

  return (
    <div className="h-full flex">
      {/* Session sidebar */}
      <div className="w-52 shrink-0">
        <SessionSidebar
          sessions={polling.sessions}
          isLoading={polling.isLoading}
          daemonOnline={polling.daemonOnline}
          focusedSessionId={liveSession?.id ?? focusedSessionId}
          currentProjectId={selectedProjectId}
          spawnCwd={spawnCwd}
          onSelectSession={onSelectSession}
          onSelectDiscoveredSession={onSelectDiscoveredSession}
          onDismissSession={onDismissSession}
          onStartFresh={onStartFresh}
          onResumeSession={onResumeSession}
          onRefresh={polling.refresh}
        />
      </div>

      {/* Main content: LiveSessionPanel (hook sessions) or TerminalCompositor (relay) */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-hidden">
          {liveSession ? (
            <LiveSessionPanel
              session={liveSession}
              onClose={onCloseLiveSession}
            />
          ) : hasRelaySessions ? (
            <TerminalCompositor
              sessions={polling.sessions}
              focusedSessionId={focusedSessionId}
              onFocusSession={onSelectSession}
            />
          ) : (
            <TerminalCompositor
              sessions={polling.sessions}
              focusedSessionId={focusedSessionId}
              onFocusSession={onSelectSession}
            />
          )}
        </div>
        <EventsTicker events={polling.events} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main AgentsPage
// ---------------------------------------------------------------------------

export function AgentsPage() {
  const [activeSection, setActiveSection] = useState<NavSection>('agents');
  const { selectedProjectId } = useProjectContext();
  const polling = useSessionPolling(selectedProjectId);
  const [focusedSessionId, setFocusedSessionId] = useState<string | null>(null);
  const [liveSession, setLiveSession] = useState<AgentSession | null>(null);

  // Derive a CWD for spawning: use the most recent session's CWD for this project,
  // or fall back to a sensible default.
  const spawnCwd = useMemo(() => {
    const projectSession = polling.sessions.find((s) => s.cwd);
    return projectSession?.cwd ?? '/mnt/sda1/Projects/00Tooling/Vlt-Bridge';
  }, [polling.sessions]);

  const handleSelectSession = useCallback((id: string) => {
    setLiveSession(null); // close live panel when selecting a relay session
    setFocusedSessionId(id);
  }, []);

  const handleSelectDiscoveredSession = useCallback((session: AgentSession) => {
    setLiveSession(session);
  }, []);

  const handleDismissSession = useCallback((id: string) => {
    dismissSession(id).then(() => polling.refresh()).catch(console.error);
  }, [polling]);

  const handleStartFresh = useCallback(
    (prompt: string) => {
      spawnSession(spawnCwd, { prompt })
        .then((result) => {
          setTimeout(() => polling.refresh(), 2000);
          if (result.session_id) {
            setLiveSession({
              id: result.session_id,
              project_id: selectedProjectId,
              name: 'New Session',
              cwd: result.cwd,
              status: 'thinking',
              model: null,
              ctx_pct: null,
              pid: 0,
              bypass_perms: true,
              source: 'managed',
              created_at: new Date().toISOString(),
              last_activity: new Date().toISOString(),
            });
          }
        })
        .catch((err) => console.error('Spawn failed:', err));
    },
    [polling, spawnCwd, selectedProjectId],
  );

  const handleResumeSession = useCallback((session: AgentSession) => {
    setLiveSession(session);
  }, []);

  return (
    <div className="h-full flex">
      {/* Left nav sidebar */}
      <aside className="w-16 border-r border-border flex flex-col items-center pt-3 pb-2 gap-0.5 shrink-0">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            onClick={() => setActiveSection(item.id)}
            className={cn(
              'w-14 rounded flex flex-col items-center justify-center py-1.5 gap-0.5 transition-colors duration-150',
              activeSection === item.id
                ? 'bg-blue-500/20 text-blue-400'
                : 'text-muted-foreground hover:text-foreground hover:bg-muted/60',
            )}
          >
            <item.icon className="h-4 w-4" />
            <span className="text-[9px] font-medium uppercase tracking-wider leading-none">
              {item.label}
            </span>
          </button>
        ))}
        <div className="flex-1" />
        <button
          onClick={polling.refresh}
          className="w-14 rounded flex flex-col items-center justify-center py-1.5 gap-0.5 text-muted-foreground hover:text-foreground hover:bg-muted/60 transition-colors"
          title={`Last refresh: ${polling.lastRefresh.toLocaleTimeString()}`}
        >
          <RefreshCw className="h-3.5 w-3.5" />
          <span className="text-[9px] font-medium uppercase tracking-wider leading-none">Refresh</span>
        </button>
      </aside>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {activeSection === 'agents' && (
          <AgentsCompositorView
            polling={polling}
            focusedSessionId={focusedSessionId}
            liveSession={liveSession}
            selectedProjectId={selectedProjectId}
            spawnCwd={spawnCwd}
            onSelectSession={handleSelectSession}
            onSelectDiscoveredSession={handleSelectDiscoveredSession}
            onDismissSession={handleDismissSession}
            onCloseLiveSession={() => setLiveSession(null)}
            onStartFresh={handleStartFresh}
            onResumeSession={handleResumeSession}
          />
        )}
        {activeSection === 'cronban' && <CronbanView />}
        {activeSection === 'connectors' && <ConnectorsView />}
      </div>
    </div>
  );
}
