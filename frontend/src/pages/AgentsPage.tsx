/**
 * AgentsPage — Niri-style terminal compositor for Claude Code sessions.
 *
 * Layout:
 *   Left nav (48px) — icons (Agents, Cronban, Connectors)
 *   Session sidebar   — relay + discovery session list
 *   Compositor        — horizontal scrolling terminal panes
 *   Events ticker     — compact event strip at bottom
 */
import { useState, useCallback } from 'react';
import { Bot, Clock, Plug, RefreshCw } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession } from '@/services/daemon-api';
import { useSessionPolling } from '@/hooks/useSessionPolling';
import { SessionSidebar } from '@/components/agents/SessionSidebar';
import { TerminalCompositor } from '@/components/agents/TerminalCompositor';
import { EventsTicker } from '@/components/agents/EventsTicker';
import { TranscriptOverlay } from '@/components/agents/TranscriptOverlay';

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
  onSelectSession,
  onSelectDiscoveredSession,
}: {
  polling: ReturnType<typeof useSessionPolling>;
  focusedSessionId: string | null;
  onSelectSession: (id: string) => void;
  onSelectDiscoveredSession: (session: AgentSession) => void;
}) {
  return (
    <div className="h-full flex">
      {/* Session sidebar */}
      <div className="w-52 shrink-0">
        <SessionSidebar
          sessions={polling.sessions}
          isLoading={polling.isLoading}
          daemonOnline={polling.daemonOnline}
          focusedSessionId={focusedSessionId}
          onSelectSession={onSelectSession}
          onSelectDiscoveredSession={onSelectDiscoveredSession}
        />
      </div>

      {/* Compositor + events */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-hidden">
          <TerminalCompositor
            sessions={polling.sessions}
            focusedSessionId={focusedSessionId}
            onFocusSession={onSelectSession}
          />
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
  const polling = useSessionPolling();
  const [focusedSessionId, setFocusedSessionId] = useState<string | null>(null);
  const [transcriptSession, setTranscriptSession] = useState<AgentSession | null>(null);

  const handleSelectSession = useCallback((id: string) => {
    setFocusedSessionId(id);
  }, []);

  const handleSelectDiscoveredSession = useCallback((session: AgentSession) => {
    setTranscriptSession(session);
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
            onSelectSession={handleSelectSession}
            onSelectDiscoveredSession={handleSelectDiscoveredSession}
          />
        )}
        {activeSection === 'cronban' && <CronbanView />}
        {activeSection === 'connectors' && <ConnectorsView />}
      </div>

      {/* Transcript overlay for discovered sessions */}
      {transcriptSession && (
        <TranscriptOverlay
          session={transcriptSession}
          open={!!transcriptSession}
          onClose={() => setTranscriptSession(null)}
        />
      )}
    </div>
  );
}
