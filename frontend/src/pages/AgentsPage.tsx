/**
 * AgentsPage — Niri-style terminal compositor for Claude Code sessions.
 *
 * Layout:
 *   Left nav (48px) — icons (Agents, Cronban, Connectors)
 *   Session sidebar   — relay + discovery session list
 *   Compositor        — horizontal scrolling terminal panes
 *   Events ticker     — compact event strip at bottom
 */
import { useState, useCallback, useMemo, useEffect } from 'react';
import { Bot, Clock, Plug, RefreshCw, Settings2, CheckCircle2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession, dismissSession, spawnSession, renameSession } from '@/services/daemon-api';
import { useSessionPolling } from '@/hooks/useSessionPolling';
import { useProjectContext } from '@/contexts/ProjectContext';
import { SessionSidebar } from '@/components/agents/SessionSidebar';
import { TerminalCompositor } from '@/components/agents/TerminalCompositor';
import { EventsTicker } from '@/components/agents/EventsTicker';
import { LiveSessionPanel } from '@/components/agents/LiveSessionPanel';
import { CronbanView as CronbanViewReal } from '@/components/cronban/CronbanView';
import { listConnectors, getConnectorConfig, saveConnectorConfig, type ConnectorInfo } from '@/services/connectors';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog';

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

function CronbanView({ projectId }: { projectId?: string }) {
  return <CronbanViewReal projectId={projectId} />;
}

function ConnectorDialog({
  connector,
  onClose,
  onSaved,
}: {
  connector: ConnectorInfo;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [fields, setFields] = useState<Record<string, string>>({});
  const [enabled, setEnabled] = useState(connector.enabled);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getConnectorConfig(connector.name)
      .then((cfg) => {
        setEnabled(cfg['__enabled'] === 'true');
        const rest: Record<string, string> = {};
        for (const [k, v] of Object.entries(cfg)) {
          if (!k.startsWith('__')) rest[k] = v;
        }
        setFields(rest);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [connector.name]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      await saveConnectorConfig(connector.name, { ...fields, __enabled: enabled ? 'true' : 'false' });
      onSaved();
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open onOpenChange={(o) => !o && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{connector.display_name} Settings</DialogTitle>
        </DialogHeader>
        {error && <Alert variant="destructive"><AlertDescription>{error}</AlertDescription></Alert>}
        {loading ? (
          <p className="text-sm text-muted-foreground py-4">Loading…</p>
        ) : (
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Enable connector</Label>
              <Switch checked={enabled} onCheckedChange={setEnabled} />
            </div>
            {connector.credential_fields.map((field) => (
              <div key={field.name} className="space-y-1">
                <Label htmlFor={`f-${field.name}`}>{field.label}</Label>
                <Input
                  id={`f-${field.name}`}
                  type={field.secret ? 'password' : 'text'}
                  placeholder={field.secret && fields[field.name] === '••••••••' ? 'Already set — enter new value to change' : field.placeholder}
                  value={fields[field.name] === '••••••••' ? '' : (fields[field.name] ?? '')}
                  onChange={(e) => setFields((prev) => ({ ...prev, [field.name]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>Cancel</Button>
          <Button onClick={handleSave} disabled={saving || loading}>{saving ? 'Saving…' : 'Save'}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function ConnectorsView() {
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [configuring, setConfiguring] = useState<ConnectorInfo | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try { setConnectors(await listConnectors()); }
    catch (e) { setError(e instanceof Error ? e.message : 'Failed to load'); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  return (
    <div className="h-full flex flex-col">
      <div className="px-4 pt-4 pb-3 flex items-center gap-2 border-b border-border">
        <Plug className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-semibold">Connectors</span>
      </div>
      <div className="p-4 space-y-3 overflow-y-auto flex-1">
        {error && <p className="text-xs text-destructive">{error}</p>}
        {loading && <p className="text-xs text-muted-foreground">Loading…</p>}
        {!loading && !error && connectors.length === 0 && (
          <p className="text-xs text-muted-foreground">No connectors registered.</p>
        )}
        <div className="grid grid-cols-2 gap-3">
          {connectors.map((c) => (
            <div key={c.name} className="rounded-md border border-border p-3 space-y-2">
              <div className="flex items-center justify-between gap-1">
                <span className="font-medium text-sm truncate">{c.display_name}</span>
                <Button variant="ghost" size="icon" className="h-6 w-6 flex-shrink-0" onClick={() => setConfiguring(c)}>
                  <Settings2 className="h-3.5 w-3.5" />
                </Button>
              </div>
              <p className="text-[11px] text-muted-foreground">{c.description}</p>
              <div>
                {c.enabled && c.configured ? (
                  <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border text-emerald-400 bg-emerald-500/15 border-emerald-500/30">
                    <CheckCircle2 className="h-2.5 w-2.5" />enabled
                  </span>
                ) : c.enabled ? (
                  <span className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border text-amber-400 bg-amber-500/15 border-amber-500/30">
                    <AlertCircle className="h-2.5 w-2.5" />needs config
                  </span>
                ) : (
                  <span className="text-[10px] px-1.5 py-0.5 rounded border text-muted-foreground bg-muted/50 border-border">
                    disabled
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
      {configuring && (
        <ConnectorDialog connector={configuring} onClose={() => setConfiguring(null)} onSaved={load} />
      )}
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
  onRenameSession,
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
  onRenameSession: (id: string, name: string) => void;
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
          onRenameSession={onRenameSession}
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

  const handleRenameSession = useCallback((id: string, name: string) => {
    renameSession(id, name).then(() => polling.refresh()).catch(console.error);
  }, [polling]);

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
            onRenameSession={handleRenameSession}
          />
        )}
        {activeSection === 'cronban' && <CronbanView projectId={selectedProjectId ?? undefined} />}
        {activeSection === 'connectors' && <ConnectorsView />}
      </div>
    </div>
  );
}
