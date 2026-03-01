/**
 * TerminalCompositor — Horizontal scrolling viewport of terminal panes.
 *
 * Niri-style: terminals laid out horizontally with scroll-snap.
 * Vertical scroll = terminal scrollback. Horizontal scroll/swipe = navigate between terminals.
 */
import { useEffect, useRef, useMemo, Component } from 'react';
import type { ReactNode } from 'react';
import { Terminal, AlertTriangle } from 'lucide-react';
import { type AgentSession } from '@/services/daemon-api';
import { useCompositorScroll } from '@/hooks/useCompositorScroll';
import { TerminalPane } from './TerminalPane';
import { CompositorNav } from './CompositorNav';

// ── Error boundary — keeps one bad pane from crashing the whole compositor ──
class TerminalPaneBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div className="h-full flex flex-col items-center justify-center gap-2 rounded-lg border border-destructive/30 bg-destructive/5 text-destructive/80 text-xs p-4 text-center">
          <AlertTriangle className="h-5 w-5" />
          <span>Terminal error</span>
          <span className="font-mono text-[10px] text-muted-foreground">{this.state.error.message}</span>
        </div>
      );
    }
    return this.props.children;
  }
}

export function TerminalCompositor({
  sessions,
  focusedSessionId,
  onFocusSession,
}: {
  sessions: AgentSession[];
  focusedSessionId: string | null;
  onFocusSession: (id: string) => void;
}) {
  const relaySessions = useMemo(
    () => sessions.filter((s) => s.source === 'relay' && s.status !== 'dead'),
    [sessions],
  );

  const {
    viewportRef,
    scrollToIndex,
    scrollPrev,
    scrollNext,
    focusedIndex,
    setFocusedIndex,
    visibleIndices,
  } = useCompositorScroll({ paneCount: relaySessions.length });

  // Guard to prevent circular sync: external prop → scrollToIndex → setFocusedIndex → onFocusSession → prop
  const syncingFromProp = useRef(false);

  // Sync external focusedSessionId → compositor scroll position
  useEffect(() => {
    if (!focusedSessionId) return;
    const idx = relaySessions.findIndex((s) => s.id === focusedSessionId);
    if (idx !== -1 && idx !== focusedIndex) {
      syncingFromProp.current = true;
      scrollToIndex(idx);
      // Reset guard after React flushes the state update
      requestAnimationFrame(() => { syncingFromProp.current = false; });
    }
  }, [focusedSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Sync compositor focus → external session ID (only from internal nav, not prop-driven)
  useEffect(() => {
    if (syncingFromProp.current) return;
    const session = relaySessions[focusedIndex];
    if (session && session.id !== focusedSessionId) {
      onFocusSession(session.id);
    }
  }, [focusedIndex]); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Empty state ────────────────────────────────────────────────────────
  if (relaySessions.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-8">
        <Terminal className="h-12 w-12 text-muted-foreground/20 mb-4" />
        <p className="text-sm text-muted-foreground mb-2">No live terminal streams</p>
        <p className="text-xs text-muted-foreground/60 max-w-[320px]">
          Sessions detected via hooks appear in the sidebar. Click one to view its transcript.
        </p>
        <p className="text-xs text-muted-foreground/40 max-w-[320px] mt-2">
          For live terminal streaming, start a session with{' '}
          <code className="font-mono bg-muted px-1 py-0.5 rounded text-foreground/50">
            vlt session-relay
          </code>
        </p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Viewport — horizontal scroll container */}
      <div
        ref={viewportRef}
        className="flex-1 flex gap-3 overflow-x-auto overflow-y-hidden px-3 py-2"
        style={{
          // No scroll-snap — lerp handles programmatic scrolling,
          // native scroll/swipe moves freely (more Niri-like)
          scrollbarWidth: 'none',
        }}
      >
        {relaySessions.map((session, idx) => (
          <div
            key={session.id}
            className="shrink-0 h-full"
            style={{
              width: 'clamp(480px, 45vw, 900px)',
            }}
          >
            <TerminalPaneBoundary>
              <TerminalPane
                session={session}
                isVisible={visibleIndices.has(idx)}
                isFocused={focusedIndex === idx}
                onFocus={() => setFocusedIndex(idx)}
              />
            </TerminalPaneBoundary>
          </div>
        ))}
      </div>

      {/* Nav controls */}
      <CompositorNav
        paneCount={relaySessions.length}
        focusedIndex={focusedIndex}
        visibleIndices={visibleIndices}
        onPrev={scrollPrev}
        onNext={scrollNext}
        onDotClick={scrollToIndex}
      />
    </div>
  );
}
