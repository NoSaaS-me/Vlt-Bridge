/**
 * useTerminalSession — Manages a single xterm.js instance + WebSocket lifecycle.
 *
 * Each relay session gets its own terminal. The terminal is created once on mount
 * and never destroyed when scrolling off-screen (retains scrollback). The WebSocket
 * connects when visible and disconnects when off-screen to save bandwidth.
 *
 * Input is only routed when `isFocused` is true.
 */
import { useRef, useEffect, useState, useCallback } from 'react';
import type { Terminal } from '@xterm/xterm';
import type { FitAddon } from '@xterm/addon-fit';
import { XTERM_THEME } from '@/components/agents/utils';

interface UseTerminalSessionOptions {
  sessionId: string;
  containerRef: React.RefObject<HTMLDivElement | null>;
  isVisible: boolean;
  isFocused: boolean;
}

interface UseTerminalSessionReturn {
  isReady: boolean;
  isConnected: boolean;
  refit: () => void;
}

export function useTerminalSession({
  sessionId,
  containerRef,
  isVisible,
  isFocused,
}: UseTerminalSessionOptions): UseTerminalSessionReturn {
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [isReady, setIsReady] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const isFocusedRef = useRef(isFocused);

  // Keep focus ref current without triggering effects
  useEffect(() => {
    isFocusedRef.current = isFocused;
  }, [isFocused]);

  // ── Initialize xterm once ──────────────────────────────────────────────
  useEffect(() => {
    if (!containerRef.current || termRef.current) return;

    let cancelled = false;

    Promise.all([
      import('@xterm/xterm'),
      import('@xterm/addon-fit'),
    ]).then(([{ Terminal }, { FitAddon }]) => {
      if (cancelled || !containerRef.current) return;

      const term = new Terminal({
        theme: XTERM_THEME,
        fontFamily: '"Cascadia Code", "Fira Code", "JetBrains Mono", monospace',
        fontSize: 13,
        lineHeight: 1.2,
        cursorBlink: false,
        allowProposedApi: true,
        scrollback: 5000,
        // Don't hard-wrap at a narrow browser-pane width — let the PTY content
        // determine wrapping. FitAddon still auto-fits cols on resize.
        cols: 220,
      });

      const fitAddon = new FitAddon();
      term.loadAddon(fitAddon);
      term.open(containerRef.current);

      // Delay initial fit to ensure container has dimensions
      requestAnimationFrame(() => {
        if (cancelled) return;
        fitAddon.fit();
        if (term.cols < 200) term.resize(200, term.rows);
      });

      termRef.current = term;
      fitRef.current = fitAddon;
      setIsReady(true);

      // Route user input → WS inject (only when focused)
      term.onData((data) => {
        if (isFocusedRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'inject', data }));
        }
      });
    });

    return () => {
      cancelled = true;
    };
  }, [containerRef, sessionId]);

  // ── ResizeObserver — fit xterm to container, but NEVER send resize to relay ──
  // The relay's PTY dimensions are owned by the user's real terminal.
  // Sending browser pane dimensions would overwrite them and break the user's terminal.
  // Enforce a minimum of 200 cols so content from wide terminals doesn't hard-wrap.
  const MIN_COLS = 200;
  useEffect(() => {
    if (!containerRef.current) return;
    const el = containerRef.current;
    const observer = new ResizeObserver(() => {
      const term = termRef.current;
      const fit = fitRef.current;
      if (!fit || !term || !el.isConnected) return;
      fit.fit();
      if (term.cols < MIN_COLS) {
        term.resize(MIN_COLS, term.rows);
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [containerRef]);

  // ── WebSocket: connect when visible, disconnect when not ───────────────
  useEffect(() => {
    if (!isReady || !isVisible) {
      // Disconnect if we go invisible
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
        setIsConnected(false);
      }
      return;
    }

    const term = termRef.current;
    if (!term) return;

    // Already connected to this session
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    term.writeln(`\x1b[2m\x1b[36m── connecting to session ${sessionId.slice(0, 8)} ──\x1b[0m`);

    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${window.location.host}/vlt/ws/sessions/${sessionId}`);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    // Show a hint if no data arrives within 6 seconds of connecting
    let noDataTimer: ReturnType<typeof setTimeout> | undefined;
    let receivedData = false;

    ws.onopen = () => {
      setIsConnected(true);
      term.writeln(`\x1b[2m\x1b[32m── connected ──\x1b[0m`);
      noDataTimer = setTimeout(() => {
        if (!receivedData) {
          term.writeln(`\x1b[2m\x1b[33m── no terminal history ──\x1b[0m`);
          term.writeln(`\x1b[2m\x1b[33m── run: vlt session-relay ──\x1b[0m`);
        }
      }, 6000);
    };

    ws.onmessage = (ev) => {
      receivedData = true;
      if (ev.data instanceof ArrayBuffer) {
        term.write(new Uint8Array(ev.data));
      } else if (typeof ev.data === 'string') {
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
      setIsConnected(false);
      wsRef.current = null;
      if (ev.code === 1008) {
        term.writeln(`\x1b[31m── session not found on daemon ──\x1b[0m`);
        term.writeln(`\x1b[2m\x1b[33m── dismiss this session from the sidebar ──\x1b[0m`);
      } else {
        term.writeln(`\x1b[2m── disconnected (${ev.code}) ──\x1b[0m`);
      }
    };

    return () => {
      clearTimeout(noDataTimer);
      ws.close();
      wsRef.current = null;
      setIsConnected(false);
    };
  }, [isReady, isVisible, sessionId]);

  // ── Cleanup on unmount ─────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      wsRef.current?.close();
      termRef.current?.dispose();
    };
  }, []);

  const refit = useCallback(() => {
    if (fitRef.current && containerRef.current?.isConnected) {
      fitRef.current.fit();
    }
  }, [containerRef]);

  return { isReady, isConnected, refit };
}
