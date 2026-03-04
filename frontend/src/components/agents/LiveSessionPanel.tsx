/**
 * LiveSessionPanel — Real-time conversation view for Claude Code sessions.
 *
 * Connects via WebSocket to stream JSONL transcript entries as they're written.
 * Supports bidirectional messaging: user types in the input bar, the daemon
 * routes it via the managed session worker (claude -p --resume) so it runs as
 * a discrete turn without touching the user's interactive terminal.
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  ArrowDown,
  Loader2,
  Send,
  WifiOff,
  X,
  ChevronDown,
  ChevronRight,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { type AgentSession, getLiveSessionWsUrl } from '@/services/daemon-api';
import { Button } from '@/components/ui/button';
import { shortPath, statusColor, timeAgo } from './utils';

// ---------------------------------------------------------------------------
// Types for JSONL entries
// ---------------------------------------------------------------------------

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type RawEntry = Record<string, any>;

interface ParsedMessage {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: ContentPart[];
  timestamp?: string;
}

interface ContentPart {
  type: 'text' | 'thinking' | 'tool_use' | 'tool_result' | 'code';
  text?: string;
  toolName?: string;
  toolInput?: string;
  toolOutput?: string;
  isError?: boolean;
}

// ---------------------------------------------------------------------------
// Entry parsing
// ---------------------------------------------------------------------------

function parseEntry(raw: RawEntry): ParsedMessage | null {
  const type = raw.type;
  if (type !== 'user' && type !== 'assistant' && type !== 'system') return null;

  const msg = raw.message;
  if (!msg) return null;

  let role = (msg.role || type) as ParsedMessage['role'];
  const parts: ContentPart[] = [];

  if (typeof msg.content === 'string') {
    parts.push({ type: 'text', text: msg.content });
  } else if (Array.isArray(msg.content)) {
    for (const block of msg.content) {
      if (block.type === 'text') {
        parts.push({ type: 'text', text: block.text || '' });
      } else if (block.type === 'thinking') {
        parts.push({ type: 'thinking', text: block.thinking || '' });
      } else if (block.type === 'tool_use') {
        parts.push({
          type: 'tool_use',
          toolName: block.name || 'unknown',
          toolInput: typeof block.input === 'string'
            ? block.input
            : JSON.stringify(block.input, null, 2),
        });
      } else if (block.type === 'tool_result') {
        const output = typeof block.content === 'string'
          ? block.content
          : Array.isArray(block.content)
            ? block.content.map((c: { text?: string }) => c.text || '').join('\n')
            : JSON.stringify(block.content);
        parts.push({
          type: 'tool_result',
          toolOutput: output,
          isError: block.is_error,
        });
      }
    }
  }

  if (parts.length === 0) return null;

  // Claude Code wraps tool results in "user" type entries. If every content
  // block is a tool_result, this is really part of the assistant's tool flow,
  // not an actual user message.
  if (role === 'user' && parts.length > 0 && parts.every((p) => p.type === 'tool_result')) {
    role = 'tool';
  }

  return {
    id: raw.uuid || `${raw.timestamp || ''}-${Math.random()}`,
    role,
    content: parts,
    timestamp: raw.timestamp,
  };
}

// ---------------------------------------------------------------------------
// Thinking indicator — animated color bars (à la Claude)
// ---------------------------------------------------------------------------

const THINKING_COLORS = [
  'bg-blue-400', 'bg-violet-400', 'bg-amber-400', 'bg-emerald-400',
  'bg-rose-400', 'bg-cyan-400', 'bg-orange-400', 'bg-indigo-400',
];

function ThinkingIndicator({ status }: { status: string }) {
  const label = status === 'executing' ? 'Working' : 'Thinking';
  return (
    <div className="rounded-md border bg-muted/30 border-border px-3 py-2.5">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-foreground/80">
        Claude
      </span>
      <div className="mt-1.5 flex flex-col gap-[3px]">
        {Array.from({ length: 4 }).map((_, row) => (
          <div key={row} className="flex gap-[3px] h-[3px]">
            {Array.from({ length: 3 + (row % 2) }).map((_, col) => {
              const ci = (row * 3 + col) % THINKING_COLORS.length;
              const widths = ['w-8', 'w-12', 'w-16', 'w-20', 'w-6', 'w-10', 'w-14'];
              const wi = (row * 5 + col * 3) % widths.length;
              // Each bar gets a different animation delay for the shimmer effect
              const delay = `${(row * 150 + col * 200) % 1000}ms`;
              return (
                <div
                  key={col}
                  className={cn(
                    'rounded-full opacity-40',
                    THINKING_COLORS[ci],
                    widths[wi],
                    'animate-pulse',
                  )}
                  style={{ animationDelay: delay, animationDuration: '1.2s' }}
                />
              );
            })}
          </div>
        ))}
      </div>
      <span className="text-[9px] text-muted-foreground/60 mt-1.5 block">{label}...</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Message bubble component
// ---------------------------------------------------------------------------

function MessageBubble({ msg }: { msg: ParsedMessage }) {
  const [thinkingOpen, setThinkingOpen] = useState(false);
  const [toolOpen, setToolOpen] = useState(false);

  const isAssistantLike = msg.role === 'assistant' || msg.role === 'tool';
  const roleLabel = msg.role === 'user' ? 'You' : isAssistantLike ? 'Claude' : 'System';
  const roleBg =
    msg.role === 'user'
      ? 'bg-blue-500/10 border-blue-500/20'
      : isAssistantLike
        ? 'bg-muted/30 border-border'
        : 'bg-amber-500/10 border-amber-500/20';
  const roleColor =
    msg.role === 'user'
      ? 'text-blue-400'
      : isAssistantLike
        ? 'text-foreground/80'
        : 'text-amber-400';

  return (
    <div className={cn('rounded-md border px-3 py-2 text-sm', roleBg)}>
      <div className="flex items-center gap-2 mb-1">
        <span className={cn('text-[10px] font-semibold uppercase tracking-wider', roleColor)}>
          {roleLabel}
        </span>
        {msg.timestamp && (
          <span className="text-[9px] text-muted-foreground">{timeAgo(msg.timestamp)}</span>
        )}
      </div>

      <div className="space-y-2">
        {msg.content.map((part, i) => {
          if (part.type === 'text') {
            return (
              <div key={i} className="text-foreground/90 whitespace-pre-wrap break-words text-[13px] leading-relaxed">
                {part.text}
              </div>
            );
          }

          if (part.type === 'thinking') {
            return (
              <div key={i}>
                <button
                  onClick={() => setThinkingOpen(!thinkingOpen)}
                  className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  {thinkingOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  <span className="italic">Thinking...</span>
                </button>
                {thinkingOpen && (
                  <div className="mt-1 pl-4 text-[12px] text-muted-foreground/70 whitespace-pre-wrap break-words max-h-60 overflow-y-auto border-l-2 border-muted">
                    {part.text}
                  </div>
                )}
              </div>
            );
          }

          if (part.type === 'tool_use') {
            return (
              <div key={i} className="rounded bg-background/60 border border-border px-2 py-1.5">
                <button
                  onClick={() => setToolOpen(!toolOpen)}
                  className="flex items-center gap-1 text-[11px] font-mono text-amber-400"
                >
                  {toolOpen ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  {part.toolName}
                </button>
                {toolOpen && part.toolInput && (
                  <pre className="mt-1 text-[11px] text-muted-foreground/80 whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
                    {part.toolInput.length > 2000 ? part.toolInput.slice(0, 2000) + '...' : part.toolInput}
                  </pre>
                )}
              </div>
            );
          }

          if (part.type === 'tool_result') {
            const output = part.toolOutput || '';
            const truncated = output.length > 500;
            return (
              <div
                key={i}
                className={cn(
                  'rounded border px-2 py-1.5 text-[11px] font-mono whitespace-pre-wrap break-words max-h-32 overflow-y-auto',
                  part.isError
                    ? 'bg-red-500/10 border-red-500/20 text-red-300'
                    : 'bg-background/40 border-border text-muted-foreground/80',
                )}
              >
                {truncated ? output.slice(0, 500) + '...' : output}
              </div>
            );
          }

          return null;
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function LiveSessionPanel({
  session,
  onClose,
}: {
  session: AgentSession;
  onClose: () => void;
}) {
  const [messages, setMessages] = useState<ParsedMessage[]>([]);
  const [status, setStatus] = useState(session.status);
  const [connected, setConnected] = useState(false);
  const [inputText, setInputText] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  // Show thinking indicator immediately after sending, before hook fires
  const [waitingForResponse, setWaitingForResponse] = useState(false);
  // Live ctx_pct from WS (overrides session prop when available)
  const [liveCtxPct, setLiveCtxPct] = useState<number | null>(session.ctx_pct);

  const wsRef = useRef<WebSocket | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, autoScroll]);

  // Detect scroll position to toggle auto-scroll
  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100;
    setAutoScroll(nearBottom);
  }, []);

  // WebSocket connection
  useEffect(() => {
    const url = getLiveSessionWsUrl(session.id);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);

        if (msg.type === 'initial') {
          // Parse all existing entries
          const parsed = (msg.entries || [])
            .map(parseEntry)
            .filter(Boolean) as ParsedMessage[];
          setMessages(parsed);
          if (msg.session?.status) setStatus(msg.session.status);
        } else if (msg.type === 'entry') {
          const parsed = parseEntry(msg.entry);
          if (parsed) {
            setMessages((prev) =>
              prev.some((m) => m.id === parsed.id) ? prev : [...prev, parsed],
            );
          }
        } else if (msg.type === 'status') {
          setStatus(msg.status);
          // Once we get a real status update, clear the optimistic waiting state
          if (msg.status === 'thinking' || msg.status === 'executing' || msg.status === 'idle') {
            setWaitingForResponse(false);
          }
        } else if (msg.type === 'ctx_pct') {
          setLiveCtxPct(msg.ctx_pct);
        }
      } catch {
        // ignore parse errors
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [session.id]);

  // Send context injection via WebSocket
  const handleSend = useCallback(() => {
    const text = inputText.trim();
    if (!text || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    wsRef.current.send(JSON.stringify({ type: 'inject', text }));
    setInputText('');
    setWaitingForResponse(true);
    inputRef.current?.focus();
  }, [inputText]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border shrink-0">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                'h-2 w-2 rounded-full',
                status === 'thinking' || status === 'executing'
                  ? 'bg-amber-400 animate-pulse'
                  : status === 'idle'
                    ? 'bg-emerald-400'
                    : 'bg-muted-foreground/50',
              )}
            />
            <span className="font-medium text-sm">{session.name}</span>
          </div>
          <span className={cn('text-[10px] px-1.5 py-0.5 rounded border', statusColor(status as AgentSession['status']))}>
            {status}
          </span>
          <span className="text-[10px] text-muted-foreground font-mono">
            {shortPath(session.cwd)}
          </span>
          {session.model && (
            <span className="text-[10px] text-muted-foreground/60">{session.model}</span>
          )}
          {liveCtxPct != null && liveCtxPct > 0 && (
            <span
              className={cn(
                'text-[10px] font-mono px-1.5 py-0.5 rounded border',
                liveCtxPct > 80
                  ? 'text-red-400 bg-red-500/10 border-red-500/20'
                  : liveCtxPct > 50
                    ? 'text-amber-400 bg-amber-500/10 border-amber-500/20'
                    : 'text-muted-foreground bg-muted/30 border-border',
              )}
            >
              ctx {liveCtxPct}%
            </span>
          )}
          {!connected && (
            <span className="flex items-center gap-1 text-[10px] text-red-400">
              <WifiOff className="h-3 w-3" /> disconnected
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground">
            {messages.length} messages
          </span>
          <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={onClose}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* Messages */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto"
        onScroll={handleScroll}
      >
        <div className="max-w-3xl mx-auto p-4 space-y-3">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <Loader2 className="h-8 w-8 text-muted-foreground/30 animate-spin mb-3" />
              <p className="text-sm text-muted-foreground">Waiting for conversation data...</p>
            </div>
          )}
          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}
          {(status === 'thinking' || status === 'executing' || waitingForResponse) && (
            <ThinkingIndicator status={status === 'executing' ? 'executing' : 'thinking'} />
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Scroll-to-bottom button */}
      {!autoScroll && (
        <div className="absolute bottom-20 right-6">
          <Button
            variant="outline"
            size="sm"
            className="rounded-full h-8 w-8 p-0 shadow-lg"
            onClick={() => {
              bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
              setAutoScroll(true);
            }}
          >
            <ArrowDown className="h-4 w-4" />
          </Button>
        </div>
      )}

      {/* Input bar */}
      {status !== 'dead' && (
        <div className="border-t border-border px-4 py-3 shrink-0">
          <div className="max-w-3xl mx-auto flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send message to Claude..."
              className="flex-1 rounded-md px-3 py-1.5 text-sm text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 bg-muted/30 border border-border focus:ring-blue-500/50"
              disabled={!connected}
            />
            <Button
              variant="outline"
              size="sm"
              onClick={handleSend}
              disabled={!inputText.trim() || !connected}
              className="h-8"
            >
              <Send className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
