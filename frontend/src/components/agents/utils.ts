/**
 * Agents page helpers — shared across compositor components.
 */
import { parseDaemonTs, type AgentSession } from '@/services/daemon-api';

export function statusColor(status: AgentSession['status']): string {
  switch (status) {
    case 'thinking':  return 'text-blue-400 bg-blue-500/15 border-blue-500/30';
    case 'executing': return 'text-amber-400 bg-amber-500/15 border-amber-500/30';
    case 'idle':      return 'text-emerald-400 bg-emerald-500/15 border-emerald-500/30';
    case 'dead':      return 'text-muted-foreground bg-muted/50 border-border';
    default:          return 'text-muted-foreground bg-muted/50 border-border';
  }
}

export function isPulsing(status: AgentSession['status']): boolean {
  return status === 'thinking' || status === 'executing';
}

export function shortPath(cwd: string): string {
  if (!cwd) return '—';
  const parts = cwd.split('/');
  return parts.slice(-2).join('/') || cwd;
}

export function timeAgo(isoStr: string): string {
  const diff = Date.now() - parseDaemonTs(isoStr).getTime();
  const secs = Math.floor(Math.abs(diff) / 1000);
  const sign = diff < 0 ? '+' : '';
  if (secs < 60) return `${sign}${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${sign}${mins}m ago`;
  return `${sign}${Math.floor(mins / 60)}h ago`;
}

/** xterm.js dark theme — GitHub-style colors */
export const XTERM_THEME = {
  background: '#0a0f1a',
  foreground: '#c9d1d9',
  cursor: 'transparent',
  cursorAccent: 'transparent',
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
} as const;

/** Event type → compact label */
export const EVENT_LABEL: Record<string, string> = {
  UserPromptSubmit: 'prompt',
  PostToolUse: 'tool',
  PreToolUse: 'pre-tool',
  Stop: 'stop',
  SessionStart: 'start',
  SessionEnd: 'end',
  SubagentStart: 'subagent',
};
