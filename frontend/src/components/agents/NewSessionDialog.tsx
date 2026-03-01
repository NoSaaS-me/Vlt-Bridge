/**
 * NewSessionDialog — Start a fresh session or resume an existing one.
 *
 * "Start Fresh" spawns a brand new Claude session with a user-provided prompt.
 * "Resume" picks an idle/dead session from the current project so the user can
 * continue a previous conversation.
 */
import { useState, useEffect, useCallback } from 'react';
import { Bot, Play, RotateCcw, Send } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { listAllSessions, type AgentSession } from '@/services/daemon-api';
import { statusColor, shortPath, timeAgo } from './utils';
import { cn } from '@/lib/utils';

export function NewSessionDialog({
  open,
  onOpenChange,
  projectId,
  spawnCwd,
  onStartFresh,
  onResume,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectId: string | null;
  spawnCwd: string;
  /** Called with user prompt when "Start Fresh" is clicked. */
  onStartFresh: (prompt: string) => void;
  /** Called when user picks a session to resume. */
  onResume: (session: AgentSession) => void;
}) {
  const [prompt, setPrompt] = useState('Hello! Ready for instructions.');
  const [resumable, setResumable] = useState<AgentSession[]>([]);
  const [loading, setLoading] = useState(false);

  // Fetch resumable sessions when dialog opens
  useEffect(() => {
    if (!open || !projectId) return;
    setLoading(true);
    listAllSessions(projectId)
      .then((all) => {
        // Show idle and dead sessions that can be resumed
        const candidates = all
          .filter((s) => s.status === 'idle' || s.status === 'dead')
          .sort(
            (a, b) =>
              new Date(b.last_activity).getTime() -
              new Date(a.last_activity).getTime(),
          );
        setResumable(candidates);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [open, projectId]);

  const handleFresh = useCallback(() => {
    const text = prompt.trim() || 'Hello! Ready for instructions.';
    onStartFresh(text);
    onOpenChange(false);
  }, [prompt, onStartFresh, onOpenChange]);

  const handleResume = useCallback(
    (session: AgentSession) => {
      onResume(session);
      onOpenChange(false);
    },
    [onResume, onOpenChange],
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md p-0 gap-0">
        <DialogHeader className="px-4 pt-4 pb-2">
          <DialogTitle className="text-sm">New Session</DialogTitle>
        </DialogHeader>

        {/* Start Fresh */}
        <div className="px-4 py-3">
          <div className="flex items-center gap-2 mb-2">
            <Play className="h-3.5 w-3.5 text-emerald-400" />
            <span className="text-xs font-medium">Start Fresh</span>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleFresh();
              }}
              placeholder="Initial prompt..."
              className="flex-1 rounded-md bg-muted/30 border border-border px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:ring-1 focus:ring-blue-500/50"
            />
            <Button
              size="sm"
              className="h-7 px-3 text-xs"
              onClick={handleFresh}
            >
              <Send className="h-3 w-3 mr-1" />
              Go
            </Button>
          </div>
          <p className="mt-1.5 text-[10px] text-muted-foreground/60">
            Creates a brand new session in{' '}
            <span className="font-mono">{shortPath(spawnCwd)}</span>
          </p>
        </div>

        <Separator />

        {/* Resume Existing */}
        <div className="px-4 py-3">
          <div className="flex items-center gap-2 mb-2">
            <RotateCcw className="h-3.5 w-3.5 text-blue-400" />
            <span className="text-xs font-medium">Resume Existing</span>
          </div>

          <div className="max-h-48 overflow-y-auto space-y-1">
            {loading ? (
              <div className="py-4 text-center text-[10px] text-muted-foreground">
                Loading sessions...
              </div>
            ) : resumable.length === 0 ? (
              <div className="py-4 text-center text-[10px] text-muted-foreground">
                No resumable sessions in this project
              </div>
            ) : (
              resumable.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleResume(s)}
                  className="w-full flex items-center gap-2 rounded-md border border-border px-2.5 py-2 text-left hover:bg-muted/60 transition-colors"
                >
                  <span
                    className={cn(
                      'h-2 w-2 rounded-full shrink-0 border',
                      statusColor(s.status),
                    )}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <Bot className="h-3 w-3 text-muted-foreground shrink-0" />
                      <span className="text-[11px] font-medium truncate">
                        {s.name || 'unnamed'}
                      </span>
                      <span
                        className={cn(
                          'text-[9px] px-1 rounded border',
                          statusColor(s.status),
                        )}
                      >
                        {s.status}
                      </span>
                    </div>
                    <div className="text-[10px] text-muted-foreground truncate">
                      {shortPath(s.cwd)} · {timeAgo(s.last_activity)}
                    </div>
                  </div>
                </button>
              ))
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
