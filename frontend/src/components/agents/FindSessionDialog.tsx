/**
 * FindSessionDialog — Searchable dialog to find and assign existing sessions
 * to the current project.
 */
import { useState, useEffect } from 'react';
import { Bot, Check } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  Command,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
} from '@/components/ui/command';
import { listAllSessions, assignSessionToProject, type AgentSession } from '@/services/daemon-api';
import { statusColor, shortPath, timeAgo } from './utils';
import { cn } from '@/lib/utils';

export function FindSessionDialog({
  open,
  onOpenChange,
  currentProjectId,
  onAssigned,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentProjectId: string;
  onAssigned: () => void;
}) {
  const [allSessions, setAllSessions] = useState<AgentSession[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    setLoading(true);
    listAllSessions()
      .then(setAllSessions)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [open]);

  const handleSelect = async (session: AgentSession) => {
    if (session.project_id === currentProjectId) return;
    try {
      await assignSessionToProject(session.id, currentProjectId);
      onAssigned();
      onOpenChange(false);
    } catch (err) {
      console.error('Failed to assign session:', err);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-lg p-0 gap-0">
        <DialogHeader className="px-4 pt-4 pb-2">
          <DialogTitle className="text-sm">Find Session</DialogTitle>
        </DialogHeader>
        <Command className="border-t border-border">
          <CommandInput placeholder="Search by name or path..." />
          <CommandList className="max-h-72">
            <CommandEmpty>
              {loading ? 'Loading sessions...' : 'No sessions found.'}
            </CommandEmpty>
            <CommandGroup>
              {allSessions.map((s) => {
                const alreadyHere = s.project_id === currentProjectId;
                return (
                  <CommandItem
                    key={s.id}
                    value={`${s.name} ${s.cwd} ${s.id}`}
                    onSelect={() => handleSelect(s)}
                    disabled={alreadyHere}
                    className="flex items-center gap-2 px-3 py-2"
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
                        <span className="text-xs font-medium truncate">
                          {s.name || 'unnamed'}
                        </span>
                        {alreadyHere && (
                          <span className="text-[9px] text-muted-foreground bg-muted px-1 rounded shrink-0">
                            already here
                          </span>
                        )}
                      </div>
                      <div className="text-[10px] text-muted-foreground truncate">
                        {shortPath(s.cwd)} · {timeAgo(s.last_activity)}
                      </div>
                    </div>
                    {alreadyHere && (
                      <Check className="h-3 w-3 text-emerald-400 shrink-0" />
                    )}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  );
}
