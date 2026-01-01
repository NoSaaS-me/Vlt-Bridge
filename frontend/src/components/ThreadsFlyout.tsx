/**
 * ThreadsFlyout - List of threads for the current project.
 * Uses the same flyout pattern as ChatPanel.
 */

import { useState, useEffect } from 'react';
import { X, Loader2, GitBranch, Clock, MessageSquare } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { listThreads } from '@/services/api';
import type { Thread } from '@/types/thread';

interface ThreadsFlyoutProps {
  projectId: string | null;
  onSelectThread?: (threadId: string) => void;
  onClose: () => void;
}

function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'active':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'archived':
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    case 'blocked':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    default:
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  }
}

export function ThreadsFlyout({
  projectId,
  onSelectThread,
  onClose,
}: ThreadsFlyoutProps) {
  const [threads, setThreads] = useState<Thread[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadThreads = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const response = await listThreads(projectId || undefined);
        setThreads(response.threads);
      } catch (err) {
        console.error('Failed to load threads:', err);
        setError(err instanceof Error ? err.message : 'Failed to load threads');
      } finally {
        setIsLoading(false);
      }
    };

    loadThreads();
  }, [projectId]);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-muted-foreground" />
            <div>
              <h2 className="font-semibold">Threads</h2>
              <p className="text-xs text-muted-foreground">
                Development reasoning chains
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onClose}
            title="Close threads panel"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1">
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="p-4 text-center text-destructive">
            <p className="text-sm">{error}</p>
            <Button
              variant="ghost"
              size="sm"
              className="mt-2"
              onClick={() => {
                setIsLoading(true);
                setError(null);
                listThreads(projectId || undefined)
                  .then((response) => setThreads(response.threads))
                  .catch((err) => setError(err.message))
                  .finally(() => setIsLoading(false));
              }}
            >
              Retry
            </Button>
          </div>
        ) : threads.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-32 text-muted-foreground p-4 text-center">
            <MessageSquare className="h-8 w-8 mb-2 opacity-50" />
            <p className="text-sm">No threads yet</p>
            <p className="text-xs mt-1">
              Use <code className="px-1 bg-muted rounded">vlt thread new</code> to create threads
            </p>
          </div>
        ) : (
          <div className="divide-y divide-border">
            {threads.map((thread) => (
              <button
                key={thread.thread_id}
                className="w-full p-4 text-left hover:bg-accent/50 transition-colors"
                onClick={() => onSelectThread?.(thread.thread_id)}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="font-medium text-sm truncate">
                      {thread.name}
                    </div>
                    <div className="flex items-center gap-2 mt-1">
                      <Badge
                        variant="outline"
                        className={`text-xs ${getStatusColor(thread.status)}`}
                      >
                        {thread.status}
                      </Badge>
                      {thread.entry_count !== undefined && (
                        <span className="text-xs text-muted-foreground">
                          {thread.entry_count} entries
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground shrink-0">
                    <Clock className="h-3 w-3" />
                    {formatRelativeTime(thread.updated_at)}
                  </div>
                </div>
              </button>
            ))}
          </div>
        )}
      </ScrollArea>

      {/* Footer */}
      <div className="p-4 border-t border-border text-xs text-muted-foreground">
        {threads.length > 0 && (
          <span>{threads.length} thread{threads.length !== 1 ? 's' : ''}</span>
        )}
      </div>
    </div>
  );
}
