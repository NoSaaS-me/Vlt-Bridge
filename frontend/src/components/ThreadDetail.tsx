/**
 * ThreadDetail - Display a thread's summary and entries.
 * 
 * Layout:
 * - Top third (33%): Thread summary/state
 * - Divider
 * - Bottom two-thirds (67%): Individual entries as items
 */

import { useState, useEffect } from 'react';
import { X, Loader2, MessageSquare, GitBranch, Clock, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { getThread } from '@/services/api';
import type { Thread } from '@/types/thread';
import ReactMarkdown from 'react-markdown';

interface ThreadDetailProps {
  threadId: string;
  onClose: () => void;
  projectId: string | null;
}

function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
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

export function ThreadDetail({ threadId, onClose }: ThreadDetailProps) {
  const [thread, setThread] = useState<Thread | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadThread = async () => {
      setIsLoading(true);
      setError(null);
      try {
        const threadData = await getThread(threadId);
        setThread(threadData);
      } catch (err) {
        console.error('Failed to load thread:', err);
        setError(err instanceof Error ? err.message : 'Failed to load thread');
      } finally {
        setIsLoading(false);
      }
    };

    if (threadId) {
      loadThread();
    }
  }, [threadId]);

  if (isLoading) {
    return (
      <div className="flex flex-col h-full bg-background">
        <div className="flex items-center justify-center h-full">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-full bg-background">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-muted-foreground" />
              <h2 className="font-semibold">Thread</h2>
            </div>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="flex items-center justify-center h-full text-destructive">
          <p>{error}</p>
        </div>
      </div>
    );
  }

  if (!thread) {
    return (
      <div className="flex flex-col h-full bg-background">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <GitBranch className="h-5 w-5 text-muted-foreground" />
              <h2 className="font-semibold">Thread</h2>
            </div>
            <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
        <div className="flex items-center justify-center h-full text-muted-foreground">
          <p>Thread not found</p>
        </div>
      </div>
    );
  }

  const entries = thread.entries || [];
  // Reverse to show newest first
  const sortedEntries = [...entries].sort((a, b) => b.sequence_id - a.sequence_id);

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="h-5 w-5 text-muted-foreground" />
            <div>
              <h2 className="font-semibold truncate max-w-md">{thread.name}</h2>
              <div className="flex items-center gap-2 mt-1">
                <Badge
                  variant="outline"
                  className={`text-xs ${getStatusColor(thread.status)}`}
                >
                  {thread.status}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {entries.length} entries
                </span>
              </div>
            </div>
          </div>
          <Button variant="ghost" size="icon" className="h-8 w-8" onClick={onClose} title="Close thread view">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 flex flex-col min-h-0">
        {/* Top third: Summary */}
        <div className="h-[33.333%] flex flex-col min-h-0">
          <div className="px-4 py-2 bg-muted/50 border-b border-border">
            <h3 className="text-sm font-medium text-muted-foreground">Thread State / Summary</h3>
          </div>
          <ScrollArea className="flex-1">
            <div className="p-4">
              {sortedEntries.length > 0 ? (
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <ReactMarkdown>
                    {sortedEntries[0]?.content || 'No summary available.'}
                  </ReactMarkdown>
                </div>
              ) : (
                <p className="text-muted-foreground text-sm">No entries in this thread yet.</p>
              )}
            </div>
          </ScrollArea>
        </div>

        {/* Divider */}
        <Separator className="bg-border" />

        {/* Bottom two-thirds: Individual entries */}
        <div className="h-[66.667%] flex flex-col min-h-0">
          <div className="px-4 py-2 bg-muted/50 border-b border-border">
            <h3 className="text-sm font-medium text-muted-foreground">Recent Thoughts / Entries</h3>
          </div>
          <ScrollArea className="flex-1">
            <div className="divide-y divide-border">
              {sortedEntries.length > 0 ? (
                sortedEntries.map((entry, index) => (
                  <div key={entry.sequence_id} className="p-4 hover:bg-accent/30 transition-colors">
                    <div className="flex items-start gap-3">
                      <div className="flex flex-col items-center gap-1">
                        <div className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium text-primary">
                          {sortedEntries.length - index}
                        </div>
                        {index < sortedEntries.length - 1 && (
                          <div className="w-px h-full bg-border" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0 pb-4">
                        <div className="flex items-center gap-2 mb-2">
                          {entry.author && (
                            <span className="flex items-center gap-1 text-xs text-muted-foreground">
                              <User className="h-3 w-3" />
                              {entry.author}
                            </span>
                          )}
                          <span className="flex items-center gap-1 text-xs text-muted-foreground">
                            <Clock className="h-3 w-3" />
                            {formatDate(entry.created_at)}
                          </span>
                        </div>
                        <div className="prose prose-sm dark:prose-invert max-w-none">
                          <ReactMarkdown>
                            {entry.content}
                          </ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center h-32 text-muted-foreground p-4 text-center">
                  <MessageSquare className="h-8 w-8 mb-2 opacity-50" />
                  <p className="text-sm">No entries yet</p>
                </div>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
