/**
 * T078: Note viewer with rendered markdown, metadata, and backlinks
 * T081-T082: Wikilink click handling and broken link styling
 */
import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Edit, Trash2, Calendar, Tag as TagIcon, ArrowLeft, Volume2, Pause, Play, Square } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import type { Note } from '@/types/note';
import type { BacklinkResult } from '@/services/api';
import { createWikilinkComponent } from '@/lib/markdown.tsx';

interface NoteViewerProps {
  note: Note;
  backlinks: BacklinkResult[];
  onEdit?: () => void;
  onDelete?: () => void;
  onWikilinkClick: (linkText: string) => void;
  ttsStatus?: 'idle' | 'loading' | 'playing' | 'paused' | 'error';
  onTtsToggle?: () => void;
  onTtsStop?: () => void;
  ttsDisabledReason?: string;
}

export function NoteViewer({
  note,
  backlinks,
  onEdit,
  onDelete,
  onWikilinkClick,
  ttsStatus = 'idle',
  onTtsToggle,
  onTtsStop,
  ttsDisabledReason,
}: NoteViewerProps) {
  // Create custom markdown components with wikilink handler
  const markdownComponents = useMemo(
    () => createWikilinkComponent(onWikilinkClick),
    [onWikilinkClick]
  );

  // Pre-process markdown to convert wikilinks to standard links
  // [[Link]] -> [Link](wikilink:Link)
  // [[Link|Alias]] -> [Alias](wikilink:Link)
  const processedBody = useMemo(() => {
    if (!note.body) return '';
    const processed = note.body.replace(/\[\[([^\]]+)\]\]/g, (_match, content) => {
      const [link, alias] = content.split('|');
      const displayText = alias || link;
      const href = `wikilink:${encodeURIComponent(link)}`;
      return `[${displayText}](${href})`;
    });
    // console.log('Processed Body:', processed);
    return processed;
  }, [note.body]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border p-4 animate-fade-in">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-3xl font-bold truncate animate-slide-in-up" style={{ animationDelay: '0.1s' }}>{note.title}</h1>
            <p className="text-sm text-muted-foreground mt-1 animate-fade-in" style={{ animationDelay: '0.2s' }}>{note.note_path}</p>
          </div>
          <div className="flex gap-2 animate-fade-in" style={{ animationDelay: '0.15s' }}>
            {onTtsToggle && (
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onTtsToggle}
                  disabled={Boolean(ttsDisabledReason) || ttsStatus === 'loading'}
                  title={ttsDisabledReason || undefined}
                >
                  {ttsStatus === 'playing' ? (
                    <Pause className="h-4 w-4 mr-2" />
                  ) : ttsStatus === 'paused' ? (
                    <Play className="h-4 w-4 mr-2" />
                  ) : (
                    <Volume2 className="h-4 w-4 mr-2" />
                  )}
                  {ttsStatus === 'playing'
                    ? 'Pause TTS'
                    : ttsStatus === 'paused'
                    ? 'Resume TTS'
                    : ttsStatus === 'loading'
                    ? 'Loading...'
                    : 'TTS Mode'}
                </Button>
                {(ttsStatus === 'playing' || ttsStatus === 'paused' || ttsStatus === 'error') && onTtsStop && (
                  <Button variant="ghost" size="sm" onClick={onTtsStop} title="Stop TTS">
                    <Square className="h-4 w-4" />
                  </Button>
                )}
              </div>
            )}
            {onEdit && (
              <Button variant="outline" size="sm" onClick={onEdit}>
                <Edit className="h-4 w-4 mr-2" />
                Edit
              </Button>
            )}
            {onDelete && (
              <Button variant="outline" size="sm" onClick={onDelete}>
                <Trash2 className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Content */}
      <ScrollArea className="flex-1 p-6">
        <div className="prose prose-slate dark:prose-invert max-w-none animate-fade-in" style={{ animationDelay: '0.3s' }}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
            urlTransform={(url) => url} // Allow all protocols including wikilink:
          >
            {processedBody}
          </ReactMarkdown>
        </div>

        <Separator className="my-8" />

        {/* Metadata Footer */}
        <div className="space-y-4 text-sm animate-fade-in" style={{ animationDelay: '0.4s' }}>
          {/* Tags */}
          {note.metadata.tags && note.metadata.tags.length > 0 && (
            <div className="flex items-start gap-2">
              <TagIcon className="h-4 w-4 mt-0.5 text-muted-foreground shrink-0" />
              <div className="flex flex-wrap gap-2">
                {note.metadata.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Timestamps */}
          <div className="flex items-center gap-4 text-muted-foreground">
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              <span>Created: {formatDate(note.created)}</span>
            </div>
            <div className="flex items-center gap-2">
              <Calendar className="h-4 w-4" />
              <span>Updated: {formatDate(note.updated)}</span>
            </div>
          </div>

          {/* Backlinks */}
          {backlinks.length > 0 && (
            <>
              <Separator className="my-4" />
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <ArrowLeft className="h-4 w-4 text-muted-foreground" />
                  <h3 className="font-semibold">
                    Backlinks ({backlinks.length})
                  </h3>
                </div>
                <div className="space-y-2 ml-6">
                  {backlinks.map((backlink) => (
                    <button
                      key={backlink.note_path}
                      className="block text-left text-primary hover:underline"
                      onClick={() => onWikilinkClick(backlink.title)}
                    >
                      {'-> '}
                      {backlink.title}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Additional metadata */}
          {note.metadata.project && (
            <div className="text-muted-foreground">
              Project: <span className="font-medium">{note.metadata.project}</span>
            </div>
          )}

          <div className="text-xs text-muted-foreground">
            Version: {note.version} | Size: {(note.size_bytes / 1024).toFixed(1)} KB
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}

