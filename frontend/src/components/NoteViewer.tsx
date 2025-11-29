/**
 * T078: Note viewer with rendered markdown, metadata, and backlinks
 * T081-T082: Wikilink click handling and broken link styling
 * T009: Font size buttons (A-, A, A+) for content adjustment
 * T037-T052: Table of Contents panel integration
 */
import { useMemo, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Edit, Trash2, Calendar, Tag as TagIcon, ArrowLeft, Volume2, Pause, Play, Square, Type, List } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import type { Note } from '@/types/note';
import type { BacklinkResult } from '@/services/api';
import { createWikilinkComponent, resetSlugCache } from '@/lib/markdown.tsx';
import { markdownToPlainText } from '@/lib/markdownToText';
import { useTableOfContents } from '@/hooks/useTableOfContents';
import { TableOfContents } from '@/components/TableOfContents';

type FontSizePreset = 'small' | 'medium' | 'large';

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
  ttsVolume?: number;
  onTtsVolumeChange?: (volume: number) => void;
  fontSize?: FontSizePreset;
  onFontSizeChange?: (size: FontSizePreset) => void;
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
  ttsVolume = 0.7,
  onTtsVolumeChange,
  fontSize = 'medium',
  onFontSizeChange,
}: NoteViewerProps) {
  // T042-T047: Table of Contents hook
  const { headings, isOpen: isTocOpen, setIsOpen: setIsTocOpen, scrollToHeading } = useTableOfContents();

  // Reset slug cache when note changes to ensure unique IDs
  useEffect(() => {
    resetSlugCache();
  }, [note.note_path]);

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

  // T031-T035: Calculate reading time from note body
  const readingTime = useMemo(() => {
    const plainText = markdownToPlainText(note.body);
    // T032: Extract word count
    const wordCount = plainText.trim().split(/\s+/).length;
    // T033: Calculate minutes at 200 WPM
    const minutes = Math.ceil(wordCount / 200);
    // T034: Return null if <1 minute (200 words threshold)
    return minutes >= 1 ? `${minutes} min read` : null;
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
            <div className="flex items-center gap-3 mb-1">
              <h1 className="text-3xl font-bold truncate animate-slide-in-up">{note.title}</h1>
              {/* T035: Render Badge with "X min read" near note title */}
              {readingTime && (
                <Badge variant="secondary" className="flex-shrink-0 animate-fade-in">
                  {readingTime}
                </Badge>
              )}
            </div>
            <p className="text-sm text-muted-foreground mt-1 animate-fade-in">{note.note_path}</p>
          </div>
          <div className="flex gap-2 animate-fade-in">
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
                {onTtsVolumeChange && (
                  <div className="flex items-center gap-2 ml-2 min-w-[120px]">
                    <Volume2 className="h-3 w-3 text-muted-foreground flex-shrink-0" />
                    <Slider
                      value={[ttsVolume * 100]}
                      onValueChange={(val: number[]) => onTtsVolumeChange(val[0] / 100)}
                      max={100}
                      step={1}
                      className="w-20"
                      title={`Volume: ${Math.round(ttsVolume * 100)}%`}
                    />
                    <span className="text-xs text-muted-foreground w-8 flex-shrink-0">
                      {Math.round(ttsVolume * 100)}%
                    </span>
                  </div>
                )}
              </div>
            )}
            {onFontSizeChange && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" title="Adjust font size">
                    <Type className="h-4 w-4 mr-2" />
                    A
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem
                    onClick={() => onFontSizeChange('small')}
                    className={fontSize === 'small' ? 'bg-accent' : ''}
                  >
                    <span className="text-xs">A-</span>
                    <span className="text-xs text-muted-foreground ml-2">Small (14px)</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => onFontSizeChange('medium')}
                    className={fontSize === 'medium' ? 'bg-accent' : ''}
                  >
                    <span className="text-sm">A</span>
                    <span className="text-xs text-muted-foreground ml-2">Medium (16px)</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    onClick={() => onFontSizeChange('large')}
                    className={fontSize === 'large' ? 'bg-accent' : ''}
                  >
                    <span className="text-lg">A+</span>
                    <span className="text-xs text-muted-foreground ml-2">Large (18px)</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {/* T044: TOC toggle button */}
            <Button
              variant={isTocOpen ? 'default' : 'outline'}
              size="sm"
              onClick={() => setIsTocOpen(!isTocOpen)}
              title="Toggle Table of Contents"
            >
              <List className="h-4 w-4 mr-2" />
              TOC
            </Button>
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

      {/* T045: Content with ResizablePanel for TOC */}
      <ResizablePanelGroup direction="horizontal" className="flex-1">
        {/* Main content panel */}
        <ResizablePanel defaultSize={isTocOpen ? 75 : 100}>
          <ScrollArea className="h-full p-6">
            <div className="prose prose-slate dark:prose-invert max-w-none animate-fade-in-smooth">
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
            <div className="space-y-4 text-sm animate-fade-in">
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
        </ResizablePanel>

        {/* T045: TOC panel with slide-in animation */}
        {isTocOpen && (
          <>
            <ResizableHandle withHandle />
            <ResizablePanel defaultSize={25} minSize={15} maxSize={40} className="animate-slide-in">
              <TableOfContents headings={headings} onHeadingClick={scrollToHeading} />
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
    </div>
  );
}

