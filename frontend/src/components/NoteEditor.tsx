/**
 * T089-T092: Note editor with split-pane and conflict handling
 */
import { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Save, X, AlertCircle } from 'lucide-react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import type { Note } from '@/types/note';
import { createWikilinkComponent } from '@/lib/markdown.tsx';
import { updateNote, APIException } from '@/services/api';

interface NoteEditorProps {
  note: Note;
  onSave: (updatedNote: Note) => void;
  onCancel: () => void;
  onWikilinkClick: (linkText: string) => void;
}

export function NoteEditor({ note, onSave, onCancel, onWikilinkClick }: NoteEditorProps) {
  const [body, setBody] = useState(note.body);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasChanges, setHasChanges] = useState(false);

  // Track if content has changed
  useEffect(() => {
    setHasChanges(body !== note.body);
  }, [body, note.body]);

  // Create custom markdown components with wikilink handler
  const markdownComponents = createWikilinkComponent(onWikilinkClick);

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);

    try {
      // T090: Send update with version for optimistic concurrency
      const updatedNote = await updateNote(note.note_path, {
        body,
        if_version: note.version,
      });
      
      onSave(updatedNote);
    } catch (err) {
      // T092: Handle 409 Conflict
      if (err instanceof APIException && err.status === 409) {
        setError(
          'This note changed since you opened it. Please reload before saving to avoid losing changes.'
        );
      } else if (err instanceof APIException) {
        setError(err.error);
      } else {
        setError('Failed to save note');
      }
      console.error('Error saving note:', err);
    } finally {
      setIsSaving(false);
    }
  };

  // T091: Cancel handler
  const handleCancel = () => {
    if (hasChanges) {
      const confirmed = window.confirm(
        'You have unsaved changes. Are you sure you want to cancel?'
      );
      if (!confirmed) return;
    }
    onCancel();
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl + S to save
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        if (hasChanges && !isSaving) {
          handleSave();
        }
      }
      // Escape to cancel
      if (e.key === 'Escape') {
        e.preventDefault();
        handleCancel();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [hasChanges, isSaving, body]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h2 className="text-xl font-semibold truncate">{note.title}</h2>
            <p className="text-xs text-muted-foreground mt-1">
              {note.note_path} | Version {note.version}
              {hasChanges && ' | Unsaved changes'}
            </p>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleCancel}
              disabled={isSaving}
            >
              <X className="h-4 w-4 mr-2" />
              Cancel
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!hasChanges || isSaving}
            >
              <Save className="h-4 w-4 mr-2" />
              {isSaving ? 'Saving...' : 'Save'}
            </Button>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <Alert variant="destructive" className="mt-4">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </div>

      {/* Split Pane Editor */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal">
          {/* Left: Markdown Source */}
          <ResizablePanel defaultSize={50} minSize={30}>
            <div className="h-full flex flex-col p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-sm font-medium text-muted-foreground">
                  Markdown Source
                </h3>
                <div className="flex gap-2">
                  <Badge variant="secondary" className="text-xs">
                    {body.length} chars
                  </Badge>
                  <Badge variant="secondary" className="text-xs">
                    {body.split('\n').length} lines
                  </Badge>
                </div>
              </div>
              <Textarea
                value={body}
                onChange={(e) => setBody(e.target.value)}
                className="flex-1 font-mono text-sm resize-none"
                placeholder="Write your markdown here..."
              />
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Right: Live Preview */}
          <ResizablePanel defaultSize={50} minSize={30}>
            <div className="h-full overflow-auto p-4">
              <div className="mb-2">
                <h3 className="text-sm font-medium text-muted-foreground">
                  Live Preview
                </h3>
              </div>
              <div className="prose prose-slate dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={markdownComponents}
                >
                  {body || '*No content*'}
                </ReactMarkdown>
              </div>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Footer with keyboard shortcuts hint */}
      <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
        <span className="mr-4">
          Tip: <kbd className="px-1.5 py-0.5 bg-muted rounded">Cmd/Ctrl+S</kbd> to save,{' '}
          <kbd className="px-1.5 py-0.5 bg-muted rounded">Esc</kbd> to cancel
        </span>
      </div>
    </div>
  );
}

