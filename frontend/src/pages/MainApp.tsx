/**
 * T080, T083-T084: Main application layout with two-pane design
 * Loads directory tree on mount and note + backlinks when path changes
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Settings as SettingsIcon } from 'lucide-react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { DirectoryTree } from '@/components/DirectoryTree';
import { SearchBar } from '@/components/SearchBar';
import { NoteViewer } from '@/components/NoteViewer';
import { NoteEditor } from '@/components/NoteEditor';
import { GraphView } from '@/components/GraphView';
import {
  listNotes,
  getNote,
  getBacklinks,
  getIndexHealth,
  createNote,
  type BacklinkResult,
  APIException,
} from '@/services/api';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import type { IndexHealth } from '@/types/search';
import type { Note, NoteSummary } from '@/types/note';
import { normalizeSlug } from '@/lib/wikilink';
import { Network } from 'lucide-react';

export function MainApp() {
  const navigate = useNavigate();
  const [notes, setNotes] = useState<NoteSummary[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [currentNote, setCurrentNote] = useState<Note | null>(null);
  const [backlinks, setBacklinks] = useState<BacklinkResult[]>([]);
  const [isLoadingNotes, setIsLoadingNotes] = useState(true);
  const [isLoadingNote, setIsLoadingNote] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);
  const [isGraphView, setIsGraphView] = useState(false);
  const [indexHealth, setIndexHealth] = useState<IndexHealth | null>(null);
  const [isNewNoteDialogOpen, setIsNewNoteDialogOpen] = useState(false);
  const [newNoteName, setNewNoteName] = useState('');

  // T083: Load directory tree on mount
  // T119: Load index health
  useEffect(() => {
    const loadData = async () => {
      setIsLoadingNotes(true);
      setError(null);
      try {
        // Load notes and index health in parallel
        const [notesList, health] = await Promise.all([
          listNotes(),
          getIndexHealth().catch(() => null), // Don't fail if health unavailable
        ]);
        
        setNotes(notesList);
        setIndexHealth(health);
        
        // Auto-select first note if available
        if (notesList.length > 0 && !selectedPath) {
          setSelectedPath(notesList[0].note_path);
        }
      } catch (err) {
        if (err instanceof APIException) {
          setError(err.error);
        } else {
          setError('Failed to load notes');
        }
        console.error('Error loading notes:', err);
      } finally {
        setIsLoadingNotes(false);
      }
    };

    loadData();
  }, []);

  // T084: Load note and backlinks when path changes
  useEffect(() => {
    if (!selectedPath) {
      setCurrentNote(null);
      setBacklinks([]);
      return;
    }

    const loadNote = async () => {
      setIsLoadingNote(true);
      setError(null);
      try {
        const [note, noteBacklinks] = await Promise.all([
          getNote(selectedPath),
          getBacklinks(selectedPath),
        ]);
        setCurrentNote(note);
        setBacklinks(noteBacklinks);
      } catch (err) {
        if (err instanceof APIException) {
          setError(err.error);
        } else {
          setError('Failed to load note');
        }
        console.error('Error loading note:', err);
        setCurrentNote(null);
        setBacklinks([]);
      } finally {
        setIsLoadingNote(false);
      }
    };

    loadNote();
  }, [selectedPath]);

  // Handle wikilink clicks
  const handleWikilinkClick = async (linkText: string) => {
    const slug = normalizeSlug(linkText);
    
    // Try to find exact match first
    let targetNote = notes.find(
      (note) => normalizeSlug(note.title) === slug
    );

    // If not found, try path-based matching
    if (!targetNote) {
      targetNote = notes.find((note) => {
        const pathSlug = normalizeSlug(note.note_path.replace(/\.md$/, ''));
        return pathSlug.endsWith(slug);
      });
    }

    if (targetNote) {
      setSelectedPath(targetNote.note_path);
    } else {
      // TODO: Show "Create note" dialog
      console.log('Note not found for wikilink:', linkText);
      setError(`Note not found: ${linkText}`);
    }
  };

  const handleSelectNote = (path: string) => {
    setSelectedPath(path);
    setError(null);
    setIsEditMode(false); // Exit edit mode when switching notes
  };

  // T093: Handle edit button click
  const handleEdit = () => {
    setIsEditMode(true);
  };

  // Handle note save from editor
  const handleNoteSave = (updatedNote: Note) => {
    setCurrentNote(updatedNote);
    setIsEditMode(false);
    setError(null);
    // Reload notes list to update modified timestamp
    listNotes().then(setNotes).catch(console.error);
  };

  // Handle editor cancel
  const handleEditCancel = () => {
    setIsEditMode(false);
  };

  // Handle create new note
  const handleCreateNote = async () => {
    if (!newNoteName.trim()) return;
    
    try {
      const notePath = newNoteName.endsWith('.md') ? newNoteName : `${newNoteName}.md`;
      const note = await createNote({
        note_path: notePath,
        title: newNoteName.replace(/\.md$/, ''),
        body: `# ${newNoteName.replace(/\.md$/, '')}\n\nStart writing your note here...`,
      });
      
      // Refresh notes list
      const notesList = await listNotes();
      setNotes(notesList);
      
      // Select the new note
      setSelectedPath(note.note_path);
      setIsNewNoteDialogOpen(false);
      setNewNoteName('');
      
      // Switch to edit mode for the new note
      setIsEditMode(true);
    } catch (err) {
      if (err instanceof APIException) {
        setError(err.error);
      } else {
        setError('Failed to create note');
      }
      console.error('Error creating note:', err);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Demo warning banner */}
      <Alert variant="destructive" className="rounded-none border-x-0 border-t-0">
        <AlertDescription className="text-center">
          DEMO ONLY - ALL DATA IS TEMPORARY AND MAY BE DELETED AT ANY TIME
        </AlertDescription>
      </Alert>
      
      {/* Top bar */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Document Viewer</h1>
          <div className="flex gap-2">
            <Dialog open={isNewNoteDialogOpen} onOpenChange={setIsNewNoteDialogOpen}>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  <Plus className="h-4 w-4 mr-2" />
                  New Note
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Create New Note</DialogTitle>
                  <DialogDescription>
                    Enter a name for your new note. The .md extension will be added automatically if not provided.
                  </DialogDescription>
                </DialogHeader>
                <div className="grid gap-4 py-4">
                  <div className="grid gap-2">
                    <label htmlFor="note-name" className="text-sm font-medium">Note Name</label>
                    <Input
                      id="note-name"
                      placeholder="my-note"
                      value={newNoteName}
                      onChange={(e) => setNewNoteName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') {
                          handleCreateNote();
                        }
                      }}
                    />
                  </div>
                </div>                <DialogFooter>
                  <Button variant="outline" onClick={() => setIsNewNoteDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreateNote} disabled={!newNoteName.trim()}>
                    Create Note
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
            <Button
              variant={isGraphView ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setIsGraphView(!isGraphView)}
              title={isGraphView ? "Switch to Note View" : "Switch to Graph View"}
            >
              <Network className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={() => navigate('/settings')}>
              <SettingsIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="horizontal">
          {/* Left sidebar */}
          <ResizablePanel defaultSize={25} minSize={15} maxSize={40}>
            <div className="h-full flex flex-col">
              <div className="p-4 space-y-4">
                <SearchBar onSelectNote={handleSelectNote} />
                <Separator />
              </div>
              <div className="flex-1 overflow-hidden">
                {isLoadingNotes ? (
                  <div className="p-4 text-center text-sm text-muted-foreground">
                    Loading notes...
                  </div>
                ) : (
                  <DirectoryTree
                    notes={notes}
                    selectedPath={selectedPath || undefined}
                    onSelectNote={handleSelectNote}
                  />
                )}
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Main content pane */}
          <ResizablePanel defaultSize={75}>
            <div className="h-full bg-background">
              {error && (
                <div className="p-4">
                  <Alert variant="destructive">
                    <AlertDescription>{error}</AlertDescription>
                  </Alert>
                </div>
              )}
              
              {isGraphView ? (
                <GraphView onSelectNote={(path) => {
                  handleSelectNote(path);
                  setIsGraphView(false);
                }} />
              ) : (
                isLoadingNote ? (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-muted-foreground">Loading note...</div>
                  </div>
                ) : currentNote ? (
                  isEditMode ? (
                    <NoteEditor
                      note={currentNote}
                      onSave={handleNoteSave}
                      onCancel={handleEditCancel}
                      onWikilinkClick={handleWikilinkClick}
                    />
                  ) : (
                    <NoteViewer
                      note={currentNote}
                      backlinks={backlinks}
                      onEdit={handleEdit}
                      onWikilinkClick={handleWikilinkClick}
                    />
                  )
                ) : (
                  <div className="flex items-center justify-center h-full">
                    <div className="text-center text-muted-foreground">
                      <p className="text-lg mb-2">Select a note to view</p>
                      <p className="text-sm">
                        {notes.length === 0
                          ? 'No notes available. Create your first note to get started.'
                          : 'Choose a note from the sidebar'}
                      </p>
                    </div>
                  </div>
                )
              )}
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Footer with Index Health */}
      <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground">
        <div className="flex items-center justify-between">
          <div>
            {indexHealth ? (
              <>
                <span className="font-medium">{indexHealth.note_count}</span> note{indexHealth.note_count !== 1 ? 's' : ''} indexed
              </>
            ) : (
              <>
                <span className="font-medium">{notes.length}</span> note{notes.length !== 1 ? 's' : ''} indexed
              </>
            )}
          </div>
          {indexHealth && indexHealth.last_incremental_update && (
            <div className="flex items-center gap-2">
              <span>Last updated:</span>
              <span className="font-medium">
                {new Date(indexHealth.last_incremental_update).toLocaleString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  hour: '2-digit',
                  minute: '2-digit',
                })}
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

