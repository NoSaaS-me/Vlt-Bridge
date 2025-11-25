/**
 * T080, T083-T084: Main application layout with two-pane design
 * Loads directory tree on mount and note + backlinks when path changes
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Settings as SettingsIcon, FolderPlus } from 'lucide-react';
import { ResizablePanelGroup, ResizablePanel, ResizableHandle } from '@/components/ui/resizable';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { DirectoryTree } from '@/components/DirectoryTree';
import { DirectoryTreeSkeleton } from '@/components/DirectoryTreeSkeleton';
import { SearchBar } from '@/components/SearchBar';
import { NoteViewer } from '@/components/NoteViewer';
import { NoteViewerSkeleton } from '@/components/NoteViewerSkeleton';
import { NoteEditor } from '@/components/NoteEditor';
import { useToast } from '@/hooks/useToast';
import {
  listNotes,
  getNote,
  getBacklinks,
  getIndexHealth,
  createNote,
  moveNote,
  type BacklinkResult,
  APIException,
} from '@/services/api';
import {
  Dialog,
  DialogClose,
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

export function MainApp() {
  const navigate = useNavigate();
  const toast = useToast();
  const [notes, setNotes] = useState<NoteSummary[]>([]);
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [currentNote, setCurrentNote] = useState<Note | null>(null);
  const [backlinks, setBacklinks] = useState<BacklinkResult[]>([]);
  const [isLoadingNotes, setIsLoadingNotes] = useState(true);
  const [isLoadingNote, setIsLoadingNote] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isEditMode, setIsEditMode] = useState(false);
  const [indexHealth, setIndexHealth] = useState<IndexHealth | null>(null);
  const [isNewNoteDialogOpen, setIsNewNoteDialogOpen] = useState(false);
  const [newNoteName, setNewNoteName] = useState('');
  const [isCreatingNote, setIsCreatingNote] = useState(false);
  const [isNewFolderDialogOpen, setIsNewFolderDialogOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [isCreatingFolder, setIsCreatingFolder] = useState(false);

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

  // Handle note dialog open change
  const handleDialogOpenChange = (open: boolean) => {
    setIsNewNoteDialogOpen(open);
    if (!open) {
      // Clear input when dialog closes
      setNewNoteName('');
    }
  };

  // Handle folder dialog open change
  const handleFolderDialogOpenChange = (open: boolean) => {
    setIsNewFolderDialogOpen(open);
    if (!open) {
      // Clear input when dialog closes
      setNewFolderName('');
    }
  };

  // Handle create new note
  const handleCreateNote = async () => {
    if (!newNoteName.trim() || isCreatingNote) return;

    setIsCreatingNote(true);
    setError(null);

    try {
      const baseName = newNoteName.replace(/\.md$/, '');
      let notePath = newNoteName.endsWith('.md') ? newNoteName : `${newNoteName}.md`;
      let attempt = 1;
      const maxAttempts = 100;

      // Retry with number suffix if note already exists
      while (attempt <= maxAttempts) {
        try {
          const note = await createNote({
            note_path: notePath,
            title: baseName,
            body: `# ${baseName}\n\nStart writing your note here...`,
          });

          // Refresh notes list
          const notesList = await listNotes();
          setNotes(notesList);

          // Select the new note
          setSelectedPath(note.note_path);
          setIsEditMode(true);
          const displayName = notePath.replace(/\.md$/, '');
          toast.success(`Note "${displayName}" created successfully`);
          break;
        } catch (err) {
          if (err instanceof APIException && err.status === 409) {
            // Note already exists, try with number suffix
            attempt++;
            if (attempt <= maxAttempts) {
              notePath = `${baseName} ${attempt}.md`;
              continue;
            } else {
              throw err;
            }
          } else {
            throw err;
          }
        }
      }
    } catch (err) {
      let errorMessage = 'Failed to create note';
      if (err instanceof APIException) {
        // Use the message field which contains the actual error description
        errorMessage = err.message || err.error;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      toast.error(errorMessage);
      console.error('Error creating note:', err);
    } finally {
      setIsCreatingNote(false);
      // Always close dialog, regardless of success or failure
      handleDialogOpenChange(false);
    }
  };

  // Handle create new folder
  const handleCreateFolder = async () => {
    if (!newFolderName.trim() || isCreatingFolder) return;

    setIsCreatingFolder(true);
    setError(null);

    try {
      // Create a placeholder note in the folder
      const folderPath = newFolderName.replace(/\/$/, ''); // Remove trailing slash if present
      const placeholderPath = `${folderPath}/.placeholder.md`;

      const note = await createNote({
        note_path: placeholderPath,
        title: 'Folder',
        body: `# ${folderPath}\n\nThis folder was created.`,
      });

      // Refresh notes list
      const notesList = await listNotes();
      setNotes(notesList);

      toast.success(`Folder "${folderPath}" created successfully`);
    } catch (err) {
      let errorMessage = 'Failed to create folder';
      if (err instanceof APIException) {
        errorMessage = err.message || err.error;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      toast.error(errorMessage);
      console.error('Error creating folder:', err);
    } finally {
      setIsCreatingFolder(false);
      // Always close dialog, regardless of success or failure
      handleFolderDialogOpenChange(false);
    }
  };

  // Handle rename note
  const handleRenameNote = async (oldPath: string, newPath: string) => {
    if (!newPath.trim()) {
      toast.error('New path cannot be empty');
      return;
    }

    try {
      // Ensure new path has .md extension
      const finalNewPath = newPath.endsWith('.md') ? newPath : `${newPath}.md`;

      await moveNote(oldPath, finalNewPath);

      // Refresh notes list
      const notesList = await listNotes();
      setNotes(notesList);

      // If renaming currently selected note, update selection
      if (selectedPath === oldPath) {
        setSelectedPath(finalNewPath);
      }

      toast.success(`Note renamed successfully`);
    } catch (err) {
      let errorMessage = 'Failed to rename note';
      if (err instanceof APIException) {
        errorMessage = err.message || err.error;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      toast.error(errorMessage);
      console.error('Error renaming note:', err);
    }
  };

  // Handle dragging file to folder
  const handleMoveNoteToFolder = async (oldPath: string, targetFolderPath: string) => {
    try {
      // Get the filename from the old path
      const fileName = oldPath.split('/').pop();
      if (!fileName) {
        toast.error('Invalid file path');
        return;
      }

      // Construct new path: targetFolder/fileName
      const newPath = targetFolderPath ? `${targetFolderPath}/${fileName}` : fileName;

      // Don't move if source and destination are the same
      if (newPath === oldPath) {
        return;
      }

      await moveNote(oldPath, newPath);

      // Refresh notes list
      const notesList = await listNotes();
      setNotes(notesList);

      // If moving currently selected note, update selection
      if (selectedPath === oldPath) {
        setSelectedPath(newPath);
      }

      toast.success(`Note moved successfully`);
    } catch (err) {
      let errorMessage = 'Failed to move note';
      if (err instanceof APIException) {
        errorMessage = err.message || err.error;
      } else if (err instanceof Error) {
        errorMessage = err.message;
      }
      toast.error(errorMessage);
      console.error('Error moving note:', err);
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
      <div className="border-b border-border p-4 animate-fade-in">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">📚 Document Viewer</h1>
          <Button variant="ghost" size="sm" onClick={() => navigate('/settings')}>
            <SettingsIcon className="h-4 w-4" />
          </Button>
=======
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
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setIsNewNoteDialogOpen(false)}>
                    Cancel
                  </Button>
                  <Button onClick={handleCreateNote} disabled={!newNoteName.trim()}>
                    Create Note
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
            <Button variant="ghost" size="sm" onClick={() => navigate('/settings')}>
              <SettingsIcon className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 overflow-hidden animate-fade-in" style={{ animationDelay: '0.1s' }}>
        <ResizablePanelGroup direction="horizontal">
          {/* Left sidebar */}
          <ResizablePanel defaultSize={25} minSize={15} maxSize={40}>
            <div className="h-full flex flex-col">
              <div className="p-4 space-y-4">
                <Dialog
                  open={isNewNoteDialogOpen}
                  onOpenChange={handleDialogOpenChange}
                >
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full">
                      <Plus className="h-4 w-4 mr-1" />
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
                    </div>
                    <DialogFooter>
                      <Button
                        variant="outline"
                        onClick={() => setIsNewNoteDialogOpen(false)}
                        disabled={isCreatingNote}
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleCreateNote}
                        disabled={!newNoteName.trim() || isCreatingNote}
                      >
                        {isCreatingNote ? 'Creating...' : 'Create Note'}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                <Dialog
                  open={isNewFolderDialogOpen}
                  onOpenChange={handleFolderDialogOpenChange}
                >
                  <DialogTrigger asChild>
                    <Button variant="outline" size="sm" className="w-full">
                      <FolderPlus className="h-4 w-4 mr-1" />
                      New Folder
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create New Folder</DialogTitle>
                      <DialogDescription>
                        Enter a name for your new folder. You can use forward slashes for nested folders (e.g., "Projects/Work").
                      </DialogDescription>
                    </DialogHeader>
                    <div className="grid gap-4 py-4">
                      <div className="grid gap-2">
                        <label htmlFor="folder-name" className="text-sm font-medium">Folder Name</label>
                        <Input
                          id="folder-name"
                          placeholder="my-folder"
                          value={newFolderName}
                          onChange={(e) => setNewFolderName(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') {
                              handleCreateFolder();
                            }
                          }}
                        />
                      </div>
                    </div>
                    <DialogFooter>
                      <Button
                        variant="outline"
                        onClick={() => setIsNewFolderDialogOpen(false)}
                        disabled={isCreatingFolder}
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleCreateFolder}
                        disabled={!newFolderName.trim() || isCreatingFolder}
                      >
                        {isCreatingFolder ? 'Creating...' : 'Create Folder'}
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
                <SearchBar onSelectNote={handleSelectNote} />
                <Separator />
              </div>
              <div className="flex-1 overflow-hidden">
                {isLoadingNotes ? (
                  <DirectoryTreeSkeleton />
                ) : (
                  <DirectoryTree
                    notes={notes}
                    selectedPath={selectedPath || undefined}
                    onSelectNote={handleSelectNote}
                    onMoveNote={handleMoveNoteToFolder}
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

              {isLoadingNote ? (
                <NoteViewerSkeleton />
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
              )}
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>

      {/* Footer with Index Health */}
      <div className="border-t border-border px-4 py-2 text-xs text-muted-foreground animate-fade-in" style={{ animationDelay: '0.2s' }}>
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

