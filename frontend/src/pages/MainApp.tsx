/**
 * T080, T083-T084: Main application layout with two-pane design
 * Loads directory tree on mount and note + backlinks when path changes
 */
import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Settings as SettingsIcon, FolderPlus, MessageCircle } from 'lucide-react';
import { useFontSize } from '@/hooks/useFontSize';
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
import { ChatPanel } from '@/components/ChatPanel';
import { useToast } from '@/hooks/useToast';
import { GraphView } from '@/components/GraphView';
import { GlowParticleEffect } from '@/components/GlowParticleEffect';
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
import { AUTH_TOKEN_CHANGED_EVENT, isDemoSession, login } from '@/services/auth';
import { synthesizeTts } from '@/services/tts';
import { markdownToPlainText } from '@/lib/markdownToText';
import { useAudioPlayer } from '@/hooks/useAudioPlayer';

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
  const [isGraphView, setIsGraphView] = useState(false);
  const [indexHealth, setIndexHealth] = useState<IndexHealth | null>(null);
  const [isNewNoteDialogOpen, setIsNewNoteDialogOpen] = useState(false);
  const [newNoteName, setNewNoteName] = useState('');
  const [isCreatingNote, setIsCreatingNote] = useState(false);
  const [isNewFolderDialogOpen, setIsNewFolderDialogOpen] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');
  const [isCreatingFolder, setIsCreatingFolder] = useState(false);
  const [isDemoMode, setIsDemoMode] = useState<boolean>(isDemoSession());
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [graphRefreshTrigger, setGraphRefreshTrigger] = useState(0);
  const [isSynthesizingTts, setIsSynthesizingTts] = useState(false);
  const ttsUrlRef = useRef<string | null>(null);
  const ttsAbortRef = useRef<AbortController | null>(null);
  // T007: Initialize font size state management
  const { fontSize, setFontSize, isFontReady } = useFontSize();
  const {
    status: ttsPlayerStatus,
    error: ttsPlayerError,
    volume: ttsVolume,
    play: playAudio,
    pause: pauseAudio,
    resume: resumeAudio,
    stop: stopAudio,
    setVolume: setTtsVolume,
  } = useAudioPlayer();
  const stopTts = () => {
    if (ttsAbortRef.current) {
      ttsAbortRef.current.abort();
      ttsAbortRef.current = null;
    }
    stopAudio();
    if (ttsUrlRef.current) {
      URL.revokeObjectURL(ttsUrlRef.current);
      ttsUrlRef.current = null;
    }
    setIsSynthesizingTts(false);
  };

  useEffect(() => {
    const handleAuthChange = () => {
      const demo = isDemoSession();
      setIsDemoMode(demo);
      if (demo) {
        setIsEditMode(false);
      }
    };
    window.addEventListener(AUTH_TOKEN_CHANGED_EVENT, handleAuthChange);
    return () => {
      window.removeEventListener(AUTH_TOKEN_CHANGED_EVENT, handleAuthChange);
    };
  }, []);

  useEffect(() => {
    if (ttsPlayerError) {
      toast.error(ttsPlayerError);
    }
  }, [ttsPlayerError, toast]);

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

  useEffect(() => {
    // Stop TTS when switching notes
    stopTts();
  }, [selectedPath]);

  useEffect(() => {
    return () => {
      stopTts();
    };
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
    console.log(`[Wikilink] Clicked: "${linkText}", Slug: "${slug}"`);
    
    // Try to find exact match first
    let targetNote = notes.find(
      (note) => normalizeSlug(note.title) === slug
    );

    // If not found, try path-based matching
    if (!targetNote) {
      targetNote = notes.find((note) => {
        const pathSlug = normalizeSlug(note.note_path.replace(/\.md$/, ''));
        // console.log(`Checking path: ${note.note_path}, Slug: ${pathSlug}`);
        return pathSlug.endsWith(slug);
      });
    }

    if (targetNote) {
      console.log(`[Wikilink] Found target: ${targetNote.note_path}`);
      setSelectedPath(targetNote.note_path);
    } else {
      // TODO: Show "Create note" dialog
      console.log('Note not found for wikilink:', linkText);
      setError(`Note not found: ${linkText}`);
    }
  };

  const handleTtsToggle = async () => {
    if (!currentNote || isSynthesizingTts) {
      return;
    }

    if (ttsPlayerStatus === 'playing') {
      pauseAudio();
      return;
    }

    if (ttsPlayerStatus === 'paused') {
      resumeAudio();
      return;
    }

    const plainText = markdownToPlainText(currentNote.body);
    if (!plainText) {
      toast.error('No readable content in this note.');
      return;
    }

    if (ttsAbortRef.current) {
      ttsAbortRef.current.abort();
    }
    const controller = new AbortController();
    ttsAbortRef.current = controller;
    setIsSynthesizingTts(true);
    try {
      const blob = await synthesizeTts(plainText, { signal: controller.signal });
      if (ttsUrlRef.current) {
        URL.revokeObjectURL(ttsUrlRef.current);
      }
      const url = URL.createObjectURL(blob);
      ttsUrlRef.current = url;
      playAudio(url);
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') {
        return;
      }
      const message = err instanceof Error ? err.message : 'Failed to generate speech.';
      toast.error(message);
    } finally {
      ttsAbortRef.current = null;
      setIsSynthesizingTts(false);
    }
  };

  const handleTtsStop = () => {
    stopTts();
  };

  const handleSelectNote = (path: string) => {
    setSelectedPath(path);
    setError(null);
    setIsEditMode(false); // Exit edit mode when switching notes
  };

  // Refresh all views when notes are changed
  const refreshAll = async () => {
    console.log('[MainApp] refreshAll called');
    try {
      // Note: Backend automatically updates index when notes are created
      // We just need to fetch the fresh data

      // Small delay to ensure backend indexing completes
      await new Promise(resolve => setTimeout(resolve, 500));

      // Fetch fresh data
      const [notesList, health] = await Promise.all([
        listNotes(),
        getIndexHealth().catch(() => null)
      ]);

      console.log('[MainApp] Fetched', notesList.length, 'notes');
      setNotes(notesList);
      setIndexHealth(health);
      setGraphRefreshTrigger(prev => {
        const newValue = prev + 1;
        console.log('[MainApp] Graph trigger:', prev, '→', newValue);
        return newValue;
      });
      console.log('[MainApp] State updated');
    } catch (err) {
      console.error('[MainApp] Error refreshing data:', err);
    }
  };

  // T093: Handle edit button click
  const handleEdit = () => {
    if (isDemoMode) {
      toast.error('Demo mode is read-only. Sign in with Hugging Face to edit notes.');
      return;
    }
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
    if (open && isDemoMode) {
      toast.error('Demo mode is read-only. Sign in with Hugging Face to create notes.');
      return;
    }
    setIsNewNoteDialogOpen(open);
    if (!open) {
      // Clear input when dialog closes
      setNewNoteName('');
    }
  };

  // Handle folder dialog open change
  const handleFolderDialogOpenChange = (open: boolean) => {
    if (open && isDemoMode) {
      toast.error('Demo mode is read-only. Sign in with Hugging Face to create folders.');
      return;
    }
    setIsNewFolderDialogOpen(open);
    if (!open) {
      // Clear input when dialog closes
      setNewFolderName('');
    }
  };

  // Handle create new note
  const handleCreateNote = async () => {
    if (isDemoMode) {
      toast.error('Demo mode is read-only. Sign in to create notes.');
      return;
    }
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
    if (isDemoMode) {
      toast.error('Demo mode is read-only. Sign in to create folders.');
      return;
    }
    if (!newFolderName.trim() || isCreatingFolder) return;

    setIsCreatingFolder(true);
    setError(null);

    try {
      // Create a placeholder note in the folder
      const folderPath = newFolderName.replace(/\/$/, ''); // Remove trailing slash if present
      const placeholderPath = `${folderPath}/.placeholder.md`;

      await createNote({
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

  // Handle dragging file to folder
  const handleMoveNoteToFolder = async (oldPath: string, targetFolderPath: string) => {
    if (isDemoMode) {
      toast.error('Demo mode is read-only. Sign in to move notes.');
      return;
    }
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

  const ttsStatus = isSynthesizingTts ? 'loading' : ttsPlayerStatus;
  const ttsDisabledReason = undefined;

  return (
    <GlowParticleEffect config="vibrant" triggerSelector="button">
      <div className="h-screen flex flex-col">
      {/* Demo warning banner */}
      <Alert variant="destructive" className="rounded-none border-x-0 border-t-0">
        <AlertDescription className="text-center">
          DEMO ONLY - ALL DATA IS TEMPORARY AND MAY BE DELETED AT ANY TIME
        </AlertDescription>
      </Alert>
      
      {/* Top bar */}
      <div className="border-b border-border p-2 animate-fade-in">
        <div className="relative flex items-center justify-center">
          <h1
            className="text-2xl tracking-[0.15em] uppercase select-none"
            style={{
              fontFamily: '"Press Start 2P", monospace',
              fontWeight: 400,
              color: "#3b82f6",
              textShadow: `
                0px 0px 1px #000000,
                1px 1px 0 #111827,
                2px 2px 0 #020617,
                3px 3px 0 #020617,
                4px 4px 0 #020617,
                5px 5px 0 #020617,
                0px 0px 4px rgba(15,23,42,0.9),
                0px 0px 10px rgba(59,130,246,0.7)
              `,
              textRendering: "geometricPrecision",
              WebkitFontSmoothing: "none",
              transform: "perspective(700px) rotateX(18deg)",
              transformOrigin: "bottom center",
            }}
          >
            Vault.MCP
          </h1>
          <div className="absolute right-0 flex gap-2">
            {isDemoMode && (
              <Button
                variant="default"
                size="sm"
                onClick={() => login()}
                title="Sign in with Hugging Face"
              >
                Sign in
              </Button>
            )}
            <Button
              variant={isChatOpen ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setIsChatOpen(!isChatOpen)}
              title={isChatOpen ? "Close Chat" : "Open AI Planning Agent"}
            >
              <MessageCircle className="h-4 w-4" />
            </Button>
            <Button
              variant={isGraphView ? "secondary" : "ghost"}
              size="sm"
              onClick={() => setIsGraphView(!isGraphView)}
              title={isGraphView ? "Switch to Note View" : "Switch to Graph View"}
              className="transition-all duration-250 ease-out"
            >
              <Network className="h-4 w-4 transition-transform duration-250" />
            </Button>
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
                    <Button variant="outline" size="sm" className="w-full" disabled={isDemoMode}>
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
                    <Button variant="outline" size="sm" className="w-full" disabled={isDemoMode}>
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

              {isGraphView ? (
                <GraphView
                  onSelectNote={(path) => {
                    handleSelectNote(path);
                    setIsGraphView(false);
                  }}
                  refreshTrigger={graphRefreshTrigger}
                />
              ) : (
                isLoadingNote || !isFontReady ? (
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
                      onEdit={isDemoMode ? undefined : handleEdit}
                      onWikilinkClick={handleWikilinkClick}
                      ttsStatus={ttsStatus}
                      onTtsToggle={handleTtsToggle}
                      onTtsStop={handleTtsStop}
                      ttsDisabledReason={ttsDisabledReason}
                      ttsVolume={ttsVolume}
                      onTtsVolumeChange={setTtsVolume}
                      fontSize={fontSize}
                      onFontSizeChange={setFontSize}
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

          {isChatOpen && (
            <>
              <ResizableHandle withHandle />
              <ResizablePanel defaultSize={25} minSize={20} maxSize={40} className="animate-slide-in">
                <ChatPanel
                  onNavigateToNote={handleSelectNote}
                  onNotesChanged={refreshAll}
                />
              </ResizablePanel>
            </>
          )}
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
    </GlowParticleEffect>
  );
}

