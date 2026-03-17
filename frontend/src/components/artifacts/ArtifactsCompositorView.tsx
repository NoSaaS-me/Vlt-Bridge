/**
 * ArtifactsCompositorView — Main layout for the Artifacts section.
 *
 * Mirrors AgentsCompositorView in structure:
 *   ArtifactSidebar (w-52) | ArtifactViewer or empty state
 */
import { useState, useEffect, useCallback } from 'react';
import { Box } from 'lucide-react';
import {
  type ArtifactData,
  listArtifacts,
  createArtifact,
  deleteArtifact,
  importArtifact,
} from '@/services/artifact-api';
import { ArtifactSidebar } from './ArtifactSidebar';
import { ArtifactViewer } from './ArtifactViewer';
import { NewArtifactDialog } from './NewArtifactDialog';

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ArtifactsCompositorViewProps {
  projectId?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ArtifactsCompositorView({ projectId }: ArtifactsCompositorViewProps) {
  const [artifacts, setArtifacts] = useState<ArtifactData[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [newDialogOpen, setNewDialogOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedArtifact = artifacts.find((a) => a.id === selectedId) ?? null;

  // ---------------------------------------------------------------------------
  // Fetch
  // ---------------------------------------------------------------------------

  const fetchArtifacts = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await listArtifacts(projectId);
      setArtifacts(data);
      // Preserve selection if artifact still exists, otherwise clear
      setSelectedId((prev) => {
        if (!prev) return data[0]?.id ?? null;
        return data.find((a) => a.id === prev) ? prev : (data[0]?.id ?? null);
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load artifacts');
    } finally {
      setIsLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchArtifacts();
  }, [fetchArtifacts]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  const handleCreate = useCallback(
    async (data: { name: string; description?: string; type: string }) => {
      const artifact = await createArtifact({
        ...data,
        project_id: projectId ?? 'default',
      });
      setArtifacts((prev) => [artifact, ...prev]);
      setSelectedId(artifact.id);
    },
    [projectId],
  );

  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteArtifact(id);
        setArtifacts((prev) => {
          const next = prev.filter((a) => a.id !== id);
          setSelectedId((sel) => {
            if (sel !== id) return sel;
            return next[0]?.id ?? null;
          });
          return next;
        });
      } catch (err) {
        console.error('Delete failed:', err);
      }
    },
    [],
  );

  const handleImport = useCallback(
    async (file: File) => {
      try {
        const artifact = await importArtifact(file, projectId);
        setArtifacts((prev) => [artifact, ...prev]);
        setSelectedId(artifact.id);
      } catch (err) {
        console.error('Import failed:', err);
      }
    },
    [projectId],
  );

  const handleStateChange = useCallback((updated: ArtifactData) => {
    setArtifacts((prev) => prev.map((a) => (a.id === updated.id ? updated : a)));
  }, []);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="h-full flex">
      {/* Sidebar */}
      <div className="w-52 shrink-0">
        <ArtifactSidebar
          artifacts={artifacts}
          selectedId={selectedId}
          onSelect={setSelectedId}
          onCreate={() => setNewDialogOpen(true)}
          onDelete={handleDelete}
          onImport={handleImport}
          isLoading={isLoading}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {error ? (
          <div className="flex-1 flex items-center justify-center text-sm text-destructive">
            {error}
          </div>
        ) : selectedArtifact ? (
          <ArtifactViewer
            artifact={selectedArtifact}
            onStateChange={handleStateChange}
          />
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center px-8">
            <Box className="h-10 w-10 text-muted-foreground/20" />
            <p className="text-sm text-muted-foreground">
              {isLoading ? 'Loading artifacts…' : 'No artifact selected'}
            </p>
            {!isLoading && (
              <p className="text-xs text-muted-foreground/60">
                Select an artifact from the sidebar or create a new one.
              </p>
            )}
          </div>
        )}
      </div>

      {/* Creation dialog */}
      <NewArtifactDialog
        open={newDialogOpen}
        onClose={() => setNewDialogOpen(false)}
        onCreate={handleCreate}
      />
    </div>
  );
}

export default ArtifactsCompositorView;
