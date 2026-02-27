/**
 * FileUpload — drag-and-drop upload dialog.
 *
 * Uses react-dropzone for file selection and XHR-based uploadAsset() from
 * services/api for individual file uploads with progress reporting.
 *
 * Each dropped file gets an editable target path (pre-populated from
 * defaultFolder + original filename). Files are uploaded sequentially.
 * Calls onUploadComplete() when all uploads finish successfully.
 */
import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, CheckCircle2, AlertCircle, Loader2, FolderOpen } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from '@/components/ui/dialog';
import { uploadAsset } from '@/services/api';

// ── Types ────────────────────────────────────────────────────────────────────

interface FileUploadProps {
  open: boolean;
  onClose: () => void;
  /** Called after all uploads complete successfully. */
  onUploadComplete: () => void;
  projectId?: string;
  /** Pre-populate target path prefix (e.g. the current folder). */
  defaultFolder?: string;
}

type UploadStatus = 'pending' | 'uploading' | 'done' | 'error';

interface QueuedFile {
  id: string;
  file: File;
  targetPath: string;
  status: UploadStatus;
  progress: number;   // 0-100
  errorMsg?: string;
}

// ── Accepted MIME types (no .md) ─────────────────────────────────────────────

const ACCEPTED_TYPES: Record<string, string[]> = {
  'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg'],
  'application/pdf': ['.pdf'],
  'audio/*': ['.mp3', '.wav', '.ogg', '.flac'],
  'video/*': ['.mp4', '.webm', '.mov'],
  'text/html': ['.html', '.htm'],
  'text/csv': ['.csv'],
  'text/plain': ['.txt'],
};

// ── Helpers ──────────────────────────────────────────────────────────────────

function buildTargetPath(file: File, defaultFolder?: string): string {
  const name = file.name;
  if (!defaultFolder) return name;
  return `${defaultFolder.replace(/\/$/, '')}/${name}`;
}

function uniqueId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

// ── Component ────────────────────────────────────────────────────────────────

export function FileUpload({
  open,
  onClose,
  onUploadComplete,
  projectId,
  defaultFolder,
}: FileUploadProps) {
  const [queue, setQueue] = useState<QueuedFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // ── Dropzone ───────────────────────────────────────────────────────────────

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const newItems: QueuedFile[] = acceptedFiles.map((file) => ({
        id: uniqueId(),
        file,
        targetPath: buildTargetPath(file, defaultFolder),
        status: 'pending' as UploadStatus,
        progress: 0,
      }));
      setQueue((prev) => [...prev, ...newItems]);
    },
    [defaultFolder]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    multiple: true,
  });

  // ── Queue management ───────────────────────────────────────────────────────

  const updateItem = (id: string, patch: Partial<QueuedFile>) => {
    setQueue((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...patch } : item))
    );
  };

  const removeItem = (id: string) => {
    setQueue((prev) => prev.filter((item) => item.id !== id));
  };

  const handlePathChange = (id: string, value: string) => {
    updateItem(id, { targetPath: value });
  };

  // ── Upload all ─────────────────────────────────────────────────────────────

  const handleUploadAllWithCheck = async () => {
    const pending = queue.filter((item) => item.status === 'pending');
    if (pending.length === 0) return;

    setIsUploading(true);
    let allSucceeded = true;

    for (const item of pending) {
      updateItem(item.id, { status: 'uploading', progress: 0 });
      try {
        await uploadAsset(
          item.file,
          item.targetPath,
          projectId,
          (pct) => updateItem(item.id, { progress: pct })
        );
        updateItem(item.id, { status: 'done', progress: 100 });
      } catch (err) {
        const msg = err instanceof Error ? err.message : 'Upload failed';
        updateItem(item.id, { status: 'error', errorMsg: msg });
        allSucceeded = false;
      }
    }

    setIsUploading(false);
    if (allSucceeded) {
      onUploadComplete();
    }
  };

  // ── Derived state ──────────────────────────────────────────────────────────

  const pendingCount = queue.filter((i) => i.status === 'pending').length;
  const doneCount = queue.filter((i) => i.status === 'done').length;
  const errorCount = queue.filter((i) => i.status === 'error').length;
  const allFinished = queue.length > 0 && pendingCount === 0 && !isUploading;

  // ── Reset on close ─────────────────────────────────────────────────────────

  const handleClose = () => {
    if (isUploading) return; // Prevent closing while uploading
    setQueue([]);
    onClose();
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <Dialog open={open} onOpenChange={(o) => !o && handleClose()}>
      <DialogContent className="max-w-2xl max-h-[80vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Upload Files
          </DialogTitle>
          <DialogDescription>
            Drop files below or click to browse. Supported formats: images, PDF,
            audio, video, HTML, CSV, TXT. Edit the target path before uploading.
            {defaultFolder && (
              <span className="ml-1 font-mono text-xs text-muted-foreground">
                Folder: {defaultFolder}
              </span>
            )}
          </DialogDescription>
        </DialogHeader>

        {/* Drop zone */}
        <div
          {...getRootProps()}
          className={`
            border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
            transition-colors duration-150 flex-shrink-0
            ${isDragActive
              ? 'border-primary bg-primary/5 text-primary'
              : 'border-border hover:border-primary/50 text-muted-foreground hover:text-foreground'
            }
          `}
        >
          <input {...getInputProps()} />
          <FolderOpen className="h-8 w-8 mx-auto mb-3 opacity-70" />
          {isDragActive ? (
            <p className="text-sm font-medium">Drop files here…</p>
          ) : (
            <p className="text-sm">
              Drag and drop files here, or{' '}
              <span className="text-primary font-medium">click to browse</span>
            </p>
          )}
        </div>

        {/* Queue list */}
        {queue.length > 0 && (
          <div className="flex-1 overflow-y-auto space-y-2 min-h-0">
            {queue.map((item) => (
              <div
                key={item.id}
                className="flex items-center gap-2 p-2 rounded-md border border-border bg-muted/30"
              >
                {/* Status icon */}
                <div className="flex-shrink-0 w-5 flex items-center justify-center">
                  {item.status === 'pending' && (
                    <span className="h-2 w-2 rounded-full bg-muted-foreground" />
                  )}
                  {item.status === 'uploading' && (
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                  )}
                  {item.status === 'done' && (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  )}
                  {item.status === 'error' && (
                    <AlertCircle className="h-4 w-4 text-destructive" />
                  )}
                </div>

                {/* File info + editable path */}
                <div className="flex-1 min-w-0 space-y-1">
                  <p className="text-xs text-muted-foreground truncate">{item.file.name}</p>
                  <Input
                    value={item.targetPath}
                    onChange={(e) => handlePathChange(item.id, e.target.value)}
                    disabled={item.status !== 'pending'}
                    className="h-7 text-xs font-mono"
                    placeholder="target/path/filename.ext"
                  />
                  {/* Progress bar */}
                  {item.status === 'uploading' && (
                    <div className="h-1 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-200"
                        style={{ width: `${item.progress}%` }}
                      />
                    </div>
                  )}
                  {/* Error message */}
                  {item.status === 'error' && item.errorMsg && (
                    <p className="text-xs text-destructive">{item.errorMsg}</p>
                  )}
                </div>

                {/* Remove button (only for pending) */}
                {item.status === 'pending' && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 flex-shrink-0"
                    onClick={() => removeItem(item.id)}
                    title="Remove"
                  >
                    <X className="h-3 w-3" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Summary line */}
        {queue.length > 0 && (
          <div className="text-xs text-muted-foreground flex-shrink-0 pt-1">
            {doneCount > 0 && (
              <span className="text-green-600 mr-2">{doneCount} uploaded</span>
            )}
            {errorCount > 0 && (
              <span className="text-destructive mr-2">{errorCount} failed</span>
            )}
            {pendingCount > 0 && (
              <span>{pendingCount} pending</span>
            )}
          </div>
        )}

        <DialogFooter className="flex-shrink-0 gap-2">
          <Button variant="outline" onClick={handleClose} disabled={isUploading}>
            {allFinished && errorCount === 0 ? 'Close' : 'Cancel'}
          </Button>

          {allFinished ? (
            <Button
              onClick={handleClose}
              disabled={errorCount > 0}
            >
              Done
            </Button>
          ) : (
            <Button
              onClick={handleUploadAllWithCheck}
              disabled={pendingCount === 0 || isUploading}
            >
              {isUploading ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Uploading…
                </>
              ) : (
                <>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload All ({pendingCount})
                </>
              )}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
