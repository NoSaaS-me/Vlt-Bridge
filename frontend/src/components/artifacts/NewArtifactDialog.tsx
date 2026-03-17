/**
 * NewArtifactDialog — Creation dialog for a new artifact.
 * Uses shadcn Dialog + Input + Textarea + Select.
 */
import { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

export interface NewArtifactDialogProps {
  open: boolean;
  onClose: () => void;
  onCreate: (data: { name: string; description?: string; type: string }) => void;
}

export function NewArtifactDialog({ open, onClose, onCreate }: NewArtifactDialogProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [type, setType] = useState<'ephemeral' | 'persistent'>('persistent');
  const [isSubmitting, setIsSubmitting] = useState(false);

  function reset() {
    setName('');
    setDescription('');
    setType('persistent');
    setIsSubmitting(false);
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setIsSubmitting(true);
    try {
      await onCreate({
        name: name.trim(),
        description: description.trim() || undefined,
        type,
      });
      reset();
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleOpenChange(open: boolean) {
    if (!open) {
      reset();
      onClose();
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle className="text-sm font-semibold">New Artifact</DialogTitle>
        </DialogHeader>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="artifact-name" className="text-xs">
              Name <span className="text-destructive">*</span>
            </Label>
            <Input
              id="artifact-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My Artifact"
              className="h-8 text-sm"
              autoFocus
              required
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="artifact-description" className="text-xs">
              Description
            </Label>
            <Textarea
              id="artifact-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="What does this artifact do?"
              className="text-sm resize-none"
              rows={3}
            />
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="artifact-type" className="text-xs">
              Type
            </Label>
            <Select value={type} onValueChange={(v) => setType(v as 'ephemeral' | 'persistent')}>
              <SelectTrigger id="artifact-type" className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="persistent" className="text-sm">
                  Persistent — saved to disk, survives restarts
                </SelectItem>
                <SelectItem value="ephemeral" className="text-sm">
                  Ephemeral — temporary, discarded on cleanup
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <DialogFooter className="gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => handleOpenChange(false)}
              disabled={isSubmitting}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              size="sm"
              disabled={!name.trim() || isSubmitting}
            >
              {isSubmitting ? 'Creating…' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default NewArtifactDialog;
