/**
 * NewArtifactDialog — Creation dialog with template picker.
 */
import { useState, useEffect } from 'react';
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
import { type ArtifactTemplate, listTemplates } from '@/services/artifact-api';

export interface NewArtifactDialogProps {
  open: boolean;
  onClose: () => void;
  onCreate: (data: { name: string; description?: string; type: string; template?: string }) => void;
}

export function NewArtifactDialog({ open, onClose, onCreate }: NewArtifactDialogProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [type, setType] = useState<'ephemeral' | 'persistent'>('persistent');
  const [template, setTemplate] = useState<string>('blank');
  const [templates, setTemplates] = useState<ArtifactTemplate[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (open) {
      listTemplates()
        .then(setTemplates)
        .catch(() => setTemplates([]));
    }
  }, [open]);

  function reset() {
    setName('');
    setDescription('');
    setType('persistent');
    setTemplate('blank');
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
        template: template === 'blank' ? undefined : template,
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
          {/* Template picker */}
          {templates.length > 0 && (
            <div className="space-y-1.5">
              <Label className="text-xs">Template</Label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  type="button"
                  onClick={() => setTemplate('blank')}
                  className={`rounded-md border p-2 text-left text-xs transition-colors ${
                    template === 'blank'
                      ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                      : 'border-border hover:border-muted-foreground/40'
                  }`}
                >
                  <div className="font-medium">Blank</div>
                  <div className="text-muted-foreground mt-0.5">Empty artifact</div>
                </button>
                {templates.map((t) => (
                  <button
                    key={t.name}
                    type="button"
                    onClick={() => setTemplate(t.name)}
                    className={`rounded-md border p-2 text-left text-xs transition-colors ${
                      template === t.name
                        ? 'border-blue-500 bg-blue-500/10 text-blue-400'
                        : 'border-border hover:border-muted-foreground/40'
                    }`}
                  >
                    <div className="font-medium">{t.description}</div>
                    <div className="text-muted-foreground mt-0.5">
                      {t.has_backend ? 'Backend + ' : ''}
                      {t.connectors.length > 0 ? t.connectors.join(', ') : 'No connectors'}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}

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
              {isSubmitting ? 'Creating...' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export default NewArtifactDialog;
