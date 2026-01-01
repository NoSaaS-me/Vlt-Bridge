/**
 * IssuesFlyout - Placeholder component for beads integration.
 * Will show issues from the bd (beads) CLI in future.
 */

import { X, AlertCircle, Construction } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface IssuesFlyoutProps {
  projectId: string | null;
  onClose: () => void;
}

export function IssuesFlyout({
  projectId: _projectId,
  onClose,
}: IssuesFlyoutProps) {
  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-muted-foreground" />
            <div>
              <h2 className="font-semibold">Issues</h2>
              <p className="text-xs text-muted-foreground">
                Track work with beads (bd)
              </p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={onClose}
            title="Close issues panel"
          >
            <X className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Content - Placeholder */}
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
        <Construction className="h-12 w-12 text-muted-foreground/50 mb-4" />
        <h3 className="text-lg font-medium mb-2">Coming Soon</h3>
        <p className="text-sm text-muted-foreground max-w-[250px]">
          Issue tracking integration with beads (bd) CLI is in development.
        </p>
        <div className="mt-4 p-3 bg-muted/50 rounded-lg">
          <p className="text-xs text-muted-foreground font-mono">
            bd quickstart
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Run this to learn the beads workflow
          </p>
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-border text-xs text-muted-foreground text-center">
        Integration with <code className="px-1 bg-muted rounded">bd</code> CLI
      </div>
    </div>
  );
}
