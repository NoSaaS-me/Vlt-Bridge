/**
 * T077: Directory tree component with collapsible folders
 * Updated: multi-format file support — notes + assets, file-type icons
 */
import React, { useState, useMemo, useEffect } from 'react';
import { ChevronRight, ChevronDown, Folder } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { NoteSummary } from '@/types/note';
import type { AssetSummary } from '@/types/asset';
import { getFileCategory, getFileIcon, type FileCategory } from '@/lib/fileTypes';
import { cn } from '@/lib/utils';

interface TreeNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  fileCategory?: FileCategory;
  children?: TreeNode[];
  note?: NoteSummary;
  asset?: AssetSummary;
}

interface DirectoryTreeProps {
  notes: NoteSummary[];
  assetSummaries?: AssetSummary[];
  selectedPath?: string;
  onSelectNote: (path: string) => void;
  onMoveNote?: (oldPath: string, newFolderPath: string) => void;
}

/**
 * Build a tree structure from flat list of note paths and optional asset paths
 */
function buildTree(notes: NoteSummary[], assets: AssetSummary[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', type: 'folder', children: [] };

  // Helper to navigate/create folders and insert a file node
  const insertFile = (filePath: string, node: TreeNode) => {
    const parts = filePath.split('/');
    let current = root;

    // Navigate/create folders
    for (let i = 0; i < parts.length - 1; i++) {
      const folderName = parts[i];
      const folderPath = parts.slice(0, i + 1).join('/');

      let folder = current.children?.find(
        (child) => child.name === folderName && child.type === 'folder'
      );

      if (!folder) {
        folder = {
          name: folderName,
          path: folderPath,
          type: 'folder',
          children: [],
        };
        current.children = current.children || [];
        current.children.push(folder);
      }

      current = folder;
    }

    // Add file node
    current.children = current.children || [];
    current.children.push(node);
  };

  // Insert notes
  for (const note of notes) {
    insertFile(note.note_path, {
      name: note.note_path.split('/').pop() || note.note_path,
      path: note.note_path,
      type: 'file',
      fileCategory: 'markdown',
      note,
    });
  }

  // Insert assets
  for (const asset of assets) {
    insertFile(asset.asset_path, {
      name: asset.asset_path.split('/').pop() || asset.asset_path,
      path: asset.asset_path,
      type: 'file',
      fileCategory: getFileCategory(asset.asset_path),
      asset,
    });
  }

  // Sort children: folders first, then files, alphabetically
  const sortChildren = (node: TreeNode) => {
    if (node.children) {
      node.children.sort((a, b) => {
        if (a.type !== b.type) {
          return a.type === 'folder' ? -1 : 1;
        }
        return a.name.localeCompare(b.name);
      });
      node.children.forEach(sortChildren);
    }
  };

  sortChildren(root);

  return root.children || [];
}

interface TreeNodeItemProps {
  node: TreeNode;
  depth: number;
  selectedPath?: string;
  onSelectNote: (path: string) => void;
  onMoveNote?: (oldPath: string, newFolderPath: string) => void;
  forceExpandState?: boolean;
}

function TreeNodeItem({ node, depth, selectedPath, onSelectNote, onMoveNote, forceExpandState }: TreeNodeItemProps) {
  const [isOpen, setIsOpen] = useState(depth < 2); // Auto-expand first 2 levels
  const [isDragOver, setIsDragOver] = useState(false);

  // T014: Use forceExpandState if provided, otherwise use local isOpen state
  // Sync local state with force state when it changes
  const effectiveIsOpen = forceExpandState ?? isOpen;

  // Update local state when force state is applied
  useEffect(() => {
    if (forceExpandState !== undefined) {
      setIsOpen(forceExpandState);
    }
  }, [forceExpandState]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (node.type === 'folder') {
      setIsDragOver(true);
    }
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    if (node.type === 'folder') {
      const draggedPath = e.dataTransfer.getData('application/note-path');
      if (draggedPath && onMoveNote) {
        onMoveNote(draggedPath, node.path);
      }
    }
  };

  const handleDragStart = (e: React.DragEvent) => {
    if (node.type === 'file') {
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('application/note-path', node.path);
    }
  };

  if (node.type === 'folder') {
    return (
      <div>
        <Button
          variant="ghost"
          className={cn(
            "w-full justify-start font-normal px-2 h-8",
            "hover:bg-accent transition-colors duration-200",
            isDragOver && "bg-accent ring-2 ring-primary"
          )}
          style={{ paddingLeft: `${depth * 12 + 8}px` }}
          onClick={() => setIsOpen(!isOpen)}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {effectiveIsOpen ? (
            <ChevronDown className="h-4 w-4 mr-1 shrink-0" />
          ) : (
            <ChevronRight className="h-4 w-4 mr-1 shrink-0" />
          )}
          <Folder className="h-4 w-4 mr-2 shrink-0 text-muted-foreground" />
          <span className="truncate">{node.name}</span>
        </Button>
        {effectiveIsOpen && node.children && (
          <div>
            {node.children.map((child) => (
              <TreeNodeItem
                key={child.path}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelectNote={onSelectNote}
                onMoveNote={onMoveNote}
                forceExpandState={forceExpandState}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  // File node
  const isSelected = node.path === selectedPath;

  // Remove .md extension for markdown files only; keep full filename for assets
  const displayName =
    node.fileCategory === 'markdown'
      ? node.name.replace(/\.md$/, '')
      : node.name;

  // Resolve the icon component based on file category
  const category = node.fileCategory ?? 'unknown';
  const IconComp = getFileIcon(category);

  return (
    <Button
      variant="ghost"
      className={cn(
        "w-full justify-start font-normal px-2 h-8",
        "hover:bg-accent transition-colors duration-200",
        isSelected && "bg-accent transition-colors duration-300 animate-highlight-pulse",
        "cursor-move"
      )}
      style={{ paddingLeft: `${depth * 12 + 8}px` }}
      onClick={() => onSelectNote(node.path)}
      draggable
      onDragStart={handleDragStart}
    >
      <IconComp className="h-4 w-4 mr-2 shrink-0 text-muted-foreground transition-colors duration-200" />
      <span className="truncate">{displayName}</span>
    </Button>
  );
}

export function DirectoryTree({ notes, assetSummaries = [], selectedPath, onSelectNote, onMoveNote }: DirectoryTreeProps) {
  const tree = useMemo(() => buildTree(notes, assetSummaries), [notes, assetSummaries]);

  // T012: Add expandAll state to DirectoryTree component
  // T013: Add collapseAll state to DirectoryTree component
  // T015: Implement expand/collapse state propagation logic
  const [expandAllState, setExpandAllState] = useState<boolean | undefined>(undefined);

  const handleExpandAll = () => {
    setExpandAllState(true);
    // Reset after transition completes (300ms)
    setTimeout(() => {
      setExpandAllState(undefined);
    }, 300);
  };

  const handleCollapseAll = () => {
    setExpandAllState(false);
    // Reset after transition completes (300ms)
    setTimeout(() => {
      setExpandAllState(undefined);
    }, 300);
  };

  if (notes.length === 0 && assetSummaries.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground text-center">
        No files found. Create your first note to get started.
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="py-2">
        {/* T016: Add "Expand All" button above directory tree */}
        {/* T017: Add "Collapse All" button above directory tree */}
        <div className="flex gap-2 px-2 pb-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleExpandAll}
            className="flex-1 text-xs"
            aria-label="Expand all folders"
          >
            Expand All
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleCollapseAll}
            className="flex-1 text-xs"
            aria-label="Collapse all folders"
          >
            Collapse All
          </Button>
        </div>
        {tree.map((node) => (
          <TreeNodeItem
            key={node.path}
            node={node}
            depth={0}
            selectedPath={selectedPath}
            onSelectNote={onSelectNote}
            onMoveNote={onMoveNote}
            forceExpandState={expandAllState}
          />
        ))}
      </div>
    </ScrollArea>
  );
}
