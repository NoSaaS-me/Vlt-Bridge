/**
 * T077: Directory tree component with collapsible folders
 */
import { useState, useMemo } from 'react';
import { ChevronRight, ChevronDown, Folder, File } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { NoteSummary } from '@/types/note';
import { cn } from '@/lib/utils';

interface TreeNode {
  name: string;
  path: string;
  type: 'file' | 'folder';
  children?: TreeNode[];
  note?: NoteSummary;
}

interface DirectoryTreeProps {
  notes: NoteSummary[];
  selectedPath?: string;
  onSelectNote: (path: string) => void;
  onMoveNote?: (oldPath: string, newFolderPath: string) => void;
}

/**
 * Build a tree structure from flat list of note paths
 */
function buildTree(notes: NoteSummary[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', type: 'folder', children: [] };

  for (const note of notes) {
    const parts = note.note_path.split('/');
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

    // Add file
    const fileName = parts[parts.length - 1];
    current.children = current.children || [];
    current.children.push({
      name: fileName,
      path: note.note_path,
      type: 'file',
      note,
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
}

function TreeNodeItem({ node, depth, selectedPath, onSelectNote, onMoveNote }: TreeNodeItemProps) {
  const [isOpen, setIsOpen] = useState(depth < 2); // Auto-expand first 2 levels
  const [isDragOver, setIsDragOver] = useState(false);

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
          {isOpen ? (
            <ChevronDown className="h-4 w-4 mr-1 shrink-0" />
          ) : (
            <ChevronRight className="h-4 w-4 mr-1 shrink-0" />
          )}
          <Folder className="h-4 w-4 mr-2 shrink-0 text-muted-foreground" />
          <span className="truncate">{node.name}</span>
        </Button>
        {isOpen && node.children && (
          <div>
            {node.children.map((child) => (
              <TreeNodeItem
                key={child.path}
                node={child}
                depth={depth + 1}
                selectedPath={selectedPath}
                onSelectNote={onSelectNote}
                onMoveNote={onMoveNote}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  // File node
  const isSelected = node.path === selectedPath;
  // Remove .md extension for display
  const displayName = node.name.replace(/\.md$/, '');

  return (
    <Button
      variant="ghost"
      className={cn(
        "w-full justify-start font-normal px-2 h-8",
        "hover:bg-accent transition-colors duration-200",
        isSelected && "bg-accent animate-highlight-pulse",
        "cursor-move"
      )}
      style={{ paddingLeft: `${depth * 12 + 8}px` }}
      onClick={() => onSelectNote(node.path)}
      draggable
      onDragStart={handleDragStart}
    >
      <File className="h-4 w-4 mr-2 shrink-0 text-muted-foreground" />
      <span className="truncate">{displayName}</span>
    </Button>
  );
}

export function DirectoryTree({ notes, selectedPath, onSelectNote, onMoveNote }: DirectoryTreeProps) {
  const tree = useMemo(() => buildTree(notes), [notes]);

  if (notes.length === 0) {
    return (
      <div className="p-4 text-sm text-muted-foreground text-center">
        No notes found. Create your first note to get started.
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="py-2">
        {tree.map((node) => (
          <TreeNodeItem
            key={node.path}
            node={node}
            depth={0}
            selectedPath={selectedPath}
            onSelectNote={onSelectNote}
            onMoveNote={onMoveNote}
          />
        ))}
      </div>
    </ScrollArea>
  );
}

