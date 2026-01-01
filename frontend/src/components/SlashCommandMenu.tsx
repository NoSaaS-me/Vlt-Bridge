import { useEffect, useState, useRef } from 'react';
import type { SlashCommand } from '@/types/oracle';
import { cn } from '@/lib/utils';
import {
  Trash2,
  FileDown,
  Layers,
  Database,
  HelpCircle,
  Settings,
  Eye,
  Brain,
  Minimize2,
} from 'lucide-react';

interface SlashCommandMenuProps {
  commands: SlashCommand[];
  onSelect: (command: SlashCommand) => void;
  onClose: () => void;
  filterText: string;
  position?: { bottom?: number; top?: number; left: number };
}

export function SlashCommandMenu({
  commands,
  onSelect,
  onClose,
  filterText,
  position,
}: SlashCommandMenuProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const menuRef = useRef<HTMLDivElement>(null);
  const itemRefs = useRef<(HTMLDivElement | null)[]>([]);

  // Filter commands based on input
  const filteredCommands = commands.filter((cmd) => {
    const searchText = filterText.toLowerCase();
    return (
      cmd.name.toLowerCase().includes(searchText) ||
      cmd.description.toLowerCase().includes(searchText)
    );
  });

  // Reset selection when filter changes
  useEffect(() => {
    setSelectedIndex(0);
  }, [filterText]);

  // Scroll selected item into view
  useEffect(() => {
    if (itemRefs.current[selectedIndex]) {
      itemRefs.current[selectedIndex]?.scrollIntoView({
        block: 'nearest',
        behavior: 'smooth',
      });
    }
  }, [selectedIndex]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, filteredCommands.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (filteredCommands[selectedIndex]) {
          onSelect(filteredCommands[selectedIndex]);
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [filteredCommands, selectedIndex, onSelect, onClose]);

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  // Compute positioning style
  const positionStyle = position
    ? {
        left: position.left,
        ...(position.bottom !== undefined ? { bottom: position.bottom } : { top: position.top }),
      }
    : undefined;

  if (filteredCommands.length === 0) {
    return (
      <div
        ref={menuRef}
        className="absolute z-50 w-80 rounded-lg border border-border bg-popover shadow-lg"
        style={positionStyle}
      >
        <div className="p-3 text-sm text-muted-foreground text-center">
          No matching commands
        </div>
      </div>
    );
  }

  return (
    <div
      ref={menuRef}
      className="absolute z-50 w-96 rounded-lg border border-border bg-popover shadow-lg overflow-hidden"
      style={positionStyle}
    >
      <div className="max-h-80 overflow-y-auto">
        {filteredCommands.map((cmd, index) => (
          <div
            key={cmd.name}
            ref={(el) => (itemRefs.current[index] = el)}
            className={cn(
              'flex items-start gap-3 px-4 py-3 cursor-pointer transition-colors',
              selectedIndex === index
                ? 'bg-accent text-accent-foreground'
                : 'hover:bg-accent/50'
            )}
            onClick={() => onSelect(cmd)}
            onMouseEnter={() => setSelectedIndex(index)}
          >
            <div className="flex-shrink-0 mt-0.5">
              <CommandIcon name={cmd.name} />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-sm">/{cmd.name}</span>
                {cmd.shortcut && (
                  <span className="text-xs text-muted-foreground bg-muted px-1.5 py-0.5 rounded">
                    {cmd.shortcut}
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-0.5 line-clamp-2">
                {cmd.description}
              </p>
            </div>
          </div>
        ))}
      </div>
      <div className="border-t border-border px-3 py-2 text-xs text-muted-foreground bg-muted/30">
        Use <kbd className="px-1 py-0.5 bg-background rounded border">↑</kbd>{' '}
        <kbd className="px-1 py-0.5 bg-background rounded border">↓</kbd> to navigate,{' '}
        <kbd className="px-1 py-0.5 bg-background rounded border">Enter</kbd> to select
      </div>
    </div>
  );
}

function CommandIcon({ name }: { name: string }) {
  const iconProps = { className: 'h-4 w-4' };

  switch (name) {
    case 'clear':
      return <Trash2 {...iconProps} />;
    case 'compact':
      return <Minimize2 {...iconProps} />;
    case 'context':
      return <Layers {...iconProps} />;
    case 'help':
      return <HelpCircle {...iconProps} />;
    case 'models':
      return <Settings {...iconProps} />;
    case 'sources':
      return <Database {...iconProps} />;
    case 'thinking':
      return <Brain {...iconProps} />;
    case 'export':
      return <FileDown {...iconProps} />;
    default:
      return <Eye {...iconProps} />;
  }
}
