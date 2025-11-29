/**
 * T038, T048-T049: Table of Contents component
 * Displays hierarchical list of document headings with smooth navigation
 */
import { ScrollArea } from '@/components/ui/scroll-area';
import { cn } from '@/lib/utils';
import type { Heading } from '@/hooks/useTableOfContents';

interface TableOfContentsProps {
  headings: Heading[];
  onHeadingClick: (id: string) => void;
}

export function TableOfContents({ headings, onHeadingClick }: TableOfContentsProps) {
  // T049: Show empty state message when no headings
  if (headings.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <div className="text-center text-muted-foreground">
          <p className="text-sm">No headings found</p>
          <p className="text-xs mt-1">Add H1, H2, or H3 headings to your note</p>
        </div>
      </div>
    );
  }

  // T048: Calculate indentation based on heading level
  const getIndentation = (level: number) => {
    // H1 = 0px, H2 = 12px, H3 = 24px
    return (level - 1) * 12;
  };

  return (
    <div className="h-full flex flex-col border-l border-border">
      <div className="p-4 border-b border-border">
        <h3 className="font-semibold text-sm">Table of Contents</h3>
      </div>
      <ScrollArea className="flex-1">
        <nav className="p-2">
          <ul className="space-y-1">
            {headings.map((heading) => (
              <li key={heading.id}>
                <button
                  onClick={() => onHeadingClick(heading.id)}
                  className={cn(
                    'w-full text-left text-sm py-1.5 px-2 rounded hover:bg-accent transition-colors',
                    'text-muted-foreground hover:text-foreground'
                  )}
                  style={{
                    paddingLeft: `${8 + getIndentation(heading.level)}px`,
                  }}
                  title={heading.text}
                >
                  <span className="line-clamp-2">{heading.text}</span>
                </button>
              </li>
            ))}
          </ul>
        </nav>
      </ScrollArea>
    </div>
  );
}
