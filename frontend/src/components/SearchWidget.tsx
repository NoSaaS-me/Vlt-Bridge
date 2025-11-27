import { ScrollArea } from '@/components/ui/scroll-area';
import { Search } from 'lucide-react';
import type { SearchResult } from '@/types/search';

interface SearchWidgetProps {
  results: SearchResult[];
  onSelectNote: (path: string) => void;
}

export function SearchWidget({ results, onSelectNote }: SearchWidgetProps) {
  return (
    <div className="flex flex-col h-full w-full">
      <div className="p-4 border-b border-border flex items-center gap-2">
        <Search className="h-4 w-4 text-muted-foreground" />
        <h2 className="text-lg font-semibold">Search Results</h2>
      </div>
      
      <ScrollArea className="flex-1 p-2">
        {results.length === 0 ? (
          <div className="p-4 text-center text-muted-foreground">
            No matching notes found.
          </div>
        ) : (
          <div className="space-y-2">
            {results.map((result) => (
              <div 
                key={result.note_path}
                className="p-3 rounded-md border border-border bg-card hover:bg-accent/50 transition-colors cursor-pointer group"
                onClick={() => onSelectNote(result.note_path)}
              >
                <h3 className="font-medium text-sm group-hover:text-primary mb-1">
                  {result.title}
                </h3>
                {result.snippet && (
                  <p 
                    className="text-xs text-muted-foreground line-clamp-2"
                    dangerouslySetInnerHTML={{ __html: result.snippet }}
                  />
                )}
                <div className="mt-2 text-[10px] text-muted-foreground font-mono">
                  {result.note_path}
                </div>
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
