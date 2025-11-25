/**
 * T079: Search bar with debounced queries and dropdown results
 */
import { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import {
  Command,
  CommandGroup,
  CommandItem,
  CommandList,
} from '@/components/ui/command';
import { Button } from '@/components/ui/button';
import { SearchResultSkeleton } from '@/components/SearchResultSkeleton';
import { searchNotes } from '@/services/api';
import type { SearchResult } from '@/types/search';

interface SearchBarProps {
  onSelectNote: (path: string) => void;
}

export function SearchBar({ onSelectNote }: SearchBarProps) {
  const [query, setQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  // Debounce search query (300ms)
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query);
    }, 300);

    return () => clearTimeout(timer);
  }, [query]);

  // Execute search when debounced query changes
  useEffect(() => {
    if (!debouncedQuery.trim()) {
      setResults([]);
      setIsOpen(false);
      return;
    }

    const performSearch = async () => {
      setIsLoading(true);
      try {
        const searchResults = await searchNotes(debouncedQuery);
        setResults(searchResults);
        setIsOpen(searchResults.length > 0);
      } catch (error) {
        console.error('Search error:', error);
        setResults([]);
        setIsOpen(false);
      } finally {
        setIsLoading(false);
      }
    };

    performSearch();
  }, [debouncedQuery]);

  const handleSelectResult = useCallback(
    (path: string) => {
      onSelectNote(path);
      setQuery('');
      setResults([]);
      setIsOpen(false);
    },
    [onSelectNote]
  );

  const handleClear = useCallback(() => {
    setQuery('');
    setResults([]);
    setIsOpen(false);
  }, []);

  return (
    <div className="relative w-full">
      <div className="relative">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search notes..."
          className="pl-8 pr-8"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => {
            if (results.length > 0) {
              setIsOpen(true);
            }
          }}
        />
        {query && (
          <Button
            variant="ghost"
            size="sm"
            className="absolute right-0 top-0 h-full px-2 hover:bg-transparent"
            onClick={handleClear}
          >
            <X className="h-4 w-4" />
          </Button>
        )}
      </div>

      {isOpen && results.length > 0 && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 animate-slide-in-down">
          <div className="bg-popover border border-border rounded-md shadow-md max-h-[400px] overflow-auto">
            <Command>
              <CommandList>
                <CommandGroup heading={`${results.length} result${results.length !== 1 ? 's' : ''}`}>
                  {results.map((result, index) => (
                    <CommandItem
                      key={result.note_path}
                      onSelect={() => handleSelectResult(result.note_path)}
                      className={cn(
                        "cursor-pointer",
                        index < 5 && `animate-stagger-${index + 1}`
                      )}
                    >
                      <div className="flex flex-col gap-1 w-full">
                        <div className="font-medium">{result.title}</div>
                        <div className="text-xs text-muted-foreground line-clamp-2">
                          {result.snippet}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {result.note_path}
                        </div>
                      </div>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </div>
        </div>
      )}

      {isLoading && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50">
          <div className="bg-popover border border-border rounded-md shadow-md p-3">
            <SearchResultSkeleton />
          </div>
        </div>
      )}
    </div>
  );
}

