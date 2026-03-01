/**
 * ProjectDropdown - Project selector with star-to-pin favorites.
 *
 * Uses Popover (not Select) so the star icon has an independent click
 * target that doesn't trigger project selection.
 * Favorites are stored in localStorage and sorted to the top.
 */

import { useState, useCallback } from 'react';
import { ChevronDown, Plus, Star } from 'lucide-react';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Separator } from '@/components/ui/separator';
import { cn } from '@/lib/utils';
import type { Project } from '@/types/project';

const FAVORITES_KEY = 'vlt-bridge-favorite-projects';

function loadFavorites(): Set<string> {
  try {
    const raw = localStorage.getItem(FAVORITES_KEY);
    return raw ? new Set(JSON.parse(raw)) : new Set();
  } catch {
    return new Set();
  }
}

function saveFavorites(favorites: Set<string>) {
  localStorage.setItem(FAVORITES_KEY, JSON.stringify([...favorites]));
}

interface ProjectDropdownProps {
  projects: Project[];
  selectedProject: Project | null;
  onSelectProject: (projectId: string) => void;
  onCreateProject: () => void;
  disabled?: boolean;
}

export function ProjectDropdown({
  projects,
  selectedProject,
  onSelectProject,
  onCreateProject,
  disabled = false,
}: ProjectDropdownProps) {
  const [open, setOpen] = useState(false);
  const [favorites, setFavorites] = useState<Set<string>>(loadFavorites);

  const toggleFavorite = useCallback((projectId: string) => {
    setFavorites((prev) => {
      const next = new Set(prev);
      if (next.has(projectId)) {
        next.delete(projectId);
      } else {
        next.add(projectId);
      }
      saveFavorites(next);
      return next;
    });
  }, []);

  // Sort: favorites first, then alphabetical within each group
  const sorted = [...projects].sort((a, b) => {
    const aFav = favorites.has(a.id) ? 0 : 1;
    const bFav = favorites.has(b.id) ? 0 : 1;
    if (aFav !== bFav) return aFav - bFav;
    return a.name.localeCompare(b.name);
  });

  const hasFavorites = sorted.some((p) => favorites.has(p.id));
  const firstNonFavIdx = hasFavorites
    ? sorted.findIndex((p) => !favorites.has(p.id))
    : -1;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          disabled={disabled}
          className={cn(
            'inline-flex items-center justify-between gap-1 rounded-md border border-input bg-background',
            'px-3 h-8 text-sm shadow-xs',
            'hover:bg-accent hover:text-accent-foreground',
            'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
            'disabled:cursor-not-allowed disabled:opacity-50',
            'w-[180px]',
          )}
        >
          <span className="truncate">{selectedProject?.name || 'Select project'}</span>
          <ChevronDown className="h-3.5 w-3.5 shrink-0 opacity-50" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-[220px] p-1" sideOffset={4}>
        <div className="max-h-64 overflow-y-auto">
          {sorted.map((project, idx) => {
            const isFav = favorites.has(project.id);
            const isSelected = project.id === selectedProject?.id;

            return (
              <div key={project.id}>
                {idx === firstNonFavIdx && <Separator className="my-1" />}
                <div
                  className={cn(
                    'flex items-center gap-1 rounded-sm px-1 py-1.5 text-sm',
                    'hover:bg-accent hover:text-accent-foreground',
                    isSelected && 'bg-accent/60',
                  )}
                >
                  {/* Star — independent click target */}
                  <button
                    type="button"
                    onClick={() => toggleFavorite(project.id)}
                    className="shrink-0 p-1 rounded hover:bg-muted/80 transition-colors"
                  >
                    <Star
                      className={cn(
                        'h-3 w-3 transition-colors',
                        isFav
                          ? 'fill-amber-400 text-amber-400'
                          : 'text-muted-foreground/30 hover:text-muted-foreground',
                      )}
                    />
                  </button>

                  {/* Project name — selects the project */}
                  <button
                    type="button"
                    className="flex-1 text-left truncate py-0.5"
                    onClick={() => {
                      onSelectProject(project.id);
                      setOpen(false);
                    }}
                  >
                    {project.name}
                    {project.note_count !== undefined && (
                      <span className="ml-1.5 text-[10px] text-muted-foreground">
                        {project.note_count}
                      </span>
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
        <Separator className="my-1" />
        <button
          type="button"
          className="flex items-center gap-2 w-full rounded-sm px-2 py-1.5 text-sm hover:bg-accent hover:text-accent-foreground transition-colors"
          onClick={() => {
            setOpen(false);
            onCreateProject();
          }}
        >
          <Plus className="h-3.5 w-3.5" />
          New Project
        </button>
      </PopoverContent>
    </Popover>
  );
}
