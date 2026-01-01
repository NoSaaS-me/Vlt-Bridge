/**
 * ProjectDropdown - Project selector dropdown using shadcn Select.
 * Allows switching between projects and creating new ones.
 */

import { Plus } from 'lucide-react';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Button } from '@/components/ui/button';
import type { Project } from '@/types/project';

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
  return (
    <Select
      value={selectedProject?.id || ''}
      onValueChange={(value) => {
        if (value === '__create__') {
          onCreateProject();
        } else {
          onSelectProject(value);
        }
      }}
      disabled={disabled}
    >
      <SelectTrigger className="w-[180px] h-8 text-sm">
        <SelectValue placeholder="Select project">
          {selectedProject?.name || 'Select project'}
        </SelectValue>
      </SelectTrigger>
      <SelectContent>
        {projects.map((project) => (
          <SelectItem key={project.id} value={project.id}>
            <div className="flex flex-col">
              <span>{project.name}</span>
              {project.note_count !== undefined && (
                <span className="text-xs text-muted-foreground">
                  {project.note_count} note{project.note_count !== 1 ? 's' : ''}
                </span>
              )}
            </div>
          </SelectItem>
        ))}
        <SelectSeparator />
        <Button
          variant="ghost"
          size="sm"
          className="w-full justify-start text-sm font-normal"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            onCreateProject();
          }}
        >
          <Plus className="h-4 w-4 mr-2" />
          New Project
        </Button>
      </SelectContent>
    </Select>
  );
}
