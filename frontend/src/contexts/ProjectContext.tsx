/**
 * ProjectContext - Provides multi-project state management.
 *
 * This context manages:
 * - List of user's projects
 * - Currently selected project
 * - Persistence of selected project to localStorage
 */

import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react';
import { fetchProjects, createProject as apiCreateProject } from '@/services/projectApi';
import type { Project, ProjectCreate } from '@/types/project';

const SELECTED_PROJECT_KEY = 'vlt-bridge-selected-project-id';

interface ProjectContextValue {
  /** List of all user's projects */
  projects: Project[];
  /** Currently selected project (null if none selected or loading) */
  selectedProject: Project | null;
  /** ID of the currently selected project */
  selectedProjectId: string | null;
  /** Set the selected project by ID */
  setSelectedProjectId: (id: string | null) => void;
  /** Whether projects are being loaded */
  isLoading: boolean;
  /** Error message if project loading failed */
  error: string | null;
  /** Refresh the projects list from API */
  refreshProjects: () => Promise<void>;
  /** Create a new project and select it */
  createProject: (data: ProjectCreate) => Promise<Project>;
}

const ProjectContext = createContext<ProjectContextValue | null>(null);

interface ProjectProviderProps {
  children: ReactNode;
}

export function ProjectProvider({ children }: ProjectProviderProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectIdState] = useState<string | null>(() => {
    // Initialize from localStorage
    return localStorage.getItem(SELECTED_PROJECT_KEY);
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Derived: get the selected project object
  const selectedProject = selectedProjectId
    ? projects.find(p => p.id === selectedProjectId) ?? null
    : null;

  // Persist selected project ID to localStorage
  const setSelectedProjectId = useCallback((id: string | null) => {
    setSelectedProjectIdState(id);
    if (id) {
      localStorage.setItem(SELECTED_PROJECT_KEY, id);
    } else {
      localStorage.removeItem(SELECTED_PROJECT_KEY);
    }
  }, []);

  // Load projects from API
  const refreshProjects = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const projectList = await fetchProjects();
      setProjects(projectList);

      // If no project is selected, or the selected project no longer exists,
      // auto-select the first project or 'default'
      if (projectList.length > 0) {
        const storedId = localStorage.getItem(SELECTED_PROJECT_KEY);
        const storedExists = storedId && projectList.some(p => p.id === storedId);

        if (!storedExists) {
          // Prefer 'default' project, otherwise first project
          const defaultProject = projectList.find(p => p.id === 'default');
          const targetProject = defaultProject || projectList[0];
          setSelectedProjectId(targetProject.id);
        }
      }
    } catch (err) {
      console.error('Failed to load projects:', err);
      setError(err instanceof Error ? err.message : 'Failed to load projects');
      // Create a fallback 'default' project in memory so UI doesn't break
      const fallbackProject: Project = {
        id: 'default',
        name: 'Default Project',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };
      setProjects([fallbackProject]);
      setSelectedProjectId('default');
    } finally {
      setIsLoading(false);
    }
  }, [setSelectedProjectId]);

  // Create a new project
  const createProject = useCallback(async (data: ProjectCreate): Promise<Project> => {
    const project = await apiCreateProject(data);
    // Refresh the list and select the new project
    await refreshProjects();
    setSelectedProjectId(project.id);
    return project;
  }, [refreshProjects, setSelectedProjectId]);

  // Load projects on mount
  useEffect(() => {
    refreshProjects();
  }, [refreshProjects]);

  const value: ProjectContextValue = {
    projects,
    selectedProject,
    selectedProjectId,
    setSelectedProjectId,
    isLoading,
    error,
    refreshProjects,
    createProject,
  };

  return (
    <ProjectContext.Provider value={value}>
      {children}
    </ProjectContext.Provider>
  );
}

/**
 * Hook to access project context.
 * Must be used within a ProjectProvider.
 */
export function useProjectContext(): ProjectContextValue {
  const context = useContext(ProjectContext);
  if (!context) {
    throw new Error('useProjectContext must be used within a ProjectProvider');
  }
  return context;
}
