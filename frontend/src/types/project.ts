/**
 * Project type definitions for multi-project architecture.
 * A project represents an isolated workspace with its own vault, threads, and indexes.
 */

/**
 * Complete project representation returned by API.
 */
export interface Project {
  id: string;           // slug: "vlt-bridge"
  name: string;         // display: "Vlt Bridge"
  description?: string;
  created_at: string;   // ISO 8601 timestamp
  updated_at: string;   // ISO 8601 timestamp
  note_count?: number;
  thread_count?: number;
}

/**
 * Request payload for creating a project.
 */
export interface ProjectCreate {
  name: string;
  description?: string;
}

/**
 * Request payload for updating a project.
 */
export interface ProjectUpdate {
  name?: string;
  description?: string;
}

/**
 * Project statistics response.
 */
export interface ProjectStats {
  note_count: number;
  thread_count: number;
  index_health?: {
    last_full_rebuild?: string;
    last_incremental_update?: string;
  };
}
