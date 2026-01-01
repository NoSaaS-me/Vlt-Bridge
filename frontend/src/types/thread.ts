/**
 * Thread type definitions for vlt thread sync.
 * Threads are reasoning chains synced from vlt-cli.
 */

/**
 * Thread entry - a single thought/log in a thread.
 */
export interface ThreadEntry {
  sequence_id: number;
  content: string;
  author?: string;
  created_at: string;
}

/**
 * Complete thread representation.
 */
export interface Thread {
  thread_id: string;
  project_id: string;
  name: string;
  status: 'active' | 'archived' | 'blocked';
  created_at: string;
  updated_at: string;
  entry_count?: number;
  entries?: ThreadEntry[];
}

/**
 * Thread list response from API.
 */
export interface ThreadListResponse {
  threads: Thread[];
  total: number;
}
