/**
 * Tag with aggregated note count.
 */
export interface Tag {
  tag_name: string;
  count: number;
}

/**
 * Index health metadata per user.
 */
export interface IndexHealth {
  user_id: string;
  note_count: number;
  last_full_rebuild: string | null;
  last_incremental_update: string | null;
}

/**
 * Full-text search result entry.
 */
export interface SearchResult {
  note_path: string;
  title: string;
  snippet: string;
  score: number;
  updated: string; // ISO 8601 timestamp
}

