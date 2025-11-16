/**
 * Note metadata/frontmatter representation.
 */
export interface NoteMetadata {
  title?: string;
  tags?: string[];
  project?: string;
  created?: string; // ISO 8601 timestamp
  updated?: string; // ISO 8601 timestamp
  [key: string]: unknown;
}

/**
 * Complete note payload returned by APIs.
 */
export interface Note {
  user_id: string;
  note_path: string;
  version: number;
  title: string;
  metadata: NoteMetadata;
  body: string;
  created: string; // ISO 8601 timestamp
  updated: string; // ISO 8601 timestamp
  size_bytes: number;
}

/**
 * Lightweight summary used for listings.
 */
export interface NoteSummary {
  note_path: string;
  title: string;
  updated: string; // ISO 8601 timestamp
}

/**
 * Request payload for creating a note.
 */
export interface NoteCreateRequest {
  note_path: string;
  title?: string;
  metadata?: NoteMetadata;
  body: string;
}

/**
 * Request payload for updating a note.
 */
export interface NoteUpdateRequest {
  title?: string;
  metadata?: NoteMetadata;
  body: string;
  if_version?: number;
}

/**
 * Wikilink and backlink representation.
 */
export interface Wikilink {
  user_id: string;
  source_path: string;
  target_path: string | null;
  link_text: string;
  is_resolved: boolean;
}

