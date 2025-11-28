export type Role = 'user' | 'assistant';

export interface SourceReference {
  path: string;
  title: string;
  snippet: string;
  score?: number;
}

export interface NoteWritten {
  path: string;
  title: string;
  action: 'created' | 'updated';
}

export interface ChatMessage {
  role: Role;
  content: string;
  timestamp: string; // ISO 8601
  sources?: SourceReference[];
  notes_written?: NoteWritten[];
  is_error?: boolean; // Frontend-only state for error messages
}

export interface ChatRequest {
  messages: ChatMessage[];
}

export interface ChatResponse {
  answer: string;
  sources: SourceReference[];
  notes_written: NoteWritten[];
}

export interface StatusResponse {
  status: 'ready' | 'building' | 'error';
  doc_count: number;
  last_updated: string | null;
}

export interface ErrorResponse {
  error: string;
  detail?: string;
}
