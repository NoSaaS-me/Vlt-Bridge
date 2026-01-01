export type SourceType = 'vault' | 'code' | 'threads';
export type StreamEventType = 'status' | 'thinking' | 'content' | 'source' | 'tool_call' | 'tool_result' | 'done' | 'error';

/**
 * Oracle API request payload
 */
export interface OracleRequest {
  question: string;
  sources?: SourceType[];
  explain?: boolean;
  max_results?: number;
  model?: string;            // Override LLM model (e.g., 'deepseek/deepseek-chat')
  thinking?: boolean;        // Enable thinking mode
  max_tokens?: number;       // Maximum tokens for context assembly
  context_id?: string;       // Context tree node ID for conversation continuity
}

/**
 * Retrieval result from any knowledge source
 */
export interface RetrievalResult {
  content: string;
  source_type: SourceType;
  source_path: string;
  score: number;
  metadata?: Record<string, unknown>;
}

/**
 * Stream event chunks from Oracle SSE endpoint
 */
export interface OracleStreamChunk {
  type: StreamEventType;
  message?: string;          // for status events
  content?: string;          // for content/thinking tokens
  source?: RetrievalResult;  // for source events
  tokens_used?: number;      // for done event
  duration_ms?: number;      // for done event
  model_used?: string;       // for done event (matches backend)
  context_id?: string;       // for done event - context node ID for next request
  error?: string;            // for error events
  tool_call?: {              // for tool_call events
    id: string;
    name: string;
    arguments: string;
    status?: string;
  };
  tool_call_id?: string;     // for tool_result events - associates result with tool call
  tool_result?: string;      // for tool_result events
}

/**
 * Slash command definition
 */
export interface SlashCommand {
  name: string;
  description: string;
  shortcut?: string;
  handler: () => void | Promise<void>;
}

/**
 * Tool call with execution result
 */
export interface ToolCallInfo {
  id: string;
  name: string;
  arguments: string;
  status?: 'pending' | 'running' | 'completed' | 'error';
  result?: string;
}

/**
 * Oracle conversation message (extends ChatMessage)
 */
export interface OracleMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  thinking?: string;         // Optional thinking/reasoning content
  sources?: RetrievalResult[];
  tool_calls?: ToolCallInfo[]; // Tool calls made during response
  model?: string;
  is_error?: boolean;
}

/**
 * Oracle session settings
 */
export interface OracleSettings {
  model?: string;
  sources?: SourceType[];
  show_thinking?: boolean;
  show_sources?: boolean;
  max_results?: number;
}
