import type { ChatMessage as ChatMessageType } from '@/types/rag';
import type { OracleMessage } from '@/types/oracle';
import { cn } from '@/lib/utils';
import { User, Bot, FilePlus, Edit, RefreshCw, ChevronDown, ChevronUp, Brain, FileCode, BookOpen, MessageSquare, Wrench, CheckCircle, Loader2, AlertCircle, Clock, Copy, Check } from 'lucide-react';
import { SourceList } from './SourceList';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { useState, useMemo, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';
import { createWikilinkComponent } from '@/lib/markdown';

interface ChatMessageProps {
  message: ChatMessageType | OracleMessage;
  onSourceClick: (path: string) => void;
  onRefreshNeeded?: () => void;
  showThinking?: boolean;
  showSources?: boolean;
  isStreaming?: boolean; // Whether this message is currently being streamed
}

/**
 * Renders a single chat message with support for:
 * - Thinking/reasoning traces (collapsible, live-streaming)
 * - Tool calls with status indicators and expandable details
 * - LaTeX math rendering (inline $...$ and block $$...$$)
 * - Markdown with GFM support
 * - Sources and citations
 */
export function ChatMessage({
  message,
  onSourceClick,
  onRefreshNeeded,
  showThinking = true,
  showSources = true,
  isStreaming = false,
}: ChatMessageProps) {
  const isUser = message.role === 'user';
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [thinkingExpanded, setThinkingExpanded] = useState(false);
  const [toolsExpanded, setToolsExpanded] = useState(true); // Default expanded to show progress
  const [expandedToolIds, setExpandedToolIds] = useState<Set<string>>(new Set());
  const [copiedToolId, setCopiedToolId] = useState<string | null>(null);

  // Create markdown components for rendering with wikilink support
  const markdownComponents = useMemo(() => createWikilinkComponent(onSourceClick), [onSourceClick]);

  // Pre-process markdown to convert wikilinks to standard links
  // [[Link]] -> [Link](wikilink:Link)
  // [[Link|Alias]] -> [Alias](wikilink:Link)
  const processWikilinks = (text: string | undefined): string => {
    if (!text) return '';
    return text.replace(/\[\[([^\]]+)\]\]/g, (_match, content) => {
      const [link, alias] = content.split('|');
      const displayText = alias || link;
      const href = `wikilink:${encodeURIComponent(link)}`;
      return `[${displayText}](${href})`;
    });
  };

  const processedContent = useMemo(() => processWikilinks(message.content), [message.content]);

  // Type guard to check if message is OracleMessage
  const isOracleMessage = (msg: ChatMessageType | OracleMessage): msg is OracleMessage => {
    return 'thinking' in msg || 'model' in msg;
  };

  const processedThinking = useMemo(() => {
    const oMsg = isOracleMessage(message) ? message : null;
    return processWikilinks(oMsg?.thinking);
  }, [message]);

  const oracleMsg = isOracleMessage(message) ? message : null;
  const hasError = Boolean(message.is_error);

  // Auto-expand thinking section when there's new thinking content during streaming
  // Show thinking even if there's an error - user should see what was thought before the error
  const hasThinking = Boolean(oracleMsg?.thinking && oracleMsg.thinking.length > 0);
  const hasContent = Boolean(message.content && message.content.length > 0);
  const hasToolCalls = Boolean(oracleMsg?.tool_calls && oracleMsg.tool_calls.length > 0);
  // Thinking is "active" when streaming and no content yet - but keep showing it on error
  const isThinkingActive = hasThinking && !hasContent && isStreaming && !hasError;

  // Auto-expand thinking while streaming, or when there's an error (so user can see what led to it)
  useEffect(() => {
    if (isThinkingActive) {
      setThinkingExpanded(true);
    }
    // Also expand thinking on error so user can debug what happened
    if (hasError && hasThinking) {
      setThinkingExpanded(true);
    }
  }, [isThinkingActive, hasError, hasThinking]);

  // Toggle individual tool expansion
  const toggleToolExpanded = (toolId: string) => {
    setExpandedToolIds(prev => {
      const next = new Set(prev);
      if (next.has(toolId)) {
        next.delete(toolId);
      } else {
        next.add(toolId);
      }
      return next;
    });
  };

  // Copy tool result to clipboard
  const copyToolResult = useCallback(async (toolId: string, result: string) => {
    try {
      await navigator.clipboard.writeText(result);
      setCopiedToolId(toolId);
      setTimeout(() => setCopiedToolId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, []);

  // Get tool status icon with appropriate styling
  const getToolStatusIcon = (status?: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-3.5 w-3.5 text-green-500" />;
      case 'running':
        return <Loader2 className="h-3.5 w-3.5 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="h-3.5 w-3.5 text-red-500" />;
      case 'pending':
        return <Clock className="h-3.5 w-3.5 text-muted-foreground" />;
      default:
        return <Loader2 className="h-3.5 w-3.5 text-muted-foreground animate-spin" />;
    }
  };

  // Get status badge color
  const getStatusBadgeVariant = (status?: string): 'default' | 'secondary' | 'destructive' | 'outline' => {
    switch (status) {
      case 'completed':
        return 'secondary';
      case 'running':
        return 'default';
      case 'error':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      console.log('[ChatMessage] Manual refresh triggered');
      if (onRefreshNeeded) {
        await onRefreshNeeded();
        console.log('[ChatMessage] Refresh completed');
      } else {
        console.error('[ChatMessage] onRefreshNeeded is undefined');
      }
    } catch (err) {
      console.error('[ChatMessage] Refresh failed:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Get source icon based on source type
  const getSourceIcon = (sourceType: string) => {
    switch (sourceType) {
      case 'vault':
        return <BookOpen className="h-3 w-3" />;
      case 'code':
        return <FileCode className="h-3 w-3" />;
      case 'thread':
      case 'threads':
        return <MessageSquare className="h-3 w-3" />;
      default:
        return <FileCode className="h-3 w-3" />;
    }
  };

  // Format JSON for display
  const formatJson = (jsonString: string): string => {
    try {
      return JSON.stringify(JSON.parse(jsonString), null, 2);
    } catch {
      return jsonString;
    }
  };

  // Truncate long text with ellipsis
  const truncateText = (text: string, maxLength: number): string => {
    if (text.length <= maxLength) return text;
    return text.slice(0, maxLength) + '...';
  };

  // Count running/completed tools
  const toolStats = useMemo(() => {
    if (!oracleMsg?.tool_calls) return { running: 0, completed: 0, total: 0 };
    const running = oracleMsg.tool_calls.filter(tc => tc.status === 'running').length;
    const completed = oracleMsg.tool_calls.filter(tc => tc.status === 'completed').length;
    return { running, completed, total: oracleMsg.tool_calls.length };
  }, [oracleMsg?.tool_calls]);

  return (
    <div className={cn("flex gap-3 p-4", isUser ? "bg-transparent" : "bg-muted/30")}>
      {/* Avatar */}
      <div className={cn(
        "h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1",
        isUser ? "bg-primary text-primary-foreground" : "bg-secondary text-secondary-foreground"
      )}>
        {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
      </div>

      <div className="flex-1 space-y-3 overflow-hidden min-w-0">
        {/* ===== THINKING SECTION ===== */}
        {!isUser && oracleMsg && hasThinking && showThinking && (
          <div className={cn(
            "border rounded-lg overflow-hidden transition-all duration-200",
            isThinkingActive
              ? "border-blue-500/50 bg-blue-500/5 shadow-sm shadow-blue-500/10"
              : hasError
                ? "border-destructive/50 bg-destructive/5" // Red border on error to show thinking context
                : "border-border bg-muted/20"
          )}>
            <button
              onClick={() => setThinkingExpanded(!thinkingExpanded)}
              className="flex items-center gap-2 w-full px-3 py-2 hover:bg-muted/40 transition-colors"
            >
              <Brain className={cn(
                "h-4 w-4 transition-colors",
                isThinkingActive ? "text-blue-500 animate-pulse" : "text-muted-foreground"
              )} />
              <span className={cn(
                "text-sm font-medium flex-1 text-left",
                isThinkingActive ? "text-blue-600 dark:text-blue-400"
                  : hasError ? "text-destructive/80"
                  : "text-muted-foreground"
              )}>
                {isThinkingActive ? 'Thinking...' : hasError ? 'Reasoning (before error)' : 'Reasoning'}
              </span>
              {/* Show thinking preview when collapsed */}
              {!thinkingExpanded && oracleMsg.thinking && (
                <span className="text-xs text-muted-foreground truncate max-w-[200px]">
                  {truncateText(oracleMsg.thinking.split('\n')[0], 50)}
                </span>
              )}
              {thinkingExpanded ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground flex-shrink-0" />
              )}
            </button>
            {thinkingExpanded && (
              <div className={cn(
                "px-3 py-2 border-t transition-colors",
                isThinkingActive ? "border-blue-500/30" : "border-border"
              )}>
                <div className="prose dark:prose-invert prose-sm max-w-none text-muted-foreground prose-p:my-1 prose-headings:my-2">
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm, remarkMath]}
                    rehypePlugins={[rehypeKatex]}
                    components={markdownComponents}
                    urlTransform={(url) => url} // Allow wikilink: protocol
                  >
                    {processedThinking}
                  </ReactMarkdown>
                  {/* Streaming cursor */}
                  {isThinkingActive && (
                    <span className="inline-block w-2 h-4 bg-blue-500 animate-pulse ml-0.5" />
                  )}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ===== TOOL CALLS SECTION ===== */}
        {!isUser && hasToolCalls && (
          <div className="border border-border rounded-lg overflow-hidden bg-muted/10">
            <button
              onClick={() => setToolsExpanded(!toolsExpanded)}
              className="flex items-center gap-2 w-full px-3 py-2 hover:bg-muted/40 transition-colors"
            >
              <Wrench className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground flex-1 text-left">
                {toolStats.total} tool{toolStats.total > 1 ? 's' : ''}
              </span>
              {/* Status summary */}
              <div className="flex items-center gap-1.5">
                {toolStats.running > 0 && (
                  <Badge variant="default" className="text-xs px-1.5 py-0 h-5 gap-1">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    {toolStats.running}
                  </Badge>
                )}
                {toolStats.completed > 0 && (
                  <Badge variant="secondary" className="text-xs px-1.5 py-0 h-5 gap-1">
                    <CheckCircle className="h-3 w-3" />
                    {toolStats.completed}
                  </Badge>
                )}
              </div>
              {toolsExpanded ? (
                <ChevronUp className="h-4 w-4 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-4 w-4 text-muted-foreground" />
              )}
            </button>
            {toolsExpanded && oracleMsg?.tool_calls && (
              <div className="border-t border-border">
                {oracleMsg.tool_calls.map((tool) => (
                  <div
                    key={tool.id}
                    className={cn(
                      "border-b border-border/50 last:border-b-0",
                      tool.status === 'running' && "bg-blue-500/5"
                    )}
                  >
                    {/* Tool Header */}
                    <button
                      onClick={() => toggleToolExpanded(tool.id)}
                      className="flex items-center gap-2 w-full px-3 py-2 hover:bg-muted/20 transition-colors"
                    >
                      {getToolStatusIcon(tool.status)}
                      <code className="text-xs font-mono text-primary font-medium">
                        {tool.name}
                      </code>
                      {/* Brief argument preview */}
                      {tool.arguments && !expandedToolIds.has(tool.id) && (
                        <span className="text-xs text-muted-foreground truncate max-w-[150px] font-mono">
                          {truncateText(tool.arguments, 30)}
                        </span>
                      )}
                      <span className="flex-1" />
                      <Badge variant={getStatusBadgeVariant(tool.status)} className="text-xs px-1.5 py-0">
                        {tool.status || 'pending'}
                      </Badge>
                      {expandedToolIds.has(tool.id) ? (
                        <ChevronUp className="h-3 w-3 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="h-3 w-3 text-muted-foreground" />
                      )}
                    </button>

                    {/* Tool Details */}
                    {expandedToolIds.has(tool.id) && (
                      <div className="px-3 py-2 bg-muted/30 space-y-2">
                        {/* Arguments */}
                        <div>
                          <div className="text-xs font-medium text-muted-foreground mb-1 flex items-center gap-1">
                            <span>Arguments</span>
                          </div>
                          <pre className="text-xs bg-background p-2 rounded border border-border overflow-x-auto max-h-32 font-mono">
                            {formatJson(tool.arguments)}
                          </pre>
                        </div>

                        {/* Result */}
                        {tool.result && (
                          <div>
                            <div className="text-xs font-medium text-muted-foreground mb-1 flex items-center justify-between">
                              <span>Result</span>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-5 px-1.5 text-xs"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  copyToolResult(tool.id, tool.result!);
                                }}
                              >
                                {copiedToolId === tool.id ? (
                                  <Check className="h-3 w-3 text-green-500" />
                                ) : (
                                  <Copy className="h-3 w-3" />
                                )}
                              </Button>
                            </div>
                            <pre className="text-xs bg-background p-2 rounded border border-border overflow-x-auto max-h-48 whitespace-pre-wrap font-mono">
                              {tool.result.length > 2000
                                ? tool.result.slice(0, 2000) + '\n... [truncated]'
                                : tool.result}
                            </pre>
                          </div>
                        )}

                        {/* Running indicator */}
                        {tool.status === 'running' && !tool.result && (
                          <div className="flex items-center gap-2 text-xs text-blue-600 dark:text-blue-400">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            <span>Executing...</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ===== MAIN CONTENT ===== */}
        <div className={cn(
          "prose dark:prose-invert prose-sm max-w-none",
          "prose-pre:bg-muted prose-pre:text-foreground",
          "prose-code:text-primary prose-code:before:content-none prose-code:after:content-none",
          "prose-p:my-2 prose-headings:my-3",
          // LaTeX styling
          "prose-math:text-foreground",
          message.is_error && "text-destructive"
        )}>
          {processedContent ? (
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={markdownComponents}
              urlTransform={(url) => url} // Allow all protocols including wikilink:
            >
              {processedContent}
            </ReactMarkdown>
          ) : (
            isUser ? null : (
              isStreaming ? (
                <span className="inline-block w-2 h-4 bg-primary animate-pulse" />
              ) : (
                <span className="text-muted-foreground">...</span>
              )
            )
          )}
        </div>

        {/* ===== MODEL BADGE ===== */}
        {!isUser && oracleMsg?.model && (
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs font-normal">
              {oracleMsg.model}
            </Badge>
          </div>
        )}

        {/* ===== NOTES WRITTEN (RAG legacy) ===== */}
        {!isUser && 'notes_written' in message && message.notes_written && message.notes_written.length > 0 && (
          <div className="space-y-2">
            <div className="flex flex-wrap gap-2">
              {message.notes_written.map((note, i) => (
                <button
                  key={i}
                  onClick={() => onSourceClick(note.path)}
                  className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-green-500/10 text-green-600 dark:text-green-400 border border-green-500/20 hover:bg-green-500/20 text-xs transition-colors"
                >
                  {note.action === 'created' ? <FilePlus className="h-3 w-3" /> : <Edit className="h-3 w-3" />}
                  <span className="font-medium">
                    {note.action === 'created' ? 'Created' : 'Updated'}: {note.title}
                  </span>
                </button>
              ))}
            </div>
            <Button
              onClick={handleRefresh}
              disabled={isRefreshing}
              size="sm"
              variant="outline"
              className="text-xs h-7"
            >
              <RefreshCw className={cn("h-3 w-3 mr-1.5", isRefreshing && "animate-spin")} />
              {isRefreshing ? 'Refreshing...' : 'Refresh Views'}
            </Button>
          </div>
        )}

        {/* ===== ORACLE SOURCES ===== */}
        {!isUser && showSources && oracleMsg?.sources && oracleMsg.sources.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs font-medium text-muted-foreground">Sources</div>
            <div className="flex flex-wrap gap-2">
              {oracleMsg.sources.map((source, i) => (
                <button
                  key={i}
                  onClick={() => onSourceClick(source.source_path)}
                  className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 text-xs transition-colors group"
                  title={`${source.source_type}: ${source.source_path}${source.score != null ? ` (score: ${source.score.toFixed(3)})` : ''}`}
                >
                  {getSourceIcon(source.source_type)}
                  <span className="font-medium">{source.source_type}</span>
                  <span className="text-muted-foreground group-hover:text-blue-600 dark:group-hover:text-blue-400">
                    {source.source_path.split('/').pop()}
                  </span>
                  {source.score != null && (
                    <Badge variant="secondary" className="text-xs px-1 py-0">
                      {source.score.toFixed(2)}
                    </Badge>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ===== LEGACY RAG SOURCES ===== */}
        {!isUser && showSources && !oracleMsg && message.sources && (
          <SourceList sources={message.sources} onSourceClick={onSourceClick} />
        )}
      </div>
    </div>
  );
}
