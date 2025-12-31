import { useState, useRef, useEffect, useMemo, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Send, Loader2, Info, Square } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ChatMessage } from './ChatMessage';
import { SlashCommandMenu } from './SlashCommandMenu';
import { streamOracle, cancelOracle, exportConversationAsMarkdown, downloadAsFile, compactHistory } from '@/services/oracle';
import { getModelSettings } from '@/services/models';
import type { OracleMessage, SlashCommand, OracleStreamChunk, SourceType } from '@/types/oracle';
import type { ModelSettings } from '@/types/models';
import { useToast } from '@/hooks/useToast';
import { Badge } from '@/components/ui/badge';

interface ChatPanelProps {
  onNavigateToNote: (path: string) => void;
  onNotesChanged?: () => void;
}

export function ChatPanel({ onNavigateToNote, onNotesChanged }: ChatPanelProps) {
  const navigate = useNavigate();
  const [messages, setMessages] = useState<OracleMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [modelSettings, setModelSettings] = useState<ModelSettings | null>(null);
  const [showThinking, setShowThinking] = useState(true);
  const [showSources, setShowSources] = useState(true);
  const [activeSources, setActiveSources] = useState<SourceType[]>(['vault', 'code', 'threads']);
  const [showCommandMenu, setShowCommandMenu] = useState(false);
  const [commandFilter, setCommandFilter] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const toast = useToast();

  // Load model settings on mount
  useEffect(() => {
    const loadModelSettings = async () => {
      try {
        const settings = await getModelSettings();
        setModelSettings(settings);
      } catch (err) {
        console.error('Failed to load model settings:', err);
      }
    };
    loadModelSettings();
  }, []);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  /**
   * Stop the current Oracle query.
   * Aborts the frontend fetch request and cancels the backend session.
   * Defined as useCallback so it can be used in slashCommands.
   */
  const handleStop = useCallback(async () => {
    // Abort the frontend request first (faster feedback)
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Cancel backend session (stops all running agents)
    try {
      await cancelOracle();
    } catch (err) {
      console.error('Failed to cancel Oracle session:', err);
      // Don't show error - frontend abort already worked
    }

    setIsLoading(false);
    setStatusMessage('');
    toast.info('Query stopped');
  }, [toast]);

  // Slash commands
  const slashCommands = useMemo<SlashCommand[]>(
    () => [
      {
        name: 'clear',
        description: 'Clear conversation history',
        shortcut: 'Ctrl+K',
        handler: () => {
          setMessages([]);
          setInput('');
          toast.success('Conversation cleared');
        },
      },
      {
        name: 'compact',
        description: 'Summarize and compress conversation',
        handler: async () => {
          const compressed = await compactHistory(messages);
          setMessages(compressed);
          toast.success('Conversation compacted');
        },
      },
      {
        name: 'context',
        description: 'Show current context sources being used',
        handler: () => {
          const contextInfo = `Active sources: ${activeSources.join(', ')}`;
          toast.info(contextInfo);
        },
      },
      {
        name: 'help',
        description: 'Show available slash commands',
        handler: () => {
          setShowCommandMenu(true);
          setCommandFilter('');
        },
      },
      {
        name: 'models',
        description: `Configure AI model (current: ${modelSettings?.oracle_model?.split('/').pop() || 'default'})`,
        handler: () => {
          navigate('/settings');
          toast.info('Navigate to Settings → AI Models to change model');
        },
      },
      {
        name: 'sources',
        description: 'Toggle source display on/off',
        handler: () => {
          setShowSources((prev) => !prev);
          toast.success(`Sources ${!showSources ? 'shown' : 'hidden'}`);
        },
      },
      {
        name: 'thinking',
        description: 'Toggle thinking mode display',
        handler: () => {
          setShowThinking((prev) => !prev);
          toast.success(`Thinking ${!showThinking ? 'shown' : 'hidden'}`);
        },
      },
      {
        name: 'export',
        description: 'Export conversation as markdown',
        handler: () => {
          const markdown = exportConversationAsMarkdown(messages);
          const filename = `oracle-conversation-${new Date().toISOString().split('T')[0]}.md`;
          downloadAsFile(markdown, filename);
          toast.success('Conversation exported');
        },
      },
      {
        name: 'stop',
        description: 'Stop current query (if running)',
        shortcut: 'Esc',
        handler: () => {
          if (isLoading) {
            handleStop();
          } else {
            toast.info('No active query to stop');
          }
        },
      },
    ],
    [messages, activeSources, showSources, showThinking, toast, modelSettings, navigate, isLoading, handleStop]
  );

  const handleSubmit = async () => {
    if (!input.trim() || isLoading) return;

    const trimmedInput = input.trim();

    // Handle slash commands
    if (trimmedInput.startsWith('/')) {
      const commandName = trimmedInput.substring(1).split(' ')[0];
      const command = slashCommands.find((cmd) => cmd.name === commandName);

      if (command) {
        command.handler();
        setInput('');
        setShowCommandMenu(false);
        return;
      }
    }

    const userMsg: OracleMessage = {
      role: 'user',
      content: trimmedInput,
      timestamp: new Date().toISOString(),
    };

    // Add user message
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);
    setStatusMessage('Searching...');

    // Create assistant message placeholder
    const assistantMsg: OracleMessage = {
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
      thinking: '',
      sources: [],
    };

    setMessages((prev) => [...prev, assistantMsg]);

    // Create abort controller for this request
    abortControllerRef.current = new AbortController();

    try {
      await streamOracle(
        {
          question: trimmedInput,
          sources: activeSources,
          max_results: 10,
        },
        (chunk: OracleStreamChunk) => {
          setMessages((prev) => {
            const updated = [...prev];
            const lastMsg = updated[updated.length - 1];

            if (lastMsg.role === 'assistant') {
              if (chunk.type === 'status') {
                setStatusMessage(chunk.message || '');
              } else if (chunk.type === 'thinking') {
                lastMsg.thinking = (lastMsg.thinking || '') + (chunk.content || '');
              } else if (chunk.type === 'content') {
                lastMsg.content += chunk.content || '';
              } else if (chunk.type === 'source' && chunk.source) {
                if (!lastMsg.sources) lastMsg.sources = [];
                lastMsg.sources.push(chunk.source);
              } else if (chunk.type === 'done') {
                lastMsg.model = chunk.model;
                setStatusMessage('');
              } else if (chunk.type === 'error') {
                lastMsg.is_error = true;
                lastMsg.content = chunk.error || 'Unknown error occurred';
              }
            }

            return updated;
          });
        },
        abortControllerRef.current.signal
      );
    } catch (err) {
      // Check if this was a user-initiated abort (stop button)
      if (err instanceof Error && err.name === 'AbortError') {
        // User cancelled - update the message to show it was stopped
        setMessages((prev) => {
          const updated = [...prev];
          const lastMsg = updated[updated.length - 1];
          if (lastMsg.role === 'assistant' && !lastMsg.content) {
            lastMsg.content = 'Query stopped by user.';
          }
          return updated;
        });
        return; // Don't show error toast for user-initiated stop
      }

      console.error('Oracle error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Failed to get response';

      setMessages((prev) => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg.role === 'assistant') {
          lastMsg.is_error = true;
          lastMsg.content = errorMessage;
        }
        return updated;
      });

      toast.error(errorMessage);
    } finally {
      setIsLoading(false);
      setStatusMessage('');
      abortControllerRef.current = null;
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setInput(value);

    // Show command menu when typing slash
    if (value.startsWith('/') && value.length > 1) {
      setShowCommandMenu(true);
      setCommandFilter(value.substring(1));
    } else {
      setShowCommandMenu(false);
      setCommandFilter('');
    }
  };

  const handleCommandSelect = (command: SlashCommand) => {
    command.handler();
    setInput('');
    setShowCommandMenu(false);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-semibold">Vlt Oracle</h2>
            <p className="text-xs text-muted-foreground">
              Multi-source intelligent context retrieval
            </p>
          </div>
          <div className="flex items-center gap-2">
            {modelSettings?.oracle_model && (
              <Badge
                variant="outline"
                className="text-xs cursor-pointer hover:bg-accent"
                onClick={() => navigate('/settings')}
                title="Click to change model"
              >
                {modelSettings.oracle_model.split('/').pop()?.replace(':free', '') || 'default'}
              </Badge>
            )}
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={() => setShowCommandMenu(true)}
              title="Show commands"
            >
              <Info className="h-4 w-4" />
            </Button>
          </div>
        </div>
        {activeSources.length < 3 && (
          <div className="mt-2 flex gap-1">
            <span className="text-xs text-muted-foreground">Active sources:</span>
            {activeSources.map((source) => (
              <Badge key={source} variant="secondary" className="text-xs">
                {source}
              </Badge>
            ))}
          </div>
        )}
      </div>

      {/* Message List */}
      <div className="flex-1 overflow-y-auto relative" ref={scrollRef}>
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-8 text-center">
            <p className="text-base">Ask Oracle anything about your project</p>
            <p className="text-sm mt-2">
              Try: "How does authentication work?" or type <kbd className="px-2 py-1 bg-muted rounded text-xs">/help</kbd>
            </p>
          </div>
        ) : (
          <div className="divide-y divide-border/50">
            {messages.map((msg, i) => (
              <ChatMessage
                key={i}
                message={msg}
                onSourceClick={onNavigateToNote}
                showThinking={showThinking}
                showSources={showSources}
              />
            ))}
            {isLoading && statusMessage && (
              <div className="p-4 flex items-center gap-2 text-muted-foreground text-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                {statusMessage}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 pb-14 border-t border-border relative">
        {showCommandMenu && (
          <SlashCommandMenu
            commands={slashCommands}
            onSelect={handleCommandSelect}
            onClose={() => setShowCommandMenu(false)}
            filterText={commandFilter}
            position={{ top: -20, left: 0 }}
          />
        )}
        <div className="flex gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question or type / for commands..."
            className="min-h-[40px] max-h-[150px] resize-none"
            rows={1}
            disabled={isLoading}
          />
          {isLoading ? (
            <Button
              onClick={handleStop}
              variant="destructive"
              size="icon"
              title="Stop query"
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button onClick={handleSubmit} disabled={!input.trim()} size="icon">
              <Send className="h-4 w-4" />
            </Button>
          )}
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            Type <kbd className="px-1 py-0.5 bg-muted rounded">/</kbd> for commands
          </span>
          {messages.length > 0 && (
            <span>{messages.length} messages</span>
          )}
        </div>
      </div>
    </div>
  );
}
